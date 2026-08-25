import os
import os.path
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import pytorch_lightning as lightning
from utilities.logger import TrainerLogger
from utilities.instantiators import instantiate, instantiate_list
from utilities.logic import get_config_path
# Force full FP32 matmul on CUDA (disable TF32) for more reproducible numerics
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Initialize logger
logger = logging.getLogger(__name__)


def _accumulate_output(batch_output: list, keys: list | None = None,
                       config_stage: DictConfig | None = None) -> dict:
    """ Accumulate results from individual batches into concatenated arrays.

        Efficiently accumulates specified keys from batch results with single-pass iteration,
        recursive concatenation handling nested dict structures. Optionally applies inverse
        transformations to all accumulated keys based on stage-specific configuration.

        Parameters
        ----------
        batch_output: list. List of dicts returned by trainer.test() or trainer.predict().
                       Each dict contains keys like 'output', 'latent', 'mask', etc.
        keys: list or None. List of keys to accumulate. If None, defaults to ['output', 'latent', 'mask'].
        config_stage: DictConfig or None. Stage-specific loader configuration containing transformations for each key.

        Returns
        -------
        dict. Dictionary with accumulated results (numpy arrays concatenated across batches).
              Inverse transformations applied to denormalize data.
              Example: {'output': {'bt_forward': array_denormalized, 'prof': array_denormalized}, ...}
    """

    # Default: Accumulate common keys
    if keys is None:
        keys = ['output', 'latent', 'mask']

    # Pre-initialize accumulator for all keys (single-pass efficiency)
    accumulated = {key: [] for key in keys}

    # Single pass: collect specified keys
    for batch_result in batch_output:
        for key in keys:
            if key in batch_result:
                accumulated[key].append(batch_result[key])
    # Remove empty keys
    accumulated = {k: v for k, v in accumulated.items() if v}

    # Recursively concatenate nested lists of arrays
    def concat_recursive(lst):
        """Recursively concatenate nested lists of arrays or dicts."""

        # If it's a list of dicts, concatenate each key
        if isinstance(lst[0], dict):
            result = {}
            for key in lst[0].keys():
                values = [item[key] for item in lst]
                result[key] = concat_recursive(values)
            return result
        # If it's a list of arrays, concatenate
        elif isinstance(lst[0], np.ndarray):
            return np.concatenate(lst)
        # Otherwise return as-is (scalars, etc.)
        else:
            return lst

    return {key: concat_recursive(value) for key, value in accumulated.items()}


def _save_output(output: dict, config_stage: DictConfig) -> None:
    """ Save output to file(s).

        Parameters
        ----------
        output: dict. Each output dict contains keys like 'output', 'latent', 'mask', etc.
        config_stage: DictConfig or None. Stage-specific loader configuration containing transformations for each key.

        Returns
        -------
        None.
    """

    # Loop over types of outputs
    for key, data in output.items():
        if hasattr(config_stage, key):
            key_config = getattr(config_stage, key)
            # Handle nested dict structure containing variables
            if isinstance(data, dict):
                for var_name, var_data in data.items():
                    if var_name in key_config and hasattr(key_config[var_name], 'save'):
                        save_config = key_config[var_name].save
                        if isinstance(save_config, DictConfig) and 'path' in save_config:
                            os.makedirs(os.path.dirname(save_config.path), exist_ok=True)
                        save_function = instantiate(save_config)
                        save_function(var_data)
            # Handle flat array structure (single variable)
            elif hasattr(key_config, 'save'):
                save_config = key_config.save
                if isinstance(save_config, DictConfig) and 'path' in save_config:
                    os.makedirs(os.path.dirname(save_config.path), exist_ok=True)
                save_function = instantiate(save_config)
                save_function(data)
    return


class Operator:
    """Class for training a neural network operator."""
    def __init__(self, config: DictConfig) -> None:
        """ Initialization of trainer and its configuration.

            Parameters
            ----------
            config: DictConfig. Hydra configuration object.

            Returns
            -------
            None.

        """
        logger.info("Reading Hydra configuration...")

        # Load config object and resolve paths
        OmegaConf.resolve(config)
        self.config = config

        # Create output directories if they don't exist
        for directory in [self.config.paths.output_dir, self.config.paths.checkpoint_dir,
                          self.config.paths.log_dir]:
            os.makedirs(directory, exist_ok=True)

        # Initialization
        self.loader = None
        self.callbacks = None
        self.trainer_logger = None
        self.trainer = None
        self.model = None

        # For reproducibility, set randomizer seed if provided
        if self.config.get("seed"):
            lightning.seed_everything(self.config.seed, workers=True)

    def _init_trainer(self, stage: str = 'train', config_loader: DictConfig | None = None) -> None:
        """ Setup trainer object.

            Parameters
            ----------
            stage: str. Stage of the training process.
                        Options are 'train', 'test', or 'predict'.
            config_loader: DictConfig or None. Configuration object for the data loader.
                           If None, uses self.config.loader.

            Returns
            -------
            None.
        """

        # Data loader
        if config_loader is None:
            config_loader = self.config.loader
        logger.info("Initializing data loader...")
        self.loader = instantiate(config_loader)
        # Generate training/validation/test sets
        self.loader.setup(stage=stage)

        # Trainer loggers: Only activated during training
        if stage == 'train':
            # Configure logger
            if self.trainer_logger is None:
                TrainerLogger(self.config.logger).configure()
                logger.info("Initializing logger(s)...")
                self.trainer_logger = instantiate_list(self.config.get("logger"), "logger")
                logger.info("Done with loggers, Initializing callbacks(s)...")
        else:
            self.trainer_logger = None

        # Callbacks: Initialized for stages train, test
        if self.callbacks is None and stage in ('train', 'test'):
            logger.info("Initializing callbacks(s)...")
            self.callbacks = instantiate_list(self.config.get("callbacks"), "callbacks")

        # Trainer
        logger.info("Waking up trainer...")
        self.trainer = instantiate(self.config.trainer, callbacks=self.callbacks, logger=self.trainer_logger)

    def _init_model(self, ckpt_path: str | None = None, strict: bool = False) -> None:
        """ Initialize model, optionally load from checkpoint, and set dtype.

            Parameters
            ----------
            ckpt_path: str or None. Path to checkpoint to load. If None, trains from scratch.
            strict: bool. If True, require exact key match when loading checkpoint.

            Returns
            -------
            None.
        """
        logger.info("Initializing model...")
        self.model = instantiate(self.config.model)

        # Load checkpoint if provided
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint...")
            self.model.load_ckpt(ckpt_path=ckpt_path, strict=strict)

        # Set dtype
        dtype = self.config.data.get('dtype', 'float32')
        self.model = self.model.to(None, dtype=getattr(torch, dtype))
        # self.model = torch.compile(self.model, mode="reduce-overhead")

    def _run_model(self) -> dict:
        """ Run model to generate a prediction.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Evaluate on test set
        logger.info("Running model on data...")
        batch_output = self.trainer.predict(self.model, self.loader)

        # Accumulate results from all batches
        logger.info("Accumulating output...")
        batch_output = _accumulate_output(batch_output, config_stage=self.loader.predict)

        return batch_output

    def train(self) -> None:
        """ Loads data, loggers, callbacks, trainer, and then trains and tests the model.
            Saves the training weights and biases in a checkpoint file.

            Parameters
            ----------
            None. Relies on self.config.

            Returns
            -------
            None. The model checkpoint (.ckpt) is stored in self.config.paths.checkpoint_dir.
        """

        # Data loader and trainer setup
        self._init_trainer(stage='train')

        # Save resolved model configuration for downstream use
        config_model_path = os.path.join(self.config.paths.checkpoint_dir, "model.yaml")
        with open(config_model_path, 'w') as f:
            OmegaConf.save(self.config.model, f)
        logger.info(f"Saving model configuration...")

        # Save resolved experiment configuration for downstream use
        config_experiment_path = os.path.join(self.config.paths.checkpoint_dir, "experiment.yaml")
        with open(config_experiment_path, 'w') as f:
            OmegaConf.save(self.config, f)
        logger.info(f"Saving experiment configuration...")

        # Model initialization based on checkpoint scenario
        ckpt_resume = self.config.get("resume_from_ckpt", None)
        ckpt_init = self.config.get("init_from_ckpt", None)

        # Resume: Trainer handles checkpoint restoration (weights + optimizer + scheduler)
        if ckpt_resume and os.path.exists(ckpt_resume):
            logger.info(f"Resuming training from checkpoint: {ckpt_resume}")
            self._init_model()  # Fresh model, trainer will restore state
            self.trainer.fit(self.model, self.loader, ckpt_path=ckpt_resume)
        # Init: Load weights only for transfer learning (new optimizer + scheduler)
        elif ckpt_init and os.path.exists(ckpt_init):
            logger.info(f"Initializing model from checkpoint: {ckpt_init}")
            self._init_model(ckpt_path=ckpt_init, strict=True)  # Load weights + set dtype
            self.trainer.fit(self.model, self.loader)
        # Scratch: Train from random initialization
        else:
            logger.info("Training model from scratch...")
            self._init_model()  # Fresh model
            self.trainer.fit(self.model, self.loader)
        logger.info("Done!")

        # Save optimal model checkpoint along with configuration
        logger.info("Saving model checkpoint...")
        self.trainer.save_checkpoint(self.model.ckpt_path, weights_only=False)

    def test(self) -> None:
        """ Loads data, callbacks, trainer, and then tests the model.
            Accumulates and saves test results.

            Parameters
            ----------
            None. Relies on self.config.

            Returns
            -------
            None. Test results are accumulated and saved to files.
        """

        # Data loader and trainer setup
        logger.info("Initializing trainer for test set...")
        self._init_trainer(stage='test')

        # Load model from checkpoint if not already loaded
        if self.model is None:
            logger.info("Initializing model...")
            self._init_model(ckpt_path=self.config.model.ckpt_path)

        # Evaluate on test set
        logger.info("Running model...")
        output = self._run_model()

        # Save results to file
        logger.info("Saving output to file...")
        _save_output(output, config_stage=self.loader.test)

    def predict(self, config_loader: DictConfig | None = None) -> dict:
        """ Predicts the output of the model on a given dataset.
            Accumulates and returns predictions.

            Parameters
            ----------
            config_loader: DictConfig or None. Configuration object for the data to predict on.
                           If None, uses self.config.loader.

            Returns
            -------
            dict. Accumulated predictions from the model with structure:
                  {'output': {'bt_forward': array, 'prof': array, ...}, ...}
        """

        # Create output directories for all result variables
        if config_loader is None:
            config_loader = self.config.loader

        # Data loader and trainer setup
        logger.info("Initializing trainer for prediction set...")
        self._init_trainer(stage='predict', config_loader=config_loader)

        # Load model from checkpoint if not already loaded
        if self.model is None:
            logger.info("Initializing model...")
            self._init_model(ckpt_path=self.config.model.ckpt_path)

        # Predict on dataset
        logger.info("Running model...")
        output = self._run_model()

        return output


def train(config: DictConfig) -> None:
    """ Train neural network based on set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Initialize trainer object
    logger.info("Initializing model...")
    model = Operator(config)

    # Train the model
    logger.info("Training model...")
    model.train()


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Train neural network based on set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Perform model training
    train(config)


if __name__ == '__main__':
    """ Train model.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        checkpoint: Training weights & biases.
    """

    main()
