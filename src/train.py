import os
import os.path
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as lightning
from utilities.logger import TrainerLogger
from utilities.instantiators import instantiate, instantiate_list
from utilities.logic import get_config_path
from src.callback.results import ResultsLogger
# Force full FP32 matmul on CUDA (disable TF32) for more reproducible numerics
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Initialize logger
logger = logging.getLogger(__name__)


class Operator:
    """Class for training a neural network operator."""
    def __init__(self, config: DictConfig) -> None:
        """ Initialization of trainer and its configuration.

            Parameters
            ----------
            config: Hydra configuration object.

            Returns
            -------
            None.

        """
        logger.info("Reading Hydra configuration...")

        # Load config object and resolve paths
        OmegaConf.resolve(config)
        self.config = config
        self.ckpt_path = os.path.join(
            config.callbacks.model_checkpoint.dirpath,
            f"{config.callbacks.model_checkpoint.filename}.ckpt"
        )

        # Create output directories if they don't exist
        for directory in [self.config.paths.output_dir, self.config.paths.checkpoint_dir,
                          self.config.paths.log_dir]:
            os.makedirs(directory, exist_ok=True)

        # Initialization
        self.data_loader = None
        self.callbacks = None
        self.trainer_logger = None
        self.trainer = None
        self.model = None

        # For reproducibility, set randomizer seed if provided
        if self.config.get("seed"):
            lightning.seed_everything(self.config.task_seed, workers=True)

    def setup(self, stage: str = 'train', loader_config: DictConfig | None = None) -> None:
        """ Setup trainer object.

            Parameters
            ----------
            stage: str. Stage of the training process.
                        Options are 'train', 'test', or 'predict'.
            loader_config: DictConfig or None. Configuration object for the data loader.
                           If None, uses self.config.loader.

            Returns
            -------
            None.
        """

        # Data loader
        if loader_config is None:
            loader_config = self.config.loader
        logger.info("Initializing data loader...")
        self.data_loader = instantiate(loader_config)
        # Generate training/validation/test sets
        self.data_loader.setup(stage=stage)

        # Trainer loggers: Only activated during training
        if stage == 'train':
            # Configure logger
            if self.trainer_logger is None:
                TrainerLogger(self.config.logger).configure()
                logger.info("Initializing logger(s)...")
                self.trainer_logger = instantiate_list(self.config.get("logger"), "logger")
                logger.info("Done with loggers, Initializing callback(s)...")
        else:
            self.trainer_logger = False

        # Callbacks: Initialized for all stages (train, test, predict)
        if self.callbacks is None:
            logger.info("Initializing callback(s)...")
            self.callbacks = instantiate_list(self.config.get("callbacks"), "callbacks")

        # Trainer
        logger.info("Waking up trainer...")
        self.trainer = instantiate(self.config.trainer, callbacks=self.callbacks, logger=self.trainer_logger)

    @staticmethod
    def _makedirs(results_config: DictConfig) -> None:
        """ Create output directories for all result variables.

            Parameters
            ----------
            results_config: DictConfig. Configuration object containing result paths
                           (e.g., results.hofx.path, results.prof.path, etc.).

            Returns
            -------
            None.
        """
        if results_config is None:
            return

        # Iterate through all result variables and create their directories
        for var_name, var_config in results_config.items():
            if isinstance(var_config, DictConfig) and 'path' in var_config:
                output_path = var_config.path
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                logger.info(f"Created output directory for {var_name}: {os.path.dirname(output_path)}")

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
            logger.info(f"Loading checkpoint: {ckpt_path}")
            self.model.load_checkpoint(ckpt_path, strict=strict)

        # Set dtype
        dtype_str = self.config.get('data.dtype', 'float32')
        self.model = self.model.to(None, dtype=getattr(torch, dtype_str))

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
        self.setup(stage='train')

        # Model initialization based on checkpoint scenario
        ckpt_resume = self.config.get("resume_from_checkpoint", None)
        ckpt_init = self.config.get("init_from_checkpoint", None)

        # Resume: Trainer handles checkpoint restoration (weights + optimizer + scheduler)
        if ckpt_resume and os.path.exists(ckpt_resume):
            logger.info(f"Resuming training from checkpoint: {ckpt_resume}")
            self._init_model()  # Fresh model, trainer will restore state
            self.trainer.fit(self.model, self.data_loader, ckpt_path=ckpt_resume)
        # Init: Load weights only for transfer learning (new optimizer + scheduler)
        elif ckpt_init and os.path.exists(ckpt_init):
            logger.info(f"Initializing model from checkpoint: {ckpt_init}")
            self._init_model(ckpt_init, strict=True)  # Load weights + set dtype
            self.trainer.fit(self.model, self.data_loader)
        # Scratch: Train from random initialization
        else:
            logger.info("Training model from scratch...")
            self._init_model()  # Fresh model
            self.trainer.fit(self.model, self.data_loader)
        logger.info("Done!")

        # Save optimal model checkpoint along with configuration
        logger.info("Saving model checkpoint...")
        self.trainer.save_checkpoint(self.ckpt_path, weights_only=False)

    def test(self) -> None:
        """ Loads data, callbacks, trainer, and then tests the model.
            Accesses test results from ResultsLogger callback.

            Parameters
            ----------
            None. Relies on self.config.

            Returns
            -------
            None. The test results are stored in ResultsLogger callback.
        """

        # Create output directories for all result variables
        self._makedirs(self.config.loader.stage.test.results)

        # Data loader and trainer setup
        self.setup(stage='test')

        # Load model from checkpoint if not already loaded
        if self.model is None:
            self._init_model(self.ckpt_path, strict=False)

        # Evaluate on test set
        logger.info("Running against test set...")
        _ = self.trainer.test(self.model, self.data_loader)

        # Access results from ResultsLogger callback
        logger.info("Retrieving test results...")
        results = {}
        for callback in self.callbacks:
            if isinstance(callback, ResultsLogger):
                results = callback.get_results("test")
        # If no callback, build manually

        # Save results to file
        logger.info("Saving results to file...")
        if hasattr(self.config.loader.stage.test, 'results'):
            # Loop over all results in the config and save them
            for result_name, result_config in self.config.loader.stage.test.results.items():
                if result_name in results and hasattr(result_config, 'save'):
                    save_function = instantiate(result_config.save)
                    save_function(results[result_name])


    def predict(self, loader_config: DictConfig | None = None) -> dict:
        """ Predicts the output of the model on a given dataset.
            Accesses predictions from ResultsLogger callback.

            Parameters
            ----------
            loader_config: DictConfig or None. Configuration object for the data to predict on.
                           If None, uses self.config.loader.

            Returns
            -------
            dict. Predictions and results from the model stored in ResultsLogger callback.
        """

        # Create output directories for all result variables
        self._makedirs(loader_config.stage.test.results if loader_config else self.config.loader.stage.test.results)

        # Data loader and trainer setup
        self.setup(stage='pred', loader_config=loader_config)

        # Load model from checkpoint if not already loaded
        if self.model is None:
            self._init_model(self.ckpt_path, strict=False)

        # Predict on dataset
        logger.info("Predicting on dataset...")
        _ = self.trainer.predict(self.model, self.data_loader)

        # Access results from ResultsLogger callback
        logger.info("Retrieving prediction results...")
        for callback in self.callbacks:
            if isinstance(callback, ResultsLogger):
                results = callback.get_results("pred")
                logger.info(f"Prediction results collected: {list(results.keys())}")
                return results

        # Fallback if no ResultsLogger found
        logger.warning("No ResultsLogger callback found. Returning empty dict.")
        return {}


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

    # Initialize trainer object
    logger.info("Initializing model...")
    forward_model = Operator(config)

    # Train the model
    logger.info("Training model...")
    forward_model.train()

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
