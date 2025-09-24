import os.path
import numpy as np
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as lightning
from utilities.logger import TrainerLogger
from utilities.instantiators import instantiate, instantiate_list
from utilities.logic import get_config_path
torch.set_float32_matmul_precision('high')

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
        self.checkpoint_path = (config.callbacks.model_checkpoint.dirpath +
                                f"/{config.callbacks.model_checkpoint.filename}.ckpt")

        # Create output directories if they don't exist
        if not os.path.exists(self.config.paths.output_dir):
            os.makedirs(self.config.paths.output_dir, exist_ok=True)
        if not os.path.exists(self.config.paths.checkpoint_dir):
            os.makedirs(self.config.paths.checkpoint_dir, exist_ok=True)
        if not os.path.exists(self.config.paths.log_dir):
            os.makedirs(self.config.paths.log_dir, exist_ok=True)
        if not os.path.exists(self.config.paths.data_dir):
            os.makedirs(self.config.paths.run_dir, exist_ok=True)

        # Initialization
        self.data_loader = None
        self.callbacks = None
        self.trainer_logger = None
        self.trainer = None
        self.model = None

        # For reproducibility, set randomizer seed if provided
        if self.config.get("seed"):
            lightning.seed_everything(self.config.task_seed, workers=True)

    def setup(self, loader_config, stage: str='train') -> None:
        """ Setup trainer object.

            Parameters
            ----------
            loader_config: DictConfig. Configuration object for the data loader.
            stage: str. Stage of the training process.
                        Options are 'train', 'test', or 'predict'.

            Returns
            -------
            None.
        """

        # Data loader
        logger.info("Initializing data loader...")
        self.data_loader = instantiate(loader_config)
        # Generate training/validation/test sets
        self.data_loader.setup(stage=stage)

        # Trainer loggers and callbacks: Only activated during training
        if stage == 'train':
            # Configure logger
            if self.trainer_logger is None:
                TrainerLogger(self.config.logger).configure()
                logger.info("Initializing logger(s)...")
                self.trainer_logger = instantiate_list(self.config.get("logger"), "logger")
                logger.info("Done with loggers, Initializing logger(s)...")

            # Callbacks
            if self.callbacks is None:
                logger.info("Initializing callback(s)...")
                self.callbacks = instantiate_list(self.config.get("callbacks"), "callbacks")

        else:
            self.trainer_logger, self.callbacks = False, None

        # Trainer
        logger.info("Waking up trainer...")
        self.trainer = instantiate(self.config.trainer, callbacks=self.callbacks, logger=self.trainer_logger)

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
        self.setup(self.config.loader, stage='train')

        # Model
        logger.info("Initializing model...")
        self.model = instantiate(self.config.model)
        if hasattr(self.config.data, 'dtype'):
            self.model = self.model.to(None, dtype=getattr(torch, self.config.data.dtype))
        else:
            self.model = self.model.to(None, dtype=getattr(torch, 'float32'))

        # Train the model ⚡
        resume_ckpt = self.config.get("resume_from_checkpoint", None)
        if resume_ckpt and os.path.exists(resume_ckpt):
            logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
            self.trainer.fit(self.model, self.data_loader, ckpt_path=resume_ckpt)
        else:
            logger.info("Training model...")
            self.trainer.fit(self.model, self.data_loader)
        logger.info("Done!")

        # Save optimal model checkpoint along with configuration
        logger.info("Saving model checkpoint...")
        self.trainer.save_checkpoint(self.checkpoint_path, weights_only=False)

    def test(self) -> None:
        """ Loads data, loggers, callbacks, trainer, and then tests the model. Saves the test results in a file.

            Parameters
            ----------
            None. Relies on self.config.

            Returns
            -------
            None. The test results are stored in self.config.paths.checkpoint_dir.
        """

        # Create output directories if they don't exist
        save_dir = os.path.dirname(self.config.loader.stage.test.results.hofx.path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Data loader and trainer setup
        self.setup(self.config.loader, stage='test')

        # Load model from checkpoint
        if self.model is None:
            logger.info("Loading model...")
            self.model = instantiate(self.config.model)
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            if hasattr(self.config.data, 'dtype'):
                self.model = self.model.to(None, dtype=getattr(torch, self.config.data.dtype))
            else:
                self.model = self.model.to(None, dtype=getattr(torch, 'float32'))

        # Evaluate on test set
        logger.info("Running against test set...")
        _ = self.trainer.test(self.model, self.data_loader)

        # Save test results to file
        logger.info("Saving results to file...")
        if hasattr(self.config.loader.stage.test, 'results'):
            # Loop over all results in the config and save them
            for result_name, result_config in self.config.loader.stage.test.results.items():
                if result_name in self.model.test_results and hasattr(result_config, 'save'):
                    save_function = instantiate(result_config.save)
                    save_function(self.model.test_results[result_name])

    def predict(self, loader_config: DictConfig) -> np.ndarray:
        """ Predicts the output of the model on a given dataset.

            Parameters
            ----------
            loader_config: DictConfig. Configuration object for the data to predict on.

            Returns
            -------
            None.
        """

        # Create output directories if they don't exist
        save_dir = os.path.dirname(loader_config.stage.test.results.hofx.path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Data loader and trainer setup
        self.setup(loader_config, stage='pred')

        # Load model from checkpoint
        if self.model is None:
            logger.info("Loading model...")
            self.model = instantiate(self.config.model)
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            if hasattr(self.config.data, 'dtype'):
                self.model = self.model.to(None, dtype=getattr(torch, self.config.data.dtype))
            else:
                self.model = self.model.to(None, dtype=getattr(torch, 'float32'))

        # Predict on dataset
        logger.info("Predicting on dataset...")
        predictions = torch.cat(self.trainer.predict(self.model, self.data_loader), dim=0).cpu().numpy()
        # Reshape based on state variables and heights
        predictions = predictions.reshape(predictions.shape[0], -1)

        return predictions


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
    logger.info("Initializing the forward model...")
    forward_model = Operator(config)

    # Train the model
    logger.info("Training the forward model...")
    forward_model.train()

if __name__ == '__main__':
    """ Train the forward model.

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
