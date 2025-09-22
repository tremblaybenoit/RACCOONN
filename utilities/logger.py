import os
import subprocess
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import webbrowser
import logging


# Initialize logger
logger = logging.getLogger(__name__)


class TrainerLogger:
    """ Class to configure and manage the logger for the training process.
    """
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the TrainerLogger class.

        Parameters
        ----------
        config: DictConfig. Configuration object for the logger.
        """
        # Load config object and resolve paths
        self.config = config

    def configure(self) -> None:
        """Configure logger for offline mode if specified in the config.

        Parameters
        ----------
        If "wandb" is in the config and "offline" is set to True, set WANDB_MODE to "offline".
        If "mlflow" is in the config and "tracking_uri" is set, set MLFLOW_TRACKING_URI to the specified URI.

        Returns
        -------
        None.
        """

        # Start UI
        if self.config.get("ui", False):
            logger.info("Starting logger UI(s)...")
            self.ui(show=self.config.ui)

        # Wandb
        if "wandb" in self.config and self.config.get("offline", False):
            # Set wandb mode to offline
            logger.info("Setting wandb mode to offline...")
            os.environ["WANDB_MODE"] = "offline"

        # MLflow
        if "mlflow" in self.config:
            # Set tracking URI
            logger.info("Setting mlflow tracking URI for offline tracking...")
            os.environ["MLFLOW_TRACKING_URI"] = "file:///" + self.config.mlflow.save_dir

    def ui(self, show: bool=False) -> None:
        """ Access logger UI via terminal commands.

        Parameters
        ----------
        show: bool. If True, open the UI in the browser.

        Returns
        -------
        None.
        """

        # Wandb
        if "wandb" in self.config:
            # Select port
            port = getattr(self.config.ports, "wandb", 8080)
            # Command for terminal
            command = "wandb server start"
            # Start the wandb UI
            logger.info("Starting wandb UI...")
            subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Open UI in the browser
            if show:
                logger.info("Opening wandb UI in browser...")
                webbrowser.open(f"http://127.0.0.1:{port}")

        # MLflow
        if "mlflow" in self.config:
            # Select port
            port = getattr(self.config.ports, "mlflow", 5000)
            # Command for terminal
            command = f"mlflow ui --backend-store-uri file:///{os.path.abspath(self.config.mlflow.save_dir)} --port {port}"
            # Start the mlflow UI
            logger.info("Starting mlflow UI...")
            subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Open UI in the browser
            if show:
                logger.info("Opening mlflow UI in browser...")
                webbrowser.open(f"http://127.0.0.1:{port}")

        # Tensorboard
        if "tensorboard" in self.config:
            # Select port
            port = getattr(self.config.ports, "tensorboard", 6006)
            # Command for terminal
            command = f"tensorboard --logdir {self.config.tensorboard.save_dir} --port {port}"
            # Start the tensorboard UI
            logger.info("Starting tensorboard UI...")
            subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Open UI in the browser
            if show:
                logger.info("Opening tensorboard UI in browser...")
                webbrowser.open(f"http://127.0.0.1:{port}")


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Main function to configure logger(s) for the training process.

        Parameters
        ----------
        config: DictConfig. Configuration object.

        Returns
        -------
        None.
    """

    # Call UI
    trainer_logger = TrainerLogger(config)
    trainer_logger.ui(show=True)


if __name__ == '__main__':
    """ Access and configure logger(s) for the training process.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        None.
    """

    main()
