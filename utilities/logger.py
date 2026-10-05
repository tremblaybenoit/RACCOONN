import os
import sys
import subprocess
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import webbrowser
import logging


# Initialize logger
logger = logging.getLogger(__name__)


def _open_url_wsl_fallback(url: str) -> bool:
    """Try webbrowser.open() then, if running under WSL, use Windows commands to open the URL.

    Parameters
    ----------
    url: str. The URL to open in a web browser.

    Returns
    -------
    bool. True if a launcher was successfully started, False otherwise.
    """

    # (a) Try webbrowser
    try:
        if webbrowser.open(url, new=2):
            return True
    except OSError:
        pass

    # (b) Detect WSL
    is_wsl = False
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/version", "r") as f:
                text = f.read()
                is_wsl = "Microsoft" in text or "microsoft" in text
        except (OSError, FileNotFoundError):
            is_wsl = False

    if is_wsl:
        # prefer cmd.exe start then powershell as fallback
        cmds = [
            ["cmd.exe", "/C", "start", "", url],
            ["powershell.exe", "-NoProfile", "-Command", "Start-Process", url],
        ]
        for cmd in cmds:
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (OSError, FileNotFoundError):
                continue

    # (c) Generic fallbacks for non-WSL Linux/Win/macOS
    if sys.platform.startswith("linux"):
        for cmd in (["xdg-open", url], ["gio", "open", url]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (OSError, FileNotFoundError):
                continue
    elif sys.platform == "darwin":
        try:
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (OSError, FileNotFoundError):
            pass
    elif os.name == "nt":
        for cmd in (["cmd", "/C", "start", "", url], ["powershell", "-NoProfile", "-Command", "Start-Process", url]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (OSError, FileNotFoundError):
                continue

    logger.warning("Could not open browser programmatically; please open URL manually: %s", url)
    return False


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

    @staticmethod
    def _fix_path(path: str) -> str:
        """Converts a potential Windows path to a WSL path if running in WSL.

        Parameters
        ----------
        path: str. The file path to convert (Windows or Unix format).

        Returns
        -------
        str. The converted path (WSL format if running in WSL, absolute path otherwise).
        """
        # Check if we are in WSL
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/version", "r") as f:
                    if "microsoft" in f.read().lower():
                        # Check if it looks like a Windows path (has : or \)
                        if ":" in path or "\\" in path:
                            # Use the 'wslpath' utility to convert C:\ to /mnt/c/
                            return subprocess.check_output(["wslpath", "-u", path]).decode().strip()
            except (OSError, FileNotFoundError, subprocess.CalledProcessError):
                pass
        return os.path.abspath(path)

    def configure(self) -> None:
        """Configure logger(s) for offline mode and UI access if specified in the config.

        Sets up environment variables for wandb offline mode and MLflow tracking URI.
        Optionally starts the logger UI(s) if configured.

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
            # Convert path for WSL compatibility
            safe_path = self._fix_path(self.config.mlflow.save_dir)
            logger.info(f"Setting mlflow tracking URI to: file://{safe_path}")

            # Ensure the directory exists before MLflow touches it
            os.makedirs(safe_path, exist_ok=True)

            # Note: file:// protocol usually takes two slashes + the absolute path
            if sys.platform.startswith("linux"):
                os.environ["MLFLOW_TRACKING_URI"] = safe_path
            else:
                os.environ["MLFLOW_TRACKING_URI"] = f"file:///{safe_path}"

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
            safe_path = self._fix_path(self.config.mlflow.save_dir)
            command = f"mlflow ui --backend-store-uri file:///{safe_path} --port {port}"
            # Start the mlflow UI
            logger.info("Starting mlflow UI...")
            subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Open UI in the browser
            if show:
                logger.info("Opening mlflow UI in browser...")
                _open_url_wsl_fallback(f"http://127.0.0.1:{port}")

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
