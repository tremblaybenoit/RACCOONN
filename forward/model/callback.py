from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from utilities.tensors import to_numpy
from utilities.plot import fig_rmse_bars
import wandb
import tempfile
import os


class FigureLogger(Callback):
    """
    Callback to log figures at the end of each validation epoch.
    """
    def __init__(self, save_every_n_epochs: int = 100, save_on_improvement: bool = True, monitor: str = "valid_loss", monitor_mode: str = "min") -> None:
        """
        Initializes the FigureLogger callback.

        Parameters
        ----------
        save_every_n_epochs: int. Frequency of saving figures in epochs. Default is 1.
        save_on_improvement: bool. Whether to save figures only when there is an improvement in the monitored metric. Default is False.
        monitor: str. The metric to monitor for improvements. Default is "val_loss".
        monitor_mode: str. The mode for monitoring the metric, either "min" or "max". Default is "min".

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Initialize list of figures
        self.figs = []

        # Parameters for saving on improvement
        self.save_every_n_epochs = None if save_every_n_epochs in (None, 0) else int(save_every_n_epochs)
        self.save_on_improvement = bool(save_on_improvement)
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.best_monitor = float("inf") if self.monitor_mode == "min" else -float("inf")

    def _get_monitor_value(self, trainer):
        """
        Retrieves the current value of the monitored metric.

        Parameters
        ----------
        trainer: pytorch_lightning.Trainer. The trainer instance.

        Returns
        -------
        float or None. The current value of the monitored metric, or None if not found.
        """

        # Prefer trainer.callback_metrics (PyTorch Lightning standard)
        val = None
        if hasattr(trainer, "callback_metrics"):
            val = trainer.callback_metrics.get(self.monitor)
        return float(val) if val is not None else None

    def _figuresaver(self, trainer, current_epoch: int) -> bool:
        """
        Determines whether to save figures based on the specified conditions.

        Parameters
        ----------
        trainer: pytorch_lightning.Trainer. The trainer instance.
        current_epoch: int. The current epoch number.

        Returns
        -------
        bool. True if figures should be saved, False otherwise.
        """

        # If neither condition requested, default save every epoch
        save_by_epoch = True
        if self.save_every_n_epochs is not None:
            save_by_epoch = (current_epoch % self.save_every_n_epochs) == 0

        save_by_improvement = True
        if self.save_on_improvement:
            current_val = self._get_monitor_value(trainer)
            if current_val is None:
                # cannot decide improvement if metric missing -> don't save on improvement
                save_by_improvement = False
            else:
                if self.monitor_mode == "min":
                    if current_val < self.best_monitor:
                        self.best_monitor = current_val
                        save_by_improvement = True
                    else:
                        save_by_improvement = False
                else:
                    if current_val > self.best_monitor:
                        self.best_monitor = current_val
                        save_by_improvement = True
                    else:
                        save_by_improvement = False

        # If both options are used, save when either condition is True
        if self.save_every_n_epochs is not None and self.save_on_improvement:
            return save_by_epoch or save_by_improvement
        # If only one is used, respect it
        if self.save_on_improvement:
            return save_by_improvement
        if self.save_every_n_epochs is not None:
            return save_by_epoch
        # default
        return True

    def _figurebuilder(self, trainer, model, tags, current_epoch):
        """
        Logs figures.

        Parameters
        ----------
        trainer: pytorch_lightning.Trainer. The trainer instance.
        model: pytorch_lightning.LightningModule. The model instance.
        tags: list of str. Tags for each figure.
        current_epoch: int. The current epoch number.

        Returns
        -------
        None. The figures are logged to the logger associated with the trainer.
        """

        # Forward model RMSE
        hofx = [to_numpy(model.metrics['hofx'][key]['rmse']) for key in model.metrics['hofx'].keys()]
        hofx_norm = [to_numpy(model.metrics['hofx_norm'][key]['rmse']) for key in model.metrics['hofx_norm'].keys()]
        self.figs.append(fig_rmse_bars(hofx, hofx_norm, labels = list(model.metrics['hofx'].keys()),
                                       title=[f"Epoch {current_epoch:02d} - Forward model errors",
                                              f"Epoch {current_epoch:02d} - Normalized forward model errors"]))

    def _figurebuffer(self, trainer, tags, current_epoch):
        """
        Adds figures to the logger at the end of each validation epoch.

        Parameters
        ----------
        trainer: pytorch_lightning.Trainer. The trainer instance.
        tags: list of str. Tags for each figure.
        current_epoch: int. The current epoch number.

        Returns
        -------
        None. The figures are logged to the logger associated with the trainer.
        """

        # Save figures to a buffer
        for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
            # TensorBoard
            if logger.__class__.__name__.lower().startswith("tensorboard"):
                for tag, fig in zip(tags, self.figs):
                    logger.experiment.add_figure(tag=tag, figure=fig, global_step=current_epoch)
            # WandB
            elif logger.__class__.__name__.lower().startswith("wandb"):
                logger.experiment.log({f"{tag}/Epoch_{current_epoch:02d}": wandb.Image(fig) for tag, fig in zip(tags, self.figs)})
            # MLflow
            elif logger.__class__.__name__.lower().startswith("mlflow"):
                for tag, fig in zip(tags, self.figs):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        filename = os.path.join(tmpdir, f"{tag}_Epoch_{current_epoch:04d}.png")
                        fig.savefig(filename)
                        logger.experiment.log_artifact(logger.run_id, filename, artifact_path="figures")

        # Close figures to free memory
        plt.close('all')
        self.figs.clear()

    def on_train_epoch_end(self, trainer, model):
        """
        Logs figures at the end of each validation epoch.

        Parameters
        ----------
        trainer: pytorch_lightning.Trainer. The trainer instance.
        model: pytorch_lightning.LightningModule. The model instance.

        Returns
        -------
        None. The figures are logged to the logger associated with the trainer.
        """

        # If in sanity checking, skip logging
        if trainer.sanity_checking:
            return

        # Epoch
        current_epoch = trainer.current_epoch
        # Decide whether to save figures
        if not self._figuresaver(trainer, current_epoch):
            return

        # Tags for each figure
        tags = ["Valid_RadianceRMSE"]
        # Call the figure builder and buffer
        self._figurebuilder(trainer, model, tags, current_epoch)
        self._figurebuffer(trainer, tags, current_epoch)
