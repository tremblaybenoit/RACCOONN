from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from utilities.plot import fig_rmse_bars
import numpy as np
import wandb
import tempfile
import os


class FigureLogger(Callback):
    """
    Callback to log figures at the end of each validation epoch.
    """
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, model):
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

        # Labels
        current_epoch = trainer.current_epoch

        # Extract results from the model
        hofx_target = np.concatenate(model.valid_results['hofx_target'], axis=0)
        hofx_pred = np.concatenate(model.valid_results['hofx_pred'], axis=0)
        # Make clear sky mask
        cloud_filter = np.concatenate(model.valid_results['cloud_filter'], axis=0)
        clrsky = ~cloud_filter

        # List of figures to log
        figs = []

        # Forward model RMSE
        figs.append(fig_rmse_bars(hofx_target, hofx_pred, clrsky, title=[f"Epoch {current_epoch:02d} - Forward model errors",
                                                                         f"Epoch {current_epoch:02d} - Normalized forward model errors"]))

        # Tags for each figure
        tags = ["RadianceRMSE"]

        # Save figures to a buffer
        for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
            # TensorBoard
            if logger.__class__.__name__.lower().startswith("tensorboard"):
                for tag, fig in zip(tags, figs):
                    logger.experiment.add_figure(tag=tag, figure=fig, global_step=current_epoch)
            # WandB
            elif logger.__class__.__name__.lower().startswith("wandb"):
                logger.experiment.log({f"{tag}/Epoch_{current_epoch:02d}": wandb.Image(fig) for tag, fig in zip(tags, figs)})
            # MLflow
            elif logger.__class__.__name__.lower().startswith("mlflow"):
                for tag, fig in zip(tags, figs):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        filename = os.path.join(tmpdir, f"{tag}_Epoch_{current_epoch:02d}.png")
                        fig.savefig(filename)
                        logger.experiment.log_artifact(logger.run_id, filename, artifact_path="figures")

        # Close figures to free memory
        plt.close('all')
