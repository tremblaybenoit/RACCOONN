from pytorch_lightning.callbacks import Callback
from forward.utilities.instantiators import instantiate
import matplotlib.pyplot as plt
from inverse.utilities.plot import fig_vertical_profiles, fig_vertical_profile, fig_rmse_bars, fig_vertical_profiles_background
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

        # Extract results from the model
        current_epoch = trainer.current_epoch
        if trainer.sanity_checking is False:
            bt_target = np.concatenate(model.valid_results['bt_target'], axis=0)
            bt_pred = np.concatenate(model.valid_results['bt_pred'], axis=0)
            prof_target = np.concatenate(model.valid_results['prof_target'], axis=0)
            prof_pred = np.concatenate(model.valid_results['prof_pred'], axis=0)
            prof_err = (prof_pred - prof_target)  # / (np.std(prof_target, axis=0) + np.finfo(float).eps)
            prof_labels = model.prof_vars if hasattr(model, 'prof_vars') else None
            pressure_levels = (
                instantiate(trainer.datamodule.sets.train.pressure.normalization, inverse_transform=True)
                (model.valid_results['pressure'][0][0]))
            clrsky = np.logical_and(prof_target[:, 5, :].sum(axis=1) == 0, prof_target[:, 6, :].sum(axis=1) == 0)
            clrsky = np.logical_and(clrsky, prof_target[:, 7, :].sum(axis=1) == 0)

            # If model.valid_background exisits, use it; otherwise, set to None
            if 'background' in model.valid_results and len(model.valid_results['background']) > 0:
               background = np.concatenate(model.valid_results['background'], axis=0)
               background_err = np.concatenate(model.valid_results['background_err'], axis=0)
               fig0 = fig_vertical_profiles_background(prof_target, prof_pred, background, background_err,
                                                       y=pressure_levels, y_label='Pressure (hPa)',
                                                       title=[f"Epoch {current_epoch:02d} - {prof_label} profiles" for prof_label in prof_labels])
            else:
                # Generate figures
                fig0 = fig_vertical_profiles(prof_target, prof_pred, y=pressure_levels, y_label='Pressure (hPa)',
                                             title=[f"Epoch {current_epoch:02d} - {prof_label} profiles" for prof_label in prof_labels])
            fig1 = fig_vertical_profile(prof_err, y=pressure_levels, y_label='Pressure (hPa)',
                                        title=[f"Epoch {current_epoch:02d} - {prof_label} profile errors" for prof_label in prof_labels])
            fig2 = fig_rmse_bars(bt_target, bt_pred, clrsky,
                                 title=[f"Epoch {current_epoch:02d} - Forward model errors",
                                        f"Epoch {current_epoch:02d} - Normalized forward model errors"])

            # Save figures to a buffer
            for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
                # TensorBoard
                if logger.__class__.__name__.lower().startswith("tensorboard"):
                    logger.experiment.add_figure(
                        tag=f"VerticalProfiles",
                        figure=fig0,
                        global_step=current_epoch
                    )
                    logger.experiment.add_figure(
                        tag=f"VerticalErrors",
                        figure=fig1,
                        global_step=current_epoch
                    )
                    logger.experiment.add_figure(
                        tag=f"RadianceRMSE",
                        figure=fig2,
                        global_step=current_epoch
                    )
                # WandB
                elif logger.__class__.__name__.lower().startswith("wandb"):
                    logger.experiment.log({
                        f"VerticalProfiles/Epoch_{current_epoch:02d}": wandb.Image(fig0),
                        f"VerticalErrors/Epoch_{current_epoch:02d}": wandb.Image(fig1),
                        f"RadianceRMSE/Epoch_{current_epoch:02d}": wandb.Image(fig2)
                    })
                # MLflow
                elif logger.__class__.__name__.lower().startswith("mlflow"):
                    for name, fig in [
                        (f"VerticalProfiles_Epoch_{current_epoch:02d}.png", fig0),
                        (f"VerticalErrors_Epoch_{current_epoch:02d}.png", fig1),
                        (f"RadianceRMSE_Epoch_{current_epoch:02d}.png", fig2)
                    ]:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            filename = os.path.join(tmpdir, name)
                            fig.savefig(filename)
                            logger.experiment.log_artifact(logger.run_id, filename, artifact_path="figures")

            # Close figures to free memory
            plt.close(fig0)
            plt.close(fig1)
            plt.close(fig2)
            # breakpoint()
