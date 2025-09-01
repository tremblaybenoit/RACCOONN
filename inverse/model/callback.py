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

        # If in sanity checking, skip logging
        if trainer.sanity_checking:
            return

        # Labels
        current_epoch = trainer.current_epoch
        prof_labels = model.prof_vars if hasattr(model, 'prof_vars') else None

        # Extract results from the model
        # Radiances
        hofx_target = np.concatenate(model.valid_results['hofx_target'], axis=0)
        hofx_pred = np.concatenate(model.valid_results['hofx_pred'], axis=0)
        # Profiles (in original units)
        prof_target = np.concatenate(model.valid_results['prof_target'], axis=0)
        prof_pred = np.concatenate(model.valid_results['prof_pred'], axis=0)
        prof_err = (prof_pred - prof_target)**2
        # Profiles (in normalized units)
        prof_norm_target = np.concatenate(model.valid_results['prof_norm_target'], axis=0)
        prof_norm_pred = np.concatenate(model.valid_results['prof_norm_pred'], axis=0)
        prof_norm_err = (prof_norm_pred - prof_norm_target)**2
        # Pressure_levels
        pressure_levels = 0.01*(
            instantiate(trainer.datamodule.stage.valid.obs.pressure.normalization, inverse_transform=True)
            (model.valid_results['pressure'][0][0]))
        # Cloud mask
        clrsky = np.logical_and(prof_target[:, 5, :].sum(axis=1) == 0, prof_target[:, 6, :].sum(axis=1) == 0)
        clrsky = np.logical_and(clrsky, prof_target[:, 7, :].sum(axis=1) == 0)

        # List of figures to log
        figs = []

        # Profiles
        if 'prof_background' in model.valid_results and len(model.valid_results['prof_background']) > 0:
           prof_background = np.concatenate(model.valid_results['prof_background'], axis=0)
           figs.append(fig_vertical_profiles_background(prof_target, prof_pred, prof_background, y=pressure_levels, y_label='Pressure (hPa)',
                                                   title=[f"Epoch {current_epoch:02d} - {prof_label} profiles" for prof_label in prof_labels]))
        else:
           # Generate figures
           figs.append(fig_vertical_profiles(prof_target, prof_pred, y=pressure_levels, y_label='Pressure (hPa)',
                                         title=[f"Epoch {current_epoch:02d} - {prof_label} profiles" for prof_label in prof_labels]))
        if 'prof_norm_background' in model.valid_results and len(model.valid_results['prof_norm_background']) > 0:
           prof_norm_background = np.concatenate(model.valid_results['prof_norm_background'], axis=0)
           figs.append(fig_vertical_profiles_background(prof_norm_target, prof_norm_pred, prof_norm_background, y=pressure_levels, y_label='Pressure (hPa)',
                                                        title=[f"Epoch {current_epoch:02d} - {prof_label} normalized profiles" for prof_label in prof_labels]))
        else:
           figs.append(fig_vertical_profiles(prof_norm_target, prof_norm_pred, y=pressure_levels, y_label='Pressure (hPa)',
                                              title=[f"Epoch {current_epoch:02d} - {prof_label} normalized profiles" for prof_label in prof_labels]))
        # Profile errors
        figs.append(fig_vertical_profile(prof_err, y=pressure_levels, y_label='Pressure (hPa)',
                                         title=[f"Epoch {current_epoch:02d} - {prof_label} profile errors" for prof_label in prof_labels]))
        figs.append(fig_vertical_profile(prof_norm_err, y=pressure_levels, y_label='Pressure (hPa)',
                                         title=[f"Epoch {current_epoch:02d} - {prof_label} normalized profile errors" for prof_label in prof_labels]))
        # Forward model RMSE
        figs.append(fig_rmse_bars(hofx_target, hofx_pred, clrsky, title=[f"Epoch {current_epoch:02d} - Forward model errors",
                                                                         f"Epoch {current_epoch:02d} - Normalized forward model errors"]))

        # Tags for each figure
        tags = ["VerticalProfiles", "VerticalNormalizedProfiles", "VerticalErrors", "VerticalNormalizedErrors", "RadianceRMSE"]

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

        # Close figures to free memory
        plt.close('all')
