from pytorch_lightning.callbacks import Callback
from utilities.instantiators import instantiate
import matplotlib.pyplot as plt
from forward.model.callback import FigureLogger as ForwardFigureLogger
from utilities.plot import fig_vertical_profiles, fig_rmse_bars
from utilities.tensors import to_numpy
import wandb
import tempfile
import os


class FigureLogger(ForwardFigureLogger):
    """
    Callback to log figures at the end of each validation epoch.
    """
    def __init__(self):
        super().__init__()

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

        # Labels
        prof_labels = model.prof_vars if hasattr(model, 'prof_vars') else None
        # Pressure_levels
        pressure_levels = 0.01 * (
            instantiate(trainer.datamodule.stage.valid.input.pressure.normalization, inverse_transform=True)
            (model.results['pressure']))

        # Profiles
        prof_mean = [to_numpy(model.metrics['prof_target']['mean']),
                     to_numpy(model.metrics['prof']['mean'])]
        prof_stdev = [to_numpy(model.metrics['prof_target']['stdev']),
                      to_numpy(model.metrics['prof']['stdev'])]
        prof_rmse = [to_numpy(model.metrics['prof']['rmse']), ]
        prof_mean_labels, prof_mean_colors = ['Target', 'Prediction'], ['#1f77b4', '#ff7f0e']
        prof_rmse_labels, prof_rmse_colors = ['Target-Prediction'], ['#ff7f0e']
        if 'prof_background' in model.metrics and len(model.metrics['prof_background']) > 0:
            prof_mean.insert(0, to_numpy(model.metrics['prof_background']['mean']))
            prof_stdev.insert(0, to_numpy(model.metrics['prof_background']['stdev']))
            prof_rmse.insert(0, to_numpy(model.metrics['prof_background']['rmse']))
            prof_mean_labels.insert(0, 'Background')
            prof_mean_colors.insert(0, '#2ca02c')
            prof_rmse_labels.insert(0, 'Target-Background')
            prof_rmse_colors.insert(0, '#2ca02c')
        # Profile Mean
        self.figs.append(fig_vertical_profiles(prof_mean, prof_mean_labels, stdev=prof_stdev,
                                               y=pressure_levels, y_label='Pressure (hPa)',
                                               x_label='Normalized profile value (no units)', color=prof_mean_colors,
                                               title=[f"Epoch {current_epoch:02d} - {prof_label}" for prof_label in
                                                      prof_labels]))
        # Profile RMSE
        self.figs.append(fig_vertical_profiles(prof_rmse, prof_rmse_labels, y=pressure_levels, y_label='Pressure (hPa)',
                                               x_label='Normalized profile RMSE (no units)', color=prof_rmse_colors,
                                               title=[f"Epoch {current_epoch:02d} - {prof_label}" for prof_label in
                                                      prof_labels]))

        # Apply parent figure builder
        super()._figurebuilder(trainer, model, tags, current_epoch)

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

        # Epoch
        current_epoch = trainer.current_epoch
        # Tags for each figure
        tags = ["Valid_ProfilesMean", "Valid_ProfilesRMSE", "Valid_RadianceRMSE"]

        # Call the figure builder and buffer
        self._figurebuilder(trainer, model, tags, current_epoch)
        self._figurebuffer(trainer, tags, current_epoch)

    def on_train_epoch_end(self, trainer, model):
        """
        Logs figures at the end of each training epoch.

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
        # Tags for each figure
        tags = ["Train_ProfilesMean", "Train_ProfilesRMSE", "Train_RadianceRMSE"]

        # Call the figure builder and buffer
        self._figurebuilder(trainer, model, tags, current_epoch)
        self._figurebuffer(trainer, tags, current_epoch)
