from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from utilities.tensors import to_numpy
from src.evaluation.plot import fig_rmse_bars, fig_vertical_profiles
from utilities.instantiators import instantiate
import wandb
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


class FigureLogger(Callback):
    """Base callback to log figures at the end of each validation epoch.

    Handles all common logic for figure saving, monitoring, and logging to different
    backends (TensorBoard, WandB, MLflow).
    """

    def __init__(self, save_every_n_epochs: int = 100, save_on_improvement: bool = True,
                 monitor: str = "valid_loss", monitor_mode: str = "min") -> None:
        """Initialize FigureLogger callback.

        Parameters
        ----------
        save_every_n_epochs : int, default 100
            Frequency of saving figures in epochs. None or 0 disables epoch-based saving.
        save_on_improvement : bool, default True
            Whether to save figures only when there is improvement in monitored metric.
        monitor : str, default "valid_loss"
            Metric to monitor for improvements.
        monitor_mode : str, default "min"
            Whether to minimize or maximize the monitored metric ("min" or "max").
        """
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
        """Retrieve current value of monitored metric.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance

        Returns
        -------
        float or None
            Current value of monitored metric, or None if not found
        """
        val = None
        if hasattr(trainer, "callback_metrics"):
            val = trainer.callback_metrics.get(self.monitor)
        return float(val) if val is not None else None

    def _figure_saver(self, trainer, current_epoch: int) -> bool:
        """Determine whether to save figures based on specified conditions.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance
        current_epoch : int
            Current epoch number

        Returns
        -------
        bool
            True if figures should be saved, False otherwise
        """
        # If neither condition requested, default save every epoch
        save_by_epoch = True
        if self.save_every_n_epochs is not None:
            save_by_epoch = (current_epoch % self.save_every_n_epochs) == 0

        save_by_improvement = True
        if self.save_on_improvement:
            current_val = self._get_monitor_value(trainer)
            if current_val is None:
                # Cannot decide improvement if metric missing -> don't save on improvement
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

    def _figure_builder(self, trainer, model, tags, current_epoch):
        """Build figures. Override in subclasses for specialized behavior.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance
        model : pytorch_lightning.LightningModule
            The model instance
        tags : list of str
            Tags for each figure
        current_epoch : int
            Current epoch number
        """
        pass  # Override in subclasses

    def _figure_buffer(self, trainer, tags, current_epoch):
        """Add figures to logger at end of validation epoch.

        Logs to all available backends (TensorBoard, WandB, MLflow) and clears figures.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance
        tags : list of str
            Tags for each figure
        current_epoch : int
            Current epoch number
        """
        # Save figures to a buffer
        for logger_inst in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
            # TensorBoard
            if logger_inst.__class__.__name__.lower().startswith("tensorboard"):
                for tag, fig in zip(tags, self.figs):
                    logger_inst.experiment.add_figure(tag=tag, figure=fig, global_step=current_epoch)
            # WandB
            elif logger_inst.__class__.__name__.lower().startswith("wandb"):
                logger_inst.experiment.log({f"{tag}/Epoch_{current_epoch:02d}": wandb.Image(fig)
                                           for tag, fig in zip(tags, self.figs)})
            # MLflow
            elif logger_inst.__class__.__name__.lower().startswith("mlflow"):
                for tag, fig in zip(tags, self.figs):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        filename = os.path.join(tmpdir, f"{tag}_Epoch_{current_epoch:04d}.png")
                        fig.savefig(filename)
                        logger_inst.experiment.log_artifact(logger_inst.run_id, filename, artifact_path="figures")

        # Close figures to free memory
        plt.close('all')
        self.figs.clear()

    def on_train_epoch_end(self, trainer, model):
        """Log figures at end of each training epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance
        model : pytorch_lightning.LightningModule
            The model instance
        """
        # If in sanity checking, skip logging
        if trainer.sanity_checking:
            return

        # Epoch
        current_epoch = trainer.current_epoch
        # Decide whether to save figures
        if not self._figure_saver(trainer, current_epoch):
            return

        # Build and log figures
        tags = self._get_tags(model, current_epoch)
        self._figure_builder(trainer, model, tags, current_epoch)
        self._figure_buffer(trainer, tags, current_epoch)

    def _get_tags(self, model, current_epoch: int):
        """Get figure tags. Override in subclasses for specialized behavior.

        Parameters
        ----------
        model : pytorch_lightning.LightningModule
            The model instance
        current_epoch : int
            Current epoch number

        Returns
        -------
        list of str
            Tags for figures
        """
        return []


class ForwardLogger(FigureLogger):
    """Callback to log radiance (forward model) RMSE figures.

    Specialized for forward models that predict radiance. Logs RMSE metrics
    in both absolute and normalized forms.
    """

    def _figure_builder(self, trainer, model, tags, current_epoch):
        """Build radiance RMSE figures.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance
        model : pytorch_lightning.LightningModule
            The model instance (should have metrics['hofx'] and metrics['hofx_norm'])
        tags : list of str
            Tags for each figure
        current_epoch : int
            Current epoch number
        """
        # Forward model RMSE
        hofx = [to_numpy(model.metrics['hofx'][key]['rmse'])
                for key in model.metrics['hofx'].keys()]
        hofx_norm = [to_numpy(model.metrics['hofx_norm'][key]['rmse'])
                     for key in model.metrics['hofx_norm'].keys()]

        self.figs.append(fig_rmse_bars(hofx, hofx_norm,
                                       labels=list(model.metrics['hofx'].keys()),
                                       title=[f"Epoch {current_epoch:02d} - Forward model errors",
                                              f"Epoch {current_epoch:02d} - Normalized forward model errors"]))

    def _get_tags(self, model, current_epoch: int):
        """Get radiance figure tags.

        Returns
        -------
        list of str
            Tags for radiance figures
        """
        return ["Valid_RadianceRMSE"]


class InverseLogger(FigureLogger):
    """Callback to log vertical profile data.

    Specialized for inverse models that predict atmospheric profiles. Logs vertical
    profiles (mean, standard deviation, RMSE) only.

    Note: For inverse models that also want radiance figures, instantiate both
    InverseLogger and ForwardLogger callbacks in config.
    """

    def _figure_builder(self, trainer, model, tags, current_epoch):
        """Build profile figures.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance (should have datamodule for pressure levels)
        model : pytorch_lightning.LightningModule
            The model instance (should have prof_vars and metrics)
        tags : list of str
            Tags for each figure
        current_epoch : int
            Current epoch number
        """
        # Get profile variable labels
        prof_labels = model.prof_vars if hasattr(model, 'prof_vars') else None

        # Get pressure levels (denormalized from datamodule)
        # TODO: Handle different kinds of pressure (update loader as necessary)
        pressure_levels = instantiate(
            trainer.datamodule.stage.valid.context.pressure.transformations.normalization,  # type: ignore
            inverse_transform=True
        )(to_numpy(model.results['pressure'])).flatten()
        # Convert from log10 space to physical space
        if model.results['pressure'].max() < 10.:
            pressure_levels = 10**pressure_levels
        pressure_levels = 0.01 * pressure_levels

        # Build profile figures (vertical profiles only)
        if 'prof' in model.metrics and len(model.metrics['prof']) > 0:
            # Extract profiles
            prof_mean = [to_numpy(model.metrics['prof_target']['mean']),
                        to_numpy(model.metrics['prof']['mean'])]
            prof_stdev = [to_numpy(model.metrics['prof_target']['stdev']),
                         to_numpy(model.metrics['prof']['stdev'])]
            prof_rmse = [to_numpy(model.metrics['prof']['rmse'])]
            prof_mean_labels, prof_mean_colors = ['Target', 'Prediction'], ['#1f77b4', '#ff7f0e']
            prof_rmse_labels, prof_rmse_colors = ['Target-Prediction'], ['#ff7f0e']

            # Add background if available
            if 'prof_background' in model.metrics and len(model.metrics['prof_background']) > 0:
                prof_mean.insert(0, to_numpy(model.metrics['prof_background']['mean']))
                prof_stdev.insert(0, to_numpy(model.metrics['prof_background']['stdev']))
                prof_rmse.insert(0, to_numpy(model.metrics['prof_background']['rmse']))
                prof_mean_labels.insert(0, 'Target-Background')
                prof_mean_colors.insert(0, '#2ca02c')
                prof_rmse_labels.insert(0, 'Target-Background')
                prof_rmse_colors.insert(0, '#2ca02c')

            # Profile Mean
            self.figs.append(fig_vertical_profiles(prof_mean, prof_mean_labels, stdev=prof_stdev,
                                                  y=pressure_levels, y_label='Pressure (hPa)',
                                                  x_label='Profile value (no units)',
                                                  color=prof_mean_colors,
                                                  title=[f"Epoch {current_epoch:02d} - {prof_label}"
                                                         for prof_label in prof_mean_labels]))
            # Profile RMSE
            self.figs.append(fig_vertical_profiles(prof_rmse, prof_rmse_labels,
                                                  y=pressure_levels, y_label='Pressure (hPa)',
                                                  x_label='Profile value (no units)',
                                                  color=prof_rmse_colors,
                                                  title=[f"Epoch {current_epoch:02d} - {prof_label}"
                                                         for prof_label in prof_rmse_labels]))

    def _get_tags(self, model, current_epoch: int):
        """Get profile figure tags.

        Returns
        -------
        list of str
            Tags for profile figures only
        """
        return ["Valid_ProfilesMean", "Valid_ProfilesRMSE"]

