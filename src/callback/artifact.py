from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from src.evaluation.plot import fig_rmse_bars, fig_vertical_profiles
from src.preprocessing.statistics import RunningStats
import numpy as np
import torch
import wandb
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


class ArtifactLogger(Callback):
    """Base callback to log figures at the end of each validation epoch.

    Uses RunningStats for incremental metric accumulation during validation,
    avoiding the need to store all validation outputs in memory.

    Handles all common logic for:
    - Metric accumulation per batch using RunningStats
    - Figure saving, monitoring, and logging to different backends
    - Epoch-based and improvement-based saving conditions
    """

    def __init__(
        self,
        statistics: dict[str, list] | None = None,
        save_every_n_epochs: int = 100,
        save_on_improvement: bool = True,
        monitor: str = "valid_loss",
        monitor_mode: str = "min"
    ) -> None:
        """
        Initialize ArtifactLogger callback.

        Parameters
        ----------
        statistics : dict of str to list, optional
            dictionary specifying which metrics to compute for each runner.
            Example: {'hofx': ['mean', 'stdev', 'rmse', 'mae']}
            Default: ['mean', 'stdev', 'rmse', 'mae']
        save_every_n_epochs : int, default 100
            Frequency of saving figures in epochs. None or 0 disables epoch-based saving.
        save_on_improvement : bool, default True
            Whether to save figures only when there is improvement in monitored metric.
        monitor : str, default "valid_loss"
            Metric to monitor for improvements.
        monitor_mode : str, default "min"
            Whether to minimize or maximize the monitored metric ("min" or "max").
        """

        # Class inheritance
        super().__init__()

        # Metric accumulation
        self.statistics = statistics or ['mean', 'stdev', 'rmse', 'mae']
        self.runners: dict[str, RunningStats] = {}
        self.metrics: dict[str, dict] = {}

        # Initialize list of figures
        self.figs = []

        # Parameters for saving on improvement
        self.save_every_n_epochs = None if save_every_n_epochs in (None, 0) else int(save_every_n_epochs)
        self.save_on_improvement = bool(save_on_improvement)
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.best_monitor = float("inf") if self.monitor_mode == "min" else -float("inf")

        # Tags for figures (set in subclasses)
        self.tags = []

    def on_validation_epoch_start(self, trainer, pl_module):
        """
        Initialize fresh RunningStats runners for this validation epoch.

        Subclasses should override to define self.runners with appropriate
        runner instances for their specific metrics.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """

        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Process each validation batch and accumulate metrics using RunningStats.

        Dynamically tracks all variables present in outputs['output'] that have
        corresponding runners initialized in on_validation_epoch_start().

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        outputs : dict
            Model's validation_step output with predictions.
        batch : dict
            Input batch with inputs, targets, and context.
        batch_idx : int
            Batch index (unused).
        dataloader_idx : int
            Dataloader index (unused).
        """

        # Skip if sanity checking
        if trainer.sanity_checking:
            return

        # Extract predictions and targets
        predictions = outputs.get('output', {})
        targets = batch.get('target', {})

        # Update runners for each variable in predictions that has a runner
        for var_name in predictions.keys():
            if var_name in self.runners:
                # Convert tensor targets to numpy if needed
                target_data = targets.get(var_name)
                if isinstance(target_data, torch.Tensor):
                    target_data = target_data.detach().cpu().numpy()

                self.runners[var_name].update(
                    data=predictions[var_name],
                    target=target_data,
                    axis=0
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Finalize metrics at end of validation epoch.

        Computes final statistics from accumulated runners and creates figures.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """

        # Skip if sanity checking
        if trainer.sanity_checking:
            return

        # Finalize metrics from runners
        self.metrics = {}
        for runner_name, runner in self.runners.items():
            self.metrics[runner_name] = runner.compute(dtype='float32')

        # Build figures from finalized metrics
        self._figure_builder(trainer, pl_module)

    def _get_monitor_value(self, trainer):
        """
        Retrieve current value of monitored metric.

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
        """
        Determine whether to save figures based on specified conditions.

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


    def _figure_buffer(self, trainer, tags, current_epoch):
        """
        Add figures to logger at end of validation epoch.

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
        """
        Log figures at end of each training epoch.

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

        # Log figures (empty if no validation was performed)
        if self.figs:
            self._figure_buffer(trainer, self.tags, current_epoch)

    def _figure_builder(self, trainer, pl_module):
        """
        Build and create figures from finalized metrics.

        Subclasses should override to create specific plots from self.metrics.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.

        The self.metrics dict contains per-runner statistics:
            {
                'runner_name': {
                    'mean': array,
                    'stdev': array,
                    'rmse': array,
                    'mae': array,
                }
            }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _figure_builder()"
        )


class ForwardLogger(ArtifactLogger):
    """
    Callback to log brightness temperature (hofx) metrics for forward models.

    Uses RunningStats to accumulate hofx prediction metrics incrementally during
    validation, then plots RMSE and other statistics at epoch end.

    Specialized for forward models that predict radiance/brightness temperature.
    """

    def __init__(
        self,
        statistics: dict[str, list] | None = None,
        save_every_n_epochs: int = 100,
        save_on_improvement: bool = True,
        monitor: str = "valid_loss",
        monitor_mode: str = "min"
    ) -> None:
        """
        Initialize ForwardLogger with hofx figure tags.

        Parameters
        ----------
        statistics : dict, optional
            Metrics to compute. Default: ['mean', 'stdev', 'rmse', 'mae']
        save_every_n_epochs : int
            Frequency for epoch-based saving.
        save_on_improvement : bool
            Whether to save on validation improvement.
        monitor : str
            Metric to monitor.
        monitor_mode : str
            Minimize or maximize ('min' or 'max').
        """

        # Class inheritance
        super().__init__(statistics, save_every_n_epochs, save_on_improvement, monitor, monitor_mode)
        # Figure tags
        self.tags = ["Valid_RadianceRMSE"]

    def on_validation_epoch_start(self, trainer, pl_module):
        """
        Initialize runners for hofx metrics.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """
        # Create runners for hofx metrics
        # hofx shape: [batch, 20] (10 channels × 2 for mean+stdev)
        self.runners = {
            'hofx': RunningStats(which=self.statistics),
            'hofx_forward': RunningStats(which=self.statistics),
        }

    def _figure_builder(self, trainer, pl_module):
        """
        Build RMSE bar figures for hofx predictions.

        Creates plots showing RMSE and normalized RMSE for each brightness
        temperature channel.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """
        # Determine which variable is available
        hofx_var = None
        if 'hofx' in self.metrics:
            hofx_var = 'hofx'
        elif 'hofx_forward' in self.metrics:
            hofx_var = 'hofx_forward'

        if hofx_var is None:
            logger.warning("No hofx or hofx_forward metrics computed for this epoch")
            return

        hofx_stats = self.metrics[hofx_var]

        # Extract metrics (already numpy arrays from RunningStats.compute())
        rmse = hofx_stats.get('rmse')

        if rmse is None:
            logger.warning(f"RMSE not computed for {hofx_var} metrics")
            return

        # Channel labels (typically 10 MW channels)
        n_channels = len(rmse) if isinstance(rmse, (list, np.ndarray)) else 1
        # Get profile variable labels
        channel_labels = [f"Ch_{i}" for i in range(n_channels)]
        try:
            if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
                channel_labels = trainer.datamodule.stage.valid.target.variables.hofx.type
        except Exception as e:
            logger.warning(f"Could not retrieve hofx labels: {e}")

        # Create figure with RMSE bars
        self.figs.append(
            fig_rmse_bars(
                [rmse],
                None,
                labels=channel_labels,
                title=[f"{hofx_var} RMSE per channel"]
            )
        )


class InverseLogger(ArtifactLogger):
    """
    Callback to log atmospheric profile metrics for inverse models.

    Uses RunningStats to accumulate profile prediction metrics incrementally
    during validation, then plots vertical profiles at epoch end.

    Specialized for inverse models that predict atmospheric profiles.
    """

    def __init__(
        self,
        statistics: dict[str, list] | None = None,
        save_every_n_epochs: int = 100,
        save_on_improvement: bool = True,
        monitor: str = "valid_loss",
        monitor_mode: str = "min"
    ) -> None:
        """
        Initialize InverseLogger with profile figure tags.

        Parameters
        ----------
        statistics : dict, optional
            Metrics to compute. Default: ['mean', 'stdev', 'rmse', 'mae']
        save_every_n_epochs : int
            Frequency for epoch-based saving.
        save_on_improvement : bool
            Whether to save on validation improvement.
        monitor : str
            Metric to monitor.
        monitor_mode : str
            Minimize or maximize ('min' or 'max').
        """
        super().__init__(statistics, save_every_n_epochs, save_on_improvement, monitor, monitor_mode)
        self.tags = ["Valid_ProfilesMean", "Valid_ProfilesRMSE"]
        # Store pressure levels from first validation batch
        self.pressure_levels = None
        # Flag to initialize static runners
        self.runners_static = True

    def on_validation_epoch_start(self, trainer, pl_module):
        """
        Initialize runners for profile metrics.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """
        # Create runners for profile metrics
        # Profile shape: [batch, vars, levels]
        self.runners = {
            'prof': RunningStats(which=self.statistics),
        }
        # Static runners
        if self.runners_static:
            self.runners['prof_target'] = RunningStats(which=self.statistics)
            self.runners['prof_prior'] = RunningStats(which=self.statistics)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Process validation batch and extract pressure levels from first batch.
        Also accumulate prof_prior statistics.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        outputs : dict
            Model's validation_step output with predictions.
        batch : dict
            Input batch with inputs, targets, and context.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        # Store pressure levels from first batch (same for all samples)
        if batch_idx == 0 and self.pressure_levels is None:
            try:
                if 'context' in batch and 'pressure' in batch['context']:
                    pressure = batch['context']['pressure']
                    # Convert tensor to numpy if needed
                    if isinstance(pressure, torch.Tensor):
                        pressure = pressure.detach().cpu().numpy()
                    # Get first sample's pressure levels (same for all batch samples)
                    self.pressure_levels = pressure[0].flatten() if pressure.ndim > 1 else pressure.flatten()
            except Exception as e:
                logger.warning(f"Could not extract pressure levels from batch: {e}")

        # Call parent's on_validation_batch_end to process prof metrics
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        # Separately accumulate prof_prior vs prof_target stats (prior vs truth)
        # Only compute once as background doesn't change across epochs
        if self.runners_static:
            # Target
            prof_target = batch.get('target', {}).get('prof', None)
            if prof_target:
                # Convert tensors to numpy if needed
                if isinstance(prof_target, torch.Tensor):
                    prof_target = prof_target.detach().cpu().numpy()
                # Update metrics
                self.runners['prof_target'].update(
                    data=prof_target,
                    target=prof_target,  # Target vs itself (for mean/stdev computation)
                    axis=0
                )
            # Prior
            prof_prior = batch.get('target', {}).get('prof_prior', None)
            if prof_prior:
                # Convert tensors to numpy if needed
                if isinstance(prof_prior, torch.Tensor):
                    prof_prior = prof_prior.detach().cpu().numpy()
                # Update metrics
                self.runners['prof_prior'].update(
                    data=prof_prior,
                    target=prof_target,  # Compare prior vs truth
                    axis=0
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Finalize metrics at end of validation epoch.

        Computes final statistics from accumulated runners and creates figures.
        Marks prof_prior as computed after first epoch (background is static).

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """

        # Skip if sanity checking
        if trainer.sanity_checking:
            return

        # Finalize metrics from runners
        for runner_name, runner in self.runners.items():
            self.metrics[runner_name] = runner.compute(dtype='float32')

        # Mark prof_prior and prof_target as computed after first epoch (both are static)
        if self.runners_static:
            self.runners_static = not self.runners_static

        # Build figures from finalized metrics
        self._figure_builder(trainer, pl_module)

    def _figure_builder(self, trainer, pl_module):
        """
        Build vertical profile figures for atmospheric predictions.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """

        # Extract profile metrics
        if 'prof' not in self.metrics:
            logger.warning("No profile metrics computed for this epoch")
            return
        prof_stats = self.metrics['prof']
        prof_prior_stats = self.metrics.get('prof_prior', {})
        prof_target_stats = self.metrics.get('prof_target', {})

        # Extract metrics (already numpy arrays from RunningStats.compute())
        # Shape after reduction: [vars, levels]
        prof_mean = prof_stats.get('mean')
        prof_stdev = prof_stats.get('stdev')
        prof_rmse = prof_stats.get('rmse')
        prof_prior_mean = prof_prior_stats.get('mean')
        prof_prior_stdev = prof_prior_stats.get('stdev')
        prof_prior_rmse = prof_prior_stats.get('rmse')
        prof_target_mean = prof_target_stats.get('mean')
        prof_target_stdev = prof_target_stats.get('stdev')

        # Get profile variable labels
        prof_labels = None
        try:
            if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
                prof_labels = trainer.datamodule.stage.valid.target.variables.prof.type
        except Exception as e:
            logger.warning(f"Could not retrieve profile labels: {e}")

        # Default labels if not available
        if prof_labels is None:
            n_vars = prof_mean.shape[0] if len(prof_mean.shape) > 1 else 1
            prof_labels = [f"Var_{i}" for i in range(n_vars)]

        # Create profile mean figure
        if prof_mean is not None or prof_target_mean is not None or prof_prior_mean is not None:
            data = []
            stdev = []
            labels = []
            colors = []
            # Target
            if prof_target_mean is not None:
                data.append(prof_target_mean)
                stdev.append(prof_target_stdev)
                labels.append('Target')
                colors.append('#56B4E9')
            # Prior
            if prof_prior_mean is not None:
                data.append(prof_prior_mean)
                stdev.append(prof_prior_stdev)
                labels.append('Prior')
                colors.append('#009E73')
            # Prediction
            if prof_mean is not None:
                data.append(prof_mean)
                stdev.append(prof_stdev)
                labels.append('Output')
                colors.append('#E69F00')
            # Figure
            if data:
                self.figs.append(
                    fig_vertical_profiles(
                        data,
                        labels=labels,
                        colors=colors,
                        stdev=stdev,
                        y=self.pressure_levels,
                        y_label='Pressure (hPa)',
                        x_label='Profile value',
                        title=[f"Mean profile: {label}" for label in prof_labels]
                    )
                )

        # Create RMSE figure comparing prediction and prior vs target
        if prof_rmse is not None or prof_prior_rmse is not None:
            data = []
            labels = []
            colors = []
            # Prior
            if prof_prior_rmse is not None:
                data.append(prof_prior_rmse)
                labels.append('Prior-Target')
                colors.append('#009E73')
            # Prediction
            if prof_rmse is not None:
                data.append(prof_rmse)
                labels.append('Output-Target')
                colors.append('#E69F00')
            # Figure
            if data:
                self.figs.append(
                    fig_vertical_profiles(
                        data,
                        labels=labels,
                        colors=colors,
                        y=self.pressure_levels,
                        y_label='Pressure (hPa)',
                        x_label='RMSE value',
                        title=[f"RMSE: {label}" for label in prof_labels]
                    )
                )
