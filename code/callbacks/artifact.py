import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from code.evaluation.plot import fig_rmse_bars, fig_vertical_profiles
from code.preprocessing.statistics import RunningStats
import numpy as np
import torch
import wandb
import tempfile
import os
import logging
from typing import Any

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
        which_statistics: list | None = None,
        save_every_n_epochs: int = 100,
        save_on_improvement: bool = True,
        monitor: str = "valid_loss",
        monitor_mode: str = "min"
    ) -> None:
        """
        Initialize ArtifactLogger callback.

        Parameters
        ----------
        which_statistics : list of str, optional
            List specifying which metrics to compute for each runner.
            Example: ['mean', 'stdev', 'rmse', 'mae']
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
        self.which_statistics = which_statistics or ['mean', 'stdev', 'rmse', 'mae']
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

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int, dataloader_idx=0) -> None:
        """
        Process each validation batch and accumulate metrics using RunningStats.

        Subclasses should override this method to extract their specific variables
        from outputs and update their runners accordingly.

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
            Data loader index (unused).
        """

        # Skip if sanity checking
        if trainer.sanity_checking:
            return

        # Subclasses should override this method
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement on_validation_batch_end()"
        )
    
    def _update_runners(self, var_name, pred_data, target_data, batch):
        """
        Update runners with processed data. Can be overridden by subclasses for custom logic.

        Parameters
        ----------
        var_name : str
            Variable name.
        pred_data : ndarray
            Processed prediction data (numpy array).
        target_data : ndarray
            Processed target data (numpy array).
        batch : dict
            Full batch dict (may contain context, masks, etc).
        """
        if var_name in self.runners:
            self.runners[var_name].update(
                data=pred_data,
                target=target_data,
                axis=0
            )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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

        # Finalize metrics from runners that have accumulated data
        self.metrics = {}
        for runner_name, runner in self.runners.items():
            # Skip runners with no accumulated data (e.g., cloud_mask in clear-sky datasets)
            if isinstance(runner._n, (int, float)) and runner._n == 0.0:
                continue
            self.metrics[runner_name] = runner.compute(dtype='float32')

        # Build figures from finalized metrics
        self._figure_builder(trainer, pl_module)

    def _get_monitor_value(self, trainer: pl.Trainer) -> float | None:
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

    def _figure_saver(self, trainer: pl.Trainer, current_epoch: int) -> bool:
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


    def _figure_buffer(self, trainer: pl.Trainer, tags: list[str], current_epoch: int) -> None:
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

    def on_train_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
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
        which_statistics: list | None = None,
        save_every_n_epochs: int = 100,
        save_on_improvement: bool = True,
        monitor: str = "valid_loss",
        monitor_mode: str = "min"
    ) -> None:
        """
        Initialize ForwardLogger with hofx figure tags.

        Parameters
        ----------
        which_statistics : list of str, optional
            List specifying which metrics to compute for each runner.
            Example: ['mean', 'stdev', 'rmse', 'mae']
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
        super().__init__(
            which_statistics, 
            save_every_n_epochs, 
            save_on_improvement, 
            monitor, 
            monitor_mode
        )

        # Figure tags
        self.tags = ["Valid_RadianceRMSE"]

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int, dataloader_idx=0) -> None:
        """
        Process hofx predictions for brightness temperature metrics.

        Handles both 'hofx' and 'hofx_mean' prediction keys, extracts mean from
        dual predictions [mean, std], and delegates mask-based filtering to _update_runners().

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        outputs : dict
            Model's validation_step output with 'hofx' or 'hofx_mean' predictions.
        batch : dict
            Input batch with targets and context.
        batch_idx : int
            Batch index (unused).
        dataloader_idx : int
            Data loader index (unused).
        """

        # Skip if sanity checking
        if trainer.sanity_checking:
            return

        # Extract predictions and targets
        predictions = outputs.get('output', {})
        targets = batch.get('target', {})

        # Extract prediction and convert to numpy
        pred_data = predictions['bt_inverse'] if 'bt_inverse' in predictions else predictions.get('bt_forward')
        if isinstance(pred_data, torch.Tensor):
            pred_data = pred_data.detach().cpu().numpy()

        # If 'bt_forward_mean' is predicted but 'bt_forward_mean' is not a target, use 'bt_forward' target
        if 'bt_forward' in targets:
            target_data = targets.get('bt_forward')
        else:
            target_data = targets.get('bt_crtm')
        
        if isinstance(target_data, torch.Tensor):
            target_data = target_data.detach().cpu().numpy()

        # Extract only matching number of channels from target if needed
        n_pred_channels = pred_data.shape[1]
        n_target_channels = target_data.shape[1]
        if n_pred_channels != n_target_channels:
            pred_data = pred_data[:, :np.min(n_pred_channels, n_target_channels)]
            target_data = target_data[:, :np.min(n_target_channels, n_target_channels)]
        elif n_pred_channels > 10:
            pred_data = pred_data[:, :n_pred_channels // 2]
            target_data = target_data[:, :n_pred_channels // 2]
        else:
            pred_data = pred_data[:, :n_pred_channels]
            target_data = target_data[:, :n_pred_channels]

        # Update runners with mask-based filtering
        self._update_runners('bt', pred_data, target_data, batch)

    def _update_runners(self, var_name, pred_data, target_data, batch):
        """
        Update hofx runners with mask-based statistics.

        Applies cloud/clear and daytime/nighttime masks to separate hofx predictions
        before accumulating statistics. This allows per-condition performance analysis.

        Parameters
        ----------
        var_name : str
            Variable name (should be 'hofx' or 'hofx_forward').
        pred_data : ndarray
            Prediction data (already extracted mean).
        target_data : ndarray
            Target data.
        batch : dict
            Batch dict with 'context' containing masks.
        """
        # Extract context masks
        context = batch.get('context', {})
        
        # Apply masks and update corresponding runners
        masks = {
            'clear': context.get('clear_mask'),
            'cloud': context.get('cloud_mask'),
            'day': context.get('daytime_mask'),
            'night': context.get('nighttime_mask')
        }
        
        for runner_key, mask in masks.items():
            if runner_key in self.runners and mask is not None:
                # Convert mask to numpy if needed
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy().astype(bool)
                
                # Extract masked data and targets using boolean indexing
                masked_pred = pred_data[mask]
                masked_target = target_data[mask] if target_data is not None else None
                
                # Only update if we have data
                if len(masked_pred) > 0:
                    self.runners[runner_key].update(
                        data=masked_pred,
                        target=masked_target,
                        axis=0
                    )


    def on_validation_epoch_start(self, trainer, pl_module):
        """
        Initialize runners for hofx metrics by mask.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """
        # Create separate runners for each mask condition
        self.runners = {
            'clear': RunningStats(which=self.which_statistics),
            'cloud': RunningStats(which=self.which_statistics),
            'day': RunningStats(which=self.which_statistics),
            'night': RunningStats(which=self.which_statistics),
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

        # Get radiance channels
        channel_candidates = ['bt_forward', 'bt_crtm']
        target_datasets = trainer.datamodule.valid.target.datasets
        channels = next(
            (target_datasets[key].type for key in channel_candidates
             if key in target_datasets),
            [f"Channel {i}" for i in range(1, 11)]
        )
        n_channels = len(channels)

        # Extract metrics (already numpy arrays from RunningStats.compute())
        rmse = []
        labels = []
        colors = []
        colors_dict = {'clear': '#D81B60', 'cloud': '#1E88E5', 'day': '#FFC107', 'night': 'r'}
        for runner_name in self.runners.keys():
            if runner_name in self.metrics:
                rmse.append(self.metrics[runner_name]['rmse'])
            else:
                rmse.append(np.zeros((n_channels,)))
            labels.append(runner_name)
            colors.append(colors_dict.get(runner_name, '#000000'))

        # Create figure with RMSE bars
        self.figs.append(
            fig_rmse_bars(
                rmse,
                channels=np.arange(0, n_channels),
                x_range=[[0, 2.0]],
                labels=labels,
                colors=colors,
                title=[f"Forward model RMSE per channel"]
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
        which_statistics: list | None = None,
        save_every_n_epochs: int = 100,
        save_on_improvement: bool = True,
        monitor: str = "valid_loss",
        monitor_mode: str = "min"
    ) -> None:
        """
        Initialize InverseLogger with profile figure tags.

        Parameters
        ----------
        which_statistics : list | None, optional
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
        super().__init__(which_statistics, save_every_n_epochs, save_on_improvement, monitor, monitor_mode)

        # Plot tags
        self.tags = ["Valid_ProfilesMean", "Valid_ProfilesRMSE"]
        # Store pressure levels from first validation batch
        self.pressure_levels = None
        # Flag to initialize static runners
        self.runners_static = True

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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
            'prof_inverse': RunningStats(which=self.which_statistics),
        }
        # Static runners
        if self.runners_static:
            self.runners['prof_target'] = RunningStats(which=self.which_statistics)
            self.runners['prof_prior'] = RunningStats(which=self.which_statistics)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int, dataloader_idx=0) -> None:
        """
        Process validation batch and extract pressure levels from first batch.
        Also accumulate prof and prof_prior statistics.

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
            Data loader index.
        """

        # Skip if sanity checking
        if trainer.sanity_checking:
            return

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

        # Extract predictions and targets
        predictions = outputs.get('output', {})
        targets = batch.get('target', {})

        # Process prof (the only variable InverseLogger cares about)
        if 'prof_inverse' in predictions:
            # Extract prof prediction and convert to numpy
            pred_data = predictions['prof_inverse']
            if isinstance(pred_data, torch.Tensor):
                pred_data = pred_data.detach().cpu().numpy()

            # Extract prof target
            target_data = targets.get('prof')
            if isinstance(target_data, torch.Tensor):
                target_data = target_data.detach().cpu().numpy()

            # Update prof runner
            if 'prof_inverse' in self.runners:
                self.runners['prof_inverse'].update(
                    data=pred_data,
                    target=target_data,
                    axis=0
                )

        # Separately accumulate prof_prior vs prof_target stats (prior vs truth)
        # Only compute once as background doesn't change across epochs
        if self.runners_static:
            # Target
            prof_target = batch.get('target', {}).get('prof', None)
            if prof_target is not None:
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
            if prof_prior is not None:
                # Convert tensors to numpy if needed
                if isinstance(prof_prior, torch.Tensor):
                    prof_prior = prof_prior.detach().cpu().numpy()
                # Update metrics
                self.runners['prof_prior'].update(
                    data=prof_prior,
                    target=prof_target,  # Compare prior vs truth
                    axis=0
                )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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

    def _figure_builder(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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
        if 'prof_inverse' not in self.metrics:
            logger.warning("No profile metrics computed for this epoch")
            return
        prof_stats = self.metrics['prof_inverse']
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
                prof_labels = trainer.datamodule.valid.target.datasets.get('prof', {}).type
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
            label = []
            color = []
            # Target
            if prof_target_mean is not None:
                data.append(prof_target_mean)
                stdev.append(prof_target_stdev)
                label.append('Target')
                color.append('#56B4E9')
            # Prior
            if prof_prior_mean is not None:
                data.append(prof_prior_mean)
                stdev.append(prof_prior_stdev)
                label.append('Prior')
                color.append('#009E73')
            # Prediction
            if prof_mean is not None:
                data.append(prof_mean)
                stdev.append(prof_stdev)
                label.append('Output')
                color.append('#E69F00')
            # Figure
            if data:
                self.figs.append(
                    fig_vertical_profiles(
                        data,
                        label=label,
                        color=color,
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
            label = []
            color = []
            # Prior
            if prof_prior_rmse is not None:
                data.append(prof_prior_rmse)
                label.append('Prior-Target')
                color.append('#009E73')
            # Prediction
            if prof_rmse is not None:
                data.append(prof_rmse)
                label.append('Output-Target')
                color.append('#E69F00')
            # Figure
            if data:
                self.figs.append(
                    fig_vertical_profiles(
                        data,
                        label=label,
                        color=color,
                        y=self.pressure_levels,
                        y_label='Pressure (hPa)',
                        x_label='RMSE value',
                        title=[f"RMSE: {label}" for label in prof_labels]
                    )
                )
