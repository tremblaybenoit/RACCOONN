import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from typing import Any


class LossLogger(Callback):
    """
    Callback to log loss components from batch step outputs.

    Replaces the pattern of embedding logging logic in models by extracting
    loss components and logging them via PyTorch Lightning's logging system.
    Supports per-variable loss breakdown for multidimensional tensors.

    For 2D/3D tensors (e.g., per-variable losses), it logs both the overall
    mean and per-variable breakdowns:
        train_loss_model_var_0, train_loss_model_var_1, ...
    """

    def __init__(self, log_per_variable: bool = True):
        """
        Initialize LossLogger callbacks.

        Parameters
        ----------
        log_per_variable : bool
            If True, log loss per variable for multidimensional loss tensors.
            Expects shapes like [batch, var] or [batch, var, level] where
            loss will be averaged over batch and level (if present) to produce
            per-variable metrics.
        """

        # Class inheritance
        super().__init__()

        # Store flag for per-variable logging
        self.log_per_variable = log_per_variable

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        """
        Log training batch losses.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            The model.
        outputs : dict
            Output dict from training_step containing 'train_loss'.
        batch : Any
            The input batch (unused).
        batch_idx : int
            Batch index (unused).
        """
        self._log_losses(pl_module, outputs, 'train')

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log training-specific metrics once per epoch (L2 norm, learning rate).

        These metrics provide insight into training stability and optimization progress.
        Only computed at epoch end to avoid redundant per-batch computation.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            The model.
        """
        self._log_train_metrics(pl_module)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int, dataloader_idx=0) -> None:
        """
        Log validation batch losses.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            The model.
        outputs : dict
            Output dict from validation_step containing 'valid_loss'.
        batch : Any
            The input batch (unused).
        batch_idx : int
            Batch index (unused).
        dataloader_idx : int
            Dataloader index (unused).
        """
        self._log_losses(pl_module, outputs, 'valid')

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any,
                          batch: Any, batch_idx: int, dataloader_idx=0) -> None:
        """
        Log test batch losses.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            The model.
        outputs : dict
            Output dict from test_step containing 'test_loss'.
        batch : Any
            The input batch (unused).
        batch_idx : int
            Batch index (unused).
        dataloader_idx : int
            Data loader index (unused).
        """
        # self._log_losses(pl_module, outputs, 'test')
        pass

    def _log_losses(self, pl_module, outputs: torch.Tensor | dict, stage: str):
        """
        Log loss components from batch outputs.

        Handles two output formats:
        1. Dict with f'{stage}_loss' key containing loss components.
           Extracts and logs each component with per-variable breakdowns.
        2. Scalar/tensor loss directly.
           Logs as '{stage}_loss_total'.

        For multidimensional loss tensors, logs both the overall mean and per-variable breakdowns.

        Parameters
        ----------
        pl_module : LightningModule
            The model.
        outputs : dict | torch.Tensor
            Output from step. Can be:
            - dict with f'{stage}_loss' key containing loss dict
            - Scalar tensor representing total loss
        stage : str
            One of 'train', 'valid', 'test'.
        """

        # Handle direct scalar/tensor loss
        if isinstance(outputs, torch.Tensor):
            value = outputs.detach().item() if outputs.ndim == 0 else outputs.detach().mean().item()
            pl_module.log(
                f'{stage}_loss',
                value,
                on_epoch=True,
                prog_bar=True,
                logger=True
            )
            return

        # Handle dictionary with multiple loss terms
        elif isinstance(outputs, dict):

            # Key associated with current stage's loss
            loss_key = f'{stage}_loss'
            pl_module.log(loss_key, outputs['loss'], on_epoch=True, prog_bar=True, logger=True)

            # If the loss wasn't computed from multiple terms
            # or multidimensional losses, then skip further logging
            if loss_key not in outputs:
                return

            # Extract dictionary of loss terms
            loss = outputs[loss_key]

            # Single loss term (multidimensional tensor)
            if isinstance(loss, np.ndarray):

                # If per-variable logging
                if self.log_per_variable:

                    # Compute mean across first dimension
                    per_var = loss.mean(axis=0)

                    # If 3D tensor, also average over level dimension
                    if per_var.ndim > 1:
                        per_var = per_var.mean(axis=-1)

                    # Log each variable
                    for i, var_loss in enumerate(per_var):
                        pl_module.log(
                            f'{stage}_loss_var_{i}',
                            var_loss.item(),
                            on_epoch=True,
                            prog_bar=False,
                            logger=True
                        )

            # Multiple loss terms
            elif not isinstance(loss, dict):

                # Log each loss component
                for key, value in loss.items():

                    # Handle scalar losses (int/float)
                    if isinstance(value, (int, float)):
                        pl_module.log(
                            f'{stage}_loss_{key}',
                            value,
                            on_epoch=True,
                            prog_bar=False,
                            logger=True
                        )

                    # Handle tensor losses
                    elif isinstance(value, np.ndarray):

                        # Scalar tensor (0D)
                        if value.ndim == 0:
                            pl_module.log(
                                f'{stage}_loss_{key}',
                                value.item(),
                                on_epoch=True,
                                prog_bar=False,
                                logger=True
                            )

                        # 1D tensor: [batch] or similar - log mean
                        elif value.ndim == 1:
                            pl_module.log(
                                f'{stage}_loss_{key}',
                                value.mean().item(),
                                on_epoch=True,
                                prog_bar=False,
                                logger=True
                            )

                        # 2D tensor: [batch, var] - log mean + per-variable
                        # 3D tensor: [batch, var, level] - log mean + per-variable (averaged over level)
                        elif value.ndim == 2 or value.ndim == 3:
                            pl_module.log(
                                f'{stage}_loss_{key}',
                                value.mean().item(),
                                on_epoch=True,
                                prog_bar=False,
                                logger=True
                            )

                            # If per-variable logging
                            if self.log_per_variable:

                                # Compute mean across first dimension
                                per_var = value.mean(axis=0)

                                # If 3D tensor, also average over level dimension
                                if per_var.ndim > 1:
                                    per_var = per_var.mean(axis=-1)

                                # Log each variable
                                for i, var_loss in enumerate(per_var):
                                    pl_module.log(
                                        f'{stage}_loss_{key}_var_{i}',
                                        var_loss.item(),
                                        on_epoch=True,
                                        prog_bar=False,
                                        logger=True
                                    )
            else:
                raise ValueError(f"Unsupported loss instance.")

    @staticmethod
    def _log_train_metrics(pl_module):
        """
        Log training-specific metrics (L2 norm, learning rate).

        These metrics provide insight into training stability and optimization progress.

        Parameters
        ----------
        pl_module : LightningModule
            The model.
        """

        # Log L2 norm of model parameters
        with torch.no_grad():
            l2_norm = sum(p.pow(2).sum() for p in pl_module.parameters()).sqrt().item()
        pl_module.log(
            'train_l2_norm',
            l2_norm,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True
        )

        # Log learning rate (if lr scheduler is configured)
        if pl_module.lr_schedulers() is not None:
            lr = pl_module.lr_schedulers().get_last_lr()[0]
            pl_module.log(
                'train_lr',
                lr,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True
            )