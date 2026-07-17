import numpy as np
import torch
from typing import Union, Any
from pytorch_lightning import LightningModule
from src.preprocessing.statistics import statistics, accumulate_statistics
from src.architecture.activation import Swish, Scale, Sine
from src.architecture.ode import PressureConditionalODEFunc
from omegaconf import DictConfig
from utilities.instantiators import instantiate
import gc
from typing import Callable


class BaseModel(LightningModule):
    """
    Lightning model template.
    """

    def __init__(self, architecture: DictConfig, optimizer: DictConfig | None = None,
                 lr_scheduler: DictConfig | None = None, loss_func: DictConfig | Callable | None = None) -> None:
        """ Initialize model.

        Parameters
        ----------
        architecture: DictConfig. Configuration object for the model architecture.
        optimizer: DictConfig. Optimizer for the model.
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).
        loss_func: DictConfig | Callable. Loss function for the model.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Model architecture
        self.architecture = instantiate(architecture)
        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler
        # Optimizer initialization
        self.optimizer = optimizer
        # Loss function
        if loss_func is not None:
            if isinstance(loss_func, DictConfig):
                self.loss_func = instantiate(loss_func)
            elif isinstance(loss_func, Callable):
                self.loss_func = loss_func
            else:
                raise ValueError("loss_func must be a DictConfig or a callable.")
        else:
            self.loss_func = None
        # Store hyperparameters
        self.save_hyperparameters(ignore=['optimizer', 'lr_scheduler', 'loss_func'])

        # Stage outputs
        self.output: dict[str, list] = {}
        self.output_metrics: dict[str, dict] = {}

    def base_step(self, batch: dict, batch_nb: int, stage: str) -> torch.Tensor | dict:
        """ Perform training/validation/test step.

            Parameters
            ----------
            batch: dict. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.
            stage: str. Current operation: "train", "valid", or "test".

            Returns
            -------
            Loss value: tensor.
        """

        # Forward pass
        step = {'outputs': self.forward(batch['input'])}
        # Stage-dependent operation: Loss
        if stage in ('train', 'valid', 'test') and self.loss_func is not None:
            # Compute loss
            loss = self.loss_func(step['outputs'], batch['target'])
            # If dictionary with multiple terms
            if isinstance(loss, dict):
                # Track total loss
                step['loss'] = loss['total']
                # Detach loss components
                step[f'{stage}_loss'] = {
                    key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                    else value for key, value in loss.items()
                }
            elif isinstance(loss, torch.Tensor):
                step['loss'] = loss.mean()
                step[f'{stage}_loss'] = loss.detach().cpu().numpy()
            else:
                step['loss'] = loss
        # Detach outputs
        step['outputs'] = {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
            else value for key, value in step['outputs'].items()
        }

        return step

    def training_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """ Perform training step.

            Parameters
            ----------
            batch: dict. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='train')

    def validation_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """ Perform validation step.

            Parameters
            ----------
            batch: dict. Batch from the validation set.
            batch_nb: int. Index of the batch out of the validation set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='valid')

    def test_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """ Perform test step.

            Parameters
            ----------
            batch: dict. Batch from the test set.
            batch_nb: int. Index of the batch out of the test set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='test')

    def on_stage_epoch_end(self):
        """ Callback to log validation results at the end of each validation epoch.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Clear the lists for the next epoch
        for k in self.output:
            self.output[k] = []
        for k in self.output_metrics:
            self.output_metrics[k] = {}

    def on_train_epoch_end(self):
        """ Callback to log training results at the end of each training epoch.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Clear the lists for the next epoch
        self.on_stage_epoch_end()
        gc.collect()

    def on_validation_epoch_end(self):
        """ Callback to log validation results at the end of each validation epoch.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Clear the lists for the next epoch
        # self.on_stage_epoch_end()
        pass

    def on_test_epoch_start(self):
        """ Perform test epoch start.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Empty lists for test results
        self.on_stage_epoch_end()

    def on_test_epoch_end(self):
        """ Perform test epoch end.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Aggregate test results and convert to numpy array
        for k in self.output:
            self.output[k] = np.concatenate(self.output[k], axis=0)  # type: ignore

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer | dict[str, Any]] | None:
        """ Instantiate optimizer.

            Parameters
            ----------
            None. Target and parameters are passed from self.optimizer_config.

            Returns
            -------
            Optimizer instance.
        """

        # Check if optimizer is defined
        if self.optimizer is not None:

            # Instantiate optimizer
            optimizer = instantiate(self.optimizer, params=self.parameters())

            # Check if learning rate scheduler is defined
            if self.lr_scheduler is not None:
                # Instantiate learning rate scheduler
                lr_scheduler = instantiate(self.lr_scheduler, optimizer=optimizer)

                # Check if the learning rate scheduler is specifically reducing on plateau
                reduce_on_plateau = isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

                # Instantiate from config object
                return {'optimizer': optimizer,
                        'lr_scheduler': {'scheduler': lr_scheduler,
                                         'interval': 'epoch',
                                         'monitor': 'valid_loss',
                                         'frequency': 1,
                                         'reduce_on_plateau': reduce_on_plateau,
                                         }
                        }
            return optimizer
        return None

    def to(self, device, dtype: torch.dtype | None = None, non_blocking: bool = False) -> 'BaseModel':
        """ Move the model and loss function to the specified device.

        Parameters
        ----------
        device: torch.device. The device to move the model and loss function to.
        dtype: torch.dtype. The desired data type of the model parameters (optional).
        non_blocking: bool. If True, and the source is in pinned memory, the

        Returns
        -------
        BaseModel. The instance with model and loss function moved to the specified device.
        """

        # Class inheritance
        super().to(device, dtype=dtype, non_blocking=non_blocking)
        # Move loss function to device
        if hasattr(self.loss_func, 'to'):
            self.loss_func = self.loss_func.to(device)
        return self
