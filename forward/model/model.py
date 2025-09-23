import numpy as np
import torch
import torch.nn as nn
from typing import Union, Any
from pytorch_lightning import LightningModule
from forward.model.activation import Swish, Scale
from omegaconf import DictConfig
from utilities.instantiators import instantiate


class BaseModel(LightningModule):
    """
    Lightning model for the CRTM emulator (Community Radiative Transfer Model) using PyTorch.
    This is translation from Keras to Pytorch of the CRTM emulator by Howard et al. (2025).
    Link: https://zenodo.org/records/13963758.
    """
    def __init__(self, optimizer: DictConfig = None, lr_scheduler: DictConfig = None, loss_func: DictConfig = None,
                 log_valid: bool = True):
        """ Initialize LightningCRTMModel.

        Parameters
        ----------
        optimizer: DictConfig. Optimizer for the model.
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).
        loss_func: DictConfig. Loss function for the model.
        log_valid: bool. If True, log validation results at the end of each validation epoch.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()
        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler
        # Optimizer initialization
        self.optimizer = optimizer
        # Loss function
        self.loss_func = instantiate(loss_func) if loss_func is not None else None
        # Store hyperparameters
        self.save_hyperparameters(ignore=['optimizer', 'lr_scheduler', 'loss_func'])

        # Validation results
        self.log_valid = log_valid
        if self.log_valid:
            self.valid_results = {'hofx_target': [],
                                  'hofx_pred': [],
                                  'cloud_filter': []}
        # Test set results
        self.test_results = {'hofx': []}

    def base_step(self, batch: dict, batch_nb: int, stage: str) -> torch.Tensor:
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
        pred = self(batch['input'])
        # Compute loss function
        loss = self.loss_func(pred, batch['target']['hofx'])

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Log metrics
            self.log(f"{stage}_loss", loss.mean(), on_epoch=True, prog_bar=True, logger=False)
            # Store test outputs
            self.test_results['hofx'].append(pred.detach().cpu().numpy())
        elif stage == 'valid':
            # Log metrics
            self.log(f"{stage}_loss", loss.mean(), on_epoch=True, prog_bar=True, logger=True)
            # Store validation outputs
            if self.log_valid:
                self.valid_results['cloud_filter'].append(batch['cloud_filter'].detach().cpu().numpy())
                self.valid_results['hofx_target'].append(batch['target']['hofx'].detach().cpu().numpy())
                self.valid_results['hofx_pred'].append(pred.detach().cpu().numpy())

        return loss

    def training_step(self, batch: dict, batch_nb: int) -> torch.Tensor:
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

    def validation_step(self, batch: dict, batch_nb: int) -> torch.Tensor:
        """ Perform validation step.

            Parameters
            ----------
            batch: tensor. Batch from the validation set.
            batch_nb: int. Index of the batch out of the validation set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='valid')

    def test_step(self, batch: dict, batch_nb: int) -> torch.Tensor:
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
        if self.log_valid:
            for k in self.valid_results:
                self.valid_results[k].clear()

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
        for k in self.test_results:
            self.test_results[k] = []

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
        for k in self.test_results:
            if k in ['prof', 'hofx']:
                self.test_results[k] = np.concatenate(self.test_results[k], axis=0)  # type: ignore

    def configure_optimizers(self) -> Union[dict[str, Union[torch.optim.Optimizer, dict[str, Any]]], None]:
        """ Instantiate optimizer.

            Parameters
            ----------
            None. Target and parameters are passed from self.optmizer_config.

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
                print('Reduce on plateau:', reduce_on_plateau)

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

    def to(self, device, dtype: torch.dtype =None, non_blocking: bool = False) -> 'BaseModel':
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

        super().to(device, dtype=dtype, non_blocking=non_blocking)
        if hasattr(self.loss_func, 'to'):
            self.loss_func = self.loss_func.to(device)
        return self


class CRTMModel(BaseModel):
    """
    Lightning model for the CRTM emulator (Community Radiative Transfer Model) using PyTorch.
    This is translation from Keras to Pytorch of the CRTM emulator by Howard et al. (2025).
    Link: https://zenodo.org/records/13963758.
    """
    def __init__(self, parameters: DictConfig, optimizer: DictConfig = None, lr_scheduler: DictConfig = None,
                 loss_func: DictConfig = None, log_valid: bool = True):
        """ Initialize LightningCRTMModel.

        Parameters
        ----------
        optimizer: DictConfig. Optimizer for the model.
        loss_func: DictConfig. Loss function for the model.
        parameters: DictConfig. Configuration object containing model parameters.
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).
        log_valid: bool. If True, log validation results at the end of each validation epoch.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func, log_valid=log_valid)

        # Input parameters
        self.nprofvars = len(parameters.data.use_prof_vars)
        self.nsurfvars = len(parameters.data.use_surf_vars)
        self.nmetavars = len(parameters.data.use_meta_vars)
        self.nlevels = int(parameters.data.nlevels)
        self.prof_vars = parameters.data.prof_vars
        self.data_dtype = getattr(torch, parameters.data.dtype)

        # Neural network parameters
        nnodes_bt = parameters.architecture.nnodes_bt
        nhidden_bt = parameters.architecture.nhidden_bt
        dropout_rate = parameters.architecture.dropout_rate
        self.max_T = parameters.data.bt_norm_max
        self.min_T = parameters.data.bt_norm_min
        self.bt_output_activation = nn.Sigmoid()
        self.std_output_activation = nn.Softplus()
        self.std_output_activation_offset = parameters.architecture.std_output_activation_offset
        self.std_scale_trainable = parameters.architecture.std_scale_trainable

        # Neural network layer components
        self.flatten = nn.Flatten()
        self.concat = lambda *tensors: torch.cat(tensors, dim=1)
        self.hidden_layers = nn.ModuleList()
        self.swish_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First dense layer
        self.hidden_layers.append(nn.Linear(self.nprofvars * self.nlevels + self.nsurfvars + self.nmetavars, nnodes_bt))
        self.swish_layers.append(Swish())
        self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Additional hidden layers
        for _ in range(nhidden_bt - 1):
            self.hidden_layers.append(nn.Linear(nnodes_bt, nnodes_bt))
            self.swish_layers.append(Swish())
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Output layers
        self.out_T = nn.Linear(nnodes_bt, 10)
        self.out_std = nn.Linear(nnodes_bt, 10)
        if self.std_scale_trainable:
            self.std_scale = Scale()
        else:
            self.std_scale = None

    def forward(self, input: dict) -> torch.Tensor:
        """ Forward pass for the model.

        Parameters
        ----------
        input: dict. Dictionary containing input tensors (profiles, surface, meta).
            profiles: torch.Tensor. Input tensor for profiles.
            surface: torch.Tensor. Input tensor for surface variables.
            meta: torch.Tensor. Input tensor for meta variables.

        Returns
        -------
        torch.Tensor. Output tensor after passing through the model.
        """

        # Reformat variables
        prof = input['prof']  # (batch, nprofvars, nlevels)
        prof = self.flatten(prof)
        x = self.concat(prof, input['surf'], input['meta'])

        # Foward pass through hidden layers
        for dense, swish, drop in zip(self.hidden_layers, self.swish_layers, self.dropout_layers):
            x = dense(x)
            x = swish(x)
            x = drop(x)

        # Mean output
        out = self.out_T(x)
        out = self.bt_output_activation(out)
        out = out * (self.max_T - self.min_T) + self.min_T

        # Std output
        out_std = self.out_std(x)
        out_std = self.std_output_activation(out_std)
        if self.std_scale is not None:
            out_std = self.std_scale(out_std)
        out_std = out_std + self.std_output_activation_offset

        # Concatenate outputs
        return torch.cat([out, out_std], dim=1)

    def predict_step(self, batch: dict, batch_nb: int):
        """ Perform prediction step.

            Parameters
            ----------
            batch: dict. Batch from the prediction set.
            batch_nb: int. Index of the batch out of the prediction set.

            Returns
            -------
            Predicted values: tensor.
        """

        # Forward pass through the model
        return self(batch['input'])
