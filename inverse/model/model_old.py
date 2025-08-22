import torch
import torch.nn as nn
from typing import Callable
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from forward.model.model import BaseModel
from forward.utilities.instantiators import instantiate
from forward.data.transformations import NormalizeProfiles
from inverse.data.transformations import min_max
import pickle
import numpy as np


class InverseEmulator(LightningModule):
    """Class for radiance transfer (PINN) inverse emulator."""
    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None):
        """ Initialize model.

        Parameters
        ----------
        optimizer : Callable. Optimizer for the model.
        loss_func : Callable. Loss function for the model.
        lr_scheduler : Callable. Learning rate scheduler for the model.

        Returns
        -------
        None.
        """

        # TODO: Complete model

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)


class PINNverseOperator(BaseModel):
    """Class for the Physics-Informed Neural Network (PINN) inverse model."""
    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None,
                 positional_encoding: DictConfig = None, activation_in: Callable = None,
                 activation_out: Callable = None, parameters: DictConfig = None, log_valid: bool = False):
        """ Initialize model.

        Parameters
        ----------
        optimizer : Callable. Optimizer for the model.
        loss_func : Callable. Loss function for the model.
        lr_scheduler : Callable. Learning rate scheduler for the model.
        positional_encoding : DictConfig. Configuration for the positional encoding.
        activation : DictConfig. Configuration for the activation function.
        parameters : DictConfig. Configuration for the model parameters.
        log_valid : bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Validation results
        self.log_valid = log_valid
        self.valid_bt_target = []
        self.valid_bt_pred = []
        self.valid_prof_target = []
        self.valid_prof_pred = []
        self.valid_pressure = []
        # Test results
        self.test_prof = []

        # Normalization transformations
        with open(parameters.data.statistics.path, 'rb') as file:
            stats = pickle.load(file)
        self.stats_profiles = {key: torch.tensor(value, dtype=torch.float32) for key, value in stats['prof'].items()}
        self.unnormalize_profiles = instantiate(min_max, stats=self.stats_profiles, inverse_transform=True, _partial_=True)
        self.stats_crtm_profiles = {'min': torch.tensor(parameters.data.prof_min, dtype=torch.float32),
                                    'max': torch.tensor(parameters.data.prof_max, dtype=torch.float32)}
        self.normalize_profiles = NormalizeProfiles(self.stats_crtm_profiles['min'], self.stats_crtm_profiles['max'],
                                                    inverse_transform=False)

        # Positional encoding
        d_input = (parameters.data.n_lat + parameters.data.n_lon + parameters.data.n_scans + parameters.data.n_surf +
                   parameters.data.n_meta + parameters.data.n_pressure)
        self.positional_encoding = instantiate(positional_encoding, d_input=d_input)
        # Input layer
        self.d_in = nn.Linear(self.positional_encoding.d_output, parameters.architecture.n_neurons)
        self.activation_in = instantiate(activation_in)
        self.dropout_in = nn.Dropout(parameters.architecture.dropout)
        # Output layer
        self.d_out = nn.Linear(parameters.architecture.n_neurons, parameters.data.n_profiles*parameters.data.n_levels)
        self.activation_out = instantiate(activation_out)
        self.n_profiles = parameters.data.n_profiles
        self.n_levels = parameters.data.n_levels
        self.prof_vars = parameters.data.prof_vars

        # Model architecture
        self.layers = nn.ModuleList([nn.Linear(parameters.architecture.n_neurons, parameters.architecture.n_neurons)
                                     for _ in range(parameters.architecture.n_layers)])
        self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(parameters.architecture.n_neurons)
                                               for _ in range(parameters.architecture.n_layers)])
        self.activations = nn.ModuleList([instantiate(activation_in) for _ in range(parameters.architecture.n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(parameters.architecture.dropout)
                                       for _ in range(parameters.architecture.n_layers)])
        self.model = nn.Sequential(
            self.d_in,
            self.activation_in,
            nn.Dropout(parameters.architecture.dropout),
            *[layer for hidden in zip(self.layers, self.batchnorm_layers, self.activations, self.dropouts) for layer in hidden],
            self.d_out,
            self.activation_out
        )

    def forward(self, x) -> torch.Tensor:
        """ Pass forward through neural network architecture.

            Parameters
            ----------
            x: tensor. Inputs: latitude, longitude, surface, and the metadata.

            Returns
            -------
            y: tensor. Outputs: predicted profiles, surface, and metadata.
        """

        # Unpack the input tensors
        lat, lon, scans, surface, meta, pressure = x

        # Concatenate inputs
        inputs = torch.cat((lat.view(-1, 1), lon.view(-1, 1), scans.view(-1, 1), surface, meta,
                            pressure.view(-1, 1)), dim=-1)
        # Apply positional encoding
        encoded_inputs = self.positional_encoding(inputs)

        # Pass through the model
        profiles = self.model(encoded_inputs)
        # Reshape profiles to match the expected output shape
        return profiles.view(-1, self.n_profiles, 1)

    def _retrieve_profiles(self, x, mask=None):
        """ Retrieve atmospheric profile over multiple pressure levels.

            Parameters
            ----------
            x: tensor. Input variables.
            mask: tensor: Mask to apply to profiles.

            Returns
            -------
            prof: tensor. Atmospheric profiles.
        """

        # Forward pass
        x_vector = [xi.repeat_interleave(x[-1].shape[-1], dim=0) for xi in x[:-1]] + [x[-1].reshape(-1, 1)]
        prof = self.forward(x_vector).view(-1, x[-1].shape[-1], self.n_profiles).transpose(1, 2)

        # Unnormalize and apply mask
        if mask is None:
            return self.unnormalize_profiles(prof)
        else:
            return self.unnormalize_profiles(prof)*mask

    def base_step(self, batch: torch.Tensor, batch_nb: int, stage: str) -> torch.Tensor:
        """ Perform training/validation/test step.

            Parameters
            ----------
            batch: tensor. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.
            stage: str. Current operation: "train", "valid", or "test".

            Returns
            -------
            Loss value: tensor.
        """

        # Check if loss function is defined
        if self.loss_func is None:
            raise ValueError("Loss function is not defined. Please provide a loss function.")

        # Extract data from batch
        x, y = batch

        # Compute mask from prof
        mask = (torch.sigmoid(1e6 * y[1]) - 0.5) * 2

        # Forward pass
        profiles = self._retrieve_profiles(x, mask=mask)

        # Compute loss function
        loss, bt_pred = self.loss_func((self.normalize_profiles(profiles), x[3], x[4]), y[0])
        # Compute L2 norm of the model parameters
        l2_norm = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)

        # Log metrics
        self.log(f"{stage}_loss", loss.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_l2_norm", l2_norm, on_epoch=True, prog_bar=False, logger=True)
        # Log individual radiance channels
        for i in range(bt_pred.shape[1]//2):
            self.log(f"{stage}_lo_{i}", loss[:, i].mean(), on_epoch=True, prog_bar=False, logger=True)

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Store test outputs
            self.test_prof.append(profiles.detach().cpu().numpy())
            self.test_hofx.append(bt_pred.detach().cpu().numpy())
        elif stage == 'valid' and self.log_valid:
            # Store validation outputs
            self.valid_bt_target.append(y[0].detach().cpu().numpy())
            self.valid_bt_pred.append(bt_pred.detach().cpu().numpy())
            self.valid_prof_target.append(y[1].detach().cpu().numpy())
            self.valid_prof_pred.append(profiles.detach().cpu().numpy())
            self.valid_pressure.append(x[5].detach().cpu().numpy())

        return loss.mean()

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
            self.valid_bt_target.clear()
            self.valid_bt_pred.clear()
            self.valid_prof_target.clear()
            self.valid_prof_pred.clear()
            self.valid_pressure.clear()

    def on_test_epoch_start(self):
        """ Perform test epoch start.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Aggregate test results and convert to numpy array
        self.test_hofx = []
        self.test_prof = []

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
        self.test_hofx = np.concatenate(self.test_hofx, axis=0)
        self.test_prof = np.concatenate(self.test_prof, axis=0)

    def to(self, device):
        """ Move model to the specified device.

            Parameters
            ----------
            device: torch.device. Device to move the model to.

            Returns
            -------
            self: PINNverseOperator. The model moved to the specified device.
        """

        super().to(device)
        # Move all tensors in stats dicts to the device
        for d in [self.stats_profiles, self.stats_crtm_profiles]:
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        d[k] = v.to(device)
        # Move normalization modules if needed
        for attr in ['normalize_profiles', 'unnormalize_profiles']:
            norm = getattr(self, attr, None)
            if hasattr(norm, 'to'):
                setattr(self, attr, norm.to(device))
        return self


class PINNvarOperator(PINNverseOperator):
    """Class for the Physics-Informed Neural Network (PINN) inverse model."""
    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None,
                 positional_encoding: DictConfig = None, activation_in: Callable = None,
                 activation_out: Callable = None, parameters: DictConfig = None, log_valid: bool = False):
        """ Initialize model.

        Parameters
        ----------
        optimizer : Callable. Optimizer for the model.
        loss_func : Callable. Loss function for the model.
        lr_scheduler : Callable. Learning rate scheduler for the model.
        positional_encoding : DictConfig. Configuration for the positional encoding.
        activation : DictConfig. Configuration for the activation function.
        parameters : DictConfig. Configuration for the model parameters.
        log_valid : bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func,
                         positional_encoding=positional_encoding, activation_in=activation_in,
                         activation_out=activation_out, parameters=parameters, log_valid=log_valid)

        if log_valid:
            self.valid_background = []
            self.valid_background_err = []

    def base_step(self, batch: torch.Tensor, batch_nb: int, stage: str):
        """ Perform training/validation/test step.

            Parameters
            ----------
            batch: tensor. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.
            stage: str. Current operation: "train", "valid", or "test".

            Returns
            -------
            Loss value: tensor.
        """

        # Check if loss function is defined
        if self.loss_func is None:
            raise ValueError("Loss function is not defined. Please provide a loss function.")

        # Extract data from batch
        x, y = batch

        # Compute mask from prof
        mask = (torch.sigmoid(1e6 * y[1]) - 0.5) * 2

        # Forward pass
        profiles = self._retrieve_profiles(x, mask=mask)

        # Compute loss function
        loss, bt_pred, obs_loss, background_loss = self.loss_func((self.normalize_profiles(profiles), x[3], x[4]),
                                                                  (y[0], y[1], y[2], y[3], profiles))

        # Log metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_lo", obs_loss.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_lb", background_loss.mean(), on_epoch=True, prog_bar=True, logger=True)
        # Log individual profile variables and radiance channels
        for i, var in enumerate(self.prof_vars):
            self.log(f"{stage}_lb_{i}_{var}", background_loss[:, i, :].mean(), on_epoch=True, prog_bar=False,
                     logger=True)
        for i in range(bt_pred.shape[1] // 2):
            self.log(f"{stage}_lo_{i}", obs_loss[:, i].mean(), on_epoch=True, prog_bar=False, logger=True)

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Store test outputs
            self.test_prof.append(profiles.detach().cpu().numpy())
            self.test_hofx.append(bt_pred.detach().cpu().numpy())
        elif stage == 'valid' and self.log_valid:
            # Store validation outputs
            self.valid_bt_target.append(y[0].detach().cpu().numpy())
            self.valid_bt_pred.append(bt_pred.detach().cpu().numpy())
            self.valid_prof_target.append(y[1].detach().cpu().numpy())
            self.valid_prof_pred.append(profiles.detach().cpu().numpy())
            self.valid_pressure.append(x[5].detach().cpu().numpy())
            self.valid_background.append(y[2].detach().cpu().numpy())
            self.valid_background_err.append(y[3].detach().cpu().numpy())
        else:
            # Compute L2 norm of the model parameters
            l2_norm = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)
            self.log(f"{stage}_l2_norm", l2_norm, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def on_validation_epoch_end(self):
        """ Callback to log validation results at the end of each validation epoch.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """
        super().on_validation_epoch_end()

        # Clear the lists for the next epoch
        if self.log_valid:
            self.valid_background.clear()
            self.valid_background_err.clear()


class PPINNvarOperator2(PINNvarOperator):
    """Class for the Physics-Informed Neural Network (PINN) inverse model."""
    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None,
                 positional_encoding: DictConfig = None, activation_in: Callable = None,
                 activation_out: Callable = None, parameters: DictConfig = None, log_valid: bool = False):
        """ Initialize model.

        Parameters
        ----------
        optimizer : Callable. Optimizer for the model.
        loss_func : Callable. Loss function for the model.
        lr_scheduler : Callable. Learning rate scheduler for the model.
        positional_encoding : DictConfig. Configuration for the positional encoding.
        activation : DictConfig. Configuration for the activation function.
        parameters : DictConfig. Configuration for the model parameters.
        log_valid : bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func,
                         positional_encoding=positional_encoding, activation_in=activation_in,
                         activation_out=activation_out, parameters=parameters, log_valid=log_valid)

    def base_step(self, batch: torch.Tensor, batch_nb: int, stage: str):
        """ Perform training/validation/test step.

            Parameters
            ----------
            batch: tensor. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.
            stage: str. Current operation: "train", "valid", or "test".

            Returns
            -------
            Loss value: tensor.
        """

        # Check if loss function is defined
        if self.loss_func is None:
            raise ValueError("Loss function is not defined. Please provide a loss function.")

        # Extract data from batch
        x, y = batch

        # Compute mask from prof
        mask = (torch.sigmoid(1e6 * y[1]) - 0.5) * 2

        # Forward pass
        profiles = self._retrieve_profiles(x, mask=mask)

        # Compute loss function
        loss, bt_pred, obs_loss, background_loss = self.loss_func((self.normalize_profiles(profiles), x[3], x[4]),
                                                                  (y[0], y[1], y[2], profiles))

        # Log metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_lo", obs_loss.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_lb", background_loss, on_epoch=True, prog_bar=True, logger=True)
        # Log individual profile variables and radiance channels
        for i in range(bt_pred.shape[1]//2):
            self.log(f"{stage}_lo_{i}", obs_loss[:, i].mean(), on_epoch=True, prog_bar=False, logger=True)

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Store test outputs
            self.test_prof.append(profiles.detach().cpu().numpy())
            self.test_hofx.append(bt_pred.detach().cpu().numpy())
        elif stage == 'valid' and self.log_valid:
            # Store validation outputs
            self.valid_bt_target.append(y[0].detach().cpu().numpy())
            self.valid_bt_pred.append(bt_pred.detach().cpu().numpy())
            self.valid_prof_target.append(y[1].detach().cpu().numpy())
            self.valid_prof_pred.append(profiles.detach().cpu().numpy())
            self.valid_pressure.append(x[5].detach().cpu().numpy())
            self.valid_background.append(y[2].detach().cpu().numpy())
            self.valid_background_err.append(y[3].detach().cpu().numpy())
        else:
            # Compute L2 norm of the model parameters
            l2_norm = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)
            self.log(f"{stage}_l2_norm", l2_norm, on_epoch=True, prog_bar=False, logger=True)

        return loss


class PPINNvarOperator3(PINNvarOperator):
    """Class for the Physics-Informed Neural Network (PINN) inverse model."""
    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None,
                 positional_encoding: DictConfig = None, activation_in: Callable = None,
                 activation_out: Callable = None, parameters: DictConfig = None, log_valid: bool = False):
        """ Initialize model.

        Parameters
        ----------
        optimizer : Callable. Optimizer for the model.
        loss_func : Callable. Loss function for the model.
        lr_scheduler : Callable. Learning rate scheduler for the model.
        positional_encoding : DictConfig. Configuration for the positional encoding.
        activation : DictConfig. Configuration for the activation function.
        parameters : DictConfig. Configuration for the model parameters.
        log_valid : bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func,
                         positional_encoding=positional_encoding, activation_in=activation_in,
                         activation_out=activation_out, parameters=parameters, log_valid=log_valid)

    def base_step(self, batch: torch.Tensor, batch_nb: int, stage: str):
        """ Perform training/validation/test step.

            Parameters
            ----------
            batch: tensor. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.
            stage: str. Current operation: "train", "valid", or "test".

            Returns
            -------
            Loss value: tensor.
        """

        # Check if loss function is defined
        if self.loss_func is None:
            raise ValueError("Loss function is not defined. Please provide a loss function.")

        # Extract data from batch
        x, y = batch

        # Compute mask from prof
        mask = (torch.sigmoid(1e6 * y[1]) - 0.5) * 2

        # Forward pass
        profiles = self._retrieve_profiles(x, mask=mask)

        # Compute loss function
        loss, bt_pred, obs_loss, background_loss = self.loss_func((self.normalize_profiles(profiles), x[3], x[4]),
                                                                  (y[0], y[1], y[2], profiles))

        # Log metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_lo", obs_loss.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_lb", background_loss, on_epoch=True, prog_bar=True, logger=True)

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Store test outputs
            self.test_prof.append(profiles.detach().cpu().numpy())
            self.test_hofx.append(bt_pred.detach().cpu().numpy())
        elif stage == 'valid' and self.log_valid:
            # Store validation outputs
            self.valid_bt_target.append(y[0].detach().cpu().numpy())
            self.valid_bt_pred.append(bt_pred.detach().cpu().numpy())
            self.valid_prof_target.append(y[1].detach().cpu().numpy())
            self.valid_prof_pred.append(profiles.detach().cpu().numpy())
            self.valid_pressure.append(x[5].detach().cpu().numpy())
            self.valid_background.append(y[2].detach().cpu().numpy())
            self.valid_background_err.append(y[3].detach().cpu().numpy())
        else:
            # Compute L2 norm of the model parameters
            l2_norm = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)
            self.log(f"{stage}_l2_norm", l2_norm, on_epoch=True, prog_bar=False, logger=True)

        return loss