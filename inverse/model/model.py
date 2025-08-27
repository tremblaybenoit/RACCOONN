import torch
import torch.nn as nn
from typing import Callable
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from forward.model.model import BaseModel
from forward.utilities.instantiators import instantiate
from forward.data.transformations import NormalizeProfiles
from inverse.data.transformations import identity
from copy import deepcopy


class InverseEmulator(LightningModule):
    """Class for radiance transfer (PINN) inverse emulator."""
    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None):
        """ Initialize model.

        Parameters
        ----------
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.

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
                 positional_encoding: Callable = None, activation_in: Callable = None, activation_out: Callable = None,
                 transform: Callable = None, inverse_transform: Callable = None, clip: Callable = None,
                 parameters: DictConfig = None, log_valid: bool = False):
        """ Initialize model.

        Parameters
        ----------
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        positional_encoding: Callable. Function for the positional encoding.
        activation_in: Callable. Activation function (in).
        activation_out: Callable. Activation function (out).
        transform: Callable. Normalization transformation.
        inverse_transform: Callable. Unnormalization transformation.
        clip: Callable. Clipping function for the model outputs.
        parameters: DictConfig. Configuration for the model parameters.
        log_valid: bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Validation results
        self.log_valid = log_valid
        # Initialize empty lists for validation and test results
        if self.log_valid:
            self.valid_results = {
                'bt_target': [],
                'bt_pred': [],
                'prof_target': [],
                'prof_pred': [],
                'pressure': [],
                'background': [],
                'background_err': [],
            }
        self.test_results['prof'] = []

        # Normalization transformations
        self.clip = clip if clip is not None else identity
        self.normalize_profiles = transform
        self.unnormalize_profiles = inverse_transform
        # Normalize profiles prior to inputting to CRTM emulator
        self.stats_crtm_profiles = {'min': torch.tensor(parameters.data.prof_min, dtype=torch.float64),
                                    'max': torch.tensor(parameters.data.prof_max, dtype=torch.float64)}
        self.normalize_crtm_profiles = NormalizeProfiles(self.stats_crtm_profiles['min'], self.stats_crtm_profiles['max'],
                                                         inverse_transform=False)

        # Model architecture
        self.n_profiles = parameters.data.n_profiles
        self.n_levels = parameters.data.n_levels
        self.prof_vars = parameters.data.prof_vars
        self.model = self._build_model(positional_encoding, activation_in, activation_out, parameters)

    def _build_model(self, positional_encoding: Callable, activation_in: Callable, activation_out: Callable,
                     parameters: DictConfig) -> nn.Module:
        """ Build the neural network model.

            Parameters
            ----------
            positional_encoding: Callable. Function for the positional encoding.
            activation_in: Callable. Activation function (in).
            activation_out: Callable. Activation function (out).
            parameters: DictConfig. Configuration for the model parameters.

            Returns
            -------
            None.
        """

        # Positional encoding
        d_input = (parameters.data.n_lat + parameters.data.n_lon + parameters.data.n_scans + parameters.data.n_pressure + parameters.data.n_clrsky)
        self.positional_encoding = instantiate(positional_encoding, d_input=d_input)
        # Input layer
        self.d_in = nn.Linear(self.positional_encoding.d_output, parameters.architecture.n_neurons)
        self.activation_in = instantiate(activation_in)
        self.dropout_in = nn.Dropout(parameters.architecture.dropout)
        # Output layer
        self.d_out = nn.Linear(parameters.architecture.n_neurons, parameters.data.n_profiles*parameters.data.n_levels)
        self.activation_out = instantiate(activation_out)

        # Model architecture
        self.layers = nn.ModuleList([nn.Linear(parameters.architecture.n_neurons, parameters.architecture.n_neurons)
                                     for _ in range(parameters.architecture.n_layers)])
        self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(parameters.architecture.n_neurons)
                                               for _ in range(parameters.architecture.n_layers)])
        self.activations = nn.ModuleList([instantiate(activation_in) for _ in range(parameters.architecture.n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(parameters.architecture.dropout)
                                       for _ in range(parameters.architecture.n_layers)])
        model = nn.Sequential(
            self.d_in,
            self.activation_in,
            nn.Dropout(parameters.architecture.dropout),
            *[layer for hidden in zip(self.layers, self.batchnorm_layers, self.activations, self.dropouts) for layer in hidden],
            self.d_out,
            self.activation_out
        )

        return model

    def forward(self, x: dict) -> torch.Tensor:
        """ Pass forward through neural network architecture.

            Parameters
            ----------
            x: tensor. Inputs: latitude, longitude, surface, and the metadata.

            Returns
            -------
            y: tensor. Outputs: predicted profiles, surface, and metadata.
        """

        # Concatenate inputs
        inputs = torch.cat((x['lat'].view(-1, 1), x['lon'].view(-1, 1), x['scans'].view(-1, 1),
                            x['pressure'].view(-1, 1)), dim=-1)
        # Apply positional encoding
        encoded_inputs = self.positional_encoding(inputs)

        # Pass through the model
        profiles = self.model(encoded_inputs)
        # Reshape profiles to match the expected output shape
        return profiles.view(-1, self.n_profiles, 1)

    def _retrieve_profiles(self, x: dict):
        """ Retrieve atmospheric profile over multiple pressure levels.

            Parameters
            ----------
            x: tensor. Input variables.

            Returns
            -------
            prof: tensor. Atmospheric profiles.
        """

        # Forward pass
        n_levels = x['pressure'].shape[-1]
        x_vector = {k: v.repeat_interleave(n_levels, dim=0) if k != 'pressure' else x['pressure'].reshape(-1, 1)
                    for k, v in x.items()}
        return self.forward(x_vector).view(-1, n_levels, self.n_profiles).transpose(1, 2)

    def _logging(self, x: dict, y: dict, stage: str, loss: dict, prof_pred: torch.Tensor, bt_pred: torch.Tensor) -> None:
        """ Log training/validation/test metrics.

            Parameters
            ----------
            loss: tensor. Loss value.
            stage: str. Current operation: "train", "valid", or "test".

            Returns
            -------
            None.
        """

        # Log total loss
        if 'total' in loss.keys():
            self.log(f"{stage}_loss", loss['total'], on_epoch=True, prog_bar=True, logger=True)
        # Log profile loss
        if 'model' in loss.keys():
            self.log(f"{stage}_loss_model", loss['model'].mean(), on_epoch=True, prog_bar=True, logger=True)
            # Log individual profile variables
            if loss['model'].ndim == 3:
                for i, var in enumerate(self.prof_vars):
                    self.log(f"{stage}_loss_model_{i}_{var}", loss['model'][:, i, :].mean(), on_epoch=True, prog_bar=False,
                             logger=True)
        # Log boundary conditions loss
        if 'bcs' in loss.keys():
            self.log(f"{stage}_loss_bcs", loss['bcs'].mean(), on_epoch=True, prog_bar=True, logger=True)
            # Log individual profile variables
            if loss['bcs'].ndim == 3:
                for i, var in enumerate(self.prof_vars):
                    self.log(f"{stage}_loss_bcs_{i}_{var}", loss['bcs'][:, i, :].mean(), on_epoch=True, prog_bar=False,
                             logger=True)
        # Log observation loss
        if 'obs' in loss.keys():
            self.log(f"{stage}_loss_obs", loss['obs'].mean(), on_epoch=True, prog_bar=True, logger=True)
            # Log individual radiance channels
            if loss['obs'].ndim == 2:
                for i in range(bt_pred.shape[1] // 2):
                    self.log(f"{stage}_loss_obs_{i}", loss['obs'][:, i].mean(), on_epoch=True, prog_bar=False, logger=True)
        # Log sanity check loss
        if 'sanity_check' in loss.keys():
            self.log(f"{stage}_loss_sanity_check", loss['sanity_check'].mean(), on_epoch=True, prog_bar=True, logger=True)
            # Log individual radiance channels
            if loss['sanity_check'].ndim == 2:
                for i in range(bt_pred.shape[1] // 2):
                    self.log(f"{stage}_loss_sanity_check_{i}", loss['sanity_check'][:, i].mean(), on_epoch=True, prog_bar=False, logger=True)

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Store test outputs
            self.test_results['prof'].append(prof_pred.detach().cpu().numpy())
            self.test_results['hofx'].append(bt_pred.detach().cpu().numpy())
        elif stage == 'valid' and self.log_valid:
            # Store validation outputs
            self.valid_results['bt_target'].append(y['hofx'].detach().cpu().numpy())
            self.valid_results['bt_pred'].append(bt_pred.detach().cpu().numpy())
            self.valid_results['prof_target'].append(y['prof'].detach().cpu().numpy()) if 'prof_norm' not in y else (
                self.valid_results['prof_target'].append(y['prof_norm'].detach().cpu().numpy()))
            self.valid_results['prof_pred'].append(prof_pred.detach().cpu().numpy())
            self.valid_results['pressure'].append(x['pressure'].detach().cpu().numpy())
            # Store background and background error if available
            if 'background' in y:
                self.valid_results['background'].append(y['background'].detach().cpu().numpy())
            if 'background_err' in y:
                self.valid_results['background_err'].append(y['background_err'].detach().cpu().numpy())
        else:
            # Compute L2 norm of the model parameters
            l2_norm = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)
            self.log(f"{stage}_l2_norm", l2_norm, on_epoch=True, prog_bar=False, logger=True)

    def base_step(self, batch: tuple[dict, dict], batch_nb: int, stage: str) -> torch.Tensor:
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

        # Extract data from batch
        x, y = batch

        # TODO: Verify handling of mask, clipping, normalization, and unnormalization

        # Compute profiles
        mask = (torch.sigmoid(1e6 * y['prof']) - 0.5) * 2
        prof_pred_norm = self._retrieve_profiles(x)
        prof_pred = self.unnormalize_profiles(prof_pred_norm)*mask

        # Compute loss function
        loss, bt_pred = self.loss_func((self.normalize_crtm_profiles(prof_pred), x['surf'], x['meta']), prof_pred_norm, y)

        # Logging
        self._logging(x, y, stage, loss, prof_pred_norm, bt_pred)

        return loss['total']

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
        for d in [self.stats_crtm_profiles]:
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        d[k] = v.to(device)
        # Move normalization modules if needed
        for attr in ['normalize_profiles', 'unnormalize_profiles', 'normalize_crtm_profiles']:
            norm = getattr(self, attr, None)
            if hasattr(norm, 'to'):
                setattr(self, attr, norm.to(device))
        return self


class PINNverseOperatorMulti(PINNverseOperator):
    """Physics-Informed Neural Network (PINN) inverse model with one neural network per profile type."""

    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None,
                 positional_encoding: Callable = None, activation_in: Callable = None, activation_out: Callable = None,
                 transform: Callable = None, inverse_transform: Callable = None, clip: Callable = None,
                 parameters: DictConfig = None, log_valid: bool = False):
        """
        Initialize model.

        Parameters
        ----------
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        positional_encoding: Callable. Function for the positional encoding.
        activation_in: Callable. Activation function (in).
        activation_out: Callable. Activation function (out).
        transform: Callable. Normalization transformation.
        inverse_transform: Callable. Unnormalization transformation.
        clip: Callable. Clipping function for the model outputs.
        parameters: DictConfig. Configuration for the model parameters.
        log_valid: bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Inherit all attributes and logic from PINNverseOperator
        super().__init__(optimizer=optimizer, loss_func=loss_func, lr_scheduler=lr_scheduler,
                         positional_encoding=positional_encoding, activation_in=activation_in,
                         activation_out=activation_out, transform=transform, inverse_transform=inverse_transform,
                         clip=clip, parameters=parameters, log_valid=log_valid)

        # Replace the single model with a list of models, one per profile type
        self.models = nn.ModuleList([
            self._build_model(
                positional_encoding, activation_in, activation_out, self._patch_profiles(deepcopy(parameters))
            )
            for _ in range(self.n_profiles)
        ])

    def _patch_profiles(self, parameters):
        """Set n_profiles=1 in parameters for sub-model construction."""
        parameters.data.n_profiles = 1
        return parameters

    def forward(self, x: dict) -> torch.Tensor:
        """
        Pass forward through all neural networks, one per profile type.

        Parameters
        ----------
        x: dict. Inputs: latitude, longitude, surface, and the metadata.

        Returns
        -------
        y: tensor. Outputs: predicted profiles for all types, concatenated.
        """
        # Concatenate inputs
        inputs = torch.cat((x['lat'].view(-1, 1), x['lon'].view(-1, 1), x['scans'].view(-1, 1),
                            x['pressure'].view(-1, 1)), dim=-1)
        # Apply positional encoding
        encoded_inputs = self.positional_encoding(inputs)
        # Each model predicts its profile type
        outputs = [model(encoded_inputs) for model in self.models]

        # Concatenate outputs along the last dimension
        return torch.cat(outputs, dim=-1)


class PINNverseOperatorMulti2(PINNverseOperator):
    """Physics-Informed Neural Network (PINN) inverse model with one neural network per profile type."""

    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None,
                 positional_encoding: Callable = None, activation_in: Callable = None, activation_out: Callable = None,
                 transform: Callable = None, inverse_transform: Callable = None, clip: Callable = None,
                 parameters: DictConfig = None, log_valid: bool = False):
        """
        Initialize model.

        Parameters
        ----------
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        positional_encoding: Callable. Function for the positional encoding.
        activation_in: Callable. Activation function (in).
        activation_out: Callable. Activation function (out).
        transform: Callable. Normalization transformation.
        inverse_transform: Callable. Unnormalization transformation.
        clip: Callable. Clipping function for the model outputs.
        parameters: DictConfig. Configuration for the model parameters.
        log_valid: bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Inherit all attributes and logic from PINNverseOperator
        super().__init__(optimizer=optimizer, loss_func=loss_func, lr_scheduler=lr_scheduler,
                         positional_encoding=positional_encoding, activation_in=activation_in,
                         activation_out=activation_out, transform=transform, inverse_transform=inverse_transform,
                         clip=clip, parameters=parameters, log_valid=log_valid)

        # Replace the single model with a list of models, one per profile type
        self.models = nn.ModuleList([
            self._build_model(
                positional_encoding, activation_in, activation_out, self._patch_profiles(deepcopy(parameters))
            )
            for _ in range(self.n_profiles)
        ])

    def _patch_profiles(self, parameters):
        """Set n_profiles=1 in parameters for sub-model construction."""
        parameters.data.n_profiles = 1
        return parameters

    def forward(self, x: dict) -> list:
        """
        Pass forward through all neural networks, one per profile type.

        Parameters
        ----------
        x: dict. Inputs: latitude, longitude, surface, and the metadata.

        Returns
        -------
        y: tensor. Outputs: predicted profiles for all types, concatenated.
        """
        # Concatenate inputs
        inputs = torch.cat([v.view(-1, 1) for v in x.values()], dim=-1)
        # Apply positional encoding
        encoded_inputs = self.positional_encoding(inputs)
        # Each model predicts its profile type
        outputs = [model(encoded_inputs) for model in self.models]

        # Concatenate outputs along the last dimension
        return outputs

    def _retrieve_profiles(self, x: dict):
        """ Retrieve atmospheric profile over multiple pressure levels.

            Parameters
            ----------
            x: tensor. Input variables.

            Returns
            -------
            prof: tensor. Atmospheric profiles.
        """

        # Forward pass
        n_levels = x['pressure'].shape[-1]
        x_vector = {k: v.repeat_interleave(n_levels, dim=0) if k != 'pressure' else x['pressure'].reshape(-1, 1)
                    for k, v in x.items()}
        return torch.cat([output.view(-1, 1, n_levels) for output in self.forward(x_vector)], dim=1)

    def base_step(self, batch: dict, batch_nb: int, stage: str) -> torch.Tensor:
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

        # Compute profiles
        prof_norm = self._retrieve_profiles(batch['coordinates'])
        prof_pred = self.unnormalize_profiles(prof_norm)

        # Compute loss function
        loss, bt_pred = self.loss_func(prof_norm, prof_pred, batch['targets'])

        # Logging
        self._logging(batch['coordinates'], batch['targets'], stage, loss, prof_norm, bt_pred)

        return loss['total']
