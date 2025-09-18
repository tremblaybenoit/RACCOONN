import torch
import torch.nn as nn
from typing import Callable
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningModule
from forward.model.model import BaseModel
from utilities.instantiators import instantiate
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
                 transform_out: ListConfig = None, parameters: DictConfig = None, log_valid: bool = False):
        """ Initialize model.

        Parameters
        ----------
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        positional_encoding: Callable. Function for the positional encoding.
        activation_in: Callable. Activation function (in).
        activation_out: Callable. Activation function (out).
        transform_out: ListConfig. List of functions to apply to the model output.
        parameters: DictConfig. Configuration for the model parameters.
        log_valid: bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func, log_valid=log_valid)

        # Initialize empty lists for validation and test results
        if self.log_valid:
            self.valid_results = {
                'hofx_target': [],
                'hofx_pred': [],
                'prof_target': [],
                'prof_pred': [],
                'prof_norm_target': [],
                'prof_norm_pred': [],
                'prof_background': [],
                'prof_norm_background': [],
                'pressure': [],
                'cloud_filter': []
            }
        # Test results
        self.test_results = {'hofx': [], 'prof': []}

        # Normalization transformations
        self.transform_out = transform_out if transform_out is not None else [identity]

        # Model architecture
        self.n_prof = parameters.data.n_prof
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
        d_input = (parameters.data.n_lat + parameters.data.n_lon + parameters.data.n_scans + parameters.data.n_pressure + parameters.data.n_cloud)
        self.positional_encoding = instantiate(positional_encoding, d_input=d_input)
        # Input layer
        self.d_in = nn.Linear(self.positional_encoding.d_output, parameters.architecture.n_neurons)
        self.activation_in = instantiate(activation_in)
        self.dropout_in = nn.Dropout(parameters.architecture.dropout)
        # Output layer
        self.d_out = nn.Linear(parameters.architecture.n_neurons, parameters.data.n_prof*parameters.data.n_levels)
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
            y: tensor. Outputs: predicted profiles.
        """

        # Concatenate inputs
        inputs = torch.cat([v.view(-1, 1) for v in x.values()], dim=-1)
        # Apply positional encoding
        encoded_inputs = self.positional_encoding(inputs)
        # Pass through the model
        profiles = self.model(encoded_inputs)
        # Reshape profiles to match the expected output shape
        return profiles.view(-1, self.n_prof, 1)

    def _retrieve_prof(self, x: dict):
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
        prof = self.forward(x_vector).view(-1, n_levels, self.n_prof).transpose(1, 2)
        return prof

    def _logging(self, stage: str, loss: dict, input: dict, target: dict, pred: dict) -> None:
        """ Log training/validation/test metrics.

            Parameters
            ----------
            stage: str. Current operation: "train", "valid", or "test".
            loss: dict. Dictionary containing the loss components.
            input: dict. Input coordinates.
            target: dict. Observations.
            pred: dict. Predictions.

            Returns
            -------
            None.
        """

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Logger flag
            logger_flag = False
            # Store test outputs
            for k, v in {'prof': pred['prof'], 'hofx': pred['hofx']}.items():
                self.test_results[k].append(v.detach().cpu().numpy())
        elif stage == 'valid':
            # Logger flag
            logger_flag = True
            # Store validation outputs for logging
            if self.log_valid:
                # Store validation outputs
                for k, v in {'prof_target': target['prof'], 'prof_pred': pred['prof'],
                             'prof_norm_target': target['prof_norm'], 'prof_norm_pred': pred['prof_norm'],
                             'hofx_target': target['hofx'], 'hofx_pred': pred['hofx'],
                             'pressure': input['pressure'], 'cloud_filter': target['cloud_filter']}.items():
                    self.valid_results[k].append(v.detach().cpu().numpy())
                # Store background if available
                if 'prof_background' in target:
                    for k, v in {'prof_background': target['prof_background'],
                                 'prof_norm_background': target['prof_norm_background']}.items():
                        self.valid_results[k].append(v.detach().cpu().numpy())
        else:
            # Logger flag
            logger_flag = True
            # Compute L2 norm of the model parameters
            l2_norm = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)
            self.log(f"{stage}_l2_norm", l2_norm, on_epoch=True, prog_bar=False, logger=logger_flag)

        # Log total loss
        if 'total' in loss:
            self.log(f"{stage}_loss", loss['total'], on_epoch=True, prog_bar=True, logger=logger_flag)
        # Log profile and boundary condition losses
        for key in ['model', 'bcs']:
            if key in loss:
                self.log(f"{stage}_loss_{key}", loss[key].mean(), on_epoch=True, prog_bar=True, logger=logger_flag)
                if loss[key].ndim == 3:
                    for i, var in enumerate(self.prof_vars):
                        self.log(f"{stage}_loss_{key}_{i}_{var}", loss[key][:, i, :].mean(), on_epoch=True,
                                 prog_bar=False, logger=logger_flag)
        # Log observation loss
        if 'obs' in loss:
            self.log(f"{stage}_loss_obs", loss['obs'].mean(), on_epoch=True, prog_bar=True, logger=logger_flag)
            if loss['obs'].ndim == 2:
                for i in range(pred['hofx'].shape[1] // 2):
                    self.log(f"{stage}_loss_obs_{i}", loss['obs'][:, i].mean(), on_epoch=True, prog_bar=False,
                             logger=logger_flag)

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
        pred = {'prof_norm': self._retrieve_prof(batch['input'])}
        # Apply output transformations
        for t, transform in enumerate(self.transform_out):
            pred['prof'] = transform(pred['prof']) if t > 0 else transform(pred['prof_norm'])

        # Compute loss function
        loss, pred['hofx'] = self.loss_func(pred, batch['target'])

        # Logging
        self._logging(stage, loss, batch['input'], batch['target'], pred)

        return loss['total']

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """ Perform prediction step.

            Parameters
            ----------
            batch: tensor. Batch from the prediction set.
            batch_idx: int. Index of the batch out of the prediction set.
            dataloader_idx: int. Index of the dataloader.

            Returns
            -------
            Predicted profiles: tensor.
        """

        # Compute profiles
        prof = self._retrieve_prof(batch['input'])
        # Apply output transformations
        for transform in self.transform_out:
            prof = transform(prof)
        return prof

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
        # Move normalization modules if needed
        for attr in ['normalize_prof', 'unnormalize_prof']:
            norm = getattr(self, attr, None)
            if hasattr(norm, 'to'):
                setattr(self, attr, norm.to(device))
        return self


class PINNverseOperators(PINNverseOperator):
    """Physics-Informed Neural Network (PINN) inverse model with one neural network per profile type."""

    def __init__(self, optimizer: DictConfig = None, loss_func: Callable = None, lr_scheduler: DictConfig = None,
                 positional_encoding: Callable = None, activation_in: Callable = None, activation_out: Callable = None,
                 transform_out: ListConfig = None, parameters: DictConfig = None, log_valid: bool = False):
        """ Initialize model.

        Parameters
        ----------
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        positional_encoding: Callable. Function for the positional encoding.
        activation_in: Callable. Activation function (in).
        activation_out: Callable. Activation function (out).
        transform_out: ListConfig. List of functions to apply to the model output.
        parameters: DictConfig. Configuration for the model parameters.
        log_valid: bool. Whether to log validation results.

        Returns
        -------
        None.
        """

        # Inherit all attributes and logic from PINNverseOperator
        super().__init__(optimizer=optimizer, loss_func=loss_func, lr_scheduler=lr_scheduler,
                         positional_encoding=positional_encoding, activation_in=activation_in,
                         activation_out=activation_out, transform_out=transform_out, parameters=parameters,
                         log_valid=log_valid)

        # Replace the single model with a list of models, one per profile type
        self.models = nn.ModuleList([
            self._build_model(
                positional_encoding, activation_in, activation_out, self._patch_profiles(deepcopy(parameters))
            )
            for _ in range(self.n_prof)
        ])

    @staticmethod
    def _patch_profiles(self, parameters):
        """Set n_prof=1 in parameters for sub-model construction."""
        parameters.data.n_prof = 1
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
        return outputs

    def _retrieve_prof(self, x: dict):
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
        prof = torch.cat([output.view(-1, 1, n_levels) for output in self.forward(x_vector)], dim=1)
        return prof if self.clip is None else self.clip(prof)
