import torch
import torch.nn as nn
from typing import Union
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningModule
from torch.nn import Sigmoid
from data.statistics import statistics, accumulate_statistics
from forward.model.model import BaseModel
from forward.model.activation import Sine
from inverse.model.encoding import IdentityPositionalEncoding
from utilities.instantiators import instantiate
from data.transformations import identity
from copy import deepcopy


class InverseEmulator(LightningModule):
    """Class for radiance transfer (PINN) inverse emulator."""
    def __init__(self, optimizer: DictConfig = None, loss_func: DictConfig = None, lr_scheduler: DictConfig = None):
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
    def __init__(self, optimizer: DictConfig = None, loss_func: DictConfig = None, lr_scheduler: DictConfig = None,
                 positional_encoding: DictConfig = None, activation_in: DictConfig = None, activation_out: DictConfig = None,
                 transform_out: ListConfig = None, parameters: DictConfig = None):
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

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Results & metrics
        self.results['prof'] = []
        self.metrics['prof'], self.metrics['prof_target'], self.metrics['prof_background'] = {}, {}, {}

        # Normalization transformations
        self.transform_out = [instantiate(t) for t in transform_out] if transform_out is not None else [identity]

        # Model architecture
        self.n_prof = parameters.data.n_prof if parameters is not None and hasattr(parameters.data, 'n_prof') \
            else 1
        self.n_levels = parameters.data.n_levels if parameters is not None and hasattr(parameters.data, 'n_levels') \
            else 1
        self.prof_vars = parameters.data.prof_vars if parameters is not None and hasattr(parameters.data, 'prof_vars') \
            else [f'var_{i}' for i in range(self.n_prof)]

        self.model = self._build_model(positional_encoding, activation_in, activation_out, parameters)

    def _build_model(self, positional_encoding: Union[DictConfig, None], activation_in: Union[DictConfig, None],
                     activation_out: Union[DictConfig, None], parameters: Union[DictConfig, None]) -> nn.Module:
        """ Build the neural network model.

            Parameters
            ----------
            positional_encoding: DictConfig. Function for the positional encoding.
            activation_in: DictConfig. Activation function (in).
            activation_out: DictConfig. Activation function (out).
            parameters: DictConfig. Configuration for the model parameters.

            Returns
            -------
            None.
        """

        # Parameters check
        dropout_rate = parameters.architecture.dropout if parameters is not None and \
            hasattr(parameters.architecture, 'dropout') else 0.0
        n_neurons = parameters.architecture.n_neurons if parameters is not None and \
            hasattr(parameters.architecture, 'n_neurons') else 128
        n_layers = parameters.architecture.n_layers if parameters is not None and \
            hasattr(parameters.architecture, 'n_layers') else 4
        n_lat = parameters.data.n_lat if parameters is not None and \
            hasattr(parameters.data, 'n_lat') else 1
        n_lon = parameters.data.n_lon if parameters is not None and \
            hasattr(parameters.data, 'n_lon') else 1
        n_scans = parameters.data.n_scans if parameters is not None and \
            hasattr(parameters.data, 'n_scans') else 1
        n_pressure = parameters.data.n_pressure if parameters is not None and \
            hasattr(parameters.data, 'n_pressure') else 1
        n_cloud = parameters.data.n_cloud if parameters is not None and \
            hasattr(parameters.data, 'n_cloud') else 0
        n_prof = parameters.data.n_prof if parameters is not None and \
            hasattr(parameters.data, 'n_prof') else 1
        n_levels = parameters.data.n_levels if parameters is not None and \
            hasattr(parameters.data, 'n_levels') else 1

        # Positional encoding
        d_input = n_lat + n_lon + n_scans + n_pressure + n_cloud
        self.positional_encoding = instantiate(positional_encoding, d_input=d_input) if positional_encoding is not None \
            else IdentityPositionalEncoding(d_input=d_input)
        # Input layer
        self.d_in = nn.Linear(self.positional_encoding.d_output, n_neurons)
        self.activation_in = instantiate(activation_in) if activation_in is not None else Sine()
        self.dropout_in = nn.Dropout(dropout_rate)
        # Output layer
        self.d_out = nn.Linear(n_neurons, n_prof*n_levels)
        self.activation_out = instantiate(activation_out) if activation_out is not None else nn.Identity()

        # Model architecture
        self.layers = nn.ModuleList([nn.Linear(n_neurons, n_neurons)
                                     for _ in range(n_layers)])
        self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(n_neurons)
                                               for _ in range(n_layers)])
        self.activations = nn.ModuleList([instantiate(activation_in) if activation_in is not None else Sine()
                                          for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate)
                                       for _ in range(n_layers)])
        model = nn.Sequential(
            self.d_in,
            self.activation_in,
            nn.Dropout(dropout_rate),
            *[layer for hidden in zip(self.layers, self.batchnorm_layers, self.activations, self.dropouts) for layer in hidden],
            self.d_out,
            self.activation_out
        )

        return model

    def _retrieve_prof(self, x: dict) -> torch.Tensor:
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

    def forward(self, x: dict):
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
        prof = self._retrieve_prof(x_vector).view(-1, n_levels, self.n_prof).transpose(1, 2)
        return prof

    def _logging_prof(self, pred: torch.Tensor, target: torch.Tensor, background: torch.Tensor=None) -> None:
        """ Log profile metrics.

            Parameters
            ----------
            pred: tensor. Predicted profiles.
            target: tensor. Target profiles.
            background: tensor. Background profiles.

            Returns
            -------
            None.
        """

        # Log mean profiles and rmse
        stats_pred = statistics(pred, axis=0, which=['mean', 'stdev', 'rmse', 'mae'], target=target)
        stats_target = statistics(target, axis=0, which=['mean', 'stdev'])
        # Check if statistics dictionaries are empty
        if self.metrics.get('prof'):
            self.metrics['prof'] = accumulate_statistics([self.metrics['prof'], stats_pred])
            self.metrics['prof_target'] = accumulate_statistics([self.metrics['prof_target'], stats_target])
        else:
            self.metrics['prof'] = stats_pred
            self.metrics['prof_target'] = stats_target

        # Log mean background profiles and rmse if available
        if background is not None:
            stats_background = statistics(background, axis=0, which=['mean', 'stdev', 'rmse', 'mae'], target=target)
            # Check if statistics dictionaries are empty
            if self.metrics.get('prof_background'):
                self.metrics['prof_background'] = accumulate_statistics([self.metrics['prof_background'], stats_background])
            else:
                self.metrics['prof_background'] = stats_background

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

        # Log pressure levels
        self.results['pressure'] = input['pressure'][0].detach().cpu().numpy()

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Logger flag
            logger_flag = False
            # Store test outputs
            for k, v in {'prof': pred['prof'], 'hofx': pred['hofx']}.items():
                self.results[k].append(v.detach().cpu().numpy())
        else:
            # Logger flag
            logger_flag = True
            # Log metrics for hofx and profiles
            self._logging_hofx(pred['hofx'], target['hofx'], target['cloud_filter'].bool(),
                               target['daytime_filter'].bool())
            self._logging_prof(pred['prof'], target['prof'], background=target.get('prof_background', None))
            # Log L2 norm of model parameters during training
            if stage == 'train':
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
                # Detailed logging per profile and variable
                if loss[key].ndim == 3:
                    for i, var in enumerate(self.prof_vars):
                        self.log(f"{stage}_loss_{key}_{i}_{var}", loss[key][:, i, :].mean(), on_epoch=True,
                                 prog_bar=False, logger=logger_flag)
                # If pressure-level filtering is involved, log only the relevant levels
                elif loss[key].ndim == 2 and self.loss_func.pressure_filter is not None:
                    n_pressure = torch.cumsum(self.loss_func.pressure_filter.sum(axis=1), dim=0)
                    for i, var in enumerate(self.prof_vars):
                        # Log loss only for the relevant pressure levels
                        start_index, end_index = n_pressure[i-1] if i > 0 else 0, n_pressure[i]
                        self.log(f"{stage}_loss_{key}_{i}_{var}", loss[key][:, start_index:end_index].mean(),
                                 on_epoch=True, prog_bar=False, logger=logger_flag)

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
        pred = {'prof': self.forward(batch['input'])}
        # Apply output transformations
        for t, transform in enumerate(self.transform_out):
            pred['prof'] = transform(pred['prof'])

        # Mask
        mask = torch.zeros_like(pred['prof'])
        mask[:, 1:2, :] = 1.0
        pred['prof'] = pred['prof'] * mask + batch['target']['prof'] * (1 - mask)

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
        prof = self.forward(batch['input'])
        # Apply output transformations
        for transform in self.transform_out:
            prof = transform(prof)
        return prof


class PINNverseOperators(PINNverseOperator):
    """Physics-Informed Neural Network (PINN) inverse model with one neural network per profile type."""

    def __init__(self, optimizer: DictConfig = None, loss_func: DictConfig = None, lr_scheduler: DictConfig = None,
                 positional_encoding: DictConfig = None, activation_in: DictConfig = None, activation_out: DictConfig = None,
                 transform_out: ListConfig = None, parameters: DictConfig = None):
        """ Initialize model.

        Parameters
        ----------
        optimizer: DictConfig. Optimizer for the model.
        loss_func: DictConfig. Loss function for the model.
        lr_scheduler: DictConfig. Learning rate scheduler for the model.
        positional_encoding: DictConfig. Function for the positional encoding.
        activation_in: DictConfig. Activation function (in).
        activation_out: DictConfig. Activation function (out).
        transform_out: ListConfig. List of functions to apply to the model output.
        parameters: DictConfig. Configuration for the model parameters.

        Returns
        -------
        None.
        """

        # Inherit all attributes and logic from PINNverseOperator
        super().__init__(optimizer=optimizer, loss_func=loss_func, lr_scheduler=lr_scheduler,
                         positional_encoding=positional_encoding, activation_in=activation_in,
                         activation_out=activation_out, transform_out=transform_out, parameters=parameters)

        # Replace the single model with a list of models, one per profile type
        self.model = nn.ModuleList([
            self._build_model(
                positional_encoding, activation_in, activation_out, self._patch_profiles(deepcopy(parameters))
            )
            for _ in range(self.n_prof)
        ])

    @staticmethod
    def _patch_profiles(parameters: DictConfig) -> DictConfig:
        """Set n_prof=1 in parameters for sub-model construction."""
        parameters.data.n_prof = 1
        return parameters

    def _retrieve_prof(self, x: dict) -> list:
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
        outputs = [model(encoded_inputs) for model in self.model]
        return outputs

    def forward(self, x: dict):
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
        prof = torch.cat([output.view(-1, 1, n_levels) for output in self._retrieve_prof(x_vector)], dim=1)
        return prof
