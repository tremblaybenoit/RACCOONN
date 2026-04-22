import numpy as np
import torch
import torch.nn as nn
from typing import Union, Any
from pytorch_lightning import LightningModule
from data.statistics import statistics, accumulate_statistics
from forward.model.activation import Swish, Scale
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from torchdiffeq import odeint
import gc


class BaseModel(LightningModule):
    """
    Lightning model for the CRTM emulator (Community Radiative Transfer Model) using PyTorch.
    This is translation from Keras to Pytorch of the CRTM emulator by Howard et al. (2025).
    Link: https://zenodo.org/records/13963758.
    """
    def __init__(self, optimizer: DictConfig = None, lr_scheduler: DictConfig = None, loss_func: DictConfig = None):
        """ Initialize LightningCRTMModel.

        Parameters
        ----------
        optimizer: DictConfig. Optimizer for the model.
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).
        loss_func: DictConfig. Loss function for the model.

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

        # Stage results
        self.results: dict[str, list] = {'hofx': []}
        self.metrics: dict[str, dict] = {'hofx': {}, 'hofx_norm': {}}

    def _logging_hofx(self, pred: torch.Tensor, target: torch.Tensor,
                      cloud_filter: torch.Tensor, daytime_filter: torch.Tensor) -> None:
        """ Compute and store metrics for hofx predictions.

            Parameters
            ----------
            pred: torch.Tensor. Predicted hofx values.
            target: torch.Tensor. Target hofx values.
            cloud_filter: torch.Tensor. Cloud filter mask for the batch.
            daytime_filter: torch.Tensor. Daytime filter mask for the batch.

            Returns
            -------
            None.
        """

        # Create masks
        mask = {
            'Clear sky': ~cloud_filter,
            'Cloudy': cloud_filter,
            'Day': daytime_filter,
            'Night': ~daytime_filter
        }

        # Aggregate metrics per batch and combine with previous batches
        for key, m in mask.items():
            # Check is there are samples in the mask
            if not torch.any(m):
                continue
            # Compute and accumulate statistics
            stats = statistics(pred[m, :10], axis=0, which=['rmse'], target=target[m, :10])
            stats_norm = statistics(pred[m, :10]/pred[m, 10:], axis=0, which=['rmse'], target=target[m, :10]/pred[m, 10:])
            # Check if the key exists in the metrics dictionary
            if key not in self.metrics['hofx']:
                self.metrics['hofx'][key] = stats
                self.metrics['hofx_norm'][key] = stats_norm
            else:
                self.metrics['hofx'][key] = accumulate_statistics([self.metrics['hofx'][key], stats])
                self.metrics['hofx_norm'][key] = accumulate_statistics([self.metrics['hofx_norm'][key], stats_norm])

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
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=pred.device)
        # Compute loss function
        loss = self.loss_func(pred, batch['target']['hofx'])*weights

        # Log loss
        self.log(f"{stage}_loss", loss.mean(), on_epoch=True, prog_bar=True, logger=stage != 'test')
        # If testing, return predictions in addition to loss
        if stage == 'test':
            self.results['hofx'].append(pred.detach().cpu().numpy())

        # Training stage
        if stage == 'train':
            # Log L2 norm of model parameters
            l2_norm = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)
            self.log(f"{stage}_l2_norm", l2_norm, on_epoch=True, prog_bar=False, logger=True)
        # Validation, test stages
        else:
            # Log metrics for hofx
            self._logging_hofx(pred, batch['target']['hofx'], batch['input']['cloud_filter'].bool(),
                               batch['input']['daytime_filter'].bool())

        return loss.mean()

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
        for k in self.results:
            self.results[k] = []
        for k in self.metrics:
            self.metrics[k] = {}

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
        for k in self.results:
            self.results[k] = np.concatenate(self.results[k], axis=0)  # type: ignore

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
                 loss_func: DictConfig = None):
        """ Initialize LightningCRTMModel.

        Parameters
        ----------
        optimizer: DictConfig. Optimizer for the model.
        loss_func: DictConfig. Loss function for the model.
        parameters: DictConfig. Configuration object containing model parameters.
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Input parameters
        self.nprofvars = len(parameters.data.use_prof_vars)
        self.nsurfvars = len(parameters.data.use_surf_vars)
        self.nmetavars = len(parameters.data.use_meta_vars)
        self.nlevels = int(parameters.data.nlevels)
        self.prof_vars = parameters.data.prof_vars

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


class CRTMModelSmooth(BaseModel):
    def __init__(self, parameters, optimizer: DictConfig = None, lr_scheduler: DictConfig = None,
                 loss_func: DictConfig = None):
        """ Initialize LightningCRTMModel.

        Parameters
        ----------
        optimizer: DictConfig. Optimizer for the model.
        loss_func: DictConfig. Loss function for the model.
        parameters: DictConfig. Configuration object containing model parameters.
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Input parameters (preserved names)
        self.nprofvars = len(parameters.data.use_prof_vars)
        self.nsurfvars = len(parameters.data.use_surf_vars)
        self.nmetavars = len(parameters.data.use_meta_vars)
        self.nlevels = int(parameters.data.nlevels)
        self.prof_vars = parameters.data.prof_vars

        # Dimension for the linear skip
        self.input_dim = self.nprofvars * self.nlevels + self.nsurfvars + self.nmetavars

        # Neural network parameters
        nnodes_bt = parameters.architecture.nnodes_bt
        nhidden_bt = parameters.architecture.nhidden_bt
        dropout_rate = parameters.architecture.dropout_rate
        self.max_T = parameters.data.bt_norm_max
        self.min_T = parameters.data.bt_norm_min

        # INVERSION FIX: We store the activation but will bypass it in the skip path
        # or use a Leaky variant if you want to keep some bounding.
        # For now, we'll keep the name for compatibility but use Identity in forward.
        self.bt_output_activation = nn.Identity()

        self.std_output_activation = nn.Softplus()
        self.std_output_activation_offset = parameters.architecture.std_output_activation_offset
        self.std_scale_trainable = parameters.architecture.std_scale_trainable

        # --- Components ---
        self.flatten = nn.Flatten()
        self.concat = lambda *tensors: torch.cat(tensors, dim=1)

        # 1. NEW: Linear Skip Connection (The Gradient Highway)
        self.skip_connection = nn.Linear(self.input_dim, 10)

        # 2. Hidden Layers (The Non-Linear Residual)
        self.hidden_layers = nn.ModuleList()
        self.swish_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First dense layer
        self.hidden_layers.append(nn.Linear(self.input_dim, nnodes_bt))
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
        # Reformat variables
        prof = input['prof']
        prof = self.flatten(prof)
        x = self.concat(prof, input['surf'], input['meta'])

        # Path 1: Linear Baseline (Skip)
        # This gives the SIREN a direct path to the radiances.
        linear_bt = self.skip_connection(x)

        # Path 2: Non-linear Hidden Layers (Residual)
        res = x
        for dense, swish, drop in zip(self.hidden_layers, self.swish_layers, self.dropout_layers):
            res = dense(res)
            res = swish(res)
            res = drop(res)

        # Mean output calculation
        # We sum the linear and residual paths before applying the range scaling
        residual_bt = self.out_T(res)

        # Combine paths
        # No Sigmoid here! We use the range mapping directly on the sum.
        out = linear_bt + residual_bt

        # Optional: We still use the norm_max/min to keep values in physical units,
        # but we don't 'squash' them through a Sigmoid first.
        # If your weights were trained with Sigmoid, this scaling might need adjustment.
        out = out * (self.max_T - self.min_T) + self.min_T

        # Std output (positivity via Softplus)
        out_std = self.out_std(res)
        out_std = self.std_output_activation(out_std)
        if self.std_scale is not None:
            out_std = self.std_scale(out_std)
        out_std = out_std + self.std_output_activation_offset

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


class CRTMModelPCA(CRTMModelSmooth):
    def __init__(self, parameters, optimizer: DictConfig = None, lr_scheduler: DictConfig = None,
                 loss_func: DictConfig = None):
        """ Initialize LightningCRTMModel.

        Parameters
        ----------
        optimizer: DictConfig. Optimizer for the model.
        loss_func: DictConfig. Loss function for the model.
        parameters: DictConfig. Configuration object containing model parameters.
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func, parameters=parameters)

        # Input parameters (preserved names)
        self.prof_scaling = parameters.data.prof_scaling

    def forward(self, input: dict) -> torch.Tensor:
        """ Forward pass for the model with profile scaling.

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

        return super().forward({'prof': input['prof']/self.prof_scaling, 'surf': input['surf'],
                                'meta': input['meta']})


class CRTMModelSiren(BaseModel):
    """
    Lightning model for the CRTM emulator (Community Radiative Transfer Model) using SIREN
    (Sinusoidal Representation Networks). This architecture is specifically designed to
    provide infinitely differentiable mappings, ensuring smooth adjoints for atmospheric inversions.
    """

    def __init__(self, parameters: DictConfig, optimizer: DictConfig = None,
                 lr_scheduler: DictConfig = None, loss_func: DictConfig = None):
        """ Initialize CRTMModelSiren.

        Parameters
        ----------
        parameters: DictConfig. Configuration object containing model parameters.
        optimizer: DictConfig. Optimizer for the model (optional).
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).
        loss_func: DictConfig. Loss function for the model (optional).

        Returns
        -------
        None.
        """
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Input parameters
        self.nprofvars = len(parameters.data.use_prof_vars)
        self.nsurfvars = len(parameters.data.use_surf_vars)
        self.nmetavars = len(parameters.data.use_meta_vars)
        self.nlevels = int(parameters.data.nlevels)
        self.input_dim = self.nprofvars * self.nlevels + self.nsurfvars + self.nmetavars

        # Neural network parameters
        nnodes_bt = parameters.architecture.nnodes_bt
        nhidden_bt = parameters.architecture.nhidden_bt
        self.max_T = parameters.data.bt_norm_max
        self.min_T = parameters.data.bt_norm_min
        self.w0 = getattr(parameters.architecture, "siren_w0", 30.0)

        # Output activations and scaling
        self.bt_output_activation = nn.Sigmoid()
        self.std_output_activation = nn.Softplus()
        self.std_output_activation_offset = parameters.architecture.std_output_activation_offset
        self.std_scale_trainable = parameters.architecture.std_scale_trainable

        # Components
        self.flatten = nn.Flatten()
        self.concat = lambda *tensors: torch.cat(tensors, dim=1)
        self.hidden_layers = nn.ModuleList()

        # Build SIREN layers with specialized initialization
        current_dim = self.input_dim
        for i in range(nhidden_bt):
            layer = nn.Linear(current_dim, nnodes_bt)
            self._siren_init(layer, is_first=(i == 0))
            self.hidden_layers.append(layer)
            current_dim = nnodes_bt

        # Output layers
        self.out_T = nn.Linear(nnodes_bt, 10)
        self.out_std = nn.Linear(nnodes_bt, 10)

        if self.std_scale_trainable:
            self.std_scale = Scale()
        else:
            self.std_scale = None

    def _siren_init(self, layer: nn.Module, is_first: bool = False):
        """ Specialized weight initialization for SIREN to ensure stable and smooth gradients.

        Parameters
        ----------
        layer: nn.Module. Linear layer to initialize.
        is_first: bool. Whether this is the input layer of the network.
        """
        with torch.no_grad():
            if is_first:
                limit = 1 / layer.in_features
            else:
                limit = np.sqrt(6 / layer.in_features) / self.w0
            layer.weight.uniform_(-limit, limit)
            layer.bias.uniform_(-limit, limit)

    def forward(self, input: dict) -> torch.Tensor:
        """ Forward pass for the SIREN model.

        Parameters
        ----------
        input: dict. Dictionary containing input tensors (prof, surf, meta).

        Returns
        -------
        torch.Tensor. Concatenated tensor of mean brightness temperatures and standard deviations.
        """
        prof = self.flatten(input['prof'])
        x = self.concat(prof, input['surf'], input['meta'])

        for layer in self.hidden_layers:
            x = torch.sin(self.w0 * layer(x))

        # Mean output scaling
        out = self.out_T(x)
        out = self.bt_output_activation(out)
        out = out * (self.max_T - self.min_T) + self.min_T

        # Uncertainty output
        out_std = self.out_std(x)
        out_std = self.std_output_activation(out_std)
        if self.std_scale is not None:
            out_std = self.std_scale(out_std)
        out_std = out_std + self.std_output_activation_offset

        return torch.cat([out, out_std], dim=1)

    def predict_step(self, batch: dict, batch_nb: int):
        """ Perform prediction step.

        Parameters
        ----------
        batch: dict. Batch containing the 'input' dictionary.
        batch_nb: int. Index of the batch.

        Returns
        -------
        torch.Tensor. Predicted values.
        """
        return self(batch['input'])


class ODEFunc(nn.Module):
    """ Internal derivative function f(x, t) for the Neural ODE integration. """

    def __init__(self, dim: int):
        """ Initialize ODEFunc.

        Parameters
        ----------
        dim: int. Dimension of the latent hidden state.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ Evaluate the derivative at time t. """
        return self.net(x)


class CRTMModelODE(BaseModel):
    """
    Lightning model for the CRTM emulator (Community Radiative Transfer Model) using Neural ODEs.
    This architecture treats the hidden state as a continuous trajectory, preventing spiky
    discontinuities in the Jacobian (adjoint).
    """

    def __init__(self, parameters: DictConfig, optimizer: DictConfig = None,
                 lr_scheduler: DictConfig = None, loss_func: DictConfig = None):
        """ Initialize CRTMModelODE.

        Parameters
        ----------
        parameters: DictConfig. Configuration object containing model parameters.
        optimizer: DictConfig. Optimizer for the model (optional).
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).
        loss_func: DictConfig. Loss function for the model (optional).

        Returns
        -------
        None.
        """
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Input parameters
        self.nprofvars = len(parameters.data.use_prof_vars)
        self.nsurfvars = len(parameters.data.use_surf_vars)
        self.nmetavars = len(parameters.data.use_meta_vars)
        self.nlevels = int(parameters.data.nlevels)
        self.input_dim = self.nprofvars * self.nlevels + self.nsurfvars + self.nmetavars

        # Scaling parameters
        nnodes_bt = parameters.architecture.nnodes_bt
        self.max_T = parameters.data.bt_norm_max
        self.min_T = parameters.data.bt_norm_min

        self.bt_output_activation = nn.Sigmoid()
        self.std_output_activation = nn.Softplus()
        self.std_output_activation_offset = parameters.architecture.std_output_activation_offset
        self.std_scale_trainable = parameters.architecture.std_scale_trainable

        # Architecture components
        self.flatten = nn.Flatten()
        self.concat = lambda *tensors: torch.cat(tensors, dim=1)

        self.input_projection = nn.Linear(self.input_dim, nnodes_bt)
        self.ode_func = ODEFunc(nnodes_bt)
        self.register_buffer('integration_time', torch.tensor([0.0, 1.0]))

        # Output layers
        self.out_T = nn.Linear(nnodes_bt, 10)
        self.out_std = nn.Linear(nnodes_bt, 10)

        if self.std_scale_trainable:
            self.std_scale = Scale()
        else:
            self.std_scale = None

    def forward(self, input: dict) -> torch.Tensor:
        """ Forward pass using Neural ODE integration.

        Parameters
        ----------
        input: dict. Dictionary containing input tensors.

        Returns
        -------
        torch.Tensor. Concatenated tensor of mean brightness temperatures and standard deviations.
        """
        prof = self.flatten(input['prof'])
        x = self.concat(prof, input['surf'], input['meta'])

        # Project to latent space
        x = torch.tanh(self.input_projection(x))

        # Solve the ODE from t=0 to t=1
        x = odeint(self.ode_func, x, self.integration_time, rtol=1e-3, atol=1e-3)[1]

        # Final BT output
        out = self.out_T(x)
        out = self.bt_output_activation(out)
        out = out * (self.max_T - self.min_T) + self.min_T

        # Final Uncertainty output
        out_std = self.out_std(x)
        out_std = self.std_output_activation(out_std)
        if self.std_scale is not None:
            out_std = self.std_scale(out_std)
        out_std = out_std + self.std_output_activation_offset

        return torch.cat([out, out_std], dim=1)

    def predict_step(self, batch: dict, batch_nb: int):
        """ Perform prediction step.

        Parameters
        ----------
        batch: dict. Batch containing the 'input' dictionary.
        batch_nb: int. Index of the batch.

        Returns
        -------
        torch.Tensor. Predicted values.
        """
        return self(batch['input'])


class ConditionalODEFunc(nn.Module):
    """
    Derivative function that treats meta/surf variables as a constant
    physical 'environment' during the integration.
    """

    def __init__(self, latent_dim: int, context_dim: int):
        super().__init__()
        # The network now sees both the evolving hidden state AND the static context
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Concatenate atmospheric state (h) with static environmental info (context)
        combined = torch.cat([h, context], dim=1)
        return self.net(combined)


class CRTMModelODE2(BaseModel):
    def __init__(self, parameters: DictConfig, optimizer=None, lr_scheduler=None, loss_func=None):
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Dimensions
        self.nprofvars = len(parameters.data.use_prof_vars)
        self.nsurfvars = len(parameters.data.use_surf_vars)
        self.nmetavars = len(parameters.data.use_meta_vars)
        self.nlevels = int(parameters.data.nlevels)

        self.latent_dim = parameters.architecture.nnodes_bt
        self.context_dim = self.nsurfvars + self.nmetavars

        # Initial projection: Maps surface level + surface variables to initial hidden state
        self.h0_projection = nn.Linear(self.nprofvars + self.context_dim, self.latent_dim)

        # Continuous block
        self.ode_func = ConditionalODEFunc(self.latent_dim, self.context_dim)
        self.register_buffer('integration_time', torch.tensor([0.0, 1.0]))

        # Output layers
        self.out_T = nn.Linear(self.latent_dim, 10)
        self.out_std = nn.Linear(self.latent_dim, 10)
        self.max_T = parameters.data.bt_norm_max
        self.min_T = parameters.data.bt_norm_min

    def forward(self, input: dict) -> torch.Tensor:
        """ Forward pass using Neural ODE integration.

        Parameters
        ----------
        input: dict. Dictionary containing input tensors.

        Returns
        -------
        torch.Tensor. Concatenated tensor of mean brightness temperatures and standard deviations.
        """

        # 1. Separate inputs
        # prof shape: (batch, nprofvars, nlevels)
        # surf shape: (batch, nsurfvars)
        # meta shape: (batch, nmetavars)
        static_context = torch.cat([input['surf'], input['meta']], dim=1)

        # 2. Set Initial State (h0) at the surface (level 0)
        surface_prof = input['prof'][:, :, 0]
        h0 = torch.tanh(self.h0_projection(torch.cat([surface_prof, static_context], dim=1)))

        # 3. Integrate with Context
        # We pass the static_context to the ode_func at every integration step
        def func_with_context(t, h):
            return self.ode_func(t, h, static_context)

        h_final = odeint(func_with_context, h0, self.integration_time, rtol=1e-3, atol=1e-3)[1]

        # 4. Output Mapping
        out = torch.sigmoid(self.out_T(h_final)) * (self.max_T - self.min_T) + self.min_T
        out_std = torch.nn.functional.softplus(self.out_std(h_final))  # Simplified for example

        return torch.cat([out, out_std], dim=1)


class ConditionalODEFunc2(nn.Module):
    """
    Derivative function (velocity) mapping the atmospheric hidden state
    at a specific pressure level to its rate of change.
    """

    def __init__(self, latent_dim: int, context_dim: int):
        super().__init__()
        # Input: latent_h + context_vars + pressure
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim + 1 + 3, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, h: torch.Tensor, context: torch.Tensor, pressure, prof) -> torch.Tensor:
        """
        Computes the velocity of the atmospheric state integration.

        Parameters
        ----------
        h: torch.Tensor (batch, latent_dim). Current hidden state.
        context: torch.Tensor (batch, context_dim). Static environmental variables.
        pressure: float. The current vertical pressure level.
        """
        # Create a tensor for the current pressure level to concatenate
        p_tensor = torch.cat([h, context, pressure.view(-1, 1), prof], dim=1)
        return self.net(p_tensor)


class CRTMModelFixedODE(BaseModel):
    """
    Lightning model for CRTM emulation using pressure-level dependent integration.

    This model treats the atmosphere as a continuous vertical column, integrating
    radiance accumulation from the surface (Level 0) to the top of the atmosphere.
    It uses a fixed-step integration weighted by the physical thickness (delta_p)
    between pressure levels.
    """

    def __init__(self, parameters: DictConfig, optimizer=None, lr_scheduler=None, loss_func=None):
        """ Initialize CRTMModelFixedODE.

        Parameters
        ----------
        parameters: DictConfig. Configuration object containing:
            - data.use_prof_vars, use_surf_vars, use_meta_vars: Lists of variables.
            - data.nlevels: Number of pressure levels.
            - data.pressure_levels: List of pressure values (hPa).
            - architecture.nnodes_bt: Latent dimension size.
        """
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # 1. Dimensions
        self.nprofvars = len(parameters.data.use_prof_vars)
        self.nsurfvars = len(parameters.data.use_surf_vars)
        self.nmetavars = len(parameters.data.use_meta_vars)
        self.nlevels = int(parameters.data.nlevels)
        self.latent_dim = parameters.architecture.nnodes_bt
        self.context_dim = self.nsurfvars + self.nmetavars

        # 2. Register Pressure Grid and Delta as Buffers
        pressure_levels = instantiate(parameters.data.pressure_levels)
        pressure_levels = (pressure_levels - pressure_levels.min())/(pressure_levels.max() - pressure_levels.min())
        pressure_grid = torch.tensor(pressure_levels, dtype=torch.float32)
        # Calculate Delta Pressure for physical integration: P_i - P_{i+1}
        # Assumes grid is ordered: surface (high pressure) to TOA (low pressure)
        delta_p = torch.abs(pressure_grid[:-1] - pressure_grid[1:])
        self.register_buffer('delta_p', delta_p)
        self.register_buffer('pressure_grid', pressure_grid)

        # 3. Model components
        self.h0_projection = nn.Linear(self.nprofvars + self.context_dim, self.latent_dim)
        self.ode_func = ConditionalODEFunc2(self.latent_dim, self.context_dim)
        self.out_T = nn.Linear(self.latent_dim, 10)
        self.out_std = nn.Linear(self.latent_dim, 10)

        self.max_T = parameters.data.bt_norm_max
        self.min_T = parameters.data.bt_norm_min
        self.bt_output_activation = nn.Sigmoid()
        self.std_output_activation = nn.Softplus()
        self.std_output_activation_offset = parameters.architecture.std_output_activation_offset
        self.std_scale_trainable = parameters.architecture.std_scale_trainable
        if self.std_scale_trainable:
            self.std_scale = Scale()
        else:
            self.std_scale = None

    def forward(self, input: dict) -> torch.Tensor:
        """ Forward pass using pressure-thickness weighted integration.

        Parameters
        ----------
        input: dict. Dictionary containing input tensors:
            - 'prof': (batch, nprofvars, nlevels)
            - 'surf': (batch, nsurfvars)
            - 'meta': (batch, nmetavars)

        Returns
        -------
        torch.Tensor. (batch, 20) concatenated mean BT and std deviation.
        """
        # Prepare static context (surface + meta)
        static_context = torch.cat([input['surf'], input['meta']], dim=1)

        # Initial condition (h at surface)
        # Use surface variables and the first profile level to anchor the integration
        h = torch.tanh(self.h0_projection(static_context))

        # Iterative integration using explicit pressure levels
        for i in range(self.nlevels - 1, 0, -1):
            # Pass the current pressure level to the network
            current_p = torch.full((h.shape[0], 1), self.pressure_grid[i], device=h.device)
            current_prof = input['prof'][:, :, i]  # Profile at current level
            v = self.ode_func(h, static_context, current_p, current_prof)

            # Weighted update by local pressure thickness
            h = h + self.delta_p[i-1] * v

        # Final output layers
        out = self.bt_output_activation(self.out_T(h)) * (self.max_T - self.min_T) + self.min_T
        out_std = self.out_std(h)
        out_std = self.std_output_activation(out_std)
        if self.std_scale is not None:
            out_std = self.std_scale(out_std)
        out_std = out_std + self.std_output_activation_offset

        return torch.cat([out, out_std], dim=1)

    def predict_step(self, batch: dict, batch_nb: int):
        return self(batch['input'])