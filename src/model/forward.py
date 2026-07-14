import numpy as np
import torch
import torch.nn as nn
from typing import Union, Any
from pytorch_lightning import LightningModule
from src.data.statistics import statistics, accumulate_statistics
from src.model.architecture.activation import Swish, Scale, Sine
from src.model.architecture.ode import PressureConditionalODEFunc
from omegaconf import DictConfig
from utilities.instantiators import instantiate
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
            stats_norm = statistics(pred[m, :10] / pred[m, 10:], axis=0, which=['rmse'],
                                    target=target[m, :10] / pred[m, 10:])
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
        loss = self.loss_func(pred, batch['target']['hofx']) * weights

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

    def to(self, device, dtype: torch.dtype = None, non_blocking: bool = False) -> 'BaseModel':
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
        if hasattr(self.loss, 'to'):
            self.loss = self.loss.to(device)
        return self


class CRTMModel(BaseModel):
    """
    Lightning model for the CRTM emulator using configurable architecture.
    This class maintains backward compatibility while using the new architecture-based approach.
    """

    def __init__(
            self,
            parameters: DictConfig = None,
            optimizer: DictConfig = None,
            scheduler: DictConfig = None,
            loss: DictConfig = None,
    ):
        """
        Initialize CRTM Model.

        Parameters
        ----------
        parameters : DictConfig. Configuration object containing model parameters (for backward compatibility).
        optimizer : DictConfig. Optimizer for the model.
        scheduler : DictConfig. Learning rate scheduler configuration.
        loss : DictConfig. Loss function for the model.
        """
        # Build architecture config from parameters (backward compatibility)
        if parameters is not None:
            architecture = DictConfig({
                '_target_': 'forward.model.architecture.CRTMArchitecture',
                'nprofvars': len(parameters.data.use_prof_vars),
                'nsurfvars': len(parameters.data.use_surf_vars),
                'nmetavars': len(parameters.data.use_meta_vars),
                'nlevels': int(parameters.data.nlevels),
                'nnodes_bt': parameters.architecture.nnodes_bt,
                'nhidden_bt': parameters.architecture.nhidden_bt,
                'dropout_rate': parameters.architecture.dropout_rate,
                'bt_norm_max': parameters.data.bt_norm_max,
                'bt_norm_min': parameters.data.bt_norm_min,
                'std_output_activation_offset': parameters.architecture.std_output_activation_offset,
                'std_scale_trainable': parameters.architecture.std_scale_trainable,
            })
        else:
            architecture = None

        # Call parent constructor
        super().__init__(optimizer=optimizer, scheduler=scheduler, loss=loss, architecture=architecture)

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
        return self.forward(batch)


class CRTMModelSmooth(BaseModel):
    """Lightning model for CRTM with smooth architecture (skip connections)."""

    def __init__(self, parameters: DictConfig = None, optimizer: DictConfig = None,
                 scheduler: DictConfig = None, loss: DictConfig = None):
        """Initialize CRTM Smooth Model."""
        # Build architecture config from parameters
        if parameters is not None:
            architecture = DictConfig({
                '_target_': 'forward.model.architecture.CRTMSmoothArchitecture',
                'nprofvars': len(parameters.data.use_prof_vars),
                'nsurfvars': len(parameters.data.use_surf_vars),
                'nmetavars': len(parameters.data.use_meta_vars),
                'nlevels': int(parameters.data.nlevels),
                'nnodes_bt': parameters.architecture.nnodes_bt,
                'nhidden_bt': parameters.architecture.nhidden_bt,
                'dropout_rate': parameters.architecture.dropout_rate,
                'bt_norm_max': parameters.data.bt_norm_max,
                'bt_norm_min': parameters.data.bt_norm_min,
                'std_output_activation_offset': parameters.architecture.std_output_activation_offset,
                'std_scale_trainable': parameters.architecture.std_scale_trainable,
            })
        else:
            architecture = None

        super().__init__(optimizer=optimizer, scheduler=scheduler, loss=loss, architecture=architecture)

    def predict_step(self, batch: dict, batch_nb: int):
        """Perform prediction step."""
        return self.forward(batch)


class CRTMModelPCA(CRTMModelSmooth):
    """CRTM model with PCA-based profile scaling."""

    def __init__(self, parameters: DictConfig = None, optimizer: DictConfig = None,
                 scheduler: DictConfig = None, loss: DictConfig = None):
        """Initialize CRTM PCA Model."""
        super().__init__(parameters=parameters, optimizer=optimizer, scheduler=scheduler, loss=loss)

        # Store profile scaling factor
        self.prof_scaling = parameters.data.prof_scaling if parameters is not None else 1.0

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass with profile scaling."""
        # Scale profiles before forward pass
        scaled_input = {
            'prof': batch['input']['prof'] / self.prof_scaling,
            'surf': batch['input']['surf'],
            'meta': batch['input']['meta']
        }
        scaled_batch = {'input': scaled_input}
        return super().forward(scaled_batch)

    def predict_step(self, batch: dict, batch_nb: int):
        """Perform prediction step."""
        return self.forward(batch)


class CRTMModelSiren(BaseModel):
    """CRTM model using SIREN architecture."""

    def __init__(self, parameters: DictConfig = None, optimizer: DictConfig = None,
                 scheduler: DictConfig = None, loss: DictConfig = None):
        """Initialize CRTM SIREN Model."""
        # Build architecture config from parameters
        if parameters is not None:
            architecture = DictConfig({
                '_target_': 'forward.model.architecture.CRTMSirenArchitecture',
                'nprofvars': len(parameters.data.use_prof_vars),
                'nsurfvars': len(parameters.data.use_surf_vars),
                'nmetavars': len(parameters.data.use_meta_vars),
                'nlevels': int(parameters.data.nlevels),
                'nnodes_bt': parameters.architecture.nnodes_bt,
                'nhidden_bt': parameters.architecture.nhidden_bt,
                'dropout_rate': parameters.architecture.dropout_rate,
                'siren_w0': getattr(parameters.architecture, 'siren_w0', 10.0),
                'bt_norm_max': parameters.data.bt_norm_max,
                'bt_norm_min': parameters.data.bt_norm_min,
                'std_output_activation_offset': parameters.architecture.std_output_activation_offset,
                'std_scale_trainable': parameters.architecture.std_scale_trainable,
            })
        else:
            architecture = None

        super().__init__(optimizer=optimizer, scheduler=scheduler, loss=loss, architecture=architecture)

    def predict_step(self, batch: dict, batch_nb: int):
        """Perform prediction step."""
        return self.forward(batch)


class ODEFunc(nn.Module):
    """
    Internal derivative function f(x, t) for the Neural ODE integration.
    LEGACY: Use forward.model.architecture.ode.ODEFunc instead.
    """

    def __init__(self, dim: int):
        """
        Initialize ODEFunc.

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
        """Evaluate the derivative at time t."""
        return self.net(x)


class ConditionalODEFunc(nn.Module):
    """
    Derivative function that treats meta/surf variables as a constant
    physical 'environment' during the integration.
    LEGACY: Use forward.model.architecture.ode.ConditionalODEFunc instead.
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


class ConditionalODEFunc2(nn.Module):
    """
    Derivative function (velocity) mapping the atmospheric hidden state
    at a specific pressure level to its rate of change.
    LEGACY: Use forward.model.architecture.ode.PressureConditionalODEFunc instead.
    """

    def __init__(self, latent_dim: int, context_dim: int):
        super().__init__()
        # Input: latent_h + context_vars + pressure + profile
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
        pressure_levels = (pressure_levels - pressure_levels.min()) / (pressure_levels.max() - pressure_levels.min())
        pressure_grid = torch.tensor(pressure_levels, dtype=torch.float32)
        # Calculate Delta Pressure for physical integration: P_i - P_{i+1}
        # Assumes grid is ordered: surface (high pressure) to TOA (low pressure)
        delta_p = torch.abs(pressure_grid[:-1] - pressure_grid[1:])
        self.register_buffer('delta_p', delta_p)
        self.register_buffer('pressure_grid', pressure_grid)

        # 3. Model components
        self.h0_projection = nn.Linear(self.context_dim, self.latent_dim)
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
            h = h + self.delta_p[i - 1] * v

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