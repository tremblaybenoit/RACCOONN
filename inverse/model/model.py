import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import ModuleList
from data.statistics import statistics, accumulate_statistics
from forward.model.model import BaseModel
from forward.model.activation import Sine
from inverse.model.encoding import IdentityPositionalEncoding
from utilities.instantiators import instantiate
from data.transformations import mean_stdev, min_max
import numpy as np
import math


class KaimingInit(nn.Module):
    def __init__(self, mode: str = 'fan_in', activation_type: str = 'swish'):
        super().__init__()
        self.mode = mode

        # Mapping smooth activations to their closest Kaiming proxy
        if activation_type.lower() in ['swish', 'silu', 'mish', 'gelu']:
            self.nonlinearity = 'leaky_relu'
            self.a = 0.01  # Small slope proxy for smooth activations
        else:
            self.nonlinearity = 'relu'
            self.a = 0

    def __call__(self, module: nn.Module):
        if hasattr(module, 'weight'):
            nn.init.kaiming_uniform_(
                module.weight,
                a=self.a,
                mode=self.mode,
                nonlinearity=self.nonlinearity
            )
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


class SwishInit:
    """
    Improved Swish weights initializer for Coordinate-based PINNs.
    Optimized to prevent the 'Flat Start' common in Swish/SiLU networks.
    """

    def __init__(self, is_first_layer: bool = False, is_head: bool = False):
        """
        Initialize SwishInit.

        Parameters
        ----------
        is_first_layer: bool. Flag indicating if the module is the first layer.
        is_head: bool. Flag indicating if the module is a head layer.
        """
        self.is_first_layer = is_first_layer
        self.is_head = is_head

    def __call__(self, module):
        """
        Swish weights initialization function.
        """
        if not hasattr(module, 'weight') or module.weight is None:
            return

        with torch.no_grad():
            dim_in = module.weight.size(1)

            if self.is_head:
                # Use Xavier Uniform for the output.
                # This keeps the whitened PCA coefficients centered at zero
                # while allowing enough range to hit +/- 2.0 std dev.
                # nn.init.xavier_uniform_(module.weight)
                nn.init.uniform_(module.weight, -1e-5, 1e-5)

            elif self.is_first_layer:
                # The "SIREN-lite" trick: Initialize the first layer with
                # a larger uniform range. This ensures the Gaussian Positional
                # Encoding is 'stretched' across the Swish non-linearity
                # from Step 0.
                scale = 1.0 / dim_in
                nn.init.uniform_(module.weight, -scale, scale)

            else:
                # For hidden layers, we use a gain of sqrt(2) ~ 1.414.
                # This treats Swish as a smooth ReLU and maintains activation
                # variance throughout the Hydra trunk.
                gain = np.sqrt(2.0)
                std = gain / np.sqrt(float(dim_in))
                nn.init.normal_(module.weight, mean=0, std=std)

            # --- Symmetry Breaking for Biases ---
            if hasattr(module, 'bias') and module.bias is not None:
                if self.is_head:
                    # Keep output biases at zero to avoid shifting PCA mean
                    nn.init.zeros_(module.bias)
                else:
                    # Small uniform noise in biases helps the network 'locate'
                    # features spatially. Without this, all neurons in a layer
                    # start by doing the exact same thing.
                    nn.init.uniform_(module.bias, -0.05, 0.05)

class NeuralFieldInit:
    """
    Custom initializer for residual MLPs with Swish activations.
    - is_first_layer: optional special handling for the first layer
    - is_head: final prediction layer (usually followed by sigmoid)
    """

    def __init__(self, is_first_layer: bool = False, is_head: bool = False):
        self.is_first_layer = is_first_layer
        self.is_head = is_head

    def __call__(self, module):
        if not isinstance(module, nn.Linear):
            return

        fan_in = module.weight.data.size(1)

        # ---- 1. First layer (optional special treatment) ----
        if self.is_first_layer:
            std = math.sqrt(2.0 / fan_in)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        # ---- 2. Head layer (output) ----
        if self.is_head:
            # Zero init is safest for sigmoid outputs
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        # ---- 3. Residual block output layers ----
        if getattr(module, "_is_residual_out", False):
            nn.init.normal_(module.weight, mean=0.0, std=1e-4)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        # ---- 4. Standard hidden layers ----
        std = math.sqrt(2.0 / fan_in) * 0.5  # scaled He init for Swish + residuals
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class XavierInit(nn.Module):
    """
    Callable Xavier/Glorot initializer.
    """
    def __init__(self, gain: float = 1.0):
        """
        Initialize Xavier/Glorot initializer.

        Parameters
        ----------
        gain: float. Gain factor for Xavier initialization.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Parameters
        self.gain = gain

    def __call__(self, module: nn.Module):
        """
        Xavier/Glorot weights initialization function.

        Parameters
        ----------
        module: nn.Module. The module to initialize.

        Returns
        -------
        None.
        """

        # Weight initialization
        if hasattr(module, 'weight'):
            nn.init.xavier_uniform_(module.weight, gain=self.gain)
        # Bias initialization
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


class SirenInit(nn.Module):
    """
    Callable SIREN initializer with configuration moved to __init__.
    """
    def __init__(self, is_first_layer: bool = False, activation=None, is_head: bool = False, w0: float = None):
        """
        Initialize SIREN initializer.

        Parameters
        ----------
        is_first_layer: bool. Flag indicating if the module is the first layer.
        activation: Callable. Activation function.
        is_head: bool. Flag indicating if the module is a head layer.
        w0: float. Frequency scaling factor.
        """

        # Class inheritance
        super().__init__()

        # Parameters
        self.is_first_layer = is_first_layer
        self.is_head = is_head
        self.w0 = w0 if w0 is not None else getattr(activation, 'w0', w0)

    def __call__(self, module):
        """
        SIREN weights initialization function.

        Parameters
        ----------
        module: nn.Module. The module to initialize.

        Returns
        -------
        None.
        """

        # Weight initialization
        with torch.no_grad():
            if hasattr(module, 'weight'):
                # Input dimension
                dim_in = module.weight.size(1)

                # First layer initialization
                if self.is_first_layer:
                    # First layer scale: w0 is critical for spatial detail
                    bound = 1.0 / dim_in
                    nn.init.uniform_(module.weight, -bound, bound)
                    # Ensure we don't multiply by w0 twice if the Sine block also does it
                    # Usually, we scale the weights by w0 here.
                    module.weight *= self.w0

                elif self.is_head:
                    # For Lanczos u-space: keep weights small so we start near the mean (x_b)
                    nn.init.uniform_(module.weight, -1e-4, 1e-4)

                else:
                    # Standard SIREN hidden layer init
                    bound = np.sqrt(6.0 / dim_in) / self.w0
                    nn.init.uniform_(module.weight, -bound, bound)

                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)


class Residual(nn.Module):
    """
    Residual block with skip connection.
    """
    def __init__(self, module: nn.Module, projection: nn.Module = None):
        """
        Initialize Residual block.

        Parameters
        ----------
        module: nn.Module. Internal block to apply.
        projection: nn.Module. Optional projection for skip connection.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Internal block
        self.module = module
        self.projection = projection if projection is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Residual block.

        Parameters
        ----------
        x: tensor. Input tensor.

        Returns
        -------
        out: tensor. Output tensor after applying the residual block.
        """

        # Skip connection: out = f(x) + x
        return self.module(x) + self.projection(x)


class Concatenate(nn.Module):
    """
    Concatenate input with the output of a module.
    """
    def __init__(self, module: nn.Module):
        """
        Initialize Concatenate block.

        Parameters
        ----------
        module: nn.Module. Internal block to apply.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Module
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Concatenate block.

        Parameters
        ----------
        x: tensor. Input tensor.

        Returns
        -------
        out: tensor. Output tensor after applying the concatenate block.
        """

        # Concatenate input with module output
        return torch.cat([x, self.module(x)], dim=-1)


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module,
                 dropout_rate: float, layernorm: nn.Module=None, init_func: nn.Module=None):
        """ Initialize a Linear Block.

        Parameters
        ----------
        in_features: int. Number of input features.
        out_features: int. Number of output features.
        activation: Callable. Activation function.
        dropout_rate: float. Dropout rate.
        layernorm: Callable. Normalization layer.
        init_func: Callable. Weight initialization function.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Internal layers
        self.linear = nn.Linear(in_features, out_features)
        self.layernorm = layernorm if layernorm is not None else nn.Identity()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

        # Apply weight initialization if provided
        if init_func is not None:
            init_func(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Linear Block.
        Parameters
        ----------
        x: tensor. Input tensor.

        Returns
        -------
        out: tensor. Output tensor after applying the linear block.
        """

        # Block operations
        out = self.linear(x)
        out = self.layernorm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class SirenFeedForwardBlock(LinearBlock):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = None,
                 dropout_rate: float = 0, init_func: nn.Module = None, layernorm: nn.Module=None):
        """ Initialize a SIREN Feed-Forward Block.

        Parameters
        ----------
        in_features: int. Number of input features.
        out_features: int. Number of output features.
        activation: Callable. Activation function.
        dropout_rate: float. Dropout rate.
        init_func: Callable. SIREN initialization function.
        layernorm: Callable. Normalization layer.

        Returns
        -------
        None.
        """

        # SIREN Activation
        activation = activation if activation is not None else Sine(w0=30.0)

        # Apply SIREN Initialization if not provided
        if init_func is None:
            init_func = SirenInit(is_first_layer=False, activation=activation)

        # Class inheritance
        super().__init__(in_features, out_features, activation, dropout_rate,
                         layernorm=layernorm, init_func=init_func)


class SirenResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = None,
                 dropout_rate: float = 0, init_func: nn.Module = None, layernorm: nn.Module=None):
        """ Initialize a SIREN Residual Block.

        Parameters
        ----------
        in_features: int. Number of neurons in the block.
        out_features: int. Number of neurons in the block.
        activation: Callable. Activation function.
        dropout_rate: float. Dropout rate.
        init_func: Callable. SIREN initialization function.
        layernorm: Callable. Normalization layer.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Internal layers for the residual path (f(x))
        linear1 = nn.Linear(in_features, in_features)
        layernorm = layernorm if layernorm is not None else nn.Identity()
        activation = activation if activation is not None else Sine(w0=30.0)
        linear2 = nn.Linear(in_features, out_features)

        # Apply SIREN Initialization to internal layers
        if init_func is None:
            init_func = SirenInit(is_first_layer=False, activation=activation)
        init_func(linear1)
        init_func(linear2)

        # Residual connection
        self.residual = Residual(nn.Sequential(
            linear1,
            layernorm,
            activation,
            linear2
        ))
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the SIREN Residual Block.

        Parameters
        ----------
        x: tensor. Input tensor.

        Returns
        -------
        out: tensor. Output tensor after applying the residual block.
        """

        out = self.residual(x)
        out = self.dropout(out)
        return out


class PredictionHeads(nn.Module):
    """
    Module for multiple prediction heads.
    """
    def __init__(self, heads: nn.ModuleList):
        """
        Initialize Prediction Heads.

        Parameters
        ----------
        heads: nn.ModuleList. List of prediction head modules.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Create heads
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Prediction Heads.

        Parameters
        ----------
        x: tensor. Input tensor.

        Returns
        -------
        out: tensor. Output tensor after applying the prediction heads.
        """

        # Apply each head and concatenate outputs
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=-1)


class ResidualMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable layers.
    """
    def __init__(self, input_layer: DictConfig, hidden_layer: DictConfig, output_layer: DictConfig,
                 positional_encoding: DictConfig = None, hidden_n_layers: int=2,
                 hidden_skip: bool=False, output_skip: bool=False, output_n_heads: int=1):
        """
        Initialize MLP.

        Parameters
        ----------
        input_layer: DictConfig. Configuration for the input layer.
        hidden_layer: DictConfig. Configuration for the hidden layers.
        output_layer: DictConfig. Configuration for the output layer.
        hidden_n_layers: int. Number of hidden layers.
        hidden_skip: bool. Flag to enable skip connections in hidden layers.
        output_skip: bool. Flag to enable skip connections in output layer.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Positional encoding
        if positional_encoding is not None:
            self.positional_encoding = instantiate(positional_encoding)
            input_layer.in_features = self.positional_encoding.d_output
        else:
            self.positional_encoding = IdentityPositionalEncoding(d_input=input_layer.in_features)
        # Input layer
        self.input_layers = instantiate(input_layer)
        # Hidden layers
        hidden_layers = nn.ModuleList()
        # Build hidden layers with optional skip connections
        for _ in range(hidden_n_layers):
            hidden_layer_instance = instantiate(hidden_layer)
            if hidden_skip:
                # Projection for skip connection if dimensions differ
                if hidden_layer_instance.in_features != hidden_layer_instance.out_features:
                    projection = nn.Linear(hidden_layer_instance.in_features,
                                           hidden_layer_instance.out_features)
                else:
                    projection = None
                hidden_layer_instance = Residual(hidden_layer_instance, projection=projection)
            hidden_layers.append(hidden_layer_instance)
        # Output layer
        if output_skip:
            self.hidden_layers = Concatenate(nn.Sequential(*hidden_layers))
            output_layer.in_features = hidden_layer.out_features + input_layer.out_features
        else:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        # Model architecture
        output_n_heads = instantiate(output_n_heads)
        if output_n_heads > 1:
            output_layers = ModuleList([instantiate(output_layer) for _ in range(output_n_heads)])
            self.output_layers = PredictionHeads(output_layers)
        else:
            self.output_layers = instantiate(output_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x: tensor. Input tensor.

        Returns
        -------
        out: tensor. Output tensor after applying the MLP.
        """

        # Pass
        x_encoded = self.positional_encoding(x)
        x_input = self.input_layers(x_encoded)
        x_hidden = self.hidden_layers(x_input)
        x_output = self.output_layers(x_hidden)
        return x_output


class HydraResidualMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable layers.
    """
    def __init__(self, input_layer: DictConfig, hidden_layer: DictConfig, output_layer: DictConfig,
                 positional_encoding: DictConfig = None, hidden_n_layers: int=2,
                 hidden_skip: bool=False, output_skip: bool=False, output_n_heads: int=1, output_n_layers: int=1, output_final_layer: DictConfig=None):
        """
        Initialize MLP.

        Parameters
        ----------
        input_layer: DictConfig. Configuration for the input layer.
        hidden_layer: DictConfig. Configuration for the hidden layers.
        output_layer: DictConfig. Configuration for the output layer.
        hidden_n_layers: int. Number of hidden layers.
        hidden_skip: bool. Flag to enable skip connections in hidden layers.
        output_skip: bool. Flag to enable skip connections in output layer.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Positional encoding
        if positional_encoding is not None:
            self.positional_encoding = instantiate(positional_encoding)
            input_layer.in_features = self.positional_encoding.d_output
        else:
            self.positional_encoding = IdentityPositionalEncoding(d_input=input_layer.in_features)
        # Input layer
        self.input_layers = instantiate(input_layer)
        # Hidden layers
        hidden_layers = nn.ModuleList()
        # Build hidden layers with optional skip connections
        for _ in range(hidden_n_layers):
            hidden_layer_instance = instantiate(hidden_layer)
            if hidden_skip:
                # Projection for skip connection if dimensions differ
                if hidden_layer_instance.in_features != hidden_layer_instance.out_features:
                    projection = nn.Linear(hidden_layer_instance.in_features,
                                           hidden_layer_instance.out_features)
                else:
                    projection = None
                hidden_layer_instance = Residual(hidden_layer_instance, projection=projection)
            hidden_layers.append(hidden_layer_instance)
        # Output layer
        if output_skip:
            self.hidden_layers = Concatenate(nn.Sequential(*hidden_layers))
        else:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        # Model architecture
        output_n_heads = instantiate(output_n_heads)
        if output_n_heads > 1:
            output_layers = ModuleList()
            for _ in range(output_n_heads):
                head_layers = []
                for l in range(output_n_layers):
                    output_layer_l = output_layer.copy()
                    if output_skip and l == 0:
                        output_layer_l.in_features = hidden_layer.out_features + input_layer.out_features
                        if output_layer_l._target_ == 'inverse.model.model.SirenResidualBlock':
                            output_layer.out_features = output_layer_l.in_features
                            output_layer_l.out_features = output_layer_l.in_features
                    elif l == 0:
                        output_layer_l.in_features = hidden_layer.out_features
                    else:
                        output_layer_l.in_features = output_layer.out_features
                    if hasattr(output_layer_l.activation, 'in_features'):
                        output_layer_l.activation.in_features = output_layer_l.out_features
                    # breakpoint()
                    head_layers.append(instantiate(output_layer_l))
                if output_final_layer is not None:
                    output_final_layer.in_features = output_layer.out_features
                    head_layers.append(instantiate(output_final_layer))
                output_layers.append(nn.Sequential(*head_layers))
            self.output_layers = PredictionHeads(output_layers)
        else:
            head_layers = []
            for l in range(output_n_layers):
                output_layer_l = output_layer.copy()
                if output_skip and l == 0:
                    output_layer_l.in_features = hidden_layer.out_features + input_layer.out_features
                    if output_layer_l._target_ == 'inverse.model.model.SirenResidualBlock':
                        output_layer.out_features = output_layer_l.in_features
                        output_layer_l.out_features = output_layer_l.in_features
                elif l == 0:
                    output_layer_l.in_features = hidden_layer.out_features
                else:
                    output_layer_l.in_features = output_layer.out_features
                if hasattr(output_layer_l.activation, 'in_features'):
                    output_layer_l.activation.in_features = output_layer_l.out_features
                # breakpoint()
                head_layers.append(instantiate(output_layer_l))
            self.output_layers = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x: tensor. Input tensor.

        Returns
        -------
        out: tensor. Output tensor after applying the MLP.
        """

        # Pass
        x_encoded = self.positional_encoding(x)
        x_input = self.input_layers(x_encoded)
        x_hidden = self.hidden_layers(x_input)
        x_output = self.output_layers(x_hidden)
        return x_output


class PINNverseOperator(BaseModel):
    """Class for the Physics-Informed Neural Network (PINN) inverse model."""
    def __init__(self, optimizer: DictConfig = None, loss_func: DictConfig = None, lr_scheduler: DictConfig = None,
                 architecture: DictConfig = None, parameters: DictConfig = None, transform: DictConfig = None):
        """ Initialize model.

        Parameters
        ----------
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        architecture: DictConfig. Configuration for the model architecture.
        parameters: DictConfig. Configuration for the model parameters.
        transform: DictConfig. Optional transform to apply to the predicted profiles before computing the loss.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func)

        # Results & metrics
        self.results['prof'] = []
        self.metrics['prof'], self.metrics['prof_target'], self.metrics['prof_background'] = {}, {}, {}
        self.metrics['prof_white'], self.metrics['prof_white_target'], self.metrics['prof_white_background'] = {}, {}, {}

        # Model parameters
        self.n_prof = instantiate(parameters.n_prof) if parameters is not None and hasattr(parameters, 'n_prof') \
            else 1
        self.n_levels = parameters.n_levels if parameters is not None and hasattr(parameters, 'n_levels') \
            else 1
        self.prof_vars = parameters.prof_vars if parameters is not None and hasattr(parameters, 'prof_vars') \
            else [f'var_{i}' for i in range(self.n_prof)]

        # Model architecture
        self.model = HydraResidualMLP(
            input_layer=architecture.input_layer,  # Pass DictConfig directly
            hidden_layer=architecture.hidden_layer,  # Pass DictConfig directly
            output_layer=architecture.output_layer,  # Pass DictConfig directly
            positional_encoding=architecture.get('positional_encoding', None),
            hidden_n_layers=architecture.get('hidden_n_layers', 2),
            hidden_skip=architecture.get('hidden_skip', False),
            output_skip=architecture.get('output_skip', False),
            output_n_heads=architecture.get('output_n_heads', 1),
            output_final_layer=architecture.get('output_final_layer', None),
            output_n_layers=architecture.get('output_n_layers', 1),
        )

        # Optional transform for predicted profiles before computing the loss
        self.transform = instantiate(transform) if transform is not None else None

    def forward(self, x: dict) -> torch.Tensor:
        """ Forward pass through the model.

            Parameters
            ----------
            x: dict. Input coordinates.

            Returns
            -------
            Predicted profiles: tensor.
        """

        # 1. Vectorized expansion
        # We create a list of tensors all shaped (Batch, n_levels, 1)
        tensors = []
        for k, v in x.items():
            if k != 'pressure':
                if v.ndim == 1:
                    # For (Batch,) -> (Batch, n_levels, 1)
                    tensors.append(v[:, None, None].expand(-1, self.n_levels, 1))
                else:
                    # For (Batch, n_levels) -> (Batch, n_levels, 1)
                    tensors.append(v.unsqueeze(-1))

        # 2. Single Concatenation
        # Shape: (Batch * n_levels, num_features)
        inputs = torch.cat(tensors, dim=-1).view(-1, len(tensors))

        # 3. Inference and Final Reshape
        # Output: (Batch, n_prof, n_levels)
        out = self.model(inputs)
        return out.view(-1, self.n_levels, self.n_prof).transpose(1, 2)

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

        # Compute profiles
        pred = {'prof': self.forward(batch['input']).contiguous()}

        # Compute loss function
        loss, pred['hofx'] = self.loss_func(pred, batch['target'], batch['input'])
        detached_loss = {k: v.detach().item() if v.ndim == 0 else v.detach() for k, v in loss.items()}

        # Logging
        self._logging(stage, detached_loss, batch['input'], batch['target'], {k: v.detach() for k, v in pred.items()})

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
        return prof

    def _logging_prof(self, pred: torch.Tensor, target: torch.Tensor, background: torch.Tensor=None, key: str='') -> None:
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
        stats_pred = {k: v.detach() for k, v in stats_pred.items()}
        stats_target = statistics(target, axis=0, which=['mean', 'stdev'])
        stats_target = {k: v.detach() for k, v in stats_target.items()}
        # Check if statistics dictionaries are empty
        if self.metrics.get('prof'+key):
            self.metrics['prof'+key] = accumulate_statistics([self.metrics['prof'+key], stats_pred])
            self.metrics['prof_target'+key] = accumulate_statistics([self.metrics['prof_target'+key], stats_target])
        else:
            self.metrics['prof'+key] = stats_pred
            self.metrics['prof_target'+key] = stats_target

        # Log mean background profiles and rmse if available
        if background is not None:
            stats_background = statistics(background, axis=0, which=['mean', 'stdev', 'rmse', 'mae'], target=target)
            stats_background = {k: v.detach() for k, v in stats_background.items()}
            # Check if statistics dictionaries are empty
            if self.metrics.get('prof_background'+key):
                self.metrics['prof_background'+key] = accumulate_statistics([self.metrics['prof_background'+key], stats_background])
            else:
                self.metrics['prof_background'+key] = stats_background

    def _logging_prof_white(self, pred: torch.Tensor, target: torch.Tensor, background: torch.Tensor=None) -> None:
        """ Log profile metrics in "white" space (i.e., without pressure-level filtering).

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
        stats_pred = {k: v.detach() for k, v in stats_pred.items()}
        stats_target = statistics(target, axis=0, which=['mean', 'stdev'])
        stats_target = {k: v.detach() for k, v in stats_target.items()}
        # Check if statistics dictionaries are empty
        if self.metrics.get('prof_white'):
            self.metrics['prof_white'] = accumulate_statistics([self.metrics['prof_white'], stats_pred])
            self.metrics['prof_white_target'] = accumulate_statistics([self.metrics['prof_white_target'], stats_target])
        else:
            self.metrics['prof_white'] = stats_pred
            self.metrics['prof_white_target'] = stats_target

        # Log mean background profiles and rmse if available
        if background is not None:
            stats_background = statistics(background, axis=0, which=['mean', 'stdev', 'rmse', 'mae'], target=target)
            stats_background = {k: v.detach() for k, v in stats_background.items()}
            # Check if statistics dictionaries are empty
            if self.metrics.get('prof_white_background'):
                self.metrics['prof_white_background'] = accumulate_statistics([self.metrics['prof_white_background'], stats_background])
            else:
                self.metrics['prof_white_background'] = stats_background

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

        # Logger flag
        logger_flag = stage != 'test'

        # If testing, return predictions in addition to loss
        if stage == 'test':
            # Log pressure levels
            self.results['pressure'] = input['pressure'][0:1].detach().cpu().numpy()
            # Store test outputs
            for k, v in {'prof': pred['prof'], 'hofx': pred['hofx']}.items():
                self.results[k].append(v.detach().cpu().numpy())
        elif stage == 'valid':
            # Log pressure levels
            self.results['pressure'] = input['pressure'][0:1].detach().cpu().numpy()
            # Log metrics for hofx and profiles
            self._logging_hofx(pred['hofx'], target['hofx'], target['cloud_filter'].bool(),
                               target['daytime_filter'].bool())
            self._logging_prof(pred['prof'], target['prof'], background=target.get('prof_background', None))
            # if 'prof_mean_stdev' in pred and 'prof_target_mean_stdev' in pred:
            #     self._logging_prof(pred['prof_mean_stdev'], pred['prof_mean_stdev'], background=pred.get('prof_background_mean_stdev', None))
            # if 'prof_min_max' in pred and 'prof_target_min_ax' in pred:
            #     self._logging_prof(pred['prof_min_max'], pred['prof_min_max'], background=pred.get('prof_background_min_max', None))
            if 'prof_white' in pred and 'prof_white' in target:
                self._logging_prof_white(pred['prof_white'], target['prof_white'], background=target.get('prof_white_background', None))
        # Log L2 norm of model parameters during training
        elif stage == 'train':
            # Compute L2 norm of the model parameters
            with torch.no_grad():
                l2_norm = sum(p.pow(2).sum() for p in self.parameters()).sqrt().item()
            self.log(f"{stage}_l2_norm", l2_norm, on_epoch=True, prog_bar=False, logger=logger_flag)
            # If using UncertaintyVarLoss, log the effective weights
            if hasattr(self.loss_func, 'log_var_obs'):
                self.log("weight_obs", torch.exp(-self.loss_func.log_var_obs), prog_bar=True, logger=logger_flag)
            if hasattr(self.loss_func, 'log_var_model'):
                self.log("weight_model", torch.exp(-self.loss_func.log_var_model), prog_bar=True, logger=logger_flag)
            if hasattr(self.loss_func, 'alpha'):
                self.log("weight_alpha", 2.0 - torch.sigmoid(self.loss_func.alpha) * 2.0, prog_bar=True, logger=logger_flag)

        # Log learning rate
        if stage == 'train' and self.lr_schedulers() is not None:
            lr = self.lr_schedulers().get_last_lr()[0]
            self.log(f"{stage}_lr", lr, on_epoch=True, prog_bar=False, logger=logger_flag)
        # Log total loss
        if 'total' in loss:
            self.log(f"{stage}_loss", loss['total'], on_epoch=True, prog_bar=True, logger=logger_flag)
        # Log profile and boundary condition losses
        for key in ['model', 'bcs', 'model_phys']:
            if key in loss:
                self.log(f"{stage}_loss_{key}", loss[key].mean(), on_epoch=True, prog_bar=True, logger=logger_flag)
                # Detailed logging per profile and variable
                if loss[key].ndim == 3:
                    for i, var in enumerate(self.prof_vars):
                        self.log(f"{stage}_loss_{key}_{i}_{var}", loss[key][:, i, :].mean(), on_epoch=True,
                                 prog_bar=False, logger=logger_flag)
                # If pressure-level filtering is involved, log only the relevant levels
                elif loss[key].ndim == 2 and hasattr(self.loss_func, 'pressure_filter') and self.loss_func.pressure_filter is not None:
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
                for i in range(loss['obs'].shape[1]):
                    self.log(f"{stage}_loss_obs_{i}", loss['obs'][:, i].mean(), on_epoch=True, prog_bar=False,
                             logger=logger_flag)

        # Log Sobolev loss
        if 'sobolev' in loss:
            self.log(f"{stage}_loss_sobolev", loss['sobolev'].mean(), on_epoch=True, prog_bar=False, logger=logger_flag)


class PINNverseOperatorP(PINNverseOperator):

    def __init__(self, optimizer: DictConfig = None, loss_func: DictConfig = None, lr_scheduler: DictConfig = None,
                 architecture: DictConfig = None, parameters: DictConfig = None, transform: DictConfig = None, stats: DictConfig = None,
                 sigmoid: bool = False, min_max: bool = True):

        # Class inheritance
        super().__init__(optimizer=optimizer, loss_func=loss_func, lr_scheduler=lr_scheduler,
                         architecture=architecture, parameters=parameters, transform=transform)

        self.metrics['prof_min_max'], self.metrics['prof_target_min_max'], self.metrics['prof_background_min_max'] = {}, {}, {}
        self.metrics['prof_mean_stdev'], self.metrics['prof_target_mean_stdev'], self.metrics['prof_background_mean_stdev'] = {}, {}, {}
        self.min_max = min_max
        self.stats = instantiate(stats) if stats is not None else None
        self.sigmoid = sigmoid
        self.sigmoid_stats = {'min': (self.stats['min'] - self.stats['mean']) / self.stats['stdev'], 'max': (self.stats['max'] - self.stats['mean'])/ self.stats['stdev']}

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

        # Norm
        if self.min_max:

            # Compute profiles
            pred = {'prof': self.forward(batch['input']).contiguous()}

            # Apply transform
            if self.transform is not None:
                pred['prof_phys'] = self.transform(pred['prof'].clone())
                pred['prof_target_phys'] = self.transform(batch['target']['prof'].clone())
                pred['prof_background_phys'] = self.transform(batch['target']['prof_background'].clone())
            else:
                pred['prof_phys'] = pred['prof'].clone()

            pred['prof_min_max'] = pred['prof'].clone()
            pred['prof_mean_stdev'] = mean_stdev(pred['prof_phys'].clone(), self.stats, axis=None)#, stdev_thresh=1.0*np.ones_like(self.stats['stdev']))
            pred['prof_background_min_max'] = batch['target']['prof_background'].clone()
            pred['prof_background_mean_stdev'] = mean_stdev(pred['prof_background_phys'].clone(), self.stats, axis=None)#, stdev_thresh=1.0*np.ones_like(self.stats['stdev']))
            pred['prof_target_min_max'] = batch['target']['prof'].clone()
            pred['prof_target_mean_stdev'] = mean_stdev(pred['prof_target_phys'].clone(), self.stats, axis=None)#, stdev_thresh=1.0*np.ones_like(self.stats['stdev']))
        else:

            # Compute profiles
            if self.sigmoid:
                raw_out = self.forward(batch['input']).contiguous()
                sig_out = torch.sigmoid(raw_out)
                prof_minmax = min_max(sig_out.clone(), self.sigmoid_stats, axis=None)
                pred = {'prof': prof_minmax}
                # pred = {'prof': min_max(torch.sigmoid(self.forward(batch['input']).contiguous().clone()), self.sigmoid_stats, axis=None)}
            else:
                pred = {'prof': self.forward(batch['input']).contiguous()}

            # Apply transform
            if self.transform is not None:
                pred['prof_phys'] = self.transform(pred['prof'].clone())
                pred['prof_target_phys'] = self.transform(batch['target']['prof'].clone())
                # pred['prof_phys'][:, 0:1, :] = pred['prof_target_phys'][:, 0:1, :].clone()
                # pred['prof_phys'][:, 1:2, :] = pred['prof_target_phys'][:, 1:2, :].clone()
                # pred['prof_phys'][:, 2:3, :] = pred['prof_target_phys'][:, 2:3, :].clone()
                pred['prof_background_phys'] = self.transform(batch['target']['prof_background'].clone())
            else:
                pred['prof_phys'] = pred['prof'].clone()

            # Compute normalized and standardized profiles for loss computation if required by the loss function
            pred['prof_min_max'] = min_max(pred['prof_phys'].clone(), self.stats, axis=1)
            pred['prof_mean_stdev'] = pred['prof'].clone()
            pred['prof_background_min_max'] = min_max(pred['prof_background_phys'].clone(), self.stats, axis=1)
            pred['prof_background_mean_stdev'] = batch['target']['prof_background'].clone()
            pred['prof_target_min_max'] = min_max(pred['prof_target_phys'].clone(), self.stats, axis=1)
            pred['prof_target_mean_stdev'] = batch['target']['prof'].clone()

        # Register standardized for plotting
        pred['prof'] = pred['prof_phys'].clone()
        batch['target']['prof'] = pred['prof_target_phys'].clone()
        batch['target']['prof_background'] = pred['prof_background_phys'].clone()

        # Compute loss function
        loss, pred['hofx'] = self.loss_func(pred, batch['target'], batch['input'])
        detached_loss = {k: v.detach().item() if v.ndim == 0 else v.detach() for k, v in loss.items()}

        # Logging
        self._logging(stage, detached_loss, batch['input'], batch['target'], {k: v.detach() for k, v in pred.items()})

        return loss['total']


class PINNverseOperatorLanczos(PINNverseOperator):
    def __init__(self, lanczos_buffers: DictConfig, optimizer: DictConfig = None, loss_func: DictConfig = None,
                 lr_scheduler: DictConfig = None, architecture: DictConfig = None,
                 parameters: DictConfig = None):
        """ Initialize model.

        Parameters
        ----------
        lanczos_buffers: dict. Lanczos buffers for profile reconstruction.
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        architecture: DictConfig. Configuration for the model architecture.
        parameters: DictConfig. Configuration for the model parameters.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func,
                         architecture=architecture, parameters=parameters)

        # Add metrics
        self.metrics['prof_lanczos'], self.metrics['prof_target_lanczos'], self.metrics[
            'prof_background_lanczos'] = {}, {}, {}

        # Lanczos Buffers
        lanczos_buffers = instantiate(lanczos_buffers)
        self.register_buffer('x_b', torch.tensor(lanczos_buffers['x_b']))
        self.register_buffer('L', torch.tensor(lanczos_buffers['L']))

    def forward(self, x: dict) -> dict:
        """ Forward pass through the model.

            Parameters
            ----------
            x: dict. Input coordinates.

            Returns
            -------
            Predicted profiles: dict.
        """

        # 1. Prepare batch metadata
        keys_coords = ['lat', 'lon', 'scans']
        inputs = torch.cat([x[k].view(-1, 1) for k in keys_coords if k in x], dim=-1)

        # 2. Lanczos Coefficient Prediction (whitened coefficients)
        u = self.model(inputs)
        # 3. Profile Reconstruction
        prof = torch.matmul(u, self.L.T) + self.x_b

        # 4. Final Reshape
        prof = prof.view(-1, self.n_prof, self.n_levels)
        return {
            'prof': prof,
            'prof_lanczos': u
        }

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
        with torch.set_grad_enabled(True):
            coords = {'lat': batch['input']['lat'].requires_grad_(True),
                      'lon': batch['input']['lon'].requires_grad_(True)}

            # Compute profiles
            pred = self.forward(batch['input'])
            pred['prof_background'] = batch['target']['prof_background'].clone()
            pred['prof_background_lanczos'] = batch['target']['prof_background_lanczos'].clone()
            pred['prof_target'] = batch['target']['prof'].clone()
            pred['prof_target_lanczos'] = batch['target']['prof_lanczos'].clone()

            # Compute loss function
            loss, pred['hofx'] = self.loss_func(pred, batch['target'], coords)

        # Logging
        self._logging(stage, loss, batch['input'], batch['target'], pred)

        return loss['total']

    def _logging(self, stage: str, loss: dict, input: dict, target: dict, pred: dict) -> None:

        # Class inheritance
        super()._logging(stage, loss, input, target, pred)

        # If testing, return predictions in addition to loss
        if stage == 'valid':
            if 'prof_lanczos' in pred and 'prof_lanczos' in target:
                self._logging_prof(pred['prof_lanczos'], target['prof_lanczos'],
                                   background=target.get('prof_background_lanczos', None), key='_lanczos')


class PINNverseOperatorPCA(PINNverseOperator):
    def __init__(self, pca_buffers: DictConfig, optimizer: DictConfig = None, loss_func: DictConfig = None,
                 lr_scheduler: DictConfig = None, architecture: DictConfig = None,
                 parameters: DictConfig = None):
        """ Initialize model.

        Parameters
        ----------
        pca_buffers: dict. PCA buffers for profile reconstruction.
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        architecture: DictConfig. Configuration for the model architecture.
        parameters: DictConfig. Configuration for the model parameters.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler, loss_func=loss_func,
                         architecture=architecture, parameters=parameters)

        # PCA Buffers
        pca_buffers = instantiate(pca_buffers)
        self.register_buffer('basis', torch.tensor(pca_buffers['basis']))  # (270, 1143)
        self.register_buffer('mu', torch.tensor(pca_buffers['mu']))  # (1143,)
        self.register_buffer('std', torch.tensor(pca_buffers['std']))  # (1143,)
        self.register_buffer('scales', torch.tensor(pca_buffers['scales']))  # (270,)

    def _logging(self, stage: str, loss: dict, input: dict, target: dict, pred: dict) -> None:

        # Logger flag
        logger_flag = stage != 'test'

        # Class inheritance
        super()._logging(stage, loss, input, target, pred)

        # Additions for PCA model
        if 'obs_phys' in loss:
            self.log(f"{stage}_loss_obs_phys", loss['obs_phys'].mean(), on_epoch=True, prog_bar=False, logger=logger_flag)
            if loss['obs_phys'].ndim == 2:
                for i in range(pred['hofx'].shape[1] // 2):
                    self.log(f"{stage}_loss_obs_phys_{i}", loss['obs_phys'][:, i].mean(), on_epoch=True, prog_bar=False,
                             logger=logger_flag)

    def forward(self, x: dict) -> dict:
        """ Forward pass through the model.

            Parameters
            ----------
            x: dict. Input coordinates.

            Returns
            -------
            Predicted profiles: dict.
        """

        # 1. Prepare batch metadata
        keys_coords = ['lat', 'lon', 'scans']
        inputs = torch.cat([x[k].view(-1, 1) for k in keys_coords if k in x], dim=-1)

        # 2. PCA Coefficient Prediction (whitened coefficients)
        w_white = self.model(inputs)
        # 3. Profile Reconstruction
        w_standardized = w_white * self.scales
        prof_standardized = torch.matmul(w_standardized, self.basis)
        prof_phys = self.mu + (prof_standardized * self.std)

        # 4. Final Reshape
        prof_phys = prof_phys.view(-1, self.n_prof, self.n_levels)
        return {
            'prof': prof_phys,
            'prof_white': w_white
        }

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
        with torch.set_grad_enabled(True):
            coords = {'lat': batch['input']['lat'].requires_grad_(True),
                      'lon': batch['input']['lon'].requires_grad_(True)}

            # Compute profiles
            pred = self.forward(batch['input'])

            # Compute loss function
            loss, pred['hofx'] = self.loss_func(pred, batch['target'], coords)

        # Logging
        self._logging(stage, loss, batch['input'], batch['target'], pred)

        return loss['total']
