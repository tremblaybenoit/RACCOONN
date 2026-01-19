import torch
import torch.nn as nn
from typing import Union, Callable
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningModule
from data.statistics import statistics, accumulate_statistics
from forward.model.model import BaseModel
from forward.model.activation import Sine
from inverse.model.encoding import IdentityPositionalEncoding
from utilities.instantiators import instantiate
from data.transformations import identity


class SirenInit:
    """
    Callable SIREN initializer with configuration moved to __init__.
    """
    def __init__(self, is_first_layer: bool = False, activation=None, is_head: bool = False, w0: float = 30.0):
        """
        Initialize SIREN initializer.

        Parameters
        ----------
        is_first_layer: bool. Flag indicating if the module is the first layer.
        activation: Callable. Activation function.
        is_head: bool. Flag indicating if the module is a head layer.
        w0: float. Frequency scaling factor.
        """

        # Parameters
        self.is_first_layer = is_first_layer
        self.is_head = is_head
        self.w0 = getattr(activation, 'w0', w0) if activation is not None else w0

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
                    bound = 1.0 / dim_in
                    nn.init.uniform_(module.weight, -bound, bound)
                    module.weight *= self.w0

                # Head/output layer initialization
                elif self.is_head:
                    # Hydra Head initialization: Small variance to maintain stability
                    # with CRTM and prevent early training divergence.
                    # bound = torch.sqrt(torch.tensor(1. / dim_in))
                    # nn.init.uniform_(module.weight, -bound, bound)
                    nn.init.xavier_uniform_(module.weight)
                    module.weight *= 1.0

                # Subsequent layer initialization
                else:
                    bound = (torch.sqrt(torch.tensor(6.0 / dim_in)) / self.w0).item()
                    nn.init.uniform_(module.weight, -bound, bound)

            # Bias initialization
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


class ResidualMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable layers.
    """
    def __init__(self, input_layer: DictConfig, hidden_layer: DictConfig, output_layer: DictConfig,
                 positional_encoding: DictConfig = None, n_hidden_layers: int=2,
                 hidden_skip: bool=False, output_skip: bool=False):
        """
        Initialize MLP.

        Parameters
        ----------
        input_layer: DictConfig. Configuration for the input layer.
        hidden_layer: DictConfig. Configuration for the hidden layers.
        output_layer: DictConfig. Configuration for the output layer.
        n_hidden_layers: int. Number of hidden layers.
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
        self.input_layer = instantiate(input_layer)
        # Hidden layers
        hidden_layers = []
        # Build hidden layers with optional skip connections
        for _ in range(n_hidden_layers):
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
        self.output_layer = instantiate(output_layer)

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
        x_input = self.input_layer(x_encoded)
        x_hidden = self.hidden_layers(x_input)
        x_output = self.output_layer(x_hidden)
        return x_output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable layers.
    """
    def __init__(self, input_layer: DictConfig, input_layernorm: DictConfig, input_activation: DictConfig,
                 input_dropout: float, input_init: DictConfig, hidden_layer: DictConfig, hidden_layernorm: DictConfig,
                 hidden_activation: DictConfig, hidden_dropout: float, hidden_init: DictConfig,
                 output_layer: DictConfig, output_layernorm: DictConfig, output_activation: DictConfig,
                 output_init: DictConfig, n_hidden_layers: int=2,
                 hidden_layer_skip: bool=False, output_layer_skip: bool=False):
        """
        Initialize MLP.

        Parameters
        ----------
        input_layer: DictConfig. Configuration for the input layer.
        input_layernorm: DictConfig. Configuration for the input layer normalization.
        input_activation: DictConfig. Configuration for the input layer activation.
        input_dropout: float. Dropout rate for the input layer.
        input_init: DictConfig. Configuration for the input layer initialization.
        hidden_layer: DictConfig. Configuration for the hidden layers.
        hidden_layernorm: DictConfig. Configuration for the hidden layer normalization.
        hidden_activation: DictConfig. Configuration for the hidden layer activation.
        hidden_dropout: float. Dropout rate for the hidden layers.
        hidden_init: DictConfig. Configuration for the hidden layer initialization.
        output_layer: DictConfig. Configuration for the output layer.
        output_layernorm: DictConfig. Configuration for the output layer normalization.
        output_activation: DictConfig. Configuration for the output layer activation.
        output_init: DictConfig. Configuration for the output layer initialization.
        n_hidden_layers: int. Number of hidden layers.
        hidden_layer_skip: bool. Flag to enable skip connections in hidden layers.
        output_layer_skip: bool. Flag to enable skip connections in output layer.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Input layer
        input_layers = LinearBlock(
            in_features=input_layer.in_features,
            out_features=input_layer.out_features,
            activation=instantiate(input_activation),
            dropout_rate=input_dropout,
            layernorm=instantiate(input_layernorm),
            init_func=instantiate(input_init)
        )
        # Hidden layers
        hidden_layers = []
        for _ in range(n_hidden_layers):
            hidden_layer_instance = LinearBlock(
                in_features=hidden_layer.in_features,
                out_features=hidden_layer.out_features,
                activation=instantiate(hidden_activation),
                dropout_rate=hidden_dropout,
                layernorm=instantiate(hidden_layernorm),
                init_func=instantiate(hidden_init)
            )
            if hidden_layer_skip:
                hidden_layer_instance = Residual(hidden_layer_instance)
            hidden_layers.append(hidden_layer_instance)
        # Output layer
        if output_layer_skip:
            output_layers = Concatenate(
                LinearBlock(
                    in_features=output_layer.in_features,
                    out_features=output_layer.out_features,
                    activation=instantiate(output_activation),
                    dropout_rate=0.0,
                    layernorm=instantiate(output_layernorm),
                    init_func=instantiate(output_init)
                )
            )
        else:
            output_layers = LinearBlock(
                in_features=output_layer.in_features,
                out_features=output_layer.out_features,
                activation=instantiate(output_activation),
                dropout_rate=0.0,
                layernorm=instantiate(output_layernorm),
                init_func=instantiate(output_init)
            )
        # Model architecture
        self.model = nn.Sequential(
            input_layers,
            *hidden_layers,
            output_layers
        )

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
        return self.model(x)


def _siren_init(module, is_first_layer: bool, activation, is_head: bool = False, w0: float = 30.0):
    """
        SIREN weights initialization function.

        Parameters
        ----------
        module: nn.Module. The module to initialize.
        is_first_layer: bool. Flag indicating if the module is the first layer.
        activation: Callable. Activation function.
        is_head: bool. Flag indicating if the module is a head layer.
        w0: float. Frequency scaling factor.

        Returns
        -------
        None.
    """

    # Adjust w0 based on activation function if available
    w0 = getattr(activation, 'w0', w0) if activation is not None else w0

    # Weight initialization
    with torch.no_grad():
        if hasattr(module, 'weight'):
            # Input dimension
            dim_in = module.weight.size(1)

            # First layer initialization
            if is_first_layer:
                bound = 1.0 / dim_in
                nn.init.uniform_(module.weight, -bound, bound)
                module.weight *= w0

            # Head/output layer initialization
            elif is_head:
                # Hydra Head initialization: Small variance to maintain stability
                # with CRTM and prevent early training divergence.
                # bound = torch.sqrt(torch.tensor(1. / dim_in))
                # nn.init.uniform_(module.weight, -bound, bound)
                nn.init.xavier_uniform_(module.weight)
                module.weight *= 1.0

            # Subsequent layer initialization
            else:
                bound = (torch.sqrt(torch.tensor(6.0 / dim_in)) / w0).item()
                nn.init.uniform_(module.weight, -bound, bound)

        # Bias initialization
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


class SirenResidualMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with SIREN Residual Blocks.
    """
    def __init__(self, input_layer: DictConfig, hidden_layer: DictConfig, output_layer: DictConfig,
                 n_hidden_layers: int=2, output_layer_skip: bool=False):
        """
        Initialize MLP with SIREN Residual Blocks.

        Parameters
        ----------
        input_layer: DictConfig. Configuration for the input layer.
        hidden_layer: DictConfig. Configuration for the hidden layers (residual blocks).
        output_layer: DictConfig. Configuration for the output layer.
        n_hidden_layers: int. Number of hidden layers (residual blocks).

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Input layer
        input_layer = instantiate(input_layer)
        # Hidden layers with SIREN Residual Blocks
        hidden_layers = []
        for _ in range(n_hidden_layers):
            hidden_layer_instance = instantiate(hidden_layer)
            hidden_layers.append(hidden_layer_instance)
        # Output layer
        if output_layer_skip:
            output_layer = Concatenate(instantiate(output_layer))
        else:
            output_layer = instantiate(output_layer)

        # Model architecture
        self.model = nn.Sequential(
            input_layer,
            *hidden_layers,
            output_layer
        )



class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: Callable,
                 dropout_rate: float, layernorm: Callable=None, init_func: Callable=None):
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


class SirenResidualBlock(nn.Module):
    def __init__(self, n_neurons: int, activation: Callable = None, dropout_rate: float = 0,
                 init_func: Callable = None, normalization: Callable=None):
        """ Initialize a SIREN Residual Block.

        Parameters
        ----------
        n_neurons: int. Number of neurons in the block.
        activation: Callable. Activation function.
        dropout_rate: float. Dropout rate.
        init_func: Callable. SIREN initialization function.
        normalization: Callable. Normalization layer.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Internal layers for the residual path (f(x))
        self.linear1 = nn.Linear(n_neurons, n_neurons)
        self.layernorm = normalization if normalization is not None else nn.Identity()
        self.activation = activation if activation is not None else Sine()
        self.linear2 = nn.Linear(n_neurons, n_neurons)
        self.dropout = nn.Dropout(dropout_rate)

        # Apply SIREN Initialization to internal layers
        if init_func is not None:
            init_func = SirenInit(is_first_layer=False, activation=self.activation)
        init_func(self.linear1)
        init_func(self.linear2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the SIREN Residual Block.

        Parameters
        ----------
        x: tensor. Input tensor.

        Returns
        -------
        out: tensor. Output tensor after applying the residual block.
        """

        # Store input for skip connection
        residual = x

        # Block 1 (Path where f(x) is computed)
        out = self.linear1(x)
        out = self.layernorm(out)
        out = self.activation(out)

        # Block 2
        out = self.linear2(out)

        # Skip connection: out = f(x) + x
        out = out + residual
        out = self.dropout(out)
        return out


class SirenResidualNetwork(nn.Module):
    def __init__(self, input_layer: DictConfig, hidden_layer: DictConfig, output_layer: DictConfig,
                 n_hidden_layers: int=2, output_layer_skip: bool=False):
        """ Initialize SIREN Residual Network.

        Parameters
        ----------
        input_layer: DictConfig. Configuration for the input layer.
        hidden_layer: DictConfig. Configuration for the hidden layers (residual blocks).
        output_layer: DictConfig. Configuration for the output layer.
        n_hidden_layers: int. Number of hidden layers (residual blocks).
        output_layer_skip: bool. Flag to enable skip connections in output layer.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Input layer
        self.input_layer = LinearBlock(
            in_features=input_layer.in_features,
            out_features=input_layer.out_features,
            activation=input_layer.activation,
            dropout_rate=input_layer.dropout_rate,
            layernorm=input_layer.layernorm,
            init_func=SirenInit(is_first_layer=True, activation=input_layer.activation)
        )

        # Hidden layers with SIREN Residual Blocks
        hidden_layers = []
        for _ in range(n_hidden_layers):
            hidden_layer_instance = SirenResidualBlock(
                n_neurons=hidden_layer.n_neurons,
                activation=hidden_layer.activation,
                dropout_rate=hidden_layer.dropout_rate,
                init_func=SirenInit(is_first_layer=False, activation=hidden_layer.activation),
                normalization=hidden_layer.normalization
            )
            hidden_layers.append(hidden_layer_instance)
        # Output layer
        if output_layer_skip:
            self.hidden_layers = Concatenate(nn.Sequential(*hidden_layers))
            output_layer.in_features = hidden_layer.out_features + input_layer.out_features
        else:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = LinearBlock(
            in_features=output_layer.in_features,
            out_features=output_layer.out_features,
            activation=output_layer.activation,
            dropout_rate=0.0,
            layernorm=output_layer.layernorm,
            init_func=SirenInit(is_first_layer=False, activation=output_layer.activation, is_head=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the SIREN Residual Network.
        Parameters
        ----------
        x: tensor. Input tensor.
        Returns
        -------
        out: tensor. Output tensor after applying the network.
        # Pass
        """
        x_input = self.input_layer(x)
        x_hidden = self.hidden_layers(x_input)
        x_output = self.output_layer(x_hidden)
        return x_output


class PINNverseOperator0(BaseModel):
    """Class for the Physics-Informed Neural Network (PINN) inverse model."""
    def __init__(self, optimizer: DictConfig = None, loss_func: DictConfig = None, lr_scheduler: DictConfig = None,
                 positional_encoding: DictConfig = None, input_layer: DictConfig = None, hidden_layer: DictConfig = None,
                 output_layer: DictConfig = None, output_transform: ListConfig = None, parameters: DictConfig = None):
        """ Initialize model.

        Parameters
        ----------
        optimizer: Callable. Optimizer for the model.
        loss_func: Callable. Loss function for the model.
        lr_scheduler: Callable. Learning rate scheduler for the model.
        positional_encoding: Callable. Function for the positional encoding.
        input_layer: Callable. Configuration for the input layer.
        hidden_layer: Callable. Configuration for the hidden layers.
        output_layer: Callable. Configuration for the output layer.
        output_transform: ListConfig. List of functions to apply to the model output.
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

        # Output transformations
        self.output_transform = [instantiate(t) for t in output_transform] if output_transform is not None else [identity]

        # Parameters
        self.n_prof = parameters.data.n_prof if parameters is not None and hasattr(parameters.data, 'n_prof') \
            else 1
        self.n_levels = parameters.data.n_levels if parameters is not None and hasattr(parameters.data, 'n_levels') \
            else 1
        self.prof_vars = parameters.data.prof_vars if parameters is not None and hasattr(parameters.data, 'prof_vars') \
            else [f'var_{i}' for i in range(self.n_prof)]

        # Model architecture
        self.model = self._build_model(positional_encoding, input_layer, hidden_layer, output_layer, parameters)

    def _build_model(self, positional_encoding: Union[DictConfig, None], input_layer: Union[DictConfig, None],
                     hidden_layer: Union[DictConfig, None], output_layer: Union[DictConfig, None],
                     parameters: Union[DictConfig, None]) -> nn.Module:
        """ Build the neural network model.

            Parameters
            ----------
            positional_encoding: DictConfig. Function for the positional encoding.
            input_layer: DictConfig. Configuration for the input layer.
            hidden_layer: DictConfig. Configuration for the hidden layers.
            output_layer: DictConfig. Configuration for the output layer.
            parameters: DictConfig. Configuration for the model parameters.

            Returns
            -------
            None.
        """

        # Positional encoding
        if positional_encoding is not None:
            self.positional_encoding = instantiate(positional_encoding)
            input_layer.in_features = self.positional_encoding.d_output
        else:
            self.positional_encoding = IdentityPositionalEncoding(d_input=input_layer.in_features)

        # Input layer
        self.input_layer = instantiate(input_layer)
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(parameters.architecture.n_layers):
            hidden_layer_instance = instantiate(hidden_layer)
            self.hidden_layers.append(hidden_layer_instance)
        # Output layer
        self.output_layer = instantiate(output_layer)

        # Model architecture
        model = nn.Sequential(
            self.input_layer,
            *self.hidden_layers,
            self.output_layer
        )
        return model


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

        # Output transformations
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
        normalization = parameters.architecture.normalization if parameters is not None and \
            hasattr(parameters.architecture, 'normalization') else None
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
        d_encoded = self.positional_encoding.d_output

        # Input layer
        self.input_layer = nn.Linear(d_encoded, n_neurons)
        self.input_activation = instantiate(activation_in) if activation_in is not None else Sine()
        self.input_normalization = instantiate(normalization) if normalization is not None else nn.Identity()
        self.input_dropout = nn.Dropout(dropout_rate)
        _siren_init(self.d_in, is_first_layer=True, activation=self.activation_in)

        # Output layer
        self.output_layer = nn.Linear(n_neurons, n_prof*n_levels)
        self.output_activation = instantiate(activation_out) if activation_out is not None else nn.Identity()
        self.output_normalization = nn.Identity()
        _siren_init(self.d_out, is_first_layer=False, activation=None, is_head=True)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers):
            hidden_layer = nn.Linear(n_neurons, n_neurons)
            hidden_normalization = instantiate(normalization) if normalization is not None else nn.Identity()
            hidden_activation = instantiate(activation_in) if activation_in is not None else Sine()
            hidden_dropout = nn.Dropout(dropout_rate)
            _siren_init(hidden_layer, is_first_layer=False, activation=hidden_activation)
            self.hidden_layers.append(nn.Sequential(hidden_layer, hidden_normalization, hidden_activation,
                                                    hidden_dropout))

        # Model architecture
        model = nn.Sequential(
            self.input_layer,
            self.input_normalization,
            self.input_activation,
            self.input_dropout,
            self.hidden_layers,
            self.output_layer,
            self.output_normalization,
            self.output_activation
        )
        return model

    def _retrieve_prof(self, x: dict, n_levels: int) -> torch.Tensor:
        """ Pass forward through neural network architecture.

            Parameters
            ----------
            x: tensor. Inputs: latitude, longitude, surface, and the metadata.
            n_levels: int. Number of pressure levels.

            Returns
            -------
            y: tensor. Outputs: predicted profiles.
        """

        # Concatenate inputs
        inputs = torch.cat([v.view(-1, 1) for v in x.values()], dim=-1)
        # Apply positional encoding
        encoded_inputs = self.positional_encoding(inputs)
        # Pass through the model
        profiles = self.model(encoded_inputs).view(-1, n_levels, self.n_prof).transpose(1, 2)
        # Reshape profiles to match the expected output shape
        return profiles

    def forward(self, x: dict):
        """ Retrieve atmospheric profile over multiple pressure levels.

            Parameters
            ----------
            x: tensor. Input variables.

            Returns
            -------
            prof: tensor. Atmospheric profiles.
        """

        # Get dimensions
        n_levels = x['pressure'].shape[-1]
        x_vector = {}

        # Reshape inputs to match expected dimensions
        for k, v in x.items():
            if k == 'pressure':
                # Pressure is (Batch, n_levels) -> (Batch * n_levels, 1)
                x_vector[k] = v.reshape(-1, 1)
            else:
                # Other variables are (Batch, ) -> (Batch * n_levels, 1)
                v_col = v.view(-1, 1)
                x_vector[k] = v_col.unsqueeze(1).expand(-1, n_levels, 1).reshape(-1, 1)

        return self._retrieve_prof(x_vector, n_levels)

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
        # Log L2 norm of model parameters during training
        elif stage == 'train':
            # Compute L2 norm of the model parameters
            l2_norm = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)
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
        for transform in self.transform_out:
            pred['prof'] = transform(pred['prof'])

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


class PINNverseOperator3(PINNverseOperator):
    """Physics-Informed Neural Network (PINN) inverse model with one neural network per profile type."""

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
        normalization = parameters.architecture.normalization if parameters is not None and \
            hasattr(parameters.architecture, 'normalization') else None
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
        d_encoded = self.positional_encoding.d_output
        # Input layer
        self.d_in = nn.Linear(d_encoded, n_neurons)
        self.activation_in = instantiate(activation_in) if activation_in is not None else Sine()
        self.dropout_in = nn.Dropout(dropout_rate)
        # Output layer
        self.d_out = nn.Linear(n_neurons, n_prof*n_levels)
        self.activation_out = instantiate(activation_out) if activation_out is not None else nn.Identity()

        # Model architecture
        self.normalization = nn.ModuleList([instantiate(normalization) if normalization is not None else nn.Identity()
                                           for _ in range(n_layers)])
        self.activations = nn.ModuleList([instantiate(activation_in) if activation_in is not None else Sine()
                                          for _ in range(n_layers)])

        # SIREN initialization
        self._siren_init(self.d_in, is_first_layer=True, activation_in=self.activation_in)

        residuals_blocks = nn.ModuleList()
        for s in range(n_layers):
            block = SirenResidualBlock(n_neurons, self.activations[s], dropout_rate, self._siren_init,
                                       self.normalization[s])
            residuals_blocks.append(block)

        model = nn.Sequential(
            self.d_in,
            self.activation_in,
            nn.Dropout(dropout_rate),
            *residuals_blocks,
            self.d_out,
            self.activation_out
        )

        return model


class PINNverseOperator4(PINNverseOperator):
    """
    Hydra-Head SIREN with Coordinate Skip Connections.
    Shared backbone learns physical correlations; independent heads specialize in variables.
    """

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
        normalization = parameters.architecture.normalization if parameters is not None and \
                                                                 hasattr(parameters.architecture,
                                                                         'normalization') else None
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
        self.positional_encoding = instantiate(positional_encoding, d_input=d_input) if positional_encoding \
            else IdentityPositionalEncoding(d_input=d_input)
        d_encoded = self.positional_encoding.d_output

        # Input layer
        self.d_in = nn.Linear(d_encoded, n_neurons)
        self.activation_in = instantiate(activation_in) if activation_in else Sine()
        self.dropout_in = nn.Dropout(dropout_rate)
        _siren_init(self.d_in, is_first_layer=True, activation_in=self.activation_in)

        # Output layer
        self.heads = nn.ModuleList([
            nn.Linear(n_neurons + d_encoded, 1) for _ in range(self.n_prof)
        ])
        self.activation_out = instantiate(activation_out) if activation_out else nn.Identity()
        # SIREN initialization (input and output layers)
        for head in self.heads:
            _siren_init(head, is_first_layer=False, activation_in=None, is_head=True)

        # Hidden layers
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            norm = instantiate(normalization) if normalization else nn.Identity()
            actf = instantiate(activation_in) if activation_in else Sine()
            block = SirenResidualBlock(n_neurons, actf, dropout_rate, _siren_init, normalization=norm)
            self.backbone.append(block)

        # Model architecture
        model = nn.ModuleDict({
            'hidden_layers': self.backbone,
            'output_layers': self.heads,
            'input_layers': self.d_in
        })
        return model

    def _retrieve_prof(self, x: dict, n_levels: int) -> torch.Tensor:
        # Explicit order to match encoding expectations: Lat, Lon, Scan, Pressure
        ordered_keys = ['lat', 'lon', 'scans', 'pressure']
        inputs = torch.cat([x[k].view(-1, 1) for k in ordered_keys], dim=-1)

        encoded_coords = self.positional_encoding(inputs)

        # Accessing via ModuleDict (self.model)
        x_latent = self.activation_in(self.model['d_in'](encoded_coords))

        for block in self.model['backbone']:
            x_latent = block(x_latent)

        head_input = torch.cat([x_latent, encoded_coords], dim=-1)

        # Iterate through heads in ModuleDict
        out = torch.cat([head(head_input) for head in self.model['heads']], dim=-1)
        out = self.activation_out(out)

        batch_size = out.shape[0] // n_levels
        return out.view(batch_size, n_levels, self.n_prof).transpose(1, 2)


class PINNverseOperatorPCA(PINNverseOperator):
    def __init__(self, pca_buffers: dict, **kwargs):

        # 1. Register PCA Buffers (from your data.pca output)
        pca_buffers = instantiate(pca_buffers)
        self.n_pca = pca_buffers['basis'].shape[0]

        super().__init__(**kwargs)

        self.register_buffer('basis', torch.tensor(pca_buffers['basis']))  # (270, 1143)
        self.register_buffer('mu', torch.tensor(pca_buffers['mu']))  # (1143,)
        self.register_buffer('std', torch.tensor(pca_buffers['std']))  # (1143,)
        self.register_buffer('scales', torch.tensor(pca_buffers['scales']))  # (270,)

    def _build_model(self, positional_encoding, activation_in, activation_out, parameters) -> nn.Module:
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
        normalization = parameters.architecture.normalization if parameters is not None and \
            hasattr(parameters.architecture, 'normalization') else None
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
        self.d_out = nn.Linear(n_neurons + d_encoded, self.n_pca)
        self.activation_out = instantiate(activation_out) if activation_out else nn.Identity()

        # Model architecture
        self.normalization = nn.ModuleList([instantiate(normalization) if normalization is not None else nn.Identity()
                                            for _ in range(n_layers)])
        self.activations = nn.ModuleList([instantiate(activation_in) if activation_in is not None else Sine()
                                          for _ in range(n_layers)])

        # SIREN initialization
        self._siren_init(self.d_in, is_first_layer=True, activation_in=self.activation_in)
        self._siren_init(self.d_out, is_first_layer=False, activation_in=None, is_head=True)

        # Build Sequential Model
        self.backbone = nn.ModuleList([
            SirenResidualBlock(
                n_neurons, self.activations[s], dropout_rate, self._siren_init,
                normalization=self.normalization[s]
            ) for s in range(n_layers)
        ])

        return nn.ModuleDict({
            'd_in': self.d_in,
            'activation_in': self.activation_in,
            'dropout_in': self.dropout_in,
            'backbone': self.backbone,
            'd_out': self.d_out,
            'activation_out': self.activation_out
        })

    def forward(self, x: dict):
        # 1. Prepare batch metadata
        keys_coords = ['lat', 'lon', 'scans']
        inputs = torch.cat([x[k].view(-1, 1) for k in keys_coords if k in x], dim=-1)
        encoded_coords = self.positional_encoding(inputs)

        # Forward through Backbone
        x_latent = self.model['activation_in'](self.model['d_in'](encoded_coords))
        for block in self.model['backbone']:
            x_latent = block(x_latent)
        # Concatenate Skip Connection
        x_out = torch.cat([x_latent, encoded_coords], dim=-1)

        # PCA Coefficient Prediction (sym-log of whitened coefficients)
        w_white = self.model['activation_out'](self.model['d_out'](x_out))

        # breakpoint()
        w_standardized = w_white * self.scales
        prof_standardized = torch.matmul(w_standardized, self.basis)
        prof_phys = self.mu + (prof_standardized * self.std)

        # 4. Final Reshape
        prof_phys = prof_phys.view(-1, 3, 127)

        return {
            'prof': prof_phys,
            'prof_white': w_white  # Don't mind the naming convention here
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

        # Compute profiles
        pred = self.forward(batch['input'])
        # Apply output transformations
        for t, transform in enumerate(self.transform_out):
            pred['prof'] = transform(pred['prof'])

        # Compute loss function
        loss, pred['hofx'] = self.loss_func(pred, batch['target'])

        # Logging
        self._logging(stage, loss, batch['input'], batch['target'], pred)

        return loss['total']
