import torch
import torch.nn as nn
from typing import Union
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from code.architecture.mlp import MLPBlocks


class ODEFunc(nn.Module):
    """
    Internal derivative function f(x, t) for Neural ODE integration.
    Basic unconditional ODE dynamics.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        n_layers: int = 2,
        activation: Union[DictConfig, nn.Module] = None,
    ):
        """
        Initialize ODEFunc.

        Parameters
        ----------
        dim : int. Dimension of the latent hidden state.
        hidden_dim : int. Hidden layer dimension (defaults to dim).
        n_layers : int. Number of layers in the ODE function.
        activation : DictConfig or nn.Module. Activation function (defaults to Tanh).
        """
        super().__init__()

        activation = activation if activation is not None else nn.Tanh()
        hidden_dim = hidden_dim if hidden_dim is not None else dim

        # Use MLPBlocks for flexible architecture
        self.net = MLPBlocks(
            in_features=dim,
            out_features=dim,
            hidden_features=hidden_dim,
            n_blocks=n_layers,
            activation=activation,
            final_activation=nn.Identity(),  # No activation on output
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the derivative at time t.

        Parameters
        ----------
        t : torch.Tensor. Current time (can be unused).
        x : torch.Tensor. Current state.

        Returns
        -------
        torch.Tensor. Time derivative dx/dt.
        """
        return self.net(x)


class ConditionalODEFunc(nn.Module):
    """
    Conditional ODE function that treats additional variables as constant
    context during integration (e.g., surface/meta variables).
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = None,
        n_layers: int = 2,
        activation: Union[DictConfig, nn.Module] = None,
    ):
        """
        Initialize ConditionalODEFunc.

        Parameters
        ----------
        latent_dim : int. Dimension of the evolving state.
        context_dim : int. Dimension of the static context.
        hidden_dim : int. Hidden layer dimension.
        n_layers : int. Number of layers.
        activation : DictConfig or nn.Module. Activation function.
        """
        super().__init__()

        activation = activation if activation is not None else nn.Tanh()
        hidden_dim = hidden_dim if hidden_dim is not None else latent_dim

        # Network sees both evolving state and static context
        self.net = MLPBlocks(
            in_features=latent_dim + context_dim,
            out_features=latent_dim,
            hidden_features=hidden_dim,
            n_blocks=n_layers,
            activation=activation,
            final_activation=nn.Identity(),
        )

    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate derivative with conditional context.

        Parameters
        ----------
        t : torch.Tensor. Current time.
        h : torch.Tensor. Current hidden state.
        context : torch.Tensor. Static context variables.

        Returns
        -------
        torch.Tensor. Time derivative dh/dt.
        """
        combined = torch.cat([h, context], dim=1)
        return self.net(combined)


class PressureConditionalODEFunc(nn.Module):
    """
    ODE function conditioned on pressure level for atmospheric modeling.
    Maps atmospheric state at a specific pressure level to its rate of change.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        profile_dim: int = 3,
        hidden_dim: int = None,
        n_layers: int = 2,
        activation: Union[DictConfig, nn.Module] = None,
    ):
        """
        Initialize PressureConditionalODEFunc.

        Parameters
        ----------
        latent_dim : int. Dimension of the latent state.
        context_dim : int. Dimension of static context (surf + meta).
        profile_dim : int. Number of profile variables at current level.
        hidden_dim : int. Hidden layer dimension.
        n_layers : int. Number of layers.
        activation : DictConfig or nn.Module. Activation function.
        """
        super().__init__()

        activation = activation if activation is not None else nn.Tanh()
        hidden_dim = hidden_dim if hidden_dim is not None else latent_dim

        # Input: latent_h + context_vars + pressure + profile_at_level
        input_dim = latent_dim + context_dim + 1 + profile_dim

        self.net = MLPBlocks(
            in_features=input_dim,
            out_features=latent_dim,
            hidden_features=hidden_dim,
            n_blocks=n_layers,
            activation=activation,
            final_activation=nn.Identity(),
        )

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        pressure: torch.Tensor,
        profile: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity of atmospheric state integration.

        Parameters
        ----------
        h : torch.Tensor. (batch, latent_dim) Current hidden state.
        context : torch.Tensor. (batch, context_dim) Static environmental variables.
        pressure : torch.Tensor. (batch, 1) Current vertical pressure level.
        profile : torch.Tensor. (batch, profile_dim) Profile variables at current level.

        Returns
        -------
        torch.Tensor. (batch, latent_dim) Velocity dh/dp.
        """
        combined = torch.cat([h, context, pressure, profile], dim=1)
        return self.net(combined)


class NeuralODEIntegrator(nn.Module):
    """
    Neural ODE integrator for atmospheric radiative transfer.
    Integrates through pressure levels to accumulate radiance.
    """

    def __init__(
        self,
        ode_func: nn.Module,
        integration_method: str = 'euler',
    ):
        """
        Initialize NeuralODEIntegrator.

        Parameters
        ----------
        ode_func : nn.Module. The ODE function to integrate.
        integration_method : str. Integration method ('euler', 'rk4', etc.).
        """
        super().__init__()
        self.ode_func = ode_func
        self.integration_method = integration_method

    def forward(
        self,
        h0: torch.Tensor,
        pressure_grid: torch.Tensor,
        delta_p: torch.Tensor,
        context: torch.Tensor = None,
        profile: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Integrate ODE through pressure levels.

        Parameters
        ----------
        h0 : torch.Tensor. Initial state at surface.
        pressure_grid : torch.Tensor. Pressure level values.
        delta_p : torch.Tensor. Pressure thickness between levels.
        context : torch.Tensor. Optional static context.
        profile : torch.Tensor. Optional profile data.

        Returns
        -------
        torch.Tensor. Final integrated state.
        """
        h = h0
        nlevels = len(pressure_grid)

        # Integrate from surface (high pressure) to TOA (low pressure)
        for i in range(nlevels - 1, 0, -1):
            # Get current pressure
            current_p = pressure_grid[i].expand(h.shape[0], 1)

            # Compute derivative
            if profile is not None:
                # Pressure-conditional ODE
                current_prof = profile[:, :, i]
                v = self.ode_func(h, context, current_p, current_prof)
            elif context is not None:
                # Context-conditional ODE
                v = self.ode_func(None, h, context)
            else:
                # Basic ODE
                v = self.ode_func(None, h)

            # Euler integration weighted by pressure thickness
            if self.integration_method == 'euler':
                h = h + delta_p[i - 1] * v
            else:
                raise NotImplementedError(f"Integration method {self.integration_method} not implemented")

        return h


class CRTMNeuralODE(nn.Module):
    """
    CRTM architecture using Neural ODE for atmospheric integration.
    Integrates radiance through pressure levels using differential equations.
    """

    def __init__(
        self,
        nprofvars: int,
        nsurfvars: int,
        nmetavars: int,
        nlevels: int,
        pressure_levels: torch.Tensor,
        latent_dim: int = 256,
        ode_hidden_dim: int = None,
        ode_n_layers: int = 2,
        ode_activation: Union[DictConfig, nn.Module] = None,
        bt_norm_max: float = 355.0,
        bt_norm_min: float = 180.0,
        std_output_activation_offset: float = 0.001,
        std_scale_trainable: bool = True,
    ):
        """
        Initialize CRTM Neural ODE Architecture.

        Parameters
        ----------
        nprofvars : int. Number of profile variables.
        nsurfvars : int. Number of surface variables.
        nmetavars : int. Number of meta variables.
        nlevels : int. Number of vertical pressure levels.
        pressure_levels : torch.Tensor. Pressure grid values.
        latent_dim : int. Dimension of latent atmospheric state.
        ode_hidden_dim : int. Hidden dimension for ODE function.
        ode_n_layers : int. Number of layers in ODE function.
        ode_activation : DictConfig or nn.Module. Activation for ODE function.
        bt_norm_max : float. Maximum brightness temperature.
        bt_norm_min : float. Minimum brightness temperature.
        std_output_activation_offset : float. Std offset.
        std_scale_trainable : bool. Trainable std scale.
        """
        super().__init__()

        # Import Scale locally to avoid circular imports
        from code.architecture.activation import Scale

        # Dimensions
        self.nprofvars = nprofvars
        self.nsurfvars = nsurfvars
        self.nmetavars = nmetavars
        self.nlevels = nlevels
        self.latent_dim = latent_dim
        self.context_dim = nsurfvars + nmetavars

        # Normalization parameters
        self.max_T = bt_norm_max
        self.min_T = bt_norm_min
        self.std_offset = std_output_activation_offset

        # Normalize and register pressure levels
        pressure_levels = (pressure_levels - pressure_levels.min()) / (
            pressure_levels.max() - pressure_levels.min()
        )
        pressure_grid = torch.tensor(pressure_levels, dtype=torch.float32)
        delta_p = torch.abs(pressure_grid[:-1] - pressure_grid[1:])
        self.register_buffer('pressure_grid', pressure_grid)
        self.register_buffer('delta_p', delta_p)

        # Model components
        # Initial state projection from context
        self.h0_projection = nn.Linear(self.context_dim, latent_dim)

        # ODE function (atmospheric dynamics)
        self.ode_func = PressureConditionalODEFunc(
            latent_dim=latent_dim,
            context_dim=self.context_dim,
            profile_dim=nprofvars,
            hidden_dim=ode_hidden_dim,
            n_layers=ode_n_layers,
            activation=ode_activation if ode_activation is not None else nn.Tanh(),
        )

        # Output heads
        self.out_T = nn.Linear(latent_dim, 10)
        self.out_std = nn.Linear(latent_dim, 10)

        # Activations
        self.bt_output_activation = nn.Sigmoid()
        self.std_output_activation = nn.Softplus()

        # Optional learnable std scaling
        self.std_scale = Scale() if std_scale_trainable else None

    def forward(self, input: dict) -> torch.Tensor:
        """
        Forward pass using pressure-thickness weighted integration.

        Parameters
        ----------
        input : dict. Dictionary containing:
            - 'prof': (batch, nprofvars, nlevels)
            - 'surf': (batch, nsurfvars)
            - 'meta': (batch, nmetavars)

        Returns
        -------
        torch.Tensor. (batch, 20) concatenated mean BT and std.
        """
        # Prepare static context (surface + meta)
        static_context = torch.cat([input['surf'], input['meta']], dim=1)

        # Initial condition (h at surface level)
        h = torch.tanh(self.h0_projection(static_context))

        # Integrate through pressure levels (surface to TOA)
        for i in range(self.nlevels - 1, 0, -1):
            # Current pressure and profile
            current_p = self.pressure_grid[i].expand(h.shape[0], 1)
            current_prof = input['prof'][:, :, i]

            # Compute velocity (derivative w.r.t. pressure)
            v = self.ode_func(h, static_context, current_p, current_prof)

            # Euler integration weighted by pressure thickness
            h = h + self.delta_p[i - 1] * v

        # Final output layers
        # Mean brightness temperature
        out = self.out_T(h)
        out = self.bt_output_activation(out)
        out = out * (self.max_T - self.min_T) + self.min_T

        # Uncertainty (standard deviation)
        out_std = self.out_std(h)
        out_std = self.std_output_activation(out_std)
        if self.std_scale is not None:
            out_std = self.std_scale(out_std)
        out_std = out_std + self.std_offset

        return torch.cat([out, out_std], dim=1)

