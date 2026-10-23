import torch
import torch.nn as nn


def _sobolev_penalty(
    pred: torch.Tensor,
    inputs: list[torch.Tensor],
    order: int,
) -> torch.Tensor:
    """
    Compute Sobolev penalty ||∂f/∂x||² (and optionally ||∂²f/∂x²||²) for a
    list of input tensors.

    Uses a Jacobian-vector product trick: grad_outputs=ones collapses all output
    dimensions, giving ∑_j ∂y_j/∂x_i for each input element x_i.  Squaring and
    averaging this over the batch gives a scalar smoothness penalty.

    Parameters
    ----------
    pred : torch.Tensor
        Network output of any shape.  Will be flattened to (Batch, -1) before
        differentiation.
    inputs : list[torch.Tensor]
        Input tensors to differentiate against.  Must have requires_grad=True.
    order : int
        Differentiation order.  1 → first-order only; 2 → first + second order.

    Returns
    -------
    torch.Tensor
        Scalar penalty.
    """

    y = pred.reshape(pred.shape[0], -1)          # (Batch, N)
    grad_outputs = torch.ones_like(y)

    # First-order gradients
    grads1 = torch.autograd.grad(
        outputs=y,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=(order >= 2),
        retain_graph=True,
        allow_unused=True,
    )

    penalty = torch.tensor(0.0, device=pred.device)
    valid_grads = []

    for g in grads1:
        if g is not None:
            penalty = penalty + g.pow(2).mean()
            valid_grads.append(g)

    if order >= 2 and valid_grads:
        # Second-order: differentiate first-order gradients again
        for g1 in valid_grads:
            grads2 = torch.autograd.grad(
                outputs=g1,
                inputs=inputs,
                grad_outputs=torch.ones_like(g1),
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )
            for g2 in grads2:
                if g2 is not None:
                    penalty = penalty + g2.pow(2).mean()

    return penalty


class SobolevHorizontalRegularization(nn.Module):
    """
    Sobolev regularization w.r.t. horizontal spatial coordinates (lat, lon).

    Penalizes ||∂f/∂lat||² + ||∂f/∂lon||², encouraging the network to produce
    spatially smooth outputs.  Optionally includes second-order terms
    (||∂²f/∂lat²||² + ||∂²f/∂lon²||²) for stronger smoothness constraints.

    The input coordinates must be present in the coords dict with
    requires_grad=True for gradients to flow.
    """

    def __init__(
        self,
        input_keys: list[str] | None = None,
        order: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        input_keys : list[str], optional
            Keys in the coords dict to differentiate against.
            Defaults to ['lat', 'lon'].
        order : int, default=1
            Differentiation order.  1 → first-order only; 2 → first + second.
        """
        super().__init__()
        self.input_keys = ['lat', 'lon'] if input_keys is None else input_keys
        self.order = order

    def __call__(self, pred: torch.Tensor, coords: dict) -> torch.Tensor:
        """
        Compute the horizontal Sobolev penalty.

        Parameters
        ----------
        pred : torch.Tensor
            Network output, e.g. prof_inverse of shape (Batch, n_prof, n_levels).
        coords : dict
            Input coordinate tensors.  Keys matching input_keys must have
            requires_grad=True.

        Returns
        -------
        torch.Tensor
            Scalar penalty.
        """
        inputs = [coords[k] for k in self.input_keys if k in coords and coords[k].requires_grad]
        if not inputs:
            return torch.tensor(0.0, device=pred.device)
        return _sobolev_penalty(pred, inputs, self.order)


class SobolevVerticalRegularization(nn.Module):
    """
    Sobolev regularization w.r.t. the vertical coordinate (pressure_log).

    Penalizes ||∂f/∂p_log||², encouraging smooth vertical profiles.  Optionally
    includes the second-order term ||∂²f/∂p_log²||² to penalize curvature as
    well as slope — useful for temperature but potentially too tight for humidity
    or ozone, which have sharp vertical features.

    Since pressure_log has shape (Batch, n_levels) and the network processes all
    levels in a single flattened forward pass, the gradient at each level is
    computed jointly and measures how sensitively the total output changes when
    the pressure coordinate at that level is perturbed.

    The pressure_log tensor must have requires_grad=True for gradients to flow.
    """

    def __init__(
        self,
        input_key: str = 'pressure_log',
        order: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        input_key : str, default='pressure_log'
            Key in the coords dict for the vertical coordinate.
        order : int, default=1
            Differentiation order.  1 → first-order only; 2 → first + second.
        """
        super().__init__()
        self.input_key = input_key
        self.order = order

    def __call__(self, pred: torch.Tensor, coords: dict) -> torch.Tensor:
        """
        Compute the vertical Sobolev penalty.

        Parameters
        ----------
        pred : torch.Tensor
            Network output, e.g. prof_inverse of shape (Batch, n_prof, n_levels).
        coords : dict
            Input coordinate tensors.  coords[input_key] must have
            requires_grad=True and shape (Batch, n_levels).

        Returns
        -------
        torch.Tensor
            Scalar penalty.
        """
        if self.input_key not in coords:
            return torch.tensor(0.0, device=pred.device)

        pressure = coords[self.input_key]
        if not pressure.requires_grad:
            return torch.tensor(0.0, device=pred.device)

        return _sobolev_penalty(pred, [pressure], self.order)
