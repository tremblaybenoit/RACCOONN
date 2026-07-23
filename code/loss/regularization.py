import torch


class SobolevRegularization(torch.nn.Module):
    """
    Sobolev Regularization loss module to penalize the magnitude of spatial gradients.
    Enforces smoothness by minimizing ||grad(f(x))||^2.
    """
    def __init__(self, input_keys: list | None = None):
        """
        Parameters
        ----------
        input_keys: list. The keys in the batch dict to differentiate against.
        """
        super().__init__()
        self.input_keys = ['lat', 'lon'] if input_keys is None else input_keys

    def __call__(self, pred: torch.Tensor, coords: dict) -> torch.Tensor:
        """
        Compute the Sobolev penalty.

        Parameters
        ----------
        pred: torch.Tensor. Predicted output (e.g., prof_white).
                            Shape: (Batch, 270)
        coords: dict. Dictionary containing input tensors with requires_grad=True.

        Returns
        -------
        torch.Tensor. Mean squared gradient across the batch.
        """

        # Ensure we have a flattened representation for differentiation
        # (Batch, N)
        y = pred.view(pred.shape[0], -1)

        # We want to find the gradient of the model output with respect to inputs.
        # Since we want a single scalar loss to minimize, we compute the gradient
        # of the sum of outputs, which is mathematically equivalent to the
        # sum of the gradients for each output feature.
        grad_outputs = torch.ones_like(y)

        # Select input tensors that exist in coords
        inputs = [coords[k] for k in self.input_keys if k in coords]

        # Calculate Jacobian-vector product
        # creates_graph=True allows the optimizer to backpropagate through this gradient
        grads = torch.autograd.grad(
            outputs=y,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )

        total_grad_loss = torch.tensor(0.0, device=pred.device)
        for g in grads:
            if g is not None:
                # g will have the same shape as the input (Batch, 1)
                # We penalize the square of the derivative
                total_grad_loss += torch.mean(g**2)

        return total_grad_loss

