import torch


def l2(pred: torch.Tensor) -> torch.Tensor:
    """ L2 loss function.

    Parameters
    ----------
    pred: torch.Tensor. Predicted tensor.

    Returns
    -------
    torch.Tensor. L2 loss over the batch.
    """
    return pred ** 2


class L2(torch.nn.Module):
    """ L2 loss module."""
    def __init__(self) -> None:
        """ Initialize the L2 module.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """

        # Class inheritance
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """ Compute the L2 loss between predicted and target tensors.

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor.
        target: torch.Tensor | None. True values.

        Returns
        -------
        torch.Tensor. L2 loss over the batch.
        """
        return l2(pred)


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Mean Squared Error loss function.

    Parameters
    ----------
    pred: torch.Tensor. Predicted tensor.
    target: torch.Tensor. True values.

    Returns
    -------
    torch.Tensor. Mean squared error over the batch.
    """
    return (pred - target) ** 2


class MSE(torch.nn.Module):
    """ Mean Squared Error loss module."""
    def __init__(self) -> None:
        """ Initialize the MSE module.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """

        # Class inheritance
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the mean squared error between predicted and target tensors.

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor.
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Mean squared error over the batch.
        """
        return mse(pred, target)

