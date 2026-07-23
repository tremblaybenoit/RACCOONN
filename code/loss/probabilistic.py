import torch
import numpy as np


def crps(pred: torch.Tensor, target: torch.Tensor, version: int=0) -> torch.Tensor:
    """
    Compute the Continuous Ranked Probability Score (CRPS) loss function.
    This function computes the CRPS for a normal distribution defined by
    the mean and standard deviation, given the true values and predicted values.

    Parameters
    ----------
    pred: torch.Tensor. Tensor containing predictions: [mean, std].
    target: torch.Tensor. True values.
    version: int. Version of the CRPS calculation to use (0 or 1 or 2).

    Returns
    -------
    torch.Tensor. Mean CRPS over the batch.
    """

    # Split input
    mu_pred = pred[:, :10]
    sigma_pred = pred[:, 10:]
    mu_target = target[:, :10]
    sigma_target = target[:, 10:]

    # Compute variance (to prevent negative sigma)
    var_pred = sigma_pred ** 2
    var_target = sigma_target ** 2

    # Minor adjustment to avoid division by zero
    eps = np.finfo(target.dtype).eps if isinstance(target, np.ndarray) else torch.finfo(target.dtype).eps
    # Normalize the errors by the standard deviation
    if version == 0:
        loc = (mu_target - mu_pred) / torch.sqrt(var_pred + eps)
    elif version == 1:
        loc = (mu_target - mu_pred) / torch.sqrt(var_pred + var_target + eps)
    elif version == 2:
        loc = (mu_target - mu_pred) / torch.sqrt(var_target + eps)
    else:
        raise ValueError("Invalid version specified. Use 0 or 1 or 2.")
    # Compute the probability density function (PDF) and cumulative distribution function (CDF)
    phi = 1.0 / np.sqrt(2.0 * np.pi) * torch.exp(-loc ** 2 / 2.0)
    pphi = 0.5 * (1.0 + torch.erf(loc / np.sqrt(2.0)))
    # Compute the CRPS for each input/target pair
    return torch.sqrt(var_pred) * (loc * (2. * pphi - 1.) + 2 * phi - 1. / np.sqrt(np.pi))


class CRPS(torch.nn.Module):
    """ Continuous Ranked Probability Score loss module."""
    def __init__(self, version: int=0) -> None:
        """ Initialize the CRPS module.

        Parameters
        ----------
        version: int. Version of the CRPS calculation to use (0 or 1 or 2).

        Returns
        -------
        None.
        """
        # Class inheritance
        super().__init__()
        # Store version
        self.version = version

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
        """ Compute the Continuous Ranked Probability Score between predicted and target normal distributions.

        Parameters
        ----------
        pred: torch.Tensor. Tensor containing predictions: [mean, std].
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Mean CRPS over the batch.
        """
        return crps(pred, target, version=self.version)


def kl_divergence_normal(mu_pred, sigma_pred, mu_target, sigma_target):
    """
    Compute the KL divergence between two normal distributions.

    Parameters
    ----------
    mu_pred: torch.Tensor. Mean of the predicted normal distribution.
    sigma_pred: torch.Tensor. Standard deviation of the predicted normal distribution.
    mu_target: torch.Tensor. Mean of the target normal distribution.
    sigma_target: torch.Tensor. Standard deviation of the target normal distribution.

    Returns
    -------
    kl_div: torch.Tensor. KL divergence between the predicted and target normal distributions.
    """

    # Compute variances
    var_pred = sigma_pred ** 2
    var_target = sigma_target ** 2

    # Compute KL divergence between two normal distributions
    return torch.log(sigma_target / sigma_pred) + (var_pred + (mu_pred - mu_target) ** 2) / (2 * var_target) - 0.5


class KLDivNormal(torch.nn.Module):
    """ KL divergence loss module."""
    def __init__(self):
        """ Initialize the KLDivNormal module.

        Returns
        -------
        None.
        """
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the KL divergence between predicted and target normal distributions.

        Parameters
        ----------
        pred: torch.Tensor. Tensor containing predictions: [mean, std].
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Mean KL divergence over the batch.
        """

        # Split input
        mu_pred = pred[:, :10]
        sigma_pred = pred[:, 10:]
        mu_target = target[:, :10]
        sigma_target = target[:, 10:]

        return kl_divergence_normal(mu_pred, sigma_pred, mu_target, sigma_target)


def wasserstein_2_normal(mu_pred, sigma_pred, mu_target, sigma_target):
    """
    Compute the Wasserstein-2 distance between two normal distributions.

    Parameters
    ----------
    mu_pred: torch.Tensor. Mean of the predicted normal distribution.
    sigma_pred: torch.Tensor. Standard deviation of the predicted normal distribution.
    mu_target: torch.Tensor. Mean of the target normal distribution.
    sigma_target: torch.Tensor. Standard deviation of the target normal distribution.

    Returns
    -------
    wasserstein_distance: torch.Tensor. Wasserstein-2 distance between the predicted and target normal distributions.
    """
    return torch.sqrt((mu_pred - mu_target) ** 2 + (sigma_pred - sigma_target) ** 2)


class Wasserstein2Normal(torch.nn.Module):
    """ Wasserstein-2 distance loss module."""
    def __init__(self):
        """ Initialize the Wasserstein2Normal module.

        Returns
        -------
        None.
        """
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the Wasserstein-2 distance between predicted and target normal distributions.

        Parameters
        ----------
        pred: torch.Tensor. Tensor containing predictions: [mean, std].
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Mean Wasserstein-2 distance over the batch.
        """

        # Split input
        mu_pred = pred[:, :10]
        sigma_pred = pred[:, 10:]
        mu_target = target[:, :10]
        sigma_target = target[:, 10:]

        return wasserstein_2_normal(mu_pred, sigma_pred, mu_target, sigma_target)

