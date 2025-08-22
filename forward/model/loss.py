import torch
import numpy as np


def crps_cost_function(pred, target):
    """
    Compute the Continuous Ranked Probability Score (CRPS) loss function.
    This function computes the CRPS for a normal distribution defined by
    the mean and standard deviation, given the true values and predicted values.

    Parameters
    ----------
    pred: torch.Tensor. Tensor containing predictions: [mean, std].
    target: torch.Tensor. True values.

    Returns
    -------
    torch.Tensor. Mean CRPS over the batch.
    """

    # Split input
    mu = pred[:, :10]
    sigma = pred[:, 10:]

    # Compute variance (to prevent negative sigma)
    var = sigma ** 2

    # Normalize the errors by the standard deviation
    loc = (target - mu) / torch.sqrt(var)
    # Compute the probability density function (PDF) and cumulative distribution function (CDF)
    phi = 1.0 / np.sqrt(2.0 * np.pi) * torch.exp(-loc ** 2 / 2.0)
    Phi = 0.5 * (1.0 + torch.erf(loc / np.sqrt(2.0)))
    # Compute the CRPS for each input/target pair
    crps = torch.sqrt(var) * (loc * (2. * Phi - 1.) + 2 * phi - 1. / np.sqrt(np.pi))

    return torch.nanmean(crps)


def mse(pred, target):
    """ Compute MSE loss between prediction and target, accounting for NaN values.

    Parameters
    ----------
    pred: torch.Tensor. Prediction tensor.
    target: torch.Tensor. Target tensor.

    Returns
    -------
    loss: torch.Tensor. MSE loss between prediction and target.
    """

    # Compute MSE loss, accounting for NaN values in the target data
    loss = torch.nanmean((pred - target)**2)

    return loss
