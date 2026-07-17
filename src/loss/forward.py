import os
import torch
from omegaconf import OmegaConf
from utilities.logic import get_config_path
from utilities.instantiators import instantiate


def resolve_path(
        path: str,
        dir: str | None = None
) -> str:
    """ Resolve relative paths to absolute; leave absolute paths unchanged.

        Parameters
        ----------
        path: str. The path to resolve.
        dir: str or None. The base directory to resolve relative paths against.

        Returns
        -------
        str. The resolved absolute path.
    """

    # If the path is absolute, pass
    if os.path.isabs(path):
        return path
    # If no base directory is specified, resolve relative to the current working directory
    if dir is None:
        return str(os.path.abspath(path))
    # Otherwise, resolve relative to the specified base directory
    return str(os.path.abspath(os.path.join(dir, path)))


class ForwardModel(torch.nn.Module):
    def __init__(
            self,
            checkpoint_path: str,
            config_path: str,
            dtype: str | None = None
    ) -> None:
        """ Initialize ForwardModel.

        Parameters
        ----------
        checkpoint_path: str. Path to the checkpoint file.
        config_path: str. Path to the configuration file.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Resolve paths only if relative
        checkpoint_path = self._resolve_path(checkpoint_path, dir=os.path.dirname(__file__))
        config_path = self._resolve_path(config_path, dir=get_config_path())

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Load the configuration
        config = OmegaConf.load(config_path)
        # Trim the configuration to only extract the model parameters
        config_forward = OmegaConf.create(
            {
                "_target_": config["_target_"],
                "parameters": config["parameters"]
            }
        )

        # Instantiate the model
        self.model = instantiate(config_forward)
        self.model.load_state_dict(checkpoint['state_dict'])
        for param in self.model.parameters():
            param.requires_grad = False
        # Set the model to evaluation mode
        self.model.eval()
        # Adjust precision
        if dtype:
            self.model.to(None, dtype=getattr(torch, dtype))

    def to(
            self,
            device,
            dtype: torch.dtype | None = None,
            non_blocking: bool = False
    ) -> 'ForwardModel':
        """ Move the model to the specified device.

        Parameters
        ----------
        device: torch.device. The device to move the model to.
        dtype: torch.dtype, optional. The desired data type of the model parameters.
        non_blocking: bool, optional. If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.

        Returns
        -------
        ForwardModel. The model itself.
        """
        self.model.to(device, dtype=dtype, non_blocking=non_blocking)
        return self

    def __call__(
            self,
            x: dict
    ) -> torch.Tensor:
        """ Call the forward model.

        Parameters
        ----------
        x: dict. Input dict.

        Returns
        -------
        torch.Tensor. Forward-modeled output tensor.
        """

        return self.model(x)
