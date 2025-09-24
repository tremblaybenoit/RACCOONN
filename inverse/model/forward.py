import torch
from omegaconf import OmegaConf
from utilities.instantiators import instantiate


class CRTMForward:
    def __init__(self, checkpoint_path, config_path):
        """ Initialize CRTM forward model.

        Parameters
        ----------
        checkpoint_path: str. Path to the checkpoint file.
        config_path: str. Path to the configuration file.

        Returns
        -------
        None.
        """
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        # Load the configuration
        config = OmegaConf.load(config_path)
        # Trim the configuration to only extract the model parameters
        config_forward = OmegaConf.create({"_target_": config._target_, "parameters": config.parameters})

        # Instantiate the model
        self.model = instantiate(config_forward)
        self.model.load_state_dict(checkpoint['state_dict'])
        # Set the model to evaluation mode
        self.model.eval()

    def to(self, device, dtype=None, non_blocking=False):
        """ Move the model to the specified device.

        Parameters
        ----------
        device: torch.device. The device to move the model to.
        dtype: torch.dtype, optional. The desired data type of the model parameters.
        non_blocking: bool, optional. If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.

        Returns
        -------
        None.
        """
        self.model.to(device, dtype=dtype, non_blocking=non_blocking)
        return self

    def __call__(self, x):
        """ Call the forward model.

        Parameters
        ----------
        x: torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Forward-modeled output tensor.
        """

        return self.model(x)
