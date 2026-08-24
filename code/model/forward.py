from code.model.base import BaseModel
from omegaconf import DictConfig
import torch


class ForwardModel(BaseModel):
    """
    Forward model for radiative transfer emulation (e.g., CRTM).

    This model follows the new modular architecture pattern where:
    - Architecture is instantiated from config and passed as a module
    - Loss computation is straightforward (pred, target) -> loss
    - Metrics collection is delegated to callbacks (not stored in model)
    - No logging logic in the model itself

    The model stores step outputs (pred, target, input) on self._step_data
    for callbacks to access and compute metrics.
    """

    def __init__(
        self,
        ckpt_path: str | DictConfig,
        architecture: DictConfig,
        optimizer: DictConfig | None = None,
        scheduler: DictConfig | None = None,
        loss: DictConfig | None = None,
        post_process: DictConfig | None = None,
    ) -> None:
        """
        Initialize ForwardModel.

        Parameters
        ----------
        ckpt_path: str | DictConfig
            Path to the checkpoint file or DictConfig containing checkpoint info.
        architecture : DictConfig
            Configuration for the model architecture.
        optimizer : DictConfig, optional
            Optimizer configuration
        scheduler : DictConfig, optional
            Learning rate scheduler configuration
        loss : DictConfig, optional
            Loss function configuration
        post_process : DictConfig, optional
            Post-processing layer configuration to transform outputs to physical space.
        """

        # Class inheritance
        super().__init__(
            ckpt_path=ckpt_path,
            architecture=architecture,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            post_process=post_process,
        )

    def _infer(self, batch: dict) -> dict:
        """ Build output structure for forward model with post-processing.

        Wraps forward model predictions with structured key and applies post-processing.

        Parameters
        ----------
        batch : dict
            Input batch containing 'input' and other batch data.

        Returns
        -------
        dict
            Dictionary with 'output' key containing predictions in physical space.
        """

        # Forward pass
        out = self.forward(batch['input'])
        if isinstance(out, torch.Tensor):
            output = {'output': {'bt_forward': out}}
        elif isinstance(out, tuple):
            output = {'output': {'bt_forward': out[0], 'bt_forward_stdev': out[1]}}
        else:
            raise ValueError("Forward model output must be a torch.Tensor or a tuple of tensors.")

        # Apply post-processing to transform outputs to physical space
        if self.post_process is not None:
            output['output'] = self.post_process(output['output'])

        return output
