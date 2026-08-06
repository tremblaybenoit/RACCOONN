from code.model.base import BaseModel
from omegaconf import DictConfig


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
        lr_scheduler: DictConfig | None = None,
        loss_func: DictConfig | None = None,
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
        lr_scheduler : DictConfig, optional
            Learning rate scheduler configuration
        loss_func : DictConfig, optional
            Loss function configuration
        """

        # Class inheritance
        super().__init__(
            ckpt_path=ckpt_path,
            architecture=architecture,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_func=loss_func,
        )

    def _infer(self, batch: dict) -> dict:
        """ Build output structure for forward model.

        Wraps forward model predictions with 'hofx' (homogenized observed radiance) key.

        Parameters
        ----------
        batch : dict
            Input batch containing 'input' and other batch data.

        Returns
        -------
        dict
            Dictionary with 'output' key containing {'hofx': predictions}.
        """
        return {'output': {'hofx': self.forward(batch['input'])}}
