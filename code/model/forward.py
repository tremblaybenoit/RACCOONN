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
        """

        # Class inheritance
        super().__init__(
            ckpt_path=ckpt_path,
            architecture=architecture,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
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

    def base_step(self, batch: dict, batch_nb: int, stage: str) -> torch.Tensor | dict:
        """ Perform training/validation/test step.

            Parameters
            ----------
            batch: dict. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.
            stage: str. Current operation: "train", "valid", or "test".

            Returns
            -------
            dict. Step output with 'output' key and optional 'loss' keys.
        """

        # Build structured output (customization point for subclasses via _infer())
        step = self._infer(batch)

        # Stage-dependent operation: Loss
        if stage in ('train', 'valid', 'test') and self.loss is not None:
            # Compute loss
            loss = self.loss(step['output']['hofx'], batch['target']['hofx'])
            # If dictionary with multiple terms
            if isinstance(loss, dict):
                # Track total loss
                step['loss'] = loss['total']
                # Detach loss components
                step[f'{stage}_loss'] = {
                    key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                    else value for key, value in loss.items()
                }
            elif isinstance(loss, torch.Tensor):
                step['loss'] = loss.mean()  # type: ignore
                step[f'{stage}_loss'] = loss.detach().cpu().numpy()
            else:
                step['loss'] = loss
        # Detach outputs
        step['output'] = {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
            else value for key, value in step['output'].items()
        }

        return step