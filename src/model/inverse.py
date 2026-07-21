import torch
from omegaconf import DictConfig, OmegaConf
from src.preprocessing.statistics import statistics, accumulate_statistics
from src.model.base import BaseModel
from src.model.forward import ForwardModel
from src.architecture.activation import Sine
from src.architecture.encoding import IdentityPositionalEncoding
from utilities.instantiators import instantiate, resolve_path
from utilities.logic import get_config_path
from src.data.transformations import mean_stdev, min_max
from typing import Callable
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)


class InverseModel(BaseModel):
    """
    Inverse model for atmospheric retrieval using Physics-Informed Neural Networks.

    This model implements:
    - Coordinate expansion: expands spatial/atmospheric inputs across pressure levels
    - Profile transformations: applies physical transformations (e.g., inverse min/max normalization)
    - Support for different output modes: min_max, mean_stdev, sigmoid

    All metrics collection and logging are delegated to callbacks.
    The model stores step outputs on self._step_data for callback access.
    """

    def __init__(
        self,
        ckpt_path: str | DictConfig,
        architecture: DictConfig,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
        loss_func: DictConfig | Callable | None = None,
        forward_model: DictConfig | ForwardModel | None = None,
    ) -> None:
        """
        Initialize InverseModel.

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
        loss_func : DictConfig | Callable, optional
            Loss function configuration
        forward_model : DictConfig | ForwardModel, optional
            Configuration for the forward model used in physics-informed loss computation.
            If DictConfig, can include 'ckpt_path' key to load pre-trained weights.
        """

        # Class inheritance
        super().__init__(
            ckpt_path=ckpt_path,
            architecture=architecture,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_func=loss_func,
        )

        # Forward model (observation operator)
        if forward_model is not None:
            if isinstance(forward_model, DictConfig):
                self.forward_model = instantiate(forward_model)
            elif isinstance(forward_model, ForwardModel):
                self.forward_model = forward_model
            else:
                raise ValueError("forward_model must be a DictConfig or a ForwardModel instance.")
            # Load checkpoint if ckpt_path is provided
            if hasattr(self.forward_model, 'ckpt_path') and self.forward_model.ckpt_path:
                self.forward_model.load_ckpt(freeze=True)  # type: ignore
        else:
            self.forward_model = None

    def forward(self, input_dict: dict) -> torch.Tensor:
        """ Perform forward pass with coordinate expansion.

        Expands input variables across pressure levels before passing to architecture.
        This is specialized for atmospheric retrieval which requires profiles at multiple
        pressure levels.

        Parameters
        ----------
        input_dict : dict
            Dictionary of input variables. May contain:
            - Variables with shape (Batch, ) which get expanded to (Batch, n_levels, 1)
            - Variables with shape (Batch, n_levels) which get reshaped to (Batch, n_levels, 1)

        Returns
        -------
        torch.Tensor
            Profile predictions with shape (Batch, n_prof, n_levels).
        """
        # Get n_levels from architecture if available, otherwise assume 1
        n_levels = getattr(self.architecture, 'n_levels', 1)  # TODO: Update

        # Vectorized expansion: Create list of tensors all shaped (Batch, n_levels, 1)
        tensors = []
        for k, v in input_dict.items():
            if v.ndim == 1:
                # For (Batch, ) → (Batch, n_levels, 1)
                tensors.append(v[:, None, None].expand(-1, n_levels, 1))
            elif v.ndim == 2:
                # For (Batch, n_levels) → (Batch, n_levels, 1)
                tensors.append(v.unsqueeze(-1))
            else:
                # Already (Batch, n_levels, ...) or other shape
                tensors.append(v)

        # Concatenate: (Batch, n_levels, num_features) → (Batch * n_levels, num_features)
        inputs = torch.cat(tensors, dim=-1).view(-1, len(tensors))

        # Inference through architecture
        out = self.architecture(inputs)

        # Reshape back to (Batch, n_prof, n_levels)
        # Assuming output shape is (Batch * n_levels, n_prof)
        batch_size = list(input_dict.values())[0].shape[0]
        n_prof = out.shape[-1] if out.ndim > 1 else 1
        return out.view(batch_size, n_levels, n_prof).transpose(1, 2)

    def _infer(self, batch: dict) -> dict:
        """ Build output structure for inverse model.

        Performs profile inversion and optionally applies forward model
        to compute observation consistency (hofx). Receives full batch
        to enable access to batch['context'] for forward model evaluation.

        Parameters
        ----------
        batch : dict
            Full batch containing 'input', 'context', and other batch data.

        Returns
        -------
        dict
            Dictionary containing {'prof': predictions, 'hofx': optional}.
        """
        # Inversion of atmospheric profiles
        output = {'prof': self.forward(batch['input'])}

        # Forward-modeled observations (requires batch['context'])
        if self.forward_model is not None and 'context' in batch:
            output['hofx_forward'] = self.forward_model(
                {
                    'prof': output['prof'],
                    'surf': batch['context']['surf'],
                    'meta': batch['context']['meta']
                }
            )

        return output
