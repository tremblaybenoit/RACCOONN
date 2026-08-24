import torch
from omegaconf import DictConfig
from code.model.forward import ForwardModel
from utilities.instantiators import instantiate
from typing import Callable
import logging

# Initialize logger
logger = logging.getLogger(__name__)


class InverseModel(ForwardModel):
    """
    Inverse model for atmospheric retrievals.

    This model implements:
    - Coordinate expansion: expands spatial/atmospheric inputs across pressure levels.
    """

    def __init__(
        self,
        ckpt_path: str | DictConfig,
        architecture: DictConfig,
        optimizer: DictConfig | None = None,
        scheduler: DictConfig | None = None,
        loss: DictConfig | Callable | None = None,
        pre_process: DictConfig | None = None,
        post_process: DictConfig | None = None,
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
        scheduler : DictConfig, optional
            Learning rate scheduler configuration
        loss : DictConfig | Callable, optional
            Loss function configuration
        post_process : DictConfig, optional
            Post-processing layer configuration to transform outputs to physical space.
        pre_process : DictConfig, optional
            Pre-processing layer configuration to transform inputs to model space.
        forward_model : DictConfig | ForwardModel, optional
            Configuration for the forward model used in physics-informed loss computation.
            If DictConfig, can include 'ckpt_path' key to load pre-trained weights.
        """

        # Class inheritance
        super().__init__(
            ckpt_path=ckpt_path,
            architecture=architecture,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            pre_process=pre_process,
            post_process=post_process,
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

    def forward(self, input_dict: dict, training_flag: bool = False) -> dict:
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
        training_flag : bool, optional
            Flag indicating whether the model is in training mode. Default is False.

        Returns
        -------
        dict
            Dictionary with 'prof' key containing profile predictions with shape (Batch, n_prof, n_levels).
        """

        # Apply pre-processing to transform inputs to model space
        if not training_flag and self.pre_process is not None:
            input_dict = self.pre_process(input_dict)

        # Get n_levels from architecture if available, otherwise assume 1
        n_levels = input_dict['pressure'].shape[-1]

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
        output_dict = {'prof': self.architecture(inputs)}

        # Reshape back to (Batch, n_prof, n_levels)
        # Assuming output shape is (Batch * n_levels, n_prof)
        batch_size = list(input_dict.values())[0].shape[0]
        n_prof = output_dict['prof'].shape[-1] if output_dict['prof'].ndim > 1 else 1
        output_dict['prof'] = output_dict['prof'].view(batch_size, n_levels, n_prof).transpose(1, 2)

        # Apply post-processing to transform outputs to physical space
        if self.post_process is not None:
            output_dict = self.post_process(output_dict)

        return output_dict

    def _infer(self, batch: dict, training_flag: bool = False) -> dict:
        """ Build output structure for inverse model with post-processing.

        Performs profile inversion and optionally applies forward model
        to compute observation consistency. Applies post-processing to transform
        outputs to physical space.

        Receives full batch to enable access to batch['context'] for forward model evaluation.

        Parameters
        ----------
        batch : dict
            Full batch containing 'input', 'context', and other batch data.
        training_flag : bool, optional
            Flag indicating whether the model is in training mode. Default is False.

        Returns
        -------
        dict
            Dictionary with 'output' key containing post-processed predictions in physical space.
        """

        # Inversion of atmospheric profiles
        output_dict = {'output': self.forward(batch['input'], training_flag=training_flag)}

        # Forward-modeled observations (requires batch['context'])
        if self.forward_model is not None and 'context' in batch:
            if all(k in batch['context'] for k in ['surf', 'meta']):
                forward_model_output = self.forward_model(
                    {
                        'prof': output_dict['output']['prof'],
                        'surf': batch['context']['surf'],
                        'meta': batch['context']['meta']
                    }
                )
                # Switch keys ('bt_forward' → 'bt_inverse') for clarity in output
                if 'bt_forward' in forward_model_output:
                    output_dict['output']['bt_inverse'] = forward_model_output['bt_forward']
                if 'bt_forward_stdev' in forward_model_output:
                    output_dict['output']['bt_inverse_stdev'] = forward_model_output['bt_forward_stdev']

        return output_dict
