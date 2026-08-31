import torch
from typing import Any
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from utilities.instantiators import instantiate
import gc
from typing import Callable
import logging
import os
import torch.nn as nn

# Initialize logger
logger = logging.getLogger(__name__)


class ForwardModel(LightningModule):
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
        loss: DictConfig | Callable | None = None,
        pre_process: DictConfig | None = None,
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
        loss : DictConfig | Callable, optional
            Loss function configuration
        pre_process : DictConfig, optional
            Pre-processing layer configuration to transform inputs to model space.
            Must instantiate to TransformationsLayer or TransformationsLayers.
            Applied to model inputs to normalize/transform data. Default None.
        post_process : DictConfig, optional
            Post-processing layer configuration to transform outputs to physical space.
                      Must instantiate to TransformationsLayer or TransformationsLayers.
                      Applied to model outputs to denormalize/transform predictions. Default None.
        """

        # Class inheritance
        super().__init__()

        # Checkpoint path
        self.ckpt_path = instantiate(ckpt_path) if isinstance(ckpt_path, DictConfig) else ckpt_path
        # Model architecture
        self.architecture = instantiate(architecture)
        # Learning rate scheduler
        self.scheduler = scheduler
        # Optimizer initialization
        self.optimizer = optimizer
        # Loss function
        if loss is not None:
            if isinstance(loss, DictConfig):
                self.loss = instantiate(loss)
            elif isinstance(loss, Callable):
                self.loss = loss
            else:
                raise ValueError("loss must be a DictConfig or a callable.")
        else:
            self.loss = None
        # Pre-processing layer (normalization/transformation to model space)
        self.pre_process: nn.Module | None = None
        if pre_process is not None:
            self.pre_process = instantiate(pre_process)
        # Post-processing layer (denormalization/transformation to physical space)
        self.post_process: nn.Module | None = None
        if post_process is not None:
            self.post_process = instantiate(post_process)

        # Store hyperparameters
        self.save_hyperparameters(ignore=['optimizer', 'scheduler', 'loss', 'post_process'])

    def forward(self, input_dict: dict, training_flag: bool = False) -> dict:
        """ Perform forward pass through architecture.

        Converts input dict to tensor and executes raw neural network prediction.
        Subclasses can override this to customize input assembly (e.g., InverseModel).

        Default behavior: concatenate all values in input_dict along last dimension.

        Parameters
        ----------
        input_dict : dict
            Dictionary of input variables. Each value should be a tensor.
        training_flag : bool, optional
            Flag indicating whether the model is in training mode. Default is False.

        Returns
        -------
        dict
            Dictionary with 'output' key containing predictions from the architecture.
        """

        # Apply pre-processing to transform inputs to model space
        if not training_flag and self.pre_process is not None:
            input_dict = self.pre_process(input_dict)

        # Reshape all tensors to 2D (batch, features) before concatenation
        reshaped = [
            x.reshape(x.size(0), -1) if x.dim() > 2 else x
            for x in input_dict.values()
        ]
        input_tensor = torch.cat(reshaped, dim=-1)

        # Inference through architecture
        out = self.architecture(input_tensor)

        # Wrap output in structured dict
        if isinstance(out, torch.Tensor):
            output_dict = {'bt_forward': out}
        elif isinstance(out, tuple):
            output_dict = {'bt_forward': out[0], 'bt_forward_stdev': out[1]}
        else:
            raise ValueError("Forward model output must be a torch.Tensor or a tuple of tensors.")

        # Apply post-processing to transform outputs to physical space
        if self.post_process is not None:
            output_dict = self.post_process(output_dict)

        return output_dict

    def _infer(self, batch: dict, training_flag: bool = False) -> dict:
        """Build output structure from batch with post-processing applied.

        Template method for customizing output structure. Receives full batch
        to enable complex logic (e.g., accessing batch['context'] for InverseModel).

        Applies post-processing to transform outputs from normalized/model space
        to physical space as part of building the final output structure.

        Subclasses override this to customize the output structure.

        Parameters
        ----------
        batch : dict
            Full batch dictionary containing 'input', 'target', 'context', etc.
        training_flag : bool, optional
            Flag indicating whether the model is in training mode. Default is False.

        Returns
        -------
        dict
            Dictionary containing model predictions in physical space.
        """

        # Forward pass
        return {'output': self.forward(batch['input'], training_flag=training_flag)}

    def base_step(self, batch: dict, batch_nb: int, stage: str) -> torch.Tensor | dict:
        """Perform training/validation/test step.

        Parameters
        ----------
        batch: dict
            Batch from the training set.
        batch_nb: int
            Index of the batch out of the training set.
        stage: str
            Current operation: "train", "valid", "test", or "predict".

        Returns
        -------
        dict
            Step output with 'output' key and optional 'loss' keys.
        """

        # Build structured output (customization point for subclasses via _infer())
        # _infer() includes post-processing to physical space
        training_flag = (stage in ('train', 'valid', 'test'))
        step = self._infer(batch, training_flag=training_flag)
        # Stage-dependent operation: Loss
        if training_flag and self.loss is not None:
            # Compute loss on post-processed outputs in physical space
            loss = self.loss(step['output'], batch)
            # If dictionary with multiple terms
            if isinstance(loss, dict):
                # Track total loss
                step['loss'] = loss['total']
                # Detach loss components
                step[f'{stage}_loss'] = {
                    key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                    else value for key, value in loss.items() if key != 'total'
                }
            elif isinstance(loss, torch.Tensor) and loss.dim() > 0:
                step['loss'] = loss.mean()  # type: ignore
                step[f'{stage}_loss'] = loss.detach().cpu().numpy()
            else:
                step['loss'] = loss
        
        # Stage-aware output conversion
        # Train/valid: keep outputs as tensors on GPU for callbacks (ForwardLogger handles conversion)
        # Test/predict: convert to numpy for saving/returning
        if stage in ('test', 'predict'):
            step['output'] = {
                key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                else value for key, value in step['output'].items()
            }

        return step

    def training_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """Perform training step.

        Parameters
        ----------
        batch: dict
            Batch from the training set.
        batch_nb: int
            Index of the batch out of the training set.

        Returns
        -------
        Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='train')

    def validation_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """Perform validation step.

        Parameters
        ----------
        batch: dict
            Batch from the validation set.
        batch_nb: int
            Index of the batch out of the validation set.

        Returns
        -------
        Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='valid')

    def test_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """Perform test step.

        Parameters
        ----------
        batch: dict
            Batch from the test set.
        batch_nb: int
            Index of the batch out of the test set.

        Returns
        -------
        Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='test')

    def predict_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """Perform prediction step.

        Provides a unified interface for prediction that returns the same
        structure as test_step.

        Parameters
        ----------
        batch: dict
            Batch from the prediction set.
        batch_nb: int
            Index of the batch out of the prediction set.

        Returns
        -------
        dict
            Predictions with structure compatible with callbacks.
        """

        return self.base_step(batch, batch_nb, stage='predict')

    def on_train_epoch_end(self):
        """Callback to log training results at the end of each training epoch.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        # Clean
        gc.collect()

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer | dict[str, Any]] | None:
        """Instantiate optimizer.

        Parameters
        ----------
        None. Target and parameters are passed from self.optimizer_config.

        Returns
        -------
        Optimizer instance.
        """

        # Check if optimizer is defined
        if self.optimizer is not None:

            # Instantiate optimizer
            optimizer = instantiate(self.optimizer, params=self.parameters())

            # Check if learning rate scheduler is defined
            if self.scheduler is not None:
                # Instantiate learning rate scheduler
                scheduler = instantiate(self.scheduler, optimizer=optimizer)

                # Check if the learning rate scheduler is specifically reducing on plateau
                reduce_on_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

                # Instantiate from config object
                return {'optimizer': optimizer,
                        'lr_scheduler': {'scheduler': scheduler,
                                         'interval': 'epoch',
                                         'monitor': 'valid_loss',
                                         'frequency': 1,
                                         'reduce_on_plateau': reduce_on_plateau,
                                         }
                        }
            return optimizer
        return None

    def to(self, device, dtype: torch.dtype | None = None, non_blocking: bool = False):
        """Move the model, loss function, and post-process layer to the specified device.

        Parameters
        ----------
        device: torch.device
            The device to move to.
        dtype: torch.dtype
            The desired data type of the model parameters (optional).
        non_blocking: bool
            If True, and the source is in pinned memory, transfers asynchronously.

        Returns
        -------
        ForwardModel
            The instance with model, loss, and post-process moved to the specified device.
        """

        # Class inheritance
        super().to(device, dtype=dtype, non_blocking=non_blocking)
        # Move loss function to device
        if hasattr(self.loss, 'to'):
            self.loss = self.loss.to(device)
        # Move post-process layer to device
        if hasattr(self.post_process, 'to'):
            self.post_process = self.post_process.to(device)
        return self

    def load_ckpt(self, ckpt_path: str | None = None, strict: bool = False, freeze: bool = False) -> 'ForwardModel':
        """Load model weights from a checkpoint file.

        Parameters
        ----------
        ckpt_path: str
            Path to the checkpoint file (.ckpt or .pt).
        strict: bool
            If True, requires all keys to match. If False, allows
            missing or extra keys. Default False (useful for test/predict).
        freeze: bool
            If True, sets the model to eval mode after loading.
            Default False. Set to True for pre-trained models to use running
            batch norm statistics without accumulating new ones. Note: To prevent
            weight updates, exclude model from optimizer (gradients still flow).

        Returns
        -------
        ForwardModel
            The instance with loaded weights (supports method chaining).

        Raises
        ------
        FileNotFoundError: If ckpt_path does not exist.
        RuntimeError: If state_dict loading fails with strict=True.
        """

        # Use provided path or fall back to instance attribute
        ckpt_path = ckpt_path or self.ckpt_path
        if not ckpt_path:
            raise ValueError("No checkpoint path provided and self.ckpt_path is not set")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

        logger.info(f"Loading checkpoint from: {ckpt_path}")

        # Load checkpoint from disk
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Extract state_dict (handle both PyTorch Lightning and raw formats)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Load into model
        self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded successfully (strict={strict})")

        # Optionally set to eval mode
        if freeze:
            self.eval()
            logger.info("Model set to eval mode (running batch norm stats, gradients enabled)")

        return self

