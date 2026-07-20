import torch
from typing import Any
from pytorch_lightning import LightningModule
from src.preprocessing.statistics import statistics, accumulate_statistics
from src.architecture.activation import Swish, Scale, Sine
from src.architecture.ode import PressureConditionalODEFunc
from omegaconf import DictConfig
from utilities.instantiators import instantiate
import gc
from typing import Callable
import logging
import os

# Initialize logger
logger = logging.getLogger(__name__)


class BaseModel(LightningModule):
    """
    Lightning model template.
    """

    def __init__(self, architecture: DictConfig, optimizer: DictConfig | None = None,
                 lr_scheduler: DictConfig | None = None, loss_func: DictConfig | Callable | None = None) -> None:
        """ Initialize model.

        Parameters
        ----------
        architecture: DictConfig. Configuration object for the model architecture.
        optimizer: DictConfig. Optimizer for the model.
        lr_scheduler: DictConfig. Configuration object for the learning rate scheduler (optional).
        loss_func: DictConfig | Callable. Loss function for the model.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Model architecture
        self.architecture = instantiate(architecture)
        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler
        # Optimizer initialization
        self.optimizer = optimizer
        # Loss function
        if loss_func is not None:
            if isinstance(loss_func, DictConfig):
                self.loss_func = instantiate(loss_func)
            elif isinstance(loss_func, Callable):
                self.loss_func = loss_func
            else:
                raise ValueError("loss_func must be a DictConfig or a callable.")
        else:
            self.loss_func = None
        # Store hyperparameters
        self.save_hyperparameters(ignore=['optimizer', 'lr_scheduler', 'loss_func'])

    def forward(self, input_dict: dict) -> torch.Tensor:
        """ Perform forward pass through architecture.

        Converts input dict to tensor and executes raw neural network prediction.
        Subclasses can override this to customize input assembly (e.g., InverseModel).

        Default behavior: concatenate all values in input_dict along last dimension.

        Parameters
        ----------
        input_dict : dict
            Dictionary of input variables. Each value should be a tensor.

        Returns
        -------
        torch.Tensor
            Predictions from the architecture (tensor).
        """

        # Concatenate all tensors along last dimension
        input_tensor = torch.cat(list(input_dict.values()), dim=-1)
        return self.architecture(input_tensor)

    def _infer(self, batch: dict) -> dict:
        """ Build output structure from batch.

        Template method for customizing output structure. Receives full batch
        to enable complex logic (e.g., accessing batch['context'] for InverseModel).

        Subclasses override this to customize the output structure.

        Parameters
        ----------
        batch : dict
            Full batch dictionary containing 'input', 'target', 'context', etc.

        Returns
        -------
        dict
            Dictionary with 'output' key containing model predictions in desired structure.
        """
        return {'output': self.forward(batch['input'])}

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
        if stage in ('train', 'valid', 'test') and self.loss_func is not None:
            # Compute loss
            loss = self.loss_func(step['output'], batch['target'])
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

    def training_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """ Perform training step.

            Parameters
            ----------
            batch: dict. Batch from the training set.
            batch_nb: int. Index of the batch out of the training set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='train')

    def validation_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """ Perform validation step.

            Parameters
            ----------
            batch: dict. Batch from the validation set.
            batch_nb: int. Index of the batch out of the validation set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='valid')

    def test_step(self, batch: dict, batch_nb: int) -> torch.Tensor | dict:
        """ Perform test step.

            Parameters
            ----------
            batch: dict. Batch from the test set.
            batch_nb: int. Index of the batch out of the test set.

            Returns
            -------
            Loss value: tensor.
        """

        return self.base_step(batch, batch_nb, stage='test')

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor | dict:
        """ Perform prediction step.

        Provides a unified interface for prediction that returns the same
        structure as test_step, enabling callback-based result accumulation
        (e.g., ResultsLogger).

            Parameters
            ----------
            batch: dict. Batch from the prediction set.
            batch_idx: int. Index of the batch out of the prediction set.

            Returns
            -------
            dict. Predictions with structure compatible with callbacks.
        """

        return self.base_step(batch, batch_idx, stage='predict')

    def on_train_epoch_end(self):
        """ Callback to log training results at the end of each training epoch.

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
        """ Instantiate optimizer.

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
            if self.lr_scheduler is not None:
                # Instantiate learning rate scheduler
                lr_scheduler = instantiate(self.lr_scheduler, optimizer=optimizer)

                # Check if the learning rate scheduler is specifically reducing on plateau
                reduce_on_plateau = isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

                # Instantiate from config object
                return {'optimizer': optimizer,
                        'lr_scheduler': {'scheduler': lr_scheduler,
                                         'interval': 'epoch',
                                         'monitor': 'valid_loss',
                                         'frequency': 1,
                                         'reduce_on_plateau': reduce_on_plateau,
                                         }
                        }
            return optimizer
        return None

    def to(self, device, dtype: torch.dtype | None = None, non_blocking: bool = False) -> 'BaseModel':
        """ Move the model and loss function to the specified device.

        Parameters
        ----------
        device: torch.device. The device to move the model and loss function to.
        dtype: torch.dtype. The desired data type of the model parameters (optional).
        non_blocking: bool. If True, and the source is in pinned memory, the

        Returns
        -------
        BaseModel. The instance with model and loss function moved to the specified device.
        """

        # Class inheritance
        super().to(device, dtype=dtype, non_blocking=non_blocking)
        # Move loss function to device
        if hasattr(self.loss_func, 'to'):
            self.loss_func = self.loss_func.to(device)
        return self

    def load_ckpt(self, ckpt_path: str, strict: bool = False, freeze: bool = False) -> 'BaseModel':
        """ Load model weights from a checkpoint file.

        Parameters
        ----------
        ckpt_path: str. Path to the checkpoint file (.ckpt or .pt).
        strict: bool. If True, requires all keys to match. If False, allows
                missing or extra keys. Default False (useful for test/predict).
        freeze: bool. If True, sets the model to eval mode after loading.
                Default False. Set to True for pre-trained models to use running
                batch norm statistics without accumulating new ones. Note: To prevent
                weight updates, exclude model from optimizer (gradients still flow).

        Returns
        -------
        BaseModel. The instance with loaded weights (supports method chaining).

        Raises
        ------
        FileNotFoundError: If ckpt_path does not exist.
        RuntimeError: If state_dict loading fails with strict=True.
        """
        # Check if checkpoint exists
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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
