import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Dict, Any
from utilities.instantiators import instantiate
import logging

logger = logging.getLogger(__name__)


class WeightedLoss(torch.nn.Module):
    """
    Flexible multi-term loss wrapper that combines multiple loss functions via weighted sum.

    Each term specifies:
    - loss: callable or DictConfig to instantiate
    - weight: lambda/scaling factor (default: 1.0)
    - input_keys: which outputs to extract
    - target_keys: which batch keys to extract as targets
    - context_keys: optional additional context from batch

    All sub-losses are combined as: total = sum(w_i * loss_i)

    Example config:
    ```yaml
    _target_: code.loss.WeightedLoss
    terms:
      obs:
        loss:
          _target_: code.loss.basic.MSE
        weight: 1.0
        input_keys: [hofx_forward]
        target_keys: [hofx]
        context_keys: []

      model:
        loss:
          _target_: code.loss.basic.MSE
        weight: 0.5
        input_keys: [prof]
        target_keys: [prof_prior]
        context_keys: [pressure_mask]
    ```
    """

    def __init__(
        self,
        terms: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Initialize WeightedLoss.

        Parameters
        ----------
        terms : Dict[str, Dict]
            Mapping of term_name -> term_config.
            Each term_config must have:
            - function: torch.nn.Module or DictConfig (instantiable to Callable)
            - weight: float, optional (scaling factor for this term, default 1.0)
            - input_keys: str or list[str] (keys to extract from outputs)
            - target_keys: str or list[str] (keys to extract from batch)
            - context_keys: str or list[str], optional (additional context from batch)
        """

        # Class inheritance
        super().__init__()

        # Initialize internal structures
        self.config = terms
        self.loss = nn.ModuleDict()
        self.weight = {}

        # Instantiate loss functionss
        for term, config_term in terms.items():
            # Validate required keys
            required = {'function', 'input_keys', 'target_keys'}
            missing = required - set(config_term.keys())
            if missing:
                raise ValueError(
                    f"Term '{term}' missing keys: {missing}. "
                    f"Must have: {required}"
                )

            # Instantiate or validate loss function
            config_loss = config_term['function']
            if isinstance(config_loss, DictConfig):
                loss_fn = instantiate(config_loss)
            elif isinstance(config_loss, nn.Module):
                loss_fn = config_loss
            else:
                raise TypeError(
                    f"Term '{term}' loss must be DictConfig or nn.Module, got {type(config_loss)}"
                )

            # Register loss module
            self.loss[term] = loss_fn
            # Default weight to 1.0 if not specified
            self.weight[term] = float(config_term.get('weight', 1.0))
            logger.info(
                f"Registered loss term '{term}' "
                f"(weight={self.weight[term]})"
            )

    def to(self, device, dtype: torch.dtype | None = None, non_blocking: bool = False):
        """Move module and sub-losses to device."""
        super().to(device, dtype=dtype, non_blocking=non_blocking)
        # Move individual loss modules
        for name, loss_fn in self.loss.items():
            if hasattr(loss_fn, 'to'):
                self.loss[name] = loss_fn.to(device)
        return self

    @staticmethod
    def _extract_values(
        source: Dict[str, torch.Tensor],
        keys: str | list[str],
        term_name: str
    ) -> torch.Tensor | tuple:
        """
        Extract and optionally concatenate tensor(s) from source dict.

        Parameters
        ----------
        source : dict
            Source dictionary (outputs, batch, etc.)
        keys : str or list[str]
            Keys to extract. If multiple, returns tuple.
        term_name : str
            Name of loss term (for logging)

        Returns
        -------
        torch.Tensor or tuple
            If single key: returns tensor directly.
            If multiple keys: returns tuple of tensors.

        Raises
        ------
        KeyError
            If any key is missing from source.
        """

        # Normalize to list
        if isinstance(keys, str):
            keys = [keys]
        elif isinstance(keys, (list, tuple)):
            keys = list(keys)
        else:
            keys = [keys]

        tensors = []
        for key in keys:
            if key not in source:
                available = list(source.keys())
                raise KeyError(
                    f"Term '{term_name}': key '{key}' not found. "
                    f"Available keys: {available}"
                )
            tensors.append(source[key])

        # Single key: return directly
        if len(tensors) == 1:
            return tensors[0]
        # Multiple keys: return as tuple (allows loss to unpack)
        else:
            return tuple(tensors)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted combination of all loss terms.

        Parameters
        ----------
        outputs : dict[str, torch.Tensor]
            Model outputs (e.g., {'hofx': tensor, 'prof': tensor})
        batch : dict
            Full batch dict with 'target' and other keys

        Returns
        -------
        dict[str, torch.Tensor]
            Loss dict with:
            - 'loss': combined loss (scalar)
            - '{term_name}': individual term loss
        """

        # Loop over loss terms
        loss_dict = {'loss': torch.tensor(0.0)}
        for term, config_term in self.config.items():
            # Extract weight and function
            weight = self.weight[term]
            loss_fn = self.loss[term]

            # Extract output tensors
            output_values = self._extract_values(
                outputs,
                config_term['input_keys'],
                term
            )

            # Extract target tensors
            target_values = self._extract_values(
                batch['target'],
                config_term['target_keys'],
                term
            )

            # Extract optional context masks and apply them
            if 'context_keys' in config_term and config_term['context_keys']:

                # Extract all masks at once
                masks = self._extract_values(
                    batch['context'],
                    config_term['context_keys'],
                    term
                )

                # If single mask, convert to tuple for uniform handling
                if not isinstance(masks, tuple):
                    masks = (masks,)

                # Apply each mask sequentially to both output and target
                for mask in masks:
                    if isinstance(output_values, tuple):
                        output_values = tuple(v[mask] for v in output_values)
                    else:
                        output_values = output_values[mask]

                    if isinstance(target_values, tuple):
                        target_values = tuple(v[mask] for v in target_values)
                    else:
                        target_values = target_values[mask]

            # Compute loss term
            if isinstance(output_values, tuple):
                term_loss = loss_fn(*output_values, *target_values)
            else:
                term_loss = loss_fn(output_values, target_values)

            # Extract scalar if needed
            if term_loss.dim() > 0:
                loss_dict['loss'] += weight * term_loss.mean()
                loss_dict[term] = term_loss.detach().cpu().numpy()
            else:
                loss_dict['loss'] += weight * term_loss

        return loss_dict
