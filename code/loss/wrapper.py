import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from typing import Dict, Any
from utilities.instantiators import instantiate
import logging

logger = logging.getLogger(__name__)


def extract_values(
    source: Dict[str, torch.Tensor],
    keys: str | list[str]
) -> torch.Tensor | tuple:
    """
    Extract and optionally concatenate tensor(s) from source dict.

    Parameters
    ----------
    source : dict
        Source dictionary (outputs, batch, batch['target'], etc.)
    keys : str or list[str]
        Keys to extract. If multiple, returns tuple.

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
    elif isinstance(keys, (list, tuple, ListConfig)):
        keys = list(keys)
    else:
        keys = [keys]

    # Extract tensors
    tensors = []
    for key in keys:
        if key not in source:
            available = list(source.keys())
            raise KeyError(
                f"Key '{key}' not found. "
                f"Available keys: {available}"
            )
        tensors.append(source[key])

    # Single key: return directly
    if len(tensors) == 1:
        return tensors[0]
    # Multiple keys: return as tuple (allows loss to unpack)
    else:
        return tuple(tensors)


class LossTerm(torch.nn.Module):
    """
    Single loss term with automatic key extraction and optional masking.

    This class encapsulates one loss function with its configuration:
    - which output keys to extract
    - which target keys to extract
    - optional context masks to apply
    """

    def __init__(
        self,
        function: DictConfig | nn.Module,
        output_keys: str | list[str],
        target_keys: str | list[str],
        context_keys: str | list[str] | None = None,
    ) -> None:
        """
        Initialize LossTerm.

        Parameters
        ----------
        function : DictConfig | nn.Module
            Loss function configuration or instance
        output_keys : str | list[str]
            Which keys to extract from model outputs
        target_keys : str | list[str]
            Which keys to extract from batch['target']
        context_keys : str | list[str], optional
            Which context masks to extract and apply from batch['context']
        """

        # Class inheritance
        super().__init__()

        # Instantiate or validate loss function
        if isinstance(function, DictConfig):
            self.loss_term = instantiate(function)
        elif isinstance(function, nn.Module):
            self.loss_term = function
        else:
            raise TypeError(
                f"function must be DictConfig or nn.Module, got {type(function)}"
            )

        # Input, target, and context variables
        self.output_keys = instantiate(output_keys)
        self.target_keys = instantiate(target_keys)
        self.context_keys = instantiate(context_keys) if context_keys else None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute loss for this term.

        Parameters
        ----------
        outputs : dict[str, torch.Tensor]
            Model outputs
        batch : dict
            Full batch dict with 'target' and 'context' keys

        Returns
        -------
        torch.Tensor
            Loss tensor (can be scalar or multi-element)
        """

        # Extract output tensors
        output_values = extract_values(
            outputs,
            self.output_keys
        )

        # Extract target tensors
        target_values = extract_values(
            batch['target'],
            self.target_keys
        )

        # Apply optional context masks
        if self.context_keys:
            # Extract all masks
            masks = extract_values(
                batch['context'],
                self.context_keys
            )

            # Convert to tuple for uniform handling
            if not isinstance(masks, tuple):
                masks = (masks,)

            # Apply each mask sequentially
            for mask in masks:
                if isinstance(output_values, tuple):
                    output_values = tuple(v[mask] for v in output_values)
                else:
                    output_values = output_values[mask]

                if isinstance(target_values, tuple):
                    target_values = tuple(v[mask] for v in target_values)
                else:
                    target_values = target_values[mask]

        # Compute loss
        if isinstance(output_values, tuple):
            if isinstance(target_values, tuple):
                loss = self.loss_term(*output_values, *target_values)
            else:
                loss = self.loss_term(*output_values, target_values)
        else:
            if isinstance(target_values, tuple):
                loss = self.loss_term(output_values, *target_values)
            else:
                loss = self.loss_term(output_values, target_values)

        return loss

    def to(self, device):
        """
        Move module and loss term to device.

        Parameters
        ----------
        device : torch.device
            Target device

        Returns
        -------
        self : LossTerms
            Returns self for chaining
        """

        # Class inheritance
        super().to(device)
        # Move loss term
        self.loss_term = self.loss_term.to(device)
        return self


class LossTerms(torch.nn.Module):
    """
    Combines multiple LossTerm instances into a single loss with weighted sum.
    """
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """
        Initialize LossTerms.

        Parameters
        ----------
        config : DictConfig
            Mapping of term -> weight, instance.
            Each instance must have:
            - _target_: code.loss.LossTerm (or pre-instantiated LossTerm)
            - function: DictConfig or nn.Module
            - output_keys: str or list[str]
            - target_keys: str or list[str]
            - context_keys: str or list[str], optional
        """

        # Class inheritance
        super().__init__()

        # Instantiate each term
        self.loss_terms = nn.ModuleDict()
        self.weights = {}
        for term, config_term in config.items():
            # Extract weight
            self.weights[term] = float(config_term.get('weight', 1.0))

            # Register loss term instance
            if hasattr(config_term, 'instance'):
                # Register loss term
                if isinstance(config_term.instance, DictConfig):
                    self.loss_terms[term] = instantiate(config_term.instance)
                elif isinstance(config_term.instance, torch.nn.Module):
                    self.loss_terms[term] = config_term.instance
                else:
                    raise TypeError(
                        f"Term '{term}' instance must be DictConfig or nn.Module, "
                        f"got {type(config_term.instance)}"
                    )
            else:
                # If no 'instance', assume config_term is a dict with '_target_'
                if isinstance(config_term, DictConfig) and '_target_' in config_term:
                    self.loss_terms[term] = instantiate(config_term)
                elif isinstance(config_term, dict) and '_target_' in config_term:
                    self.loss_terms[term] = instantiate(DictConfig(config_term))
                else:
                    raise TypeError(
                        f"Term '{term}' must have 'instance' or be a dict with '_target_', "
                        f"got {type(config_term)}"
                    )
            logger.info(
                f"Registered loss term '{term}' (weight={self.weights[term]})"
            )

    def to(self, device):
        """
        Move module and all loss terms to device.

        Parameters
        ----------
        device : torch.device
            Target device

        Returns
        -------
        self : LossTerms
            Returns self for chaining
        """

        # Class inheritance
        super().to(device)

        # Move individual loss terms
        for term, loss_term in self.loss_terms.items():
            if hasattr(loss_term, 'to'):
                self.loss_terms[term] = loss_term.to(device)
        return self

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
            Model outputs
        batch : dict
            Full batch dict with 'target' and 'context' keys

        Returns
        -------
        dict[str, torch.Tensor]
            Loss dict with:
            - 'total': weighted sum of all terms
            - '{term}': raw loss for each term
            - '{term}_weighted': weighted loss for each term
        """

        loss_dict = {}
        total = torch.tensor(0.0, device=outputs[list(outputs.keys())[0]].device)

        # Compute each loss term
        for term, loss_term in self.loss_terms.items():
            # Get weight
            weight = self.weights[term]

            # Compute loss (LossTerm returns raw tensor)
            term_loss = loss_term(outputs, batch)

            # Extract scalar
            if term_loss.dim() > 0:
                term_scalar = term_loss.mean()
            else:
                term_scalar = term_loss

            # Store individual and weighted losses
            loss_dict[term] = term_loss.detach().cpu().numpy()
            total = total + weight * term_scalar

        # Add total to output
        loss_dict['total'] = total

        return loss_dict
