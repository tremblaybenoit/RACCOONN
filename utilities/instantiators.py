from typing import List, Union, Callable, Any
import hydra
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
from functools import partial
import logging


# Initialize logger
logger = logging.getLogger(__name__)


def instantiate(config: Any, **kwargs):
    """ Instantiate data module or module from config.

        Parameters
        ----------
        config: A DictConfig object containing configurations, or a callable class/function.
        kwargs: Optional keyword arguments. This includes:
                _partial_ for partial instantiation;
                _recursive_ for recursive instantiation.

        Returns
        -------
        Instance of module.
    """

    # If DictConfig object is provided
    if isinstance(config, DictConfig):

        # Instantiate module
        return hydra.utils.instantiate(config, **kwargs)

    # If callable object is provided
    elif callable(config) and not isinstance(config, partial):

        # Check for partial instantiation
        partial_flag = kwargs.pop("_partial_", False)

        # If partial, return partially instantiated object
        if partial_flag:
            return partial(config, **kwargs)

        # If not partial, return the callable object
        return config(**kwargs)
    else:
        return config


def instantiate_list(config_list: DictConfig, obj_type: str, **kwargs) -> List[Union[Callback, Logger]]:
    """ Instantiates callback(s) from config.

        Parameters
        ----------
        config_list: A DictConfig object containing callback configuration(s).
        obj_type: Type of objects to instantiate ("callback" or "logger").
        kwargs: Optional keyword arguments.

        Returns
        -------
        config_obj: A list of instantiated objects (callback(s) or logger(s)).
    """

    # Initialize empty list of objects (as there can be more than one)
    config_obj: List[Union[Callback, Logger]] = []

    # No objects found
    if not config_list:
        # log.warning(f"No {obj_type} config found! Skipping...")
        return config_obj

    # Config provided, but in the wrong format
    if not isinstance(config_list, DictConfig):
        log.error(f"{obj_type} config must be a DictConfig!")
        raise TypeError(f"{obj_type} config must be a DictConfig!")

    # If multiple objects are provided
    for _, config in config_list.items():
        # Instantiate individual callbacks
        if isinstance(config, DictConfig) and "_target_" in config:
            # Log
            # log.info(f"Instantiating {obj_type} <{config._target_}>")
            # Add to list
            config_obj.append(hydra.utils.instantiate(config, **kwargs))

    return config_obj


def instantiate_callbacks(callback_config: DictConfig, **kwargs) -> List[Callback]:
    """ Instantiates callback(s) from config.

        Parameters
        ----------
        callback_config: A DictConfig object containing callback configuration(s).
        kwargs: Optional keyword arguments.

        Returns
        -------
        A list of instantiated callbacks.
    """
    return instantiate_list(callback_config, "callbacks", **kwargs)


def instantiate_loggers(logger_config: DictConfig, **kwargs) -> List[Logger]:
    """ Instantiates logger(s) from config.

        Parameters
        ----------
        logger_config: A DictConfig object containing callback configuration(s).
        kwargs: Optional keyword arguments.

        Returns
        -------
        A list of instantiated loggers.
    """
    return instantiate_list(logger_config, "logger", **kwargs)
