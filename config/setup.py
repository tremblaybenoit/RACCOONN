import os
from typing import Optional, Any
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
import argparse


def read_hydra_as_dict(config_path: str,
                       config_name: str,
                       version_base: Optional[str | None] = None,
                       overrides: Optional[str | list[str] | None] = None,
                       return_hydra_config: bool = True,
                       verbose: bool = False) -> Any:
    """ Read complete Hydra configuration and return as dictionary.

         Parameters
        ----------
        config_path: str. Directory containing hydra config file.
        config_name: str. Config filename.
        version_base: Optional[str]; default=None. Version number.
        overrides: Optional[Union[str, List[str]]]; default=None. Experiment overriding the hydra config file contents.
                   Can be a single string or a list of strings.
        return_hydra_config: bool; default=True. Whether to extract hydra config.
        verbose: bool; default=False. Flag to print config file contents.

         Returns
        -------
        Any
            Dictionary containing all configs.
    """

    # Convert single override to list
    if isinstance(overrides, str):
        overrides = [overrides]
    elif overrides is None:
        overrides = []

    # Manually initialize Hydra and compose the configuration
    with initialize(version_base=version_base, config_path=config_path):

        if overrides:
            config = compose(config_name=config_name,
                             overrides=overrides,
                             return_hydra_config=return_hydra_config)
        else:
            config = compose(config_name=config_name,
                             return_hydra_config=return_hydra_config)

        # Manually set missing mandatory values if missing
        if 'num' not in config.hydra.job:
            config.hydra.job.num = 1  # Set a default value for hydra.job.num
        if 'output_dir' not in config.hydra.runtime:
            config.hydra.runtime.output_dir = config.hydra.run.dir

        # Set hydra config
        HydraConfig.instance().set_config(config)

        # Resolve paths
        config_dict = OmegaConf.to_container(config, resolve=True)
        if config_dict is not None and 'hydra' in config_dict:
            del config_dict['hydra']

        # Print contents
        if verbose:
            print(OmegaConf.to_yaml(config_dict))

        return config_dict


def setup_directories_from_hydra(config_path: str,
                                 config_name: str,
                                 overrides: Optional[str | list[str] | None] = None,
                                 verbose: bool = False) -> None:
    """
    Reads the Hydra configuration, extracts the paths, and creates the necessary directories.

    Parameters
    ----------
    config_path : str
        Path to the Hydra configuration folder.
    config_name : str
        Name of the configuration file.
    overrides : Optional[str | list[str] | None], optional
        Experiment to override in the configuration.
        Can be a single string or a list of strings.
    verbose : bool, optional
        Prints the configuration if True.

    Returns
    -------
    None
    """

    # Read Hydra configuration as dictionary
    hydra_config = read_hydra_as_dict(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
        verbose=verbose
    )
    paths_config = hydra_config['paths']
    dirs = ['task_dir', 'output_dir', 'checkpoint_dir', 'log_dir', 'run_dir', 'data_dir']

    # Create directories based on the paths configuration
    for d in dirs:
        if d in paths_config:
            os.makedirs(paths_config[d], exist_ok=True)
        else:
            raise KeyError(f"Directory '{d}' not found in paths configuration.")

    return


if __name__ == "__main__":
    """ Read complete Hydra configuration and build directory dependencies.

        Parameters
        ----------
        config_path: str. Path to the Hydra configuration folder (default: ../config).
        config_name: str. Name of the configuration file (default: default).
        overrides: Hydra overrides as positional arguments (e.g., +experiment=inverse_default).

        Examples
        --------
        python -m config.setup +experiment=inverse_default
        python -m config.setup -config_path=./config +experiment=forward_default
        python -m config.setup -verbose +experiment=inverse_default
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', type=str, default="../config",
                        help='Path to configuration file containing all model hyperparameters.')
    parser.add_argument('-config_name', type=str, default="default",
                        help='Name of the configuration file containing all model hyperparameters.')
    parser.add_argument('-verbose', action='store_true',
                        help='Flag to print the configuration file contents.')
    parser.add_argument('overrides', nargs='*', default=[],
                        help='Hydra overrides (e.g., +experiment=inverse_default key=value).')
    args = parser.parse_args()

    # Setup directories from Hydra configuration
    setup_directories_from_hydra(args.config_path, args.config_name,
                                 overrides=args.overrides if args.overrides else None,
                                 verbose=args.verbose)
