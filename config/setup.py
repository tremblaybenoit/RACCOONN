import os
from typing import Optional, Any
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
import argparse


def get_filenames(
    config: dict[str, Any] | list[Any] | tuple[Any, ...] | str,
    exts: tuple[str, ...] = ('.npy', '.npz', '.pkl', '.txt', '.ckpt', '.csv', '.json', '.nc'),
    exclude_keys: set[str] | None = None
) -> list[str]:
    """ Recursively find all file path strings in a nested config dict/list.

        Parameters
        ----------
        config: dict, list, tuple, or str. The configuration to search.
        exts: tuple of str. File extensions to look for.
        exclude_keys: set of str. Keys to exclude from the search.

        Returns
        -------
        list of str: Sorted list of unique file paths found in the configuration.
    """

    # Initialize exclude_keys if not provided
    if exclude_keys is None:
        exclude_keys = set()

    # Helper function for recursive search (accumulate in set to avoid duplicates)
    def _recursively_find(obj: Any, exts: tuple[str, ...], exclude_keys: set[str], paths: set[str]) -> None:
        """ Recursively search for file paths in the given object.

            Parameters
            ----------
            obj: Any. The object to search (can be dict, list, tuple, or str).
            exts: tuple of str. File extensions to look for.
            exclude_keys: set of str. Keys to exclude from the search.
            paths: set of str. Accumulator for found file paths.

            Returns
            -------
            None.
        """

        # Parse dictionary
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in exclude_keys:
                    _recursively_find(v, exts, exclude_keys, paths)
        # Parse list/tuple
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _recursively_find(item, exts, exclude_keys, paths)
        # Parse string
        elif isinstance(obj, str):
            # Use tuple endswith
            if obj.endswith(exts):
                paths.add(obj)  # Set automatically prevents duplicates

    # Accumulate in set (no duplicates), sort once at the end
    paths = set()
    _recursively_find(config, exts, exclude_keys, paths)
    return sorted(paths)


def get_directories(file_paths: list[str]) -> list[str]:
    """ Extract unique parent directories from a list of file paths.

        Parameters
        ----------
        file_paths: list[str]. List of file paths (output from get_filenames).

        Returns
        -------
        list[str]: Sorted list of unique parent directories.
    """
    directories = set()
    for file_path in file_paths:
        # Get parent directory
        dir_path = os.path.dirname(file_path)
        # Only add if not empty (file_path was not already a directory)
        if dir_path:
            directories.add(dir_path)

    return sorted(directories)


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

    # Create paths from paths_config
    paths_config = hydra_config['paths']
    dirs = ['task_dir', 'output_dir', 'checkpoint_dir', 'log_dir', 'run_dir', 'data_dir']

    # Create directories based on the paths configuration
    for d in dirs:
        if d in paths_config:
            os.makedirs(paths_config[d], exist_ok=True)
        else:
            raise KeyError(f"Directory '{d}' not found in paths configuration.")

    # Extract and create directories from config file paths
    # config_filenames = get_filenames(hydra_config)
    # config_directories = get_directories(config_filenames)
    # for d in config_directories:
    #     os.makedirs(d, exist_ok=True)

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
