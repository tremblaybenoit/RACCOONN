import os
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
import argparse


def get_key_in_dict(key, dictionary):
    """
    Extracts a key from a dictionary.

    Parameters
    ----------
    key : str
        Key to extract.
    dictionary : dict
        Dictionary to extract from.

    Returns
    -------
    dict
        Dictionary containing the key.
    """

    # Check if key is in dictionary
    if check_key_in_dict(key, dictionary):
        return dictionary[key]
    # Raise an error if key is not in dictionary
    else:
        raise KeyError(f"Key '{key}' not found in dictionary.")


# Check if key is in dictionary
def check_key_in_dict(key, dictionary):
    """
    Check if a key is in a dictionary.

    Parameters
    ----------
    key : str
        Key to check.
    dictionary : dict
        Dictionary to check.

    Returns
    -------
    bool
        True if the key is in the dictionary, False otherwise.
    """
    return key in dictionary


def read_hydra_as_dict(config_path, config_name, version_base=None, overrides=None, return_hydra_config=True,
                       verbose=False):
    """ Read complete Hydra configuration and return as dictionnary.

        Parameters
        ----------
        config_path: str. Directory containing hydra config file.
        config_name: str. Config filename.
        version_base: float; default=None. Version number.
        overrides: str; default=None. Experiment overriding the hydra config file contents.
        return_hydra_config: bool; default=True. Whether to extract hydra config.
        verbose: bool; default=False. Flag to print config file contents.

        Returns
        -------
        config_dict: Dictionnary containing all configs.
    """

    # Manually initialize Hydra and compose the configuration
    with initialize(version_base=version_base, config_path=config_path):

        if overrides is not None:
            config = compose(config_name=config_name,
                             overrides=[overrides],
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
        if check_key_in_dict('hydra', config_dict):
            del config_dict['hydra']

        # Print contents
        if verbose:
            print(OmegaConf.to_yaml(config_dict))

        return config_dict


def setup_directories_from_hydra(config_path, config_name, overrides=None, verbose=False):
    """
    Reads the Hydra configuration, extracts the paths, and creates the necessary directories.

    Parameters
    ----------
    config_path : str. Path to the Hydra configuration folder.
    config_name : str. Name of the configuration file.
    overrides : str, optional. Experiment to override in the configuration.
    verbose : bool, optional. Prints the configuration if True.

    Returns
    -------
    None.
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
    for dir in dirs:
        if dir in paths_config:
            os.makedirs(paths_config[dir], exist_ok=True)
        else:
            raise KeyError(f"Directory '{dir}' not found in paths configuration.")

    return


if __name__ == "__main__":
    """ Read complete Hydra configuration and build directory dependencies.

        Parameters
        ----------
        config_path: str. Directory containing hydra config file.
        config_name: str. Config filename.
        experiment: str; default=None. Experiment overriding the hydra config file contents.

        Returns
        -------
        config_as_dict: Dictionnary containing all configs.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', type=str, default="../config",
                        help='Path to configuration file containing all model hyperparameters.')
    parser.add_argument('-config_name', type=str, default="default",
                        help='Name of the configuration file containing all model hyperparameters.')
    parser.add_argument('-overrides', type=str, default=None,
                        help='Name of the experiment that overrides the main hydra configuration.')
    parser.add_argument('-verbose', type=bool, default=False,
                        help='Flag to print the configuration file contents.')
    args = parser.parse_args()

    # Setup directories from Hydra configuration
    setup_directories_from_hydra(args.config_path, args.config_name, args.overrides, verbose=args.verbose)
