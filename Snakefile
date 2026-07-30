from config.setup import read_hydra_as_dict, setup_directories_from_hydra
from typing import Any, Dict, List, Optional, Set, Tuple, Union


#########################################################################################################
# CONFIGURATION
#########################################################################################################

# Hydra/Snakemake config
config_path = config.get("-config-path", "../config")
config_name = config.get("-config-name", "default")
config_experiment = config.get("+experiment", None)
config_experiment = f"+experiment={config_experiment}" if config_experiment is not None else ""

# Create necessary directories
setup_directories_from_hydra(config_path=config_path, config_name=config_name, overrides=config_experiment)
# Hydra configuration file
config_hydra = read_hydra_as_dict(config_path=config_path, config_name=config_name, overrides=config_experiment)

# Data, preprocessing, loader, and model configurations (from Snakemake config file)
config_data = config_hydra["data"]
config_preprocessing = config_hydra.get("preprocessing", {})
config_loader = config_hydra["loader"]
config_model = config_hydra["model"]

#########################################################################################################
# HELPER FUNCTIONS
#########################################################################################################

def get_filenames(
    config: Union[Dict[str, Any], List[Any], Tuple[Any, ...], str],
    exts: Tuple[str, ...] = ('.npy', '.npz', '.pkl', '.txt', '.ckpt', '.csv', '.json', '.nc'),
    exclude_keys: Optional[Set[str]] = None
) -> List[str]:
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
    def _recursively_find(obj: Any, exts: Tuple[str, ...], exclude_keys: Set[str], paths: Set[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in exclude_keys:
                    _recursively_find(v, exts, exclude_keys, paths)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _recursively_find(item, exts, exclude_keys, paths)
        elif isinstance(obj, str):
            # Use tuple endswith
            if obj.endswith(exts):
                paths.add(obj)  # Set automatically prevents duplicates

    # Accumulate in set (no duplicates), sort once at the end
    paths = set()
    _recursively_find(config, exts, exclude_keys, paths)
    return sorted(paths)


#########################################################################################################
# RULES
#########################################################################################################

# Preprocessing rules (dynamic)
if config_preprocessing:
    # Loop over preprocessing operations
    for prep_type, config_prep in config_preprocessing.items():

        # If the preprocessing operation is executable
        if '_target_' in config_prep:

            rule:
                name: f"{prep_type}"
                input:
                    # Input data
                    data = get_filenames(config_prep.get('input', {}))
                params:
                    # Hydra configuration
                    config_name = config_name,
                    experiment = config_experiment
                output:
                    results = get_filenames(config_prep.get('output', {}))
                shell:
                    f"""
                    python -m code.preprocessing.{prep_type} --config-name={params.config_name} {params.experiment}
                    """

        # Else loop over each preprocessing step in the current operation
        else:

            for step_name, config_step in config_prep.items():

                if isinstance(config_step, dict) and '_target_' in config_step:

                    rule:
                        name: f"{prep_type}_{step_name}"
                        input:
                            # Input data
                            data = get_filenames(config_step.get('input', {}))
                        params:
                            # Hydra configuration
                            config_name = config_name,
                            experiment = config_experiment
                        output:
                            results = get_filenames(config_step.get('output', {}))
                        shell:
                            f"""
                            python -m code.preprocessing.{prep_type} --config-name={params.config_name} {params.experiment}
                            """


# Training rule
if 'train' in config_loader:
    rule:
        name: "train"
        input:
            # Input data
            data = set(get_filenames(config_loader['train']) + get_filenames(config_loader['valid'])),
            # Model dependencies (if any)
            model = get_filenames(config_model, exclude_keys={'ckpt_path'})
        params:
            # Hydra configuration
            config_name = config_name,
            experiment = config_experiment
        output:
            # Model checkpoint
            checkpoint = config_model['ckpt_path']
        shell:
            """
            python -m code.train --config-name={params.config_name} {params.experiment}
            """

# Test rule
if 'test' in config_loader:
    rule:
        name: "test"
        input:
            # Input data
            data = get_filenames(config_loader['test'], exclude_keys={'output', 'latent'}),
            # Model dependencies (if any)
            model = get_filenames(config_model)
        params:
            # Hydra configuration
            config_name = config_name,
            experiment = config_experiment
        output:
            # Output results
            results = get_filenames(config_loader['test'], exclude_keys={'input', 'context', 'target'})
        shell:
            """
            python -m code.test --config-name={params.config_name} {params.experiment}
            """

# Prediction rule
if 'predict' in config_loader:
    rule:
        name: "predict"
        input:
            # Input data
            data = get_filenames(config_loader['predict'], exclude_keys={'output', 'latent'}),
            # Model checkpoint
            checkpoint = config_model['ckpt_path']
        params:
            # Hydra configuration
            config_name = config_name,
            experiment = config_experiment
        output:
            # Output results
            results = get_filenames(config_loader['predict'], exclude_keys={'input', 'context', 'target'})
        shell:
            """
            python -m code.predict --config-name={params.config_name} {params.experiment}
            """