from config.setup import read_hydra_as_dict, setup_directories_from_hydra


seen_dependencies = set()
def find_file_paths(config, exts=('.npy', '.npz', '.pkl', '.txt', '.ckpt', '.csv', '.json'), exclude_keys=None,
                    track_seen_dependencies=False):
    """ Recursively find all file path strings in a nested config dict/list.

        Parameters
        ----------
        config: dict, list, or str. The configuration to search.
        exts: tuple of str. File extensions to look for.
        exclude_keys: set of str. Keys to exclude from the search.
        track_seen_dependencies: bool. If True, only return unique file paths across multiple calls.

        Returns
        -------
        paths: list of str. List of file paths found in the configuration.
    """

    # Initialize exclude_keys if not provided
    if exclude_keys is None:
        exclude_keys = set()

    # Initialize list to store file paths
    paths = []
    # Recursively search through the config
    if isinstance(config,dict):
        for k, v in config.items():
            if k in exclude_keys:
                continue
            paths.extend(find_file_paths(v, exts, exclude_keys, track_seen_dependencies))
    elif isinstance(config,list):
        for v in config:
            paths.extend(find_file_paths(v, exts, exclude_keys, track_seen_dependencies))
    elif isinstance(config,str):
        if any(config.endswith(ext) for ext in exts):
            if track_seen_dependencies:
                if config not in seen_dependencies:
                    seen_dependencies.add(config)
                    paths.append(config)
            else:
                paths.append(config)
    return sorted(paths)

#########################################################################################################
# CONFIGURATION
#########################################################################################################

# Hydra/Snakemake config
hydra_config_path = "../config"
hydra_config_name = "default"
hydra_experiment = "inverse_operator_train_000"
hydra_dependencies = False

# Create necessary directories
setup_directories_from_hydra(config_path=hydra_config_path, config_name=hydra_config_name,
                             overrides=f"+experiment={hydra_experiment}")

# Hydra configuration file
hydra_config = read_hydra_as_dict(config_path=hydra_config_path, config_name=hydra_config_name,
                                  overrides=f"+experiment={hydra_experiment}")
# Data configuration file (from Snakemake config file)
data_config = hydra_config.get("data", {})
prep_config = hydra_config.get("preparation", {})
loader_config = hydra_config.get("loader", {}).get("stage", {})
# Paths configuration file (from Snakemake config file)
paths_config = hydra_config["paths"]
# Callback configuration file (from Snakemake config file)
checkpoint_config = hydra_config["callbacks"]["model_checkpoint"]

breakpoint()