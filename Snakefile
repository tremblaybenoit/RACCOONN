from config.setup import read_hydra_as_dict, setup_directories_from_hydra


#########################################################################################################
# CONFIGURATION
#########################################################################################################

# Hydra/Snakemake config
hydra_config_path = config.get("hydra-config-path", "../config")
hydra_config_name = config.get("hydra-config-name", "default")
hydra_experiment = config.get("hydra-experiment", None)
hydra_dependencies = config.get("hydra-dependencies", False)

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

#########################################################################################################
# TARGET
#########################################################################################################

# Train model and get test results (if applicable)
rule model_train:
    input:
        checkpoint = f"{checkpoint_config['dirpath']}/{checkpoint_config['filename']}.ckpt",
        test_out = [loader_config['test']['results'][output]['load']['path']
                    for output in loader_config['test']['results']] if 'test' in loader_config else ''

# TODO: Untested.
# Make predictions with the trained model
if 'predict' in loader_config:
    rule model_predict:
        input:
            # Model checkpoint
            checkpoint = f"{checkpoint_config['dirpath']}/{checkpoint_config['filename']}.ckpt",
            # Prediction results
            pred_out = [loader_config['predict']['results'][output]['load']['path']
                        for output in loader_config['predict']['results']]

#########################################################################################################
# HELPER FUNCTIONS
#########################################################################################################

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
# RULES
#########################################################################################################

# Data download
rule data:
    params:
        # Hydra configuration
        config_name = hydra_config_name,
        experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
    output:
        # Downloaded data
        data = set([path for stage_config in data_config['stage'].values() for var in stage_config['vars'].values()
                    for path in find_file_paths(var['load'], exclude_keys='normalization')])
    shell:
        """
        python -m forward.utilities.download \
        --config-name={params.config_name} \
        {params.experiment}
        """

# Preparation: All steps in prep_config
for prep_type, step_dict in prep_config.items():
    # Set module and exclude_keys based on prep_type
    if prep_type == 'statistics':
        module_to_run = 'inverse.data.preparation.statistics'
        exclude_keys = 'normalization'
    elif prep_type == 'covariance':
        module_to_run = 'inverse.data.preparation.covariance'
        exclude_keys = None
    else:
        continue  # Skip unknown prep types or handle as needed

    for step, step_config in step_dict.items():
        rule:
            name: f"{prep_type}_{step}"
            input:
                data = set([path for vars_config in step_config['input'].values()
                            for path in find_file_paths(vars_config, exclude_keys=exclude_keys, track_seen_dependencies=hydra_dependencies)])
            params:
                config_name = hydra_config_name,
                experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
            output:
                out = step_config['output']['path']
            shell:
                f"""
                python -m {module_to_run} \
                --config-name={{params.config_name}} \
                {{params.experiment}}
                """

# Training step
if 'train' in loader_config:
    rule train:
        input:
            # Input coordinates and observations
            data = set([path for stage in ['train', 'valid'] for vars_config in loader_config[stage].values()
                        for path in find_file_paths(vars_config, track_seen_dependencies=hydra_dependencies)]),
            # Data statistics and other preparation outputs
            preparation = set([path for step in prep_config for step_config in prep_config[step].values()
                               for path in find_file_paths(step_config['output'], track_seen_dependencies=hydra_dependencies)]),
        params:
            # Hydra configuration
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            # Model checkpoint
            checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
        shell:
            """
            python -m inverse.train \
            --config-name={params.config_name} \
            {params.experiment}
            """

# Test step
if 'test' in loader_config:
    rule test:
        input:
            # Input coordinates and observations
            data = set([path for path in find_file_paths(loader_config['test'], exclude_keys='results',
                track_seen_dependencies=hydra_dependencies)]),
            # Data statistics and other preparation outputs
            preparation = set([path for step in prep_config for step_config in prep_config[step].values()
                               for path in find_file_paths(step_config['output'], track_seen_dependencies=hydra_dependencies)]),
            # Model checkpoint
            checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
        params:
            # Hydra configuration
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            # Output results
            test_out = [var['load']['path'] for var in loader_config['test']['results'].values()]
        shell:
            """
            python -m inverse.test \
            --config-name={params.config_name} \
            {params.experiment}
            """

# Prediction step
# TODO: Untested.
if 'predict' in loader_config:
    rule predict:
        input:
            # Input coordinates
            data = set([path for path in find_file_paths(loader_config['predict'], exclude_keys='results',
                track_seen_dependencies=hydra_dependencies)]),
            # Preparation
            preparation = set([path for step in prep_config for step_config in prep_config[step].values()
                               for path in find_file_paths(step_config['output'], track_seen_dependencies=hydra_dependencies)]),
            # Model checkpoint
            checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
        params:
            # Hydra configuration
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            # Output results
            predict_out = [var['load']['path'] for var in loader_config['predict']['results'].values()]
        shell:
            """
            python -m inverse.predict \
            --config-name={params.config_name} \
            {params.experiment}
            """