from config.setup import read_hydra_as_dict, setup_directories_from_hydra, get_filenames
import sys


#########################################################################################################
# CONFIGURATION
#########################################################################################################

# Hydra/Snakemake config
config_path = config.get("-config-path", "../config")
config_name = config.get("-config-name", "default")
config_experiment = config.get("experiment", None)
config_experiment = f"+experiment={config_experiment}" if config_experiment is not None else ""

# Create necessary directories
setup_directories_from_hydra(config_path=config_path, config_name=config_name, overrides=config_experiment)
# Hydra configuration file
config_hydra = read_hydra_as_dict(config_path=config_path, config_name=config_name, overrides=config_experiment)

# Data, preprocessing, loader, and model configurations (from Snakemake config file)
config_data = config_hydra["data"]
config_preprocessing = config_hydra.get("preprocessing", {})
config_loader = config_hydra["loader"]["stage"]
config_model = config_hydra["model"]


#########################################################################################################
# RULES
#########################################################################################################

# Preprocessing rules (dynamic)
if config_preprocessing:
    # Loop over preprocessing operations
    for prep_type, config_prep in config_preprocessing.items():

        # If the preprocessing operation is executable
        if '_target_' in config_prep:

            # Statistics exceptions
            if prep_type in ('statistics', 'recast'):
                data_filenames = get_filenames(config_prep.get('input', {}),
                                               exclude_keys={'transformations', *config_prep.get('exclude', {})})
            else:
                data_filenames = get_filenames(config_prep.get('input', {}))

            rule:
                name: f"{prep_type}"
                input:
                    # Input data
                    data = data_filenames
                params:
                    # Hydra configuration
                    config_name = config_name,
                    experiment = config_experiment,
                    operation = prep_type
                output:
                    results = get_filenames(config_prep.get('output', {}), exclude_keys={'transformations'})
                shell:
                    """
                    python -m code.preprocessing.{params.operation} --config-name={params.config_name} {params.experiment}
                    """

        # Else loop over each preprocessing step in the current operation
        else:

            for step_name, config_step in config_prep.items():

                if isinstance(config_step, dict) and '_target_' in config_step:


                    # Statistics exceptions
                    if prep_type in ('statistics', 'recast'):
                        data_filenames = get_filenames(config_step.get('input', {}),
                                                       exclude_keys={'transformations', *config_step.get('exclude', {})})
                    else:
                        data_filenames = get_filenames(config_step.get('input', {}))

                    # Delete overrides for other steps in the same preprocessing operation
                    other_steps = [s for s in config_prep.keys() if s != step_name]
                    delete_overrides = " ".join([f"~preprocessing.{prep_type}.{s}" for s in other_steps])

                    rule:
                        name: f"{prep_type}_{step_name}"
                        input:
                            # Input data
                            data = data_filenames
                        params:
                            # Hydra configuration
                            config_name = config_name,
                            experiment = config_experiment,
                            overrides = delete_overrides,
                            operation = prep_type
                        output:
                            results = get_filenames(config_step.get('output', {}), exclude_keys={'transformations'})
                        shell:
                            """
                            python -m code.preprocessing.{params.operation} --config-name={params.config_name} \
                            {params.experiment} {params.overrides}
                            """


# Training rule
if 'train' in config_loader:
    rule:
        name: "train"
        input:
            # Input data
            data = set(get_filenames(config_loader['train'], exclude_keys={'output', 'latent'}) + get_filenames(config_loader['valid'], exclude_keys={'output', 'latent'})),
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
            results = get_filenames(config_loader['test'], exclude_keys={'input', 'context', 'target', 'transformations'})
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
            checkpoint = config_model['ckpt_path'],
            # Test results
            results = get_filenames(config_loader['test'], exclude_keys={'input', 'context', 'target', 'transformations'})
        params:
            # Hydra configuration
            config_name = config_name,
            experiment = config_experiment
        output:
            # Output results
            results = get_filenames(config_loader['predict'], exclude_keys={'input', 'context', 'target', 'transformations'})
        shell:
            """
            python -m code.predict --config-name={params.config_name} {params.experiment}
            """