from config.setup import read_hydra_as_dict, setup_directories_from_hydra


#########################################################################################################
# CONFIGURATION
#########################################################################################################

# Hydra/Snakemake config
hydra_config_path = config.get("hydra-config-path", "../config")
hydra_config_name = config.get("hydra-config-name", "default")
hydra_experiment = config.get("hydra-experiment", None)

# Create necessary directories
setup_directories_from_hydra(config_path=hydra_config_path, config_name=hydra_config_name,
                             overrides=f"+experiment={hydra_experiment}")

# Hydra configuration file
hydra_config = read_hydra_as_dict(config_path=hydra_config_path, config_name=hydra_config_name,
                                  overrides=f"+experiment={hydra_experiment}")
# Data configuration file (from Snakemake config file)
data_config = hydra_config.get("data", {})
vars_config = data_config.get("vars", {})
prep_config = hydra_config.get("preparation", {})
stage_config = hydra_config.get("loader", {}).get("stage", {})
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
        test_out = [stage_config['test']['results'][output]['path']
                    for output in stage_config['test']['results']] if 'test' in stage_config else ''

# TODO: Untested.
# Make predictions with the trained model
if 'predict' in stage_config:
    rule model_predict:
        input:
            # Model checkpoint
            checkpoint = f"{checkpoint_config['dirpath']}/{checkpoint_config['filename']}.ckpt",
            # Prediction results
            pred_out = [stage_config['predict']['results'][output]['path']
                        for output in stage_config['predict']['results']]

#########################################################################################################
# RULES
#########################################################################################################

# Data download
rule data:
    params:
        # Hydra configuration
        config_name=hydra_config_name,
        experiment=f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
    output:
        # Downloaded data
        data = [var['path'] for var in vars_config.values() if 'path' in var]
    shell:
        """
        python -m forward.utilities.download \
        --config-name={params.config_name} \
        {params.experiment}
        """

# Preparation: Statistics
if 'statistics' in prep_config:
    # Multiple steps (e.g., for different data sources)
    for step, statistics_config in prep_config['statistics'].items():
        rule:
            name: f"statistics_{step}"
            input:
                # Coordinates and observations
                data = [var['path'] for var in statistics_config['input']['vars'].values() if 'path' in var]
            params:
                # Hydra configuration
                config_name = hydra_config_name,
                experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
            output:
                # Data statistics
                out = statistics_config['output']['path']
            shell:
                """
                python -m inverse.data.preparation.statistics \
                --config-name={params.config_name} \
                {params.experiment}
                """

# Preparation: Covariance matrices
if 'covariance' in prep_config:

    def covariance_inputs(cov_config):
        """ Helper function to get input files for covariance calculation

            Parameters:
            cov_config (dict): Covariance configuration dictionary.

            Returns:
            list: List of input file paths.
        """
        # Initialize input list
        inputs = []
        # Add variable files and statistics file if present
        if 'input' in cov_config and 'vars' in cov_config['input']:
            # Variables
            for var in cov_config['input']['vars'].values():
                # Only add if 'path' key exists
                if 'path' in var:
                    # Add variable file
                    inputs.append(var['path'])
                    # Check normalization file
                    if 'normalization' in var and 'stats' in var['normalization']:
                        # Add normalization file if it is among the arguments of the normalization function
                        inputs.append(var['normalization']['stats']['_args_'][0])
                else:
                    # If the variable is derived from others, add those dependencies
                    for var_dependency in var.get('load', {}):
                        # Add dependency file if 'path' key exists
                        if 'path' in var_dependency:
                            inputs.append(var_dependency['path'])
                        # Check normalization file
                        if 'normalization' in var_dependency and 'stats' in var_dependency['normalization']:
                            # Add normalization file if it is among the arguments of the normalization function
                            inputs.append(var_dependency['normalization']['stats']['_args_'][0])

        return inputs

    # Multiple steps (e.g., for model and observations)
    for step, covariance_config in prep_config['covariance'].items():
        rule:
            name: f"covariance_{step}"
            input:
                # Data statistics
                # data = covariance_inputs(covariance_config)
                statistics = [prep_config['statistics'][var]['output']['path'] for var in prep_config['statistics']],
            params:
                # Hydra configuration
                config_name = hydra_config_name,
                experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
            output:
                # Output covariance matrices
                out = covariance_config['output']['path']
            shell:
                """
                python -m inverse.data.preparation.covariance \
                --config-name={params.config_name} \
                {params.experiment}
                """

# Training step
if 'train' in stage_config:
    train_config = stage_config['train']
    valid_config = stage_config['valid']
    rule train:
        input:
            # Input coordinates and observations
            # train_coords = [var['path'] for var in train_config['coords'].values() if 'path' in var],
            # train_obs = [var['path'] for var in train_config['obs'].values() if 'path' in var],
            # valid_coords = [var['path'] for var in valid_config['coords'].values() if 'path' in var],
            # valid_obs = [var['path'] for var in valid_config['obs'].values() if 'path' in var],
            # Data statistics and other preparation outputs
            preparation = [prep_config[step][substep]['output']['path'] for step in prep_config
                           for substep in prep_config[step]],
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
if 'test' in stage_config:
    test_config = stage_config['test']
    rule test:
        input:
            # Input coordinates and observations
            # test_coords = [var['path'] for var in test_config['coords'].values() if 'path' in var],
            # test_obs = [var['path'] for var in test_config['obs'].values() if 'path' in var],
            # Data statistics and other preparation outputs
            # preparation = [prep_config[step][substep]['output']['path'] for step in prep_config
            #                for substep in prep_config[step]],
            # Model checkpoint
            checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
        params:
            # Hydra configuration
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            # Output results
            test_out = [var['path'] for var in test_config['results'].values()]
        shell:
            """
            python -m inverse.test \
            --config-name={params.config_name} \
            {params.experiment}
            """

# Prediction step
if 'predict' in stage_config:
    predict_config = stage_config['predict']
    rule predict:
        input:
            # Input coordinates
            predict_coords = [var['path'] for var in predict_config['coords'].values() if 'path' in var],
            # Data statistics
            statistics = prep_config['statistics']['data']['output']['path'],
            # Model checkpoint
            checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
        params:
            # Hydra configuration
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            # Output results
            predict_out = [var['path'] for var in predict_config['results'].values()]
        shell:
            """
            python -m inverse.predict \
            --config-name={params.config_name} \
            {params.experiment}
            """