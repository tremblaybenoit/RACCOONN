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
vars_config = hydra_config["data"]["vars"]
prep_config = hydra_config["preparation"]
stage_config = hydra_config["loader"]["stage"]
# Paths configuration file (from Snakemake config file)
paths_config = hydra_config["paths"]
# Callback configuration file (from Snakemake config file)
checkpoint_config = hydra_config["callbacks"]["model_checkpoint"]

#########################################################################################################
# TARGET
#########################################################################################################

# Inverse model
rule model_train:
    input:
        checkpoint = f"{checkpoint_config['dirpath']}/{checkpoint_config['filename']}.ckpt",
        test_results = [stage_config['test']['results'][output]['path']
                        for output in stage_config['test']['results']] if 'test' in stage_config else ''

# TODO: Untested.
rule model_predict:
    input:
        checkpoint = f"{checkpoint_config['dirpath']}/{checkpoint_config['filename']}.ckpt",
        pred_results = [stage_config['predict']['results'][output]['path']
                        for output in stage_config['predict']['results']] if 'predict' in stage_config else ''

#########################################################################################################
# RULES
#########################################################################################################

# Data download
rule download:
    params:
        config_name=hydra_config_name,
        experiment=f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
    output:
        [vars_config[var]['path'] for var in vars_config if 'path' in vars_config[var]]
    shell:
        """
        python -m forward.utilities.download \
        --config-name={params.config_name} \
        {params.experiment}
        """

# Preparation: Statistics
if 'statistics' in prep_config:
    rule statistics:
        input:
            [prep_config['statistics']['input']['vars'][var]['path'] if 'path' in prep_config['statistics']['input']['vars'][var] else None
             for var in prep_config['statistics']['input']['vars']]
        params:
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            out = prep_config['statistics']['output']['path']
        shell:
            """
            python -m inverse.data.preparation.statistics \
            --config-name={params.config_name} \
            {params.experiment}
            """

# Preparation: Covariance matrices
if 'covariance_model' in prep_config or 'covariance_obs' in prep_config:
    rule error_covariance:
        input:
            [prep_config['statistics']['input']['vars'][var]['path'] for var in (['prof', 'hofx'] if 'covariance_obs' in prep_config else [])],
            stats = prep_config['statistics']['output']['path']
        params:
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            out_model = prep_config['covariance_model']['output']['path'] if 'covariance_model' in prep_config else None,
            out_observation = prep_config['covariance_obs']['output']['path'] if 'covariance_obs' in prep_config else None,
        shell:
            """
            python -m inverse.data.preparation.covariance \
            --config-name={params.config_name} \
            {params.experiment}
            """

# Training step
if 'train' in stage_config:
    rule train:
        input:
            [stage_config['train']['coords'][var]['path'] for var in stage_config['train']['coords'] if 'path' in stage_config['train']['coords'][var]],
            [stage_config['train']['obs'][var]['path'] for var in stage_config['train']['obs'] if 'path' in stage_config['train']['obs'][var]],
            [stage_config['valid']['coords'][var]['path'] for var in stage_config['valid']['coords'] if 'path' in stage_config['valid']['coords'][var]],
            [stage_config['valid']['obs'][var]['path'] for var in stage_config['valid']['obs'] if 'path' in stage_config['valid']['obs'][var]],
            [prep_config[step_name]['output']['path'] for step_name in prep_config]
        params:
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
        shell:
            """
            python -m inverse.train \
            --config-name={params.config_name} \
            {params.experiment}
            """

# Test step
if 'test' in stage_config:
    rule test:
        input:
            [stage_config['test']['coords'][var]['path'] for var in stage_config['test']['coords'] if 'path' in stage_config['test']['coords'][var]],
            [stage_config['test']['obs'][var]['path'] for var in stage_config['test']['obs'] if 'path' in stage_config['test']['obs'][var]],
            [prep_config[step_name]['output']['path'] for step_name in prep_config],
            checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
        params:
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            results = [stage_config['test']['results'][output]['path']
                       for output in stage_config['test']['results']] if 'test' in stage_config else ''
        shell:
            """
            python -m inverse.test \
            --config-name={params.config_name} \
            {params.experiment}
            """

# Prediction step
if 'predict' in stage_config:
    rule predict:
        input:
            [stage_config['pred']['coords'][var]['path'] for var in stage_config['pred']['coords'] if 'path' in stage_config['pred']['coords'][var]],
            statistics = prep_config['statistics']['output']['path'],
            checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
        params:
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            results = [stage_config['test']['results'][output]['path']
                       for output in stage_config['test']['results']] if 'test' in stage_config else ''
        shell:
            """
            python -m inverse.predict \
            --config-name={params.config_name} \
            {params.experiment}
            """