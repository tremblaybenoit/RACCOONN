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
prep_config = hydra_config["data"]["preparation"]
train_config = hydra_config["data"]["loader"]["train"]
valid_config = hydra_config["data"]["loader"]["valid"]
test_config = hydra_config["data"]["loader"]["test"]
pred_config = hydra_config["data"]["loader"]["pred"]
# Paths configuration file (from Snakemake config file)
paths_config = hydra_config["paths"]
# Callback configuration file (from Snakemake config file)
checkpoint_config = hydra_config["callbacks"]["model_checkpoint"]

#########################################################################################################
# TARGET
#########################################################################################################

# Inverse model
rule inverse:
    input:
        checkpoint = f"{checkpoint_config['dirpath']}/{checkpoint_config['filename']}.ckpt"

#########################################################################################################
# RULES
#########################################################################################################

# Data download
rule data_download:
    params:
        config_name=hydra_config_name,
        experiment=f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
    output:
        [vars_config[var]['path'] for var in vars_config]
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
            [prep_config['statistics']['input']['vars'][var]['path'] for var in prep_config['statistics']['input']['vars']]
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
if 'covariance_model' in prep_config or 'covariance_observation' in prep_config:
    rule covariance_matrices:
        input:
            prof = prep_config['statistics']['input']['vars']['prof']['load']['path'],
            hofx = prep_config['statistics']['input']['vars']['hofx']['load']['path'] if 'covariance_observation' in prep_config else None,
            stats = prep_config['statistics']['output']['path']
        params:
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
        output:
            out_model = prep_config['covariance_model']['output']['path'] if 'covariance_model' in prep_config else None,
            out_observation = prep_config['covariance_observation']['output']['path'] if 'covariance_observation' in prep_config else None,
        shell:
            """
            python -m inverse.data.preparation.covariance \
            --config-name={params.config_name} \
            {params.experiment}
            """

# Training step
rule train:
    input:
        [train_config['coords'][var]['path'] if 'path' in train_config['coords'][var] else None for var in train_config['coords']],
        [train_config['obs'][var]['path'] if 'path' in train_config['obs'][var] else None for var in train_config['obs']],
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
