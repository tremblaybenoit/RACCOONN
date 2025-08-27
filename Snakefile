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
data_config = hydra_config["data"]
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
# JOBS
#########################################################################################################

# Loop over preparation steps
# for step_nb, step_name in enumerate(data_config['preparation']):
#     rule:
#         name: f"{step_name}"
#         input:
#             get_path_from_config(radiance_config['preparation'][step_name]['input'], key='path')
#         output:
#             get_path_from_config(radiance_config['preparation'][step_name]['output'], key='path', flag_directories=True)
#         params:
#             config_name = hydra_config_name,
#             experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else "",
#             target = step_name,
#             condition = f"~data.dataset.state.preparation.{step_name}" if check_key_in_dict(step_name, state_config['preparation']) else "",
#             log_path = paths_config['log_dir'],
#             run_path = paths_config['run_dir']
#         shell:
#             """
#             python -c "import os; os.makedirs('{params.log_path}', exist_ok=True)" && \
#             python -c "import os; os.makedirs('{params.run_path}', exist_ok=True)" && \
#             python -m retrieval.data.{params.target} \
#             --config-name={params.config_name} \
#             {params.experiment} \
#             {params.condition}
#             """

# Preparation steps
for step_nb, step_name in enumerate(data_config['preparation']):
    rule:
        name: step_name
        input:
            lat = data_config['loader']['coordinates']['lat']['load']['_args_'][0],
            lon = data_config['loader']['coordinates']['lon']['load']['_args_'][0],
            scans = data_config['loader']['coordinates']['scans']['load']['_args_'][0],
            pressure = data_config['loader']['coordinates']['pressure']['load']['_args_'][0],
            prof  = data_config['loader']['targets']['prof']['load']['_args_'][0],
            surf  = data_config['loader']['targets']['surf']['load']['_args_'][0],
            meta  = data_config['loader']['targets']['meta']['load']['_args_'][0],
            hofx  = data_config['loader']['targets']['hofx']['load']['_args_'][0],
        params:
            config_name = hydra_config_name,
            experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else "",
            target = step_name,
            # condition= f"~data.dataset.state.preparation.{step_name}" if check_key_in_dict(step_name, state_config['preparation']) else "",
        output:
            out = data_config['preparation'][step_name]['output']['path']
        shell:
            """
            python -m inverse.data.preparation.{params.target} \
            --config-name={params.config_name} \
            {params.experiment} \
            """

# TODO: Check if f strings are necessary here
rule train:
    input:
        lat = data_config['loader']['coordinates']['lat']['load']['_args_'][0],
        lon = data_config['loader']['coordinates']['lon']['load']['_args_'][0],
        scans = data_config['loader']['coordinates']['scans']['load']['_args_'][0],
        pressure = data_config['loader']['coordinates']['pressure']['load']['_args_'][0],
        prof = data_config['loader']['targets']['prof']['load']['_args_'][0],
        surf = data_config['loader']['targets']['surf']['load']['_args_'][0],
        meta = data_config['loader']['targets']['meta']['load']['_args_'][0],
        hofx = data_config['loader']['targets']['hofx']['load']['_args_'][0],
        stats = data_config['preparation']['statistics']['output']['path']
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
