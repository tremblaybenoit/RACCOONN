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
                             experiment=hydra_experiment)

# Hydra configuration file
hydra_config = read_hydra_as_dict(config_path=hydra_config_path, config_name=hydra_config_name,
                                  experiment=hydra_experiment)
# Data configuration file (from Snakemake config file)
data_config = hydra_config["data"]
# Paths configuration file (from Snakemake config file)
paths_config = hydra_config["paths"]
# Callback configuration file (from Snakemake config file)
checkpoint_config = hydra_config["callbacks"]["model_checkpoint"]

#########################################################################################################
# TARGET
#########################################################################################################

# Foward model
rule forward:
    input:
        checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt",
        test_hofx= f"{data_config['sets']['test']['results']['hofx']['save']['_args_'][0]}"

# Inverse model
rule inverse:
    input:
        checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt",
        test_prof= f"{data_config['sets']['test']['results']['prof']['save']['_args_'][0]}",
        test_hofx=f"{data_config['sets']['test']['results']['hofx']['save']['_args_'][0]}"

#########################################################################################################
# JOBS
#########################################################################################################

# Training
rule train_forward:
    input:
        train_prof  = f"{data_config['sets']['train']['prof']['load']['_args_'][0]}",
        train_surf  = f"{data_config['sets']['train']['surf']['load']['_args_'][0]}",
        train_meta  = f"{data_config['sets']['train']['meta']['load']['_args_'][0]}",
        train_hofx  = f"{data_config['sets']['train']['hofx']['load']['_args_'][0]}",
        train_lat   = f"{data_config['sets']['train']['lat']['load']['_args_'][0]}",
        train_lon   = f"{data_config['sets']['train']['lon']['load']['_args_'][0]}",
        train_scans = f"{data_config['sets']['train']['scans']['load']['_args_'][0]}",
        train_pres  = f"{data_config['sets']['train']['pressure']['load']['_args_'][0]}",
        valid_prof  = f"{data_config['sets']['valid']['prof']['load']['_args_'][0]}",
        valid_surf  = f"{data_config['sets']['valid']['surf']['load']['_args_'][0]}",
        valid_meta  = f"{data_config['sets']['valid']['meta']['load']['_args_'][0]}",
        valid_hofx  = f"{data_config['sets']['valid']['hofx']['load']['_args_'][0]}",
        valid_lat   = f"{data_config['sets']['valid']['lat']['load']['_args_'][0]}",
        valid_lon   = f"{data_config['sets']['valid']['lon']['load']['_args_'][0]}",
        valid_scans = f"{data_config['sets']['valid']['scans']['load']['_args_'][0]}",
        valid_pres  = f"{data_config['sets']['valid']['pressure']['load']['_args_'][0]}"
    params:
        config_name = hydra_config_name,
        experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
    output:
        checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt"
    shell:
        """
        python -m forward.train \
        --config-name={params.config_name} \
        {params.experiment}
        """

rule test_forward:
    input:
        checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt",
        test_prof  = f"{data_config['sets']['test']['prof']['load']['_args_'][0]}",
        test_surf  = f"{data_config['sets']['test']['surf']['load']['_args_'][0]}",
        test_meta  = f"{data_config['sets']['test']['meta']['load']['_args_'][0]}",
        test_hofx  = f"{data_config['sets']['test']['hofx']['load']['_args_'][0]}",
        test_lat   = f"{data_config['sets']['test']['lat']['load']['_args_'][0]}",
        test_lon   = f"{data_config['sets']['test']['lon']['load']['_args_'][0]}",
        test_scans = f"{data_config['sets']['test']['scans']['load']['_args_'][0]}",
        test_pres  = f"{data_config['sets']['test']['pressure']['load']['_args_'][0]}"
    params:
        config_name = hydra_config_name,
        experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
    output:
        test_hofx = f"{data_config['sets']['test']['results']['hofx']['save']['_args_'][0]}"
    shell:
        """
            python -m forward.test \
            --config-name={params.config_name} \
            {params.experiment}
            """

#rule statistics_inverse:
#    input:
#        train_prof=f"{data_config['sets']['train']['prof']['load']['_args_'][0]}",
#        train_surf=f"{data_config['sets']['train']['surf']['load']['_args_'][0]}",
#        train_meta=f"{data_config['sets']['train']['meta']['load']['_args_'][0]}",
#        train_hofx=f"{data_config['sets']['train']['hofx']['load']['_args_'][0]}",
#        train_lat=f"{data_config['sets']['train']['lat']['load']['_args_'][0]}",
#        train_lon=f"{data_config['sets']['train']['lon']['load']['_args_'][0]}",
#        train_scans=f"{data_config['sets']['train']['scans']['load']['_args_'][0]}",
#        train_pres=f"{data_config['sets']['train']['pressure']['load']['_args_'][0]}"
#    params:
#        config_name=hydra_config_name,
#        experiment=f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
#    shell:
#        """
#        python -m inverse.data.statistics \
#        --config-name={params.config_name} \
#        {params.experiment}
#        """

rule train_inverse:
    input:
        train_prof  = f"{data_config['sets']['train']['prof']['load']['_args_'][0]}",
        train_surf  = f"{data_config['sets']['train']['surf']['load']['_args_'][0]}",
        train_meta  = f"{data_config['sets']['train']['meta']['load']['_args_'][0]}",
        train_hofx  = f"{data_config['sets']['train']['hofx']['load']['_args_'][0]}",
        train_lat   = f"{data_config['sets']['train']['lat']['load']['_args_'][0]}",
        train_lon   = f"{data_config['sets']['train']['lon']['load']['_args_'][0]}",
        train_scans = f"{data_config['sets']['train']['scans']['load']['_args_'][0]}",
        train_pres  = f"{data_config['sets']['train']['pressure']['load']['_args_'][0]}",
        valid_prof  = f"{data_config['sets']['valid']['prof']['load']['_args_'][0]}",
        valid_surf  = f"{data_config['sets']['valid']['surf']['load']['_args_'][0]}",
        valid_meta  = f"{data_config['sets']['valid']['meta']['load']['_args_'][0]}",
        valid_hofx  = f"{data_config['sets']['valid']['hofx']['load']['_args_'][0]}",
        valid_lat   = f"{data_config['sets']['valid']['lat']['load']['_args_'][0]}",
        valid_lon   = f"{data_config['sets']['valid']['lon']['load']['_args_'][0]}",
        valid_scans = f"{data_config['sets']['valid']['scans']['load']['_args_'][0]}",
        valid_pres  = f"{data_config['sets']['valid']['pressure']['load']['_args_'][0]}"
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

rule test_inverse:
    input:
        checkpoint = f"{paths_config['checkpoint_dir']}/{checkpoint_config['filename']}.ckpt",
        test_prof  = f"{data_config['sets']['test']['prof']['load']['_args_'][0]}",
        test_surf  = f"{data_config['sets']['test']['surf']['load']['_args_'][0]}",
        test_meta  = f"{data_config['sets']['test']['meta']['load']['_args_'][0]}",
        test_hofx  = f"{data_config['sets']['test']['hofx']['load']['_args_'][0]}",
        test_lat   = f"{data_config['sets']['test']['lat']['load']['_args_'][0]}",
        test_lon   = f"{data_config['sets']['test']['lon']['load']['_args_'][0]}",
        test_scans = f"{data_config['sets']['test']['scans']['load']['_args_'][0]}",
        test_pres  = f"{data_config['sets']['test']['pressure']['load']['_args_'][0]}"
    params:
        config_name = hydra_config_name,
        experiment = f"+experiment={hydra_experiment}" if hydra_experiment is not None else ""
    output:
        test_prof = f"{data_config['sets']['test']['results']['prof']['save']['_args_'][0]}",
        test_hofx = f"{data_config['sets']['test']['results']['hofx']['save']['_args_'][0]}"
    shell:
        """
        python -m inverse.test \
        --config-name={params.config_name} \
        {params.experiment}
        """