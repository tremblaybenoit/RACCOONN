import logging
import os
import json
from omegaconf import OmegaConf, DictConfig
import hydra
from utilities.logic import get_config_path
from config.setup import get_filenames

# Initialize logger
logger = logging.getLogger(__name__)


def aggregate_rule_files(config_hydra: dict, config_preprocessing: dict, config_loader: dict, config_model: dict) -> dict:
    """ Aggregate input and output filenames for each Snakemake rule.

        Parameters
        ----------
        config_hydra: dict. Resolved Hydra configuration dictionary (resolved from DictConfig).
        config_preprocessing: dict. Preprocessing configuration.
        config_loader: dict. Loader configuration.
        config_model: dict. Model configuration.

        Returns
        -------
        dict: Dictionary with rule names as keys and {'inputs': [...], 'outputs': [...]} as values.
    """

    rules = {}

    # Process preprocessing rules
    if config_preprocessing:
        logger.info("Processing preprocessing rules...")
        for prep_type, config_prep in config_preprocessing.items():
            logger.info(f"\n  Preprocessing type: {prep_type}")

            # If the preprocessing operation is executable (has _target_)
            if '_target_' in config_prep:
                logger.info(f"    Rule: {prep_type} (single step)")

                # Handle statistics exclude list for single-step case
                if prep_type in ('statistics',):
                    exclude_set = {'transformations'}
                    exclude_list = config_prep.get('exclude', None)
                    if exclude_list is not None and isinstance(exclude_list, list):
                        exclude_set.update(exclude_list)
                    input_files = get_filenames(config_prep.get('input', {}), exclude_keys=exclude_set)
                else:
                    input_files = get_filenames(config_prep.get('input', {}), exclude_keys={'transformations'})

                output_files = get_filenames(config_prep.get('output', {}), exclude_keys={'transformations'})
                rules[prep_type] = {
                    'inputs': sorted(input_files),
                    'outputs': sorted(output_files),
                    'type': 'preprocessing_single'
                }
                logger.info(f"      Inputs: {len(input_files)} files")
                logger.info(f"      Outputs: {len(output_files)} files")

            # Else loop over each preprocessing step
            else:
                logger.info(f"    Multiple steps in {prep_type}:")
                for step_name, config_step in config_prep.items():
                    if '_target_' in config_step:
                        rule_name = f"{prep_type}_{step_name}"
                        logger.info(f"      Step: {step_name}")

                        # Handle statistics exclude list for multi-step case
                        if prep_type in ('statistics', ):
                            exclude_set = {'transformations'}
                            exclude_list = config_step.get('exclude', None)
                            if exclude_list is not None and isinstance(exclude_list, list):
                                exclude_set.update(exclude_list)
                            input_files = get_filenames(config_step.get('input', {}), exclude_keys=exclude_set)
                        else:
                            input_files = get_filenames(config_step.get('input', {}))

                        output_files = get_filenames(config_step.get('output', {}), exclude_keys={'transformations'})
                        # breakpoint()
                        rules[rule_name] = {
                            'inputs': sorted(input_files),
                            'outputs': sorted(output_files),
                            'type': 'preprocessing_step'
                        }
                        logger.info(f"        Inputs: {len(input_files)} files")
                        logger.info(f"        Outputs: {len(output_files)} files")

    # Process training rule
    if 'train' in config_loader:
        logger.info("\nProcessing training rule...")
        train_files = get_filenames(config_loader['train'])
        valid_files = get_filenames(config_loader['valid'])
        model_files = get_filenames(config_model, exclude_keys={'ckpt_path'})
        input_files = sorted(set(train_files + valid_files + model_files))
        output_files = [config_model['ckpt_path']] if 'ckpt_path' in config_model else []
        rules['train'] = {
            'inputs': input_files,
            'outputs': output_files,
            'type': 'training'
        }
        logger.info(f"  Inputs: {len(input_files)} files")
        logger.info(f"  Outputs: {len(output_files)} files")

    # Process test rule
    if 'test' in config_loader:
        logger.info("\nProcessing test rule...")
        test_files = get_filenames(config_loader['test'], exclude_keys={'output', 'latent'})
        model_files = get_filenames(config_model)
        input_files = sorted(test_files + model_files)
        output_files = sorted(get_filenames(config_loader['test'], exclude_keys={'input', 'context', 'target', 'transformations'}))
        rules['test'] = {
            'inputs': input_files,
            'outputs': output_files,
            'type': 'testing'
        }
        logger.info(f"  Inputs: {len(input_files)} files")
        logger.info(f"  Outputs: {len(output_files)} files")

    # Process predict rule
    if 'predict' in config_loader:
        logger.info("\nProcessing predict rule...")
        predict_files = get_filenames(config_loader['predict'], exclude_keys={'output', 'latent'})
        ckpt_files = [config_model['ckpt_path']] if 'ckpt_path' in config_model else []
        input_files = sorted(predict_files + ckpt_files)
        output_files = sorted(get_filenames(config_loader['predict'], exclude_keys={'input', 'context', 'target', 'transformations'}))
        rules['predict'] = {
            'inputs': input_files,
            'outputs': output_files,
            'type': 'prediction'
        }
        logger.info(f"  Inputs: {len(input_files)} files")
        logger.info(f"  Outputs: {len(output_files)} files")

    return rules


def check_cyclic_dependencies(rules: dict) -> list:
    """ Check for cyclic dependencies where a rule's output is also its input.

        Parameters
        ----------
        rules: dict. Dictionary of rules with inputs and outputs.

        Returns
        -------
        list: List of cyclic dependency issues found.
    """

    issues = []
    logger.info("\n")
    logger.info("="*80)
    logger.info("CHECKING FOR CYCLIC DEPENDENCIES")
    logger.info("="*80)

    for rule_name, rule_info in rules.items():
        inputs = set(rule_info['inputs'])
        outputs = set(rule_info['outputs'])

        # Check for self-cycles (output is also input)
        overlap = inputs & outputs
        if overlap:
            issues.append({
                'rule': rule_name,
                'type': 'self_cycle',
                'files': sorted(overlap)
            })
            logger.warning(f"CYCLIC: {rule_name} has overlapping input/output:")
            for f in sorted(overlap):
                logger.warning(f"  - {f}")

    # Check for dependency chains
    logger.info("\n")
    logger.info("="*80)
    logger.info("CHECKING FOR MISSING INPUT FILES")
    logger.info("="*80)

    all_outputs = set()
    for rule_info in rules.values():
        all_outputs.update(rule_info['outputs'])

    for rule_name, rule_info in rules.items():
        inputs = set(rule_info['inputs'])
        # Find inputs that are not produced by any rule and not existing files
        missing = []
        for inp in inputs:
            if inp not in all_outputs and not os.path.exists(inp):
                missing.append(inp)

        if missing:
            issues.append({
                'rule': rule_name,
                'type': 'missing_input',
                'files': sorted(missing)
            })
            logger.warning(f"MISSING INPUT: {rule_name} requires non-existent files:")
            for f in sorted(missing):
                logger.warning(f"  - {f}")

    return issues


def check_rule_dependencies(rules: dict) -> dict:
    """ Map which rules depend on which other rules.

        Parameters
        ----------
        rules: dict. Dictionary of rules with inputs and outputs.

        Returns
        -------
        dict: Dependency map showing which rules feed into which other rules.
    """

    dependencies = {}
    logger.info("\n")
    logger.info("="*80)
    logger.info("RULE DEPENDENCY MAP")
    logger.info("="*80)

    # Build output-to-rule mapping
    output_to_rule = {}
    for rule_name, rule_info in rules.items():
        for output_file in rule_info['outputs']:
            output_to_rule[output_file] = rule_name

    # Find dependencies
    for rule_name, rule_info in rules.items():
        deps = []
        for input_file in rule_info['inputs']:
            if input_file in output_to_rule:
                source_rule = output_to_rule[input_file]
                if source_rule != rule_name and source_rule not in deps:
                    deps.append(source_rule)

        if deps:
            dependencies[rule_name] = deps
            logger.info(f"{rule_name} depends on: {', '.join(deps)}")

    return dependencies


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Analyze Snakemake rule structure for debugging.

        Parameters
        ----------
        config: DictConfig. Complete Hydra configuration.

        Returns
        -------
        None.
    """

    # Resolve DictConfig to plain dictionary (same as read_hydra_as_dict does internally)
    config_hydra = OmegaConf.to_container(config, resolve=True)

    # Extract configs (same as Snakefile)
    config_data = config_hydra["data"]
    config_preprocessing = config_hydra.get("preprocessing", {})
    config_loader = config_hydra["loader"]["stage"]
    config_model = config_hydra["model"]
    config_paths = config_hydra["paths"]

    logger.info("="*80)
    logger.info("SNAKEMAKE RULE ANALYSIS")
    logger.info("="*80)

    # Aggregate files per rule
    rules = aggregate_rule_files(config_hydra, config_preprocessing, config_loader, config_model)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total rules: {len(rules)}")
    for rule_name, rule_info in sorted(rules.items()):
        logger.info(f"  {rule_name:25} - Inputs: {len(rule_info['inputs']):3} | Outputs: {len(rule_info['outputs']):3}")

    # Check for issues
    cyclic_issues = check_cyclic_dependencies(rules)
    dependencies = check_rule_dependencies(rules)

    # Report issues
    if cyclic_issues:
        logger.error(f"\nFound {len(cyclic_issues)} issue(s):")
        for issue in cyclic_issues:
            logger.error(f"\n  {issue['type'].upper()}: {issue['rule']}")
            for f in issue['files'][:5]:
                logger.error(f"    - {f}")
            if len(issue['files']) > 5:
                logger.error(f"    ... and {len(issue['files']) - 5} more")
    else:
        logger.info("\nNo cyclic dependencies detected!")

    # Save detailed report
    output_path = os.path.join(config_paths["data_dir"], 'snakemake_analysis.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'rules': {rule_name: {
                'inputs': rule_info['inputs'],
                'outputs': rule_info['outputs'],
                'type': rule_info['type']
            } for rule_name, rule_info in rules.items()},
            'dependencies': dependencies,
            'issues': cyclic_issues
        }, f, indent=2)
    logger.info(f"\nDetailed report saved to: {output_path}")


if __name__ == '__main__':
    """ Analyze Snakemake rule structure for debugging cyclic dependencies and missing inputs.
    
        Usage:
        ------
        python -m code.evaluation.snakemake --config-name=default +experiment=inverse_clear
    """

    main()

