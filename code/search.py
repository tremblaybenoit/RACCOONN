import gc
import logging
import os
from typing import Any
import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from code.train import Operator
from utilities.logic import get_config_path

logger = logging.getLogger(__name__)


def _clone_config(config: DictConfig) -> DictConfig:
    """Deep-clone a Hydra config without resolving interpolations."""
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _to_float_metric(value: Any) -> float:
    """Convert a Lightning callback metric to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _apply_trial_params(cfg: DictConfig, trial: optuna.Trial) -> None:
    """Apply Optuna-suggested architecture/optimizer hyperparameters."""

    width = trial.suggest_categorical("width", [64, 128, 256])
    hidden_blocks = trial.suggest_int("hidden_blocks", 1, 4)
    head_hidden = trial.suggest_categorical("head_hidden", [16, 32, 64, 128])
    head_blocks = trial.suggest_int("head_blocks", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.15)

    freq_xy = trial.suggest_categorical("freq_xy", [4, 8, 12])
    freq_p = trial.suggest_categorical("freq_pressure", [8, 16, 24, 32])

    enc_skip_hidden = trial.suggest_categorical("enc_skip_hidden", [False, True])
    enc_skip_output = trial.suggest_categorical("enc_skip_output", [False, True])

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    # Architecture
    cfg.architecture.input_layer.out_features = width
    cfg.architecture.input_layer.dropout_rate = dropout

    cfg.architecture.hidden_layer.in_features = width
    cfg.architecture.hidden_layer.out_features = width
    cfg.architecture.hidden_layer.hidden_features = width
    cfg.architecture.hidden_layer.n_blocks = hidden_blocks
    cfg.architecture.hidden_layer.dropout_rate = dropout

    cfg.architecture.output_layer.head.in_features = width
    cfg.architecture.output_layer.head.hidden_features = head_hidden
    cfg.architecture.output_layer.head.n_blocks = head_blocks
    cfg.architecture.output_layer.head.dropout_rate = dropout

    cfg.architecture.positional_encoding.num_freqs = [freq_xy, freq_xy, freq_p]
    cfg.architecture.enc_skip_hidden = enc_skip_hidden
    cfg.architecture.enc_skip_output = enc_skip_output

    # Optimizer
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay


def _objective(base_config: DictConfig, trial: optuna.Trial) -> float:
    cfg = _clone_config(base_config)

    # Give each trial unique run artifacts/checkpoints
    cfg.paths.run_id = f"optuna_{trial.number:04d}"
    if hasattr(cfg, "task_name"):
        cfg.task_name = f"{cfg.task_name}_optuna"

    # Do not open browser UIs during search
    if hasattr(cfg, "logger") and cfg.logger is not None and hasattr(cfg.logger, "ui"):
        cfg.logger.ui = False

    # Optional max-epoch override for faster search
    max_epochs = cfg.get("optuna", {}).get("max_epochs")
    if max_epochs is not None:
        cfg.trainer.max_epochs = int(max_epochs)

    _apply_trial_params(cfg, trial)

    operator = Operator(cfg)
    operator.train()

    metric = operator.trainer.callback_metrics.get("valid_loss")
    if metric is None:
        raise RuntimeError("valid_loss not found in callback_metrics after training.")

    value = _to_float_metric(metric)
    if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError(f"Non-finite valid_loss: {value}")

    # Help Python release trial resources
    del operator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return value


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Run Optuna architecture search on top of a Hydra experiment config.

    Example:
      python -m code.search +experiment=inverse_clear_log_split2_transform_0046 \
        optuna.n_trials=30 optuna.max_epochs=300
    """
    optuna_cfg = config.get("optuna", {})

    n_trials = int(optuna_cfg.get("n_trials", 20))
    direction = str(optuna_cfg.get("direction", "minimize"))
    sampler_seed = int(optuna_cfg.get("seed", 42))

    study_name = str(optuna_cfg.get("study_name", "inverse_architecture_search"))
    storage = optuna_cfg.get("storage", None)
    if storage is None:
        # Default persistent local storage
        db_path = os.path.abspath(os.path.join(config.paths.task_dir, f"{study_name}.db"))
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        storage = f"sqlite:///{db_path}"

    logger.info("Creating Optuna study '%s' at %s", study_name, storage)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )

    logger.info("Starting Optuna with n_trials=%d", n_trials)
    study.optimize(lambda t: _objective(config, t), n_trials=n_trials)

    logger.info("Best value: %.8f", study.best_value)
    logger.info("Best params: %s", study.best_params)


if __name__ == "__main__":
    main()

