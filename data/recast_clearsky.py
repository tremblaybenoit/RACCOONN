import numpy as np
from data.filters import clearsky_filter, pressure_filter
import hydra
from utilities.instantiators import instantiate
from omegaconf import DictConfig
import os
from utilities.logic import get_config_path
import logging
from utilities.tensors import to_numpy


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute statistics of a given dataset.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    out_bases = {
        "float64": "../../GOES_ML-main/Data_cs_float64",
        "float32": "../../GOES_ML-main/Data_cs_float32",
    }
    if hasattr(config.data, "out_root"):
        out_bases["float64"] = os.path.join(config.data.out_root, "float64")
        out_bases["float32"] = os.path.join(config.data.out_root, "float32")

    for stage_name, stage_cfg in config.data.stage.items():
        stage_dir = str(stage_cfg.dir)
        # use only the final directory name for output to avoid nesting the original data path
        stage_name_on_disk = os.path.basename(os.path.normpath(stage_dir))
        logger.info("Processing stage `%s` (source dir `%s`, output subdir `%s`)", stage_name, stage_dir,
                    stage_name_on_disk)

        vars_cfg = stage_cfg.vars
        prof_mask = None
        prof_non_zero = None

        if "prof" in vars_cfg:
            vars_cfg["prof"].load.split = None
            vars_cfg["prof"].load.dtype = "float64"
            prof_arr = instantiate(vars_cfg["prof"].load)

            prof_mask = clearsky_filter(prof_arr).astype(np.bool_, copy=False)

            cleared = prof_arr[prof_mask]
            non_zero = []
            for i in range(prof_arr.shape[1]):
                if cleared[:, i, :].sum() > 0:
                    non_zero.append(i)
            prof_non_zero = non_zero
            prof_arr = None
            cleared = None
            logger.info("Stage `%s`: prof non-zero profile indices: %s", stage_name, prof_non_zero)

        for precision in ("float32",): #, "float32"):
            out_root = out_bases[precision]
            out_stage_dir = os.path.join(out_root, stage_name_on_disk)
            os.makedirs(out_stage_dir, exist_ok=True)
            save_dtype = np.float32 if precision == "float32" else np.float64

            for var_name, var_cfg in vars_cfg.items():
                if var_name in ("h",):
                    if hasattr(var_cfg.load, "split"):
                        var_cfg.load.split = None
                    if hasattr(var_cfg.load, "dtype"):
                        var_cfg.load.dtype = precision
                    data = instantiate(var_cfg.load)
                    if data.dtype != save_dtype:
                        data = to_numpy(data, dtype=save_dtype)
                    logger.info("Var dtype: %s", data.dtype)

                    if prof_mask is not None and data.shape[0] == prof_mask.shape[0]:
                        data = data[prof_mask]
                    logger.info("Var dtype: %s", data.dtype)

                    if var_name == 'hofx':
                        print(data.min(), data.max(), data.mean(), data.var())

                    if (var_name == "prof" or var_name == "prof_background") and prof_non_zero is not None and data.ndim >= 3:
                        data = data[:, prof_non_zero, :]

                    out_path = os.path.join(out_stage_dir, f"{var_name}.npy")
                    if var_name == "cloud_filter":
                        # Save as boolean array
                        np.save(out_path, data.astype(np.bool_, copy=False))
                    else:
                        np.save(out_path, data.astype(save_dtype, copy=False))
                    logger.info("Saved `%s` (stage `%s`, precision `%s`) -> `%s`", var_name, stage_name, precision,
                                out_path)

    # Compute background from newly computed profiles
    prof_train = np.load(os.path.join(out_root, 'Train2/prof.npy'))
    prof_valid = np.load(os.path.join(out_root, 'Val2/prof.npy'))
    prof_test = np.load(os.path.join(out_root, 'Test2/prof.npy'))
    prof_stack = np.concatenate([prof_train, prof_valid, prof_test], axis=0)
    prof_background = np.mean(prof_stack, axis=0, keepdims=True)
    prof_filter = pressure_filter(prof_stack)
    np.save(os.path.join(out_root, 'Train2/prof_background.npy'), prof_background)
    np.save(os.path.join(out_root, 'Val2/prof_background.npy'), prof_background)
    np.save(os.path.join(out_root, 'Test2/prof_background.npy'), prof_background)
    np.save(os.path.join(out_root, 'Train2/prof_increment.npy'), prof_train-prof_background)
    np.save(os.path.join(out_root, 'Val2/prof_increment.npy'), prof_valid-prof_background)
    np.save(os.path.join(out_root, 'Test2/prof_increment.npy'), prof_test-prof_background)
    np.save(os.path.join(out_root, 'pressure_filter.npy'), prof_filter)

    return

if __name__ == '__main__':
    """ Recast a given dataset.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        Recasted dataset saved to disk.
    """

    main()
