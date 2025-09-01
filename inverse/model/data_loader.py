import numpy as np
from omegaconf import DictConfig, ListConfig
from forward.model.data_loader import BaseDataloader
from forward.utilities.instantiators import instantiate
from inverse.data.transformations import identity
from torch.utils.data import Dataset
from typing import Union
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def load_coords(config: DictConfig, split: np.ndarray = None) -> dict:
    """ Load and normalize coordinates.

        Parameters
        ----------
        config: DictConfig. Configuration object for the coordinates.
        split : np.ndarray. Array of indices for the specified stage.

        Returns
        -------
        dict. Dictionary containing the loaded and normalized coordinates.
    """

    # Initialize dictionary to hold coordinates
    x = {}

    # Load and normalize coordinates
    for c in config:
        # Extract normalization function
        f_norm = instantiate(config[c]['normalization']) if hasattr(config[c], 'normalization') else identity
        # Load and normalize coordinates
        x[c] = f_norm(np.array(instantiate(config[c]['load']), dtype=np.float64))

    # Extract number of coordinate points and number of scans
    if 'lat' not in config or 'lon' not in config or 'scans' not in config:
        raise ValueError("Coordinates 'lat', 'lon', and 'scans' must be provided in the configuration.")
    n_coords, n_scans = x['lat'].shape[0], x['scans'].shape[0]

    # Repeat or tile lat, lon and t to match the number of samples (n_coords * n_timesteps)
    x['lat'] = np.tile(x['lat'], n_scans)
    x['lon'] = np.tile(x['lon'], n_scans)
    x['scans'] = np.repeat(x['scans'], n_coords, axis=0)

    # Apply split if available
    if split is not None:
        breakpoint()
        for c in x:
            breakpoint()
            if x[c].shape[0] == len(split):
                breakpoint()
                x[c] = x[c][split]

    return x


def load_obs(config: DictConfig, split: np.ndarray = None) -> dict:
    """ Load and normalize variables.

        Parameters
        ----------
        config: DictConfig. Configuration object for the variables.
        split : np.ndarray. Array of indices for the specified stage.

        Returns
        -------
        dict. Dictionary containing the loaded and normalized variables.
    """

    # Initialize dictionary to hold variables
    x = {}

    # Load variables (if provided)
    for c in config:
        # Extract normalization function
        f_norm = instantiate(config[c]['normalization']) if hasattr(config[c], 'normalization') else identity
        # Load and normalize variable
        data = f_norm(np.array(instantiate(config[c]['load']), dtype=np.float64))
        # Apply split if available
        if split is not None and data.shape[0] == len(split):
            data = data[split]
        x[c] = data

    return x


class InverseDataloader(BaseDataloader):
    def __init__(self, stage: DictConfig, batch_size: int = 32, num_workers: int = None,
                 persistent_workers: bool = True, pin_memory: bool = True) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        stage: DictConfig. Configuration object for the dataset at each stage (train, valid, test, pred).
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
        persistent_workers : bool. If True, the data loader will not shutdown the worker processes after a dataset has been consumed.
        pin_memory : bool. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

        Returns
        -------
        None.
        """

        #  Class inheritance
        super().__init__(batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)

        # Data sets
        self.stage = stage

    def setup(self, stage: str):
        """ Set up the dataset for training, validation, testing, or prediction.

            Parameters
            ----------
            stage : str. Stage of the model ('train', 'valid', 'test', 'predict').

            Returns
            -------
            None.
        """

        # Load datasets
        if stage == 'train':
            # Training/validation data
            self.ds_train, self.ds_valid = self.stage.train, self.stage.valid
        elif stage == 'test':
            # Test/prediction data
            self.ds_test = self.stage.test
        elif stage == 'pred':
            # Prediction data
            self.ds_pred = self.stage.pred


class InverseDataset(Dataset):
    """Lazy loader for the inverse dataset."""

    def __init__(self, x: dict) -> None:
        """Initialize the lazy loader.

        Parameters
        ----------
        x : dict. Dictionary containing the data.

        Returns
        -------
        None.
        """

        # Store data
        self.x = x

    def __len__(self) -> int:
        """ Get the length of the dataset.

        Returns
        -------
        int. Length of the dataset.
        """
        # Return the length of the first input tensor
        first_profile = next(iter(self.x.values()))
        first_var = next(iter(first_profile.values()))
        return len(first_var)

    def __getitem__(self, idx: int) -> Union[tuple, dict]:
        """ Get data

        Returns
        -------
        Dataset object.
        """

        # Get the data at the specified index
        return {k: {kk: vv[idx] if vv.shape[0] == self.__len__() else vv
                    for kk, vv in v.items()} for k, v in self.x.items()}


class InverseCRTMDataset(InverseDataset):
    """Lazy loader for the inverse dataset."""

    def __init__(self, coords: DictConfig, prof_type: ListConfig, split: DictConfig = None,
                 cloud_filter: DictConfig = None, obs: DictConfig = None, results: DictConfig = None) -> None:
        """Initialize the lazy loader.

        Parameters
        ----------
        coords : DictConfig. Configuration object for the coordinates.
        prof_type : ListConfig. List of profile types to load.
        split : DictConfig. Indices for the specified stage.
        cloud_filter : DictConfig. Boolean mask to filter out clear-sky profiles.
        obs : DictConfig. Configuration object for the variables.
        results : DictConfig. Configuration object for the results.

        Returns
        -------
        None.
        """

        # Store profile types
        self.prof_type = prof_type

        # Store results configuration
        self.results = results

        # Extract splitting
        split = instantiate(split) if split is not None else None
        if cloud_filter is not None:
            cloud_filter = instantiate(cloud_filter)
            split = split[cloud_filter[split]] if split is not None else np.where(cloud_filter)[0]

        # Load coordinates
        x = {'coords': load_coords(coords, split=split)}

        # Load variables (if provided)
        if obs is not None:
            x['obs'] = load_obs(obs, split=split)

        # Class inheritance
        super().__init__(x)
