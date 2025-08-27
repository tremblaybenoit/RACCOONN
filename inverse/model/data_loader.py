import numpy as np
from omegaconf import DictConfig
from forward.model.data_loader import CRTMDataloader, CRTMDataset, BaseDataloader
from forward.utilities.instantiators import instantiate
from inverse.data.transformations import identity
from torch.utils.data import Dataset
from typing import Union
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def assign_coordinates(lat: np.ndarray, lon: np.ndarray, scans: np.ndarray) -> dict:
    """ Assign coordinates to the dataset.

    Parameters
    ----------
    lat : np.ndarray. Latitude coordinates.
    lon : np.ndarray. Longitude coordinates.
    scans : np.ndarray. Scans performed.

    Returns
    -------
    lat_all: np.ndarray. Sampled latitude coordinates.
    lon_all: np.ndarray. Sampled longitude coordinates.
    scans_all: np.ndarray. Sampled scans.
    """

    # Extract dimensions
    n_coords, n_scans = lat.shape[0], scans.shape[0]  # Number of coordinates
    # Repeat or tile lat, lon and t to match the number of samples (n_coords * n_timesteps)
    lat_all = np.tile(lat, n_scans)
    lon_all = np.tile(lon, n_scans)
    scans_all = np.repeat(scans, n_coords, axis=0)

    return {'lat': lat_all, 'lon': lon_all, 'scans': scans_all}


class PINNverseDataloader(BaseDataloader):
    def __init__(self, sets: DictConfig, batch_size: int = 32, num_workers: int = None, pin_memory: bool = True) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        sets: DictConfig. Configuration object for the dataset sets.
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
        pin_memory : bool. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

        Returns
        -------
        None.
        """

        #  Class inheritance
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        # Data sets
        self.sets = sets

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
            self.ds_train, self.ds_valid = self._make_dataset(self.sets['train']), self._make_dataset(self.sets['valid'])
        elif stage == 'test':
            # Test/prediction data
            self.ds_test = self._make_dataset(self.sets[stage])
        elif stage == 'pred':
            # Prediction data
            self.ds_pred = self._make_dataset(self.sets[stage])

    def _make_dataset(self, data: DictConfig) -> Dataset:
        """ Create a dataset for the specified stage.

        Parameters
        ----------
        data: DictConfig. Configuration object for the dataset.

        Returns
        -------
        VarDataset. Dataset object containing the input and output data.
        """

        # Asemble dataset in a dictionary.
        dataset = {}

        # Assign coordinates
        coordinates = assign_coordinates(np.array(instantiate(data['coordinates']['lat']['load']), dtype=np.float64),
                                         np.array(instantiate(data['coordinates']['lon']['load']), dtype=np.float64),
                                         np.array(instantiate(data['coordinates']['scans']['load']), dtype=np.float64))

        # Split
        if hasattr(data, 'split'):
            split = data['split']['range']
            if hasattr(data['split'], 'cloud_filter'):
                cloud_filter = np.array(instantiate(data['split']['cloud_filter']['load']), dtype=bool)
                split = cloud_filter[split]
        else:
            split = None

        # Store coordinates and targets (if available)
        labels = ['coordinates'] + ['targets' if hasattr(data, 'targets') else []]
        for label in labels:
            for var, config in data[label].items():
                f_norm = instantiate(config['normalization']) if hasattr(config, 'normalization') else identity
                if var in ['lat', 'lon', 'scans']:
                    dataset[label][var] = f_norm(coordinates[var])
                elif var == 'pressure':
                    dataset[label][var] = np.tile(f_norm(0.01*np.array(instantiate(config['load']), dtype=np.float64)), (self.batch_size, 1))
                else:
                    dataset[label][var] = f_norm(np.array(instantiate(config['load']), dtype=np.float64))
                # Apply split
                if split is not None and var not in ['pressure'] and dataset[label][var].shape[0] == len(split):
                    dataset[label][var] = dataset[label][var][split]

        return VarDataset(dataset)


class VarDataset(Dataset):
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
        return {k: {kk: vv[idx] if kk != 'pressure' else vv for kk, vv in v.items()} for k, v in self.x.items()}
