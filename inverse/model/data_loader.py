import numpy as np
from omegaconf import DictConfig
from forward.model.data_loader import BaseDataloader
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
    return {'lat': np.tile(lat, n_scans), 'lon': np.tile(lon, n_scans), 'scans': np.repeat(scans, n_coords, axis=0)}


class InverseDataloader(BaseDataloader):
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
            self.ds_train, self.ds_valid = self._make_dataset('train'), self._make_dataset('valid')
        elif stage == 'test':
            # Test/prediction data
            self.ds_test = self._make_dataset(stage)
        elif stage == 'pred':
            # Prediction data
            self.ds_pred = self._make_dataset(stage)

    def _get_split(self, stage: str) -> Union[np.ndarray, None]:
        """ Get the split for the specified stage.

        Parameters
        ----------
        stage : str. Stage of the model ('train', 'valid', 'test', 'pred').

        Returns
        -------
        np.ndarray. Array of indices for the specified stage.
        """

        if hasattr(self.sets, 'split') and stage in ['train', 'valid', 'test']:
            split = self.sets['split'][stage]
            if hasattr(self.sets['split'], 'cloud_filter'):
                cloud_filter = np.array(instantiate(self.sets['split']['cloud_filter']['load']), dtype=bool)
                split = cloud_filter[split]
            return split
        return None

    def _process_variable(self, data: np.ndarray, f_norm: Callable = None, split: np.ndarray = None) -> np.ndarray:
        """ Process the variables in the dataset.

        Parameters
        ----------
        data : DictConfig. Configuration object for the dataset.
        split : np.ndarray. Array of indices for the specified stage.

        Returns
        -------
        dict. Dictionary containing the processed variables.
        """

        # Normalization function
        f_norm = f_norm if f_norm is not None else identity

        # Initialize dictionary to store processed variables
        processed_data = f_norm(data)
        # Apply split if available
        if split is not None and processed_data.shape[0] == len(split):
            processed_data = processed_data[split]
        return processed_data

    def _process_variables(self, data: DictConfig, coordinates: dict = None, split: np.ndarray = None) -> dict:
        """ Process the variables in the dataset.

        Parameters
        ----------
        data : DictConfig. Configuration object for the dataset.
        coordinates : dict. Dictionary containing the coordinates.
        split : np.ndarray. Array of indices for the specified stage.

        Returns
        -------
        dict. Dictionary containing the processed variables.
        """

        # Initialize dictionary to store processed variables
        processed_data = {}

        # Process each variable in the dataset
        for var, config in data.items():
            # Normalization function
            f_norm = instantiate(config['normalization']) if hasattr(config, 'normalization') else identity
            # If coordinates, assign directly
            if coordinates is not None and var in coordinates:
                var_data = f_norm(coordinates[var])
            elif var == 'pressure':
                var_data = np.tile(f_norm(0.01*np.array(instantiate(config['load']), dtype=np.float64)), (self.batch_size, 1))
            else:
                var_data = f_norm(np.array(instantiate(config['load']), dtype=np.float64))
            # Apply split if available
            if split is not None and var not in ['pressure'] and var_data.shape[0] == len(split):
                var_data = var_data[split]
            # Store processed variable
            processed_data[var] = var_data

        return processed_data

    def _make_dataset(self, stage: str) -> Dataset:
        """ Create a dataset for the specified stage.

        Parameters
        ----------
        stage : str. Stage of the model ('train', 'valid', 'test', 'pred').

        Returns
        -------
        InverseDataset. Dataset object containing the input and output data.
        """

        # Asemble dataset in a dictionary.
        dataset = {}

        # Assign coordinates
        coordinates = assign_coordinates(np.array(instantiate(self.sets['coordinates']['lat']['load']), dtype=np.float64),
                                         np.array(instantiate(self.sets['coordinates']['lon']['load']), dtype=np.float64),
                                         np.array(instantiate(self.sets['coordinates']['scans']['load']), dtype=np.float64))

        # Split
        split = self._get_split(stage)

        # Store coordinates
        dataset['coordinates'] = self._process_variables(self.sets['coordinates'], coordinates=coordinates, split=split)
        # Store targets if available
        if hasattr(self.sets, 'targets') and stage in ['train', 'valid', 'test']:
            dataset['targets'] = self._process_variables(self.sets['targets'], split=split)

        # Assemble dataset
        return InverseDataset(dataset)

    def _assign_coordinates(self, coordinates: DictConfig) -> dict:
        """ Assign coordinates to the dataset.

        Parameters
        ----------
        coordinates : DictConfig. Configuration object for the coordinates.

        Returns
        -------
        dict. Dictionary containing the assigned coordinates.
        """

        # Extract coordinates from configuration
        lat = np.array(instantiate(self.sets['lat']['load']), dtype=np.float64)
        lon = np.array(instantiate(self.sets['lon']['load']), dtype=np.float64)
        scans = np.array(instantiate(self.sets['scans']['load']), dtype=np.float64)
        pressure = np.array(instantiate(self.sets['coordinates']['pressure']['load']), dtype=np.float64)

        # Assign coordinates
        return {'lat': np.tile(lat, scans.shape[0]), 'lon': np.tile(lon, scans.shape[0]), 'scans': np.repeat(scans, lat.shape[0], axis=0), 'pressure': np.tile(0.01*pressure, (self.batch_size, 1))}

    def _make_dataset2(self, coordinates: DictConfig, targets: DictConfig = None, split: np.ndarray) -> Dataset:
        """ Create a dataset for the specified stage.

        Parameters
        ----------


        Returns
        -------
        InverseDataset. Dataset object containing the input and output data.
        """

        # Asemble dataset in a dictionary.
        dataset = {}

        # Assign coordinates
        coordinates = assign_coordinates(np.array(instantiate(self.sets['coordinates']['lat']['load']), dtype=np.float64),
                                         np.array(instantiate(self.sets['coordinates']['lon']['load']), dtype=np.float64),
                                         np.array(instantiate(self.sets['coordinates']['scans']['load']), dtype=np.float64))
        coordinates['pressure'] = np.tile(0.01*np.array(instantiate(self.sets['coordinates']['pressure']['load']), dtype=np.float64), (self.batch_size, 1))

        # Split
        split = self._get_split(stage)

        # Store coordinates
        dataset['coordinates'] = self._process_variables(self.sets['coordinates'], coordinates=coordinates, split=split)
        # Store targets if available
        if hasattr(self.sets, 'targets') and stage in ['train', 'valid', 'test']:
            dataset['targets'] = self._process_variables(self.sets['targets'], split=split)

        # Assemble dataset
        return InverseDataset(dataset)

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
        return {k: {kk: vv[idx] if kk != 'pressure' else vv for kk, vv in v.items()} for k, v in self.x.items()}
