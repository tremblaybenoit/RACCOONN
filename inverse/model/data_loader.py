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


class PINNverseDataloader(CRTMDataloader):
    def __init__(self, dir: str, sets: DictConfig, inputs: Union[list, str], outputs: Union[list, str] = None,
                 batch_size: int = 32, num_workers: int = None, pin_memory: bool = True, clrsky_filter: bool = False) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        dir : str. Path to the directory containing the dataset files.
        sets: DictConfig. Configuration object for the dataset sets.
        inputs : list or str. List of input variables to be used in the model.
        outputs : list or str. List of output variables to be used in the model. If None, all outputs will be used.
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
        pin_memory : bool. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        clrsky_filter : bool. If True, filter the dataset to include only cloudy profiles.

        Returns
        -------
        None.
        """

        #  Class inheritance
        super().__init__(dir=dir, sets=sets, inputs=inputs, outputs=outputs, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=pin_memory)
        # Path to data
        self.ds_path = dir
        # Data sets
        self.sets = sets
        # Clear sky
        self.clrsky_filter = clrsky_filter

        # Masks and filters
        data = np.array(instantiate(self.sets['train']['prof']['load']), dtype=np.float64)
        index_clrsky = np.logical_and(data[:, 5, :].sum(axis=1) == 0, data[:, 6, :].sum(axis=1) == 0)
        index_clrsky = np.logical_and(index_clrsky, data[:, 7, :].sum(axis=1) == 0)
        self.index_cloudy = ~index_clrsky
        self.prof_mask = data.var(axis=0, keepdims=True) > 0
        data = None  # Free memory

    def _make_dataset(self, stage: str, inputs: Union[list, str], outputs: Union[list, str] = None) -> Dataset:
        """ Create a dataset for the specified stage.

        Parameters
        ----------
        stage : str. Stage of the model ('train', 'valid', 'test', 'pred').
        inputs : Union[list, str]. List of input data files or a single input data file.
        outputs : Union[list, str]. List of output data files or a single output data file.

        Returns
        -------
        CRTMDataset. Dataset object containing the input and output data.
        """

        # Assign coordinates
        coordinates = assign_coordinates(np.array(instantiate(self.sets[stage]['lat']['load']), dtype=np.float64),
                                         np.array(instantiate(self.sets[stage]['lon']['load']), dtype=np.float64),
                                         np.array(instantiate(self.sets[stage]['scans']['load']), dtype=np.float64))

        # Inputs
        x = {}
        for input in inputs:
            f_norm = instantiate(self.sets[stage][input]['normalization']) \
                if hasattr(self.sets[stage][input], 'normalization') else identity
            if input in ['lat', 'lon', 'scans']:
                x[input] = f_norm(coordinates[input][self.sets[stage]['split']['start']:self.sets[stage]['split']['end']])
            elif input == 'pressure':
                x[input] = np.tile(f_norm(0.01*np.array(instantiate(self.sets[stage][input]['load']), dtype=np.float64)),
                                   (self.sets[stage]['split']['end']-self.sets[stage]['split']['start'], 1))
            else:
                x[input] = f_norm(np.array(instantiate(self.sets[stage][input]['load']), dtype=np.float64)
                                  [self.sets[stage]['split']['start']:self.sets[stage]['split']['end']])
        # If clrsky is True, filter inputs based on clrsky mask
        if self.clrsky_filter:
            for key in x.keys():
                x[key] = x[key][self.index_cloudy[self.sets[stage]['split']['start']:self.sets[stage]['split']['end']]]

        # TODO: Verify which normalization function to use for background
        # If outputs include background, compute it
        if 'background' in outputs:
            # Compute background
            var = 'prof_norm' if 'prof_norm' in self.sets[stage] else 'prof'
            # Load the profile data and apply normalization
            f_norm = instantiate(self.sets[stage][var]['normalization']) \
                if hasattr(self.sets[stage][var], 'normalization') else identity
            xt = f_norm(np.array(instantiate(self.sets[stage][var]['load']), dtype=np.float64))
            xb = (np.zeros_like(xt[self.sets[stage]['split']['start']:self.sets[stage]['split']['end']]) +
                  np.nanmean(xt[self.index_cloudy], axis=0, keepdims=True))  # Compute mean and replace NaN with 0
            xb_err = np.zeros_like(xb) + np.nanstd(xt[self.index_cloudy], axis=0, keepdims=True)  # TODO: Fix axis
            background = {'background': xb, 'background_err': xb_err}
        else:
            background = None

        # Outputs
        y = {}
        if outputs is not None:
            for output in outputs:
                if background is not None and output in ['background', 'background_err']:
                    y[output] = background[output]
                elif output == 'prof_mask':
                    y[output] = np.tile(self.prof_mask.astype(np.float64),
                                        (self.sets[stage]['split']['end'] - self.sets[stage]['split']['start'], 1, 1))
                else:
                    f_norm = instantiate(self.sets[stage][output]['normalization']) \
                        if hasattr(self.sets[stage][output], 'normalization') else identity
                    y[output]= f_norm(np.array(instantiate(self.sets[stage][output]['load']), dtype=np.float64)
                                      [self.sets[stage]['split']['start']:self.sets[stage]['split']['end']])
            # If clrsky is True, filter outputs based on clrsky mask
            if self.clrsky_filter:
                for key in y.keys():
                    y[key] = y[key][self.index_cloudy[self.sets[stage]['split']['start']:self.sets[stage]['split']['end']]]

            return CRTMDataset(x, y)
        return CRTMDataset(x)


class VarDataloader(BaseDataloader):
    def __init__(self, sets: DictConfig, cloud_filter: bool = False, prof_filter: bool = False,
                 batch_size: int = 32, num_workers: int = None, pin_memory: bool = True) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        sets: DictConfig. Configuration object for the dataset sets.
        inputs : list or str. List of input variables to be used in the model.
        outputs : list or str. List of output variables to be used in the model. If None, all outputs will be used.
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
        # Clear sky
        self.cloud_filter = cloud_filter

        # Masks and filters
        data = np.array(instantiate(self.sets['train']['prof']['load']), dtype=np.float64)
        index_clrsky = np.logical_and(data[:, 5, :].sum(axis=1) == 0, data[:, 6, :].sum(axis=1) == 0)
        index_clrsky = np.logical_and(index_clrsky, data[:, 7, :].sum(axis=1) == 0)
        self.index_cloudy = ~index_clrsky
        self.prof_mask = data.var(axis=0, keepdims=True) > 0
        self.prof_bcs = data[0:1, ...] * (1.0-self.prof_mask.astype(np.float64))  # Boundary conditions from the first profile
        data = None  # Free memory

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

        # Assign coordinates
        coordinates = assign_coordinates(np.array(instantiate(data['coordinates']['lat']['load']), dtype=np.float64),
                                         np.array(instantiate(data['coordinates']['lon']['load']), dtype=np.float64),
                                         np.array(instantiate(data['coordinates']['scans']['load']), dtype=np.float64))

        # Asemble dataset on a per profile type basis.


        # Initialize empty dataset
        coordinates = {}
        dataset = {}

        # Store coordinates
        for coordinate in data['coordinates']:
            f_norm = instantiate(data['coordinates'][coordinate]['normalization']) \
                if hasattr(data['coordinates'][coordinate], 'normalization') else identity
            if coordinate in ['lat', 'lon', 'scans']:
                coordinates[coordinate] = f_norm(coordinates[coordinate]
                                                 [data['split']['start']:data['split']['end']])
            elif coordinate == 'pressure':
                coordinates[coordinate] = np.tile(f_norm(0.01*np.array(instantiate(data['coordinates'][coordinate]['load']),
                                                                       dtype=np.float64)),
                                                  (data['split']['end']-data['split']['start'], 1))
            else:
                coordinates[coordinate] = f_norm(np.array(instantiate(data['coordinates'][coordinate]['load']),
                                                          dtype=np.float64)
                                                 [data['split']['start']:data['split']['end']])

        # Inputs
        x = {}
        for input in inputs:
            f_norm = instantiate(self.sets[stage][input]['normalization']) \
                if hasattr(self.sets[stage][input], 'normalization') else identity
            if input in ['lat', 'lon', 'scans']:
                x[input] = f_norm(coordinates[input][self.sets[stage]['split']['start']:self.sets[stage]['split']['end']])
            elif input == 'pressure':
                x[input] = np.tile(f_norm(0.01*np.array(instantiate(self.sets[stage][input]['load']), dtype=np.float64)),
                                   (self.sets[stage]['split']['end']-self.sets[stage]['split']['start'], 1))
            else:
                x[input] = f_norm(np.array(instantiate(self.sets[stage][input]['load']), dtype=np.float64)
                                  [self.sets[stage]['split']['start']:self.sets[stage]['split']['end']])
        # If clrsky is True, filter inputs based on clrsky mask
        if self.clrsky_filter:
            for key in x.keys():
                x[key] = x[key][self.index_cloudy[self.sets[stage]['split']['start']:self.sets[stage]['split']['end']]]

        # TODO: Verify which normalization function to use for background
        # If outputs include background, compute it
        if 'background' in outputs:
            # Compute background
            var = 'prof_norm' if 'prof_norm' in self.sets[stage] else 'prof'
            # Load the profile data and apply normalization
            f_norm = instantiate(self.sets[stage][var]['normalization']) \
                if hasattr(self.sets[stage][var], 'normalization') else identity
            xt = f_norm(np.array(instantiate(self.sets[stage][var]['load']), dtype=np.float64))
            xb = (np.zeros_like(xt[self.sets[stage]['split']['start']:self.sets[stage]['split']['end']]) +
                  np.nanmean(xt[self.index_cloudy], axis=0, keepdims=True))  # Compute mean and replace NaN with 0
            xb_err = np.zeros_like(xb) + np.nanstd(xt[self.index_cloudy], axis=0, keepdims=True)  # TODO: Fix axis
            background = {'background': xb, 'background_err': xb_err}
        else:
            background = None

        # Outputs
        y = {}
        if outputs is not None:
            for output in outputs:
                if background is not None and output in ['background', 'background_err']:
                    y[output] = background[output]
                elif output == 'prof_mask':
                    y[output] = np.tile(self.prof_mask.astype(np.float64),
                                        (self.sets[stage]['split']['end'] - self.sets[stage]['split']['start'], 1, 1))
                else:
                    f_norm = instantiate(self.sets[stage][output]['normalization']) \
                        if hasattr(self.sets[stage][output], 'normalization') else identity
                    y[output]= f_norm(np.array(instantiate(self.sets[stage][output]['load']), dtype=np.float64)
                                      [self.sets[stage]['split']['start']:self.sets[stage]['split']['end']])
            # If clrsky is True, filter outputs based on clrsky mask
            if self.clrsky_filter:
                for key in y.keys():
                    y[key] = y[key][self.index_cloudy[self.sets[stage]['split']['start']:self.sets[stage]['split']['end']]]

            return CRTMDataset(x, y)
        return CRTMDataset(x)

class MultiVarDataloader(BaseDataloader):
    def __init__(self, sets: DictConfig, cloud_filter: bool = False, prof_filter: bool = False,
                 batch_size: int = 32, num_workers: int = None, pin_memory: bool = True) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        sets: DictConfig. Configuration object for the dataset sets.
        cloud_filter : bool. If True, filter the dataset to include only cloudy profiles.
        prof_filter : bool. If True, filter the dataset to include only profiles with non-zero variance.
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

        # Cloud filter  # TODO: This does not make sense in the context of predictions
        data = np.array(instantiate(self.sets['train']['prof']['load']), dtype=np.float64)
        index_clrsky = np.logical_and(data[:, 5, :].sum(axis=1) == 0, data[:, 6, :].sum(axis=1) == 0)
        index_clrsky = np.logical_and(index_clrsky, data[:, 7, :].sum(axis=1) == 0)
        self.cloud_filter = ~index_clrsky if cloud_filter else np.ones_like(index_clrsky, dtype=bool)
        # Pressure filter
        data_var = data.var(axis=0, keepdims=True)
        self.prof_filter = data_var > 0 if prof_filter else np.ones_like(data_var, dtype=bool)
        # Free memory
        data, data_var = None, None

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

        # Assign coordinates
        coordinates = assign_coordinates(np.array(instantiate(data['coordinates']['lat']['load']), dtype=np.float64),
                                         np.array(instantiate(data['coordinates']['lon']['load']), dtype=np.float64),
                                         np.array(instantiate(data['coordinates']['scans']['load']), dtype=np.float64))

        # Asemble dataset on a per profile type basis.
        dataset = {}
        for p, prof_type in enumerate(data['prof_type']):
            # Store coordinates
            for var, config in (data['coordinates'] + data['targets'] if hasattr(data, 'targets') else None).items():
                f_norm = instantiate(config['normalization']) if hasattr(config, 'normalization') else identity
                if var in ['lat', 'lon', 'scans']:
                    dataset[prof_type][var] = f_norm(coordinates[var])
                elif var == 'pressure':
                    dataset[prof_type][var] = f_norm(0.01*np.array(instantiate(config['load']), dtype=np.float64))
                    if hasattr(data, 'filters'):
                        if 'prof_filter' in data['filters'] and self.prof_filter is not None:
                            prof_filter = self.prof_filter[0, self.sets['train']['targets']['prof']['type'].index(prof_type)]
                            dataset[prof_type][var] = dataset[prof_type][var][prof_filter]
                        if 'prof_filter' in data['filters']:
                            prof_filter = np.array(instantiate(config['load']), dtype=bool)[0, self.sets['train']['targets']['prof']['type'].index(prof_type)]
                            dataset[prof_type][var] = dataset[prof_type][var][prof_filter]
                else:
                    dataset[prof_type][var] = f_norm(np.array(instantiate(config['load']), dtype=np.float64))
                # Apply split
                if hasattr(data, 'split'):
                    split = data['split'] if self.cloud_filter is None else self.cloud_filter[data['split']]
                    dataset[prof_type][var] = dataset[prof_type][var][split] if var != 'pressure' else dataset[prof_type][var]

        # Store variables
        vars = {**(data['filters'] if hasattr(data, 'filters') else {}),
                **data['coordinates'],
                **(data['targets'] if hasattr(data, 'targets') else {})}
        if hasattr(data, 'split'):
            split = data['split']
            if hasattr(data['filters'], 'cloud_filter'):
                cloud_filter = np.array(instantiate(data['filters']['cloud_filter']['load']), dtype=bool)

        for p, prof_type in enumerate(data['prof_type']):
            for var, config in vars.items():
                f_norm = instantiate(config['normalization']) if hasattr(config, 'normalization') else identity
                if var in ['lat', 'lon', 'scans']:
                    dataset[prof_type][var] = f_norm(coordinates[var])
                elif var == 'pressure':
                    dataset[prof_type][var] = f_norm(0.01*np.array(instantiate(config['load']), dtype=np.float64))
                    if hasattr(dataset, 'prof_filter'):
                        prof_filter = np.array(instantiate(config['load']), dtype=bool)[0, self.sets['train']['targets']['prof']['type'].index(prof_type)]
                        if self.prof_filter:
                            dataset[prof_type][var] = dataset[prof_type][var][prof_filter]
                else:
                    dataset[prof_type][var] = f_norm(np.array(instantiate(config['load']), dtype=np.float64))
                # Apply split
                if hasattr(data, 'split'):
                    split = data['split'] if self.cloud_filter is None else self.cloud_filter[data['split']]
                    dataset[prof_type][var] = dataset[prof_type][var][split] if var != 'pressure' else dataset[prof_type][var]

        return VarDataset(dataset)


class MultiVarDataloader2(BaseDataloader):
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

    def _make_dataset(self, stage: str) -> Dataset:
        """ Create a dataset for the specified stage.

        Parameters
        ----------
        stage : str. Stage of the model ('train', 'valid', 'test', 'pred').

        Returns
        -------
        VarDataset. Dataset object containing the input and output data.
        """

        # Asemble dataset in a dictionary.
        dataset = {}

        # Assign coordinates
        coordinates = assign_coordinates(np.array(instantiate(self.sets['coordinates']['lat']['load']), dtype=np.float64),
                                         np.array(instantiate(self.sets['coordinates']['lon']['load']), dtype=np.float64),
                                         np.array(instantiate(self.sets['coordinates']['scans']['load']), dtype=np.float64))

        # Split
        if hasattr(self.sets, 'split') and stage in ['train', 'valid', 'test']:
            split = self.sets['split'][stage]
            if hasattr(self.sets['split'], 'cloud_filter'):
                cloud_filter = np.array(instantiate(self.sets['split']['cloud_filter']['load']), dtype=bool)
                split = cloud_filter[split]
        else:
            split = None

        # Store coordinates and targets (if available)
        labels = ['coordinates'] + ['targets' if stage in ['train', 'valid', 'test'] else []]
        for label in labels:
            for var, config in self.sets[label].items():
                f_norm = instantiate(config['normalization']) if hasattr(config, 'normalization') else identity
                if var in ['lat', 'lon', 'scans']:
                    dataset[label][var] = f_norm(coordinates[var])
                elif var == 'pressure':
                    dataset[label][var] = np.tile(f_norm(0.01*np.array(instantiate(config['load']), dtype=np.float64)), (self.batch_size, 1))
                else:
                    dataset[label][var] = f_norm(np.array(instantiate(config['load']), dtype=np.float64))
                # Apply split
                if split is not None and var not in ['pressure']:
                    dataset[label][var] = dataset[label][var][split]

        return VarDataset(dataset)


class MultiVarDataloader3(BaseDataloader):
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
                if split is not None and var not in ['pressure']:
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
