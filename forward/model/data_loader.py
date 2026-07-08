import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from data.io import load_var_and_normalize, load_var
from utilities.instantiators import instantiate
import os
import numpy as np
import torch
import torch.multiprocessing as mp
# Set this BEFORE any dataloader starts
mp.set_sharing_strategy('file_system')
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BaseDataloader(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int | None = None,
                 persistent_workers: bool = True, pin_memory: bool = True, shuffle: bool = True) -> None:
        """ Base dataloader class.

        Parameters
        ----------
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
        persistent_workers : bool. If True, the data loader will keep workers alive between epochs.
        pin_memory : bool. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        shuffle : bool. If True, the data loader will shuffle the data at every epoch.

        Returns
        -------
        None.
        """

        #  Class inheritance
        super().__init__()

        # Number of cpus
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        # Persistent workers for faster data loading
        self.persistent_workers = persistent_workers
        # Neural network training batch size
        self.batch_size = batch_size
        # Pin memory for faster data transfer
        self.pin_memory = pin_memory
        # Shuffle data at every epoch
        self.shuffle = shuffle

        # Datasets
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        self.ds_pred = None

    def train_dataloader(self) -> DataLoader:
        """ Loads training set.

            Parameters
            ----------
            None.

            Returns
            -------
            Training set (inputs & outputs).

        """
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        """ Load validation set.

            Parameters
            ----------
            None.

            Returns
            -------
            Validation set (inputs & outputs).

        """
        return DataLoader(self.ds_valid, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def test_dataloader(self) -> DataLoader:
        """ Load test set.

            Parameters
            ----------
            None.

            Returns
            -------
            Test set (inputs & outputs).

        """
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def predict_dataloader(self) -> DataLoader:
        """ Load prediction set.

            Parameters
            ----------
            None.

            Returns
            -------
            Prediction set (inputs & outputs if available).

        """
        return DataLoader(self.ds_pred, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)


class Dataloader(BaseDataloader):
    def __init__(self, stage: DictConfig, batch_size: int = 32, num_workers: int | None = None,
                 persistent_workers: bool = True, pin_memory: bool = True) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        stage: DictConfig. Configuration object for the dataset at each stage (train, valid, test, pred).
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
        persistent_workers : bool. If True, the data loader will not shut down the worker processes after a dataset has been consumed.
        pin_memory : bool. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

        Returns
        -------
        None.
        """

        #  Class inheritance
        super().__init__(batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)

        # Data sets
        self.ds_stage = stage

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
            self.ds_train, self.ds_valid = instantiate(self.ds_stage.train), instantiate(self.ds_stage.valid)
        elif stage == 'test':
            # Test/prediction data
            self.ds_test = instantiate(self.ds_stage.test)
        elif stage == 'pred':
            # Prediction data
            self.ds_pred = instantiate(self.ds_stage.predict)


class BaseDataset(Dataset):
    """Base dataset class."""

    def __init__(self, x: dict) -> None:
        """Initialize the dataset class.

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

            Parameters
            ----------
            None.

            Returns
            -------
            int. Length of the dataset.
        """
        # Return the length of the first input tensor
        first_dict = next(iter(self.x.values()))
        first_var = next(iter(first_dict.values()))
        return len(first_var)

    def __getitem__(self, idx: int) -> dict:
        """ Get item from data

            Parameters
            ----------
            idx : int. Index of the item to retrieve.

            Returns
            -------
            Dataset object.
        """

        # Get the data at the specified index
        return {k: {kk: vv[idx] if vv.shape[0] == self.__len__() else vv.squeeze()
                    for kk, vv in v.items()} for k, v in self.x.items()}


class TorchDataset(Dataset):
    """ Dataset using torch shared memory tensors. """
    def __init__(self, input: DictConfig, target: DictConfig = None, results: DictConfig = None) -> None:
        """ Initialize the dataset.

            Parameters
            ----------
            input : DictConfig. Configuration for input variables.
            target : DictConfig. Configuration for target variables (optional).
            results : DictConfig. Configuration for results (optional).

            Returns
            -------
            None.
        """

        def _load(var_dict: DictConfig) -> torch.Tensor:
            """ Load and normalize variables from configuration.

                Parameters
                ----------
                var_dict : DictConfig. Configuration for variable.

                Returns
                -------
                torch.Tensor. Loaded and normalized tensor in shared memory.
            """
            arr = load_var_and_normalize(var_dict)
            # if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
            t = torch.from_numpy(arr)
            t.share_memory_()
            return t

        # Automate Input Block Loading
        self.input_blocks, self.input_blocks_keys = {}, []
        self.input_consts, self.input_consts_keys = {}, []
        # We store the keys in a list once so __getitem__ doesn't have to look them up
        input_keys = list(input.keys())
        for key in input_keys:
            data = _load(input[key])
            if data.shape[0] == 1:
                self.input_consts[key] = data.squeeze()
                self.input_consts_keys.append(key)
            elif key == 'pressure':
                # Special case: always treat pressure as constant
                self.input_consts[key] = data
                self.input_consts_keys.append(key)
            else:
                self.input_blocks[key] = data
                self.input_blocks_keys.append(key)
        # 2. Automate Target Block Loading
        self.target_blocks, self.target_blocks_keys = {}, []
        self.target_consts, self.target_consts_keys = {}, []
        self.has_targets = target is not None
        if self.has_targets:
            target_keys = list(target.keys())
            for key in target_keys:
                data = _load(target[key])
                if data.shape[0] == 1:
                    self.target_consts[key] = data.squeeze()
                    self.target_consts_keys.append(key)
                else:
                    self.target_blocks[key] = data
                    self.target_blocks_keys.append(key)

        # Dataset length (assuming 'prof' or the first key exists)
        self._len = self.input_blocks[self.input_blocks_keys[0]].shape[0]

        # Store results config
        self.results = results

    def __len__(self):
        """ Get the length of the dataset."""
        return int(self._len)

    def __getitem__(self, idx: int) -> dict:
        """ Get item from dataset.

            Parameters
            ----------
            idx : int. Index of the item to retrieve.

            Returns
            -------
            Dataset object.
        """

        # Use dictionary comprehension over pre-cached keys
        input_dict = {k: self.input_blocks[k][idx] for k in self.input_blocks_keys}
        for k in self.input_consts_keys:
            input_dict[k] = self.input_consts[k]
        out = {'input': input_dict}

        # Add target dictionary only if it exists
        if self.has_targets:
            # Automates any number of target keys (hofx, cloud_filter, etc.)
            target_dict = {k: self.target_blocks[k][idx] for k in self.target_blocks_keys}
            for k in self.target_consts_keys:
                target_dict[k] = self.target_consts[k]
            out['target'] = target_dict

        return out


class VariableDataset(Dataset):
    """ Dataset for a single variable loaded from a file with transformations.

        Loads a single .npy file, applies a transformation pipeline (preprocessing,
        normalization, etc.) in order, and stores the result in shared memory tensors.

        This is similar to GeoSatDataset but loads an entire file at once (not lazy)
        since RACCOONN uses single files per variable rather than multiple files per sample.
    """

    def __init__(
        self,
        load: DictConfig,
        transformations: DictConfig | None = None,
        **kwargs,
    ) -> None:
        """ Initialize VariableDataset.

            Parameters
            ----------
            load       : DictConfig. Loading function config (e.g. data.io.load_npy)
                         with _target_ specifying the loader function.
            transformations: DictConfig or None. Contains transformation steps for this variable:
                         - preprocessing: initial transformations (if any)
                         - normalization: statistical normalization (if any)
                         - ... additional steps as needed, all with _partial_: true
            **kwargs   : Additional fields from config (path, type, etc.) passed but not used.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Store load config for later use
        self.load_cfg = load

        # Build transformation pipeline in order (preprocessing, then normalization, then others)
        self.pipeline = []
        if transformations is not None:
            # Add preprocessing if available
            if hasattr(transformations, 'preprocessing'):
                self.pipeline.append(instantiate(transformations.preprocessing))
            # Add normalization if available
            if hasattr(transformations, 'normalization'):
                self.pipeline.append(instantiate(transformations.normalization))
            # Add any other transformation steps (in case there are more)
            for key in transformations.keys():
                if key not in ['preprocessing', 'normalization']:
                    self.pipeline.append(instantiate(transformations[key]))

        # Load and transform data once at initialization
        self.data = self._load_and_transform()

    def _load_and_transform(self) -> torch.Tensor:
        """ Load data from file and apply transformation pipeline.

            Returns
            -------
            torch.Tensor. Transformed data tensor in shared memory.
        """

        # Load data using the instantiated load config
        arr = instantiate(self.load_cfg)

        # Apply transformations in sequence
        for transform in self.pipeline:
            arr = transform(arr)

        # Convert to contiguous array for efficient memory layout
        arr = np.ascontiguousarray(arr)

        # Create torch tensor from numpy array
        t = torch.from_numpy(arr)

        # Enable sharing across processes
        t.share_memory_()

        return t

    def __len__(self) -> int:
        """ Return length of the dataset.

            Returns
            -------
            int: Number of samples (first dimension of data tensor).
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Get item from dataset.

            Parameters
            ----------
            idx : int. Index to retrieve.

            Returns
            -------
            torch.Tensor. Data at the given index.
        """
        return self.data[idx]


class TorchDatasets(Dataset):
    """ Dataset combining multiple VariableDatasets for input and target.

        Each entry in `input` (and optionally `target`) is a VariableDataset config
        with its own `_target_` pointing to VariableDataset and nested configs
        (load, transformations, etc.). TorchDatasets instantiates each one independently
        and combines them into a single flat index space, returning input and target
        dictionaries with aligned samples.
    """

    def __init__(
        self,
        input: DictConfig,
        target: DictConfig | None = None,
        results: DictConfig | None = None,
    ) -> None:
        """ Initialize TorchDatasets.

            Parameters
            ----------
            input  : DictConfig. Ordered mapping of variable name → VariableDataset config.
                     Each entry should carry _target_ and _recursive_: false so nested
                     configs (load, transformations, path, etc.) are passed as-is and
                     resolved inside VariableDataset's __init__.
            target : DictConfig or None. Same structure as input. Must contain the same
                     number of entries in the same order as input, or None if no targets.
            results: DictConfig or None. Results configuration for saving outputs.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Instantiate input VariableDatasets; preserve order via keys list
        self.input_keys = list(input.keys())
        self.input_datasets = {
            key: instantiate(cfg)
            for key, cfg in input.items()
        }

        # Instantiate target VariableDatasets if provided; aligned 1:1 with input
        self.target_keys = list(target.keys()) if target is not None else None
        self.target_datasets = (
            {key: instantiate(cfg) for key, cfg in target.items()}
            if target is not None else None
        )

        # Store results config
        self.results = results

        # Infer dataset length from first input variable
        self._len = len(next(iter(self.input_datasets.values())))

    def __len__(self) -> int:
        """ Return length of the dataset.

            Returns
            -------
            int: Number of samples.
        """
        return self._len

    def __getitem__(self, idx: int) -> dict:
        """ Get sample at index, assembling input and target dictionaries.

            Parameters
            ----------
            idx : int. Index to retrieve.

            Returns
            -------
            dict with keys:
                - 'input': dict mapping variable name → tensor
                - 'target': dict mapping variable name → tensor (if targets provided)
        """

        # Build input dictionary by fetching from each input variable dataset
        input_dict = {
            key: self.input_datasets[key][idx]
            for key in self.input_keys
        }

        out = {'input': input_dict}

        # Add target dictionary if targets are provided
        if self.target_datasets is not None:
            target_dict = {
                key: self.target_datasets[key][idx]
                for key in self.target_keys
            }
            out['target'] = target_dict

        return out
