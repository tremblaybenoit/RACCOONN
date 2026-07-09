import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
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


class UnivariateDataset(Dataset):
    """ Base class for eager-loaded single-variable datasets with transformation pipelines.

        Loads data from a file once at initialization, applies a transformation pipeline
        (preprocessing, normalization, etc.) in order, and stores the result in shared
        memory tensors for efficient multiprocessing. Subclasses control how indexing
        works and handle different data shapes.
    """

    def __init__(
        self,
        load: DictConfig,
        transformations: DictConfig | None = None,
        **kwargs,
    ) -> None:
        """ Initialize UnivariateDataset.

            Parameters
            ----------
            load           : DictConfig. Loading function config (e.g. data.io.load_npy)
                             with _target_ specifying the loader function.
            transformations: DictConfig or None. Contains transformation steps for this variable:
                             - preprocessing: initial transformations (if any)
                             - normalization: statistical normalization (if any)
                             - ... additional steps as needed, all with _partial_: true
            **kwargs       : Additional fields from config passed but not used by base class.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Store load config for later use
        self.load_cfg = load

        # Build transformation pipeline in order
        self.pipeline = []
        self.pipeline_inverse = []
        if transformations is not None:
            # Add any transformation steps
            for key in transformations.keys():
                self.pipeline.append(instantiate(transformations[key]))
                # Check if transformation has an inverse operation
                if hasattr(transformations[key], 'inverse_transform'):
                    # If so, store the transformation but with inverse_transform set to True
                    cfg = transformations[key].copy()
                    cfg.inverse_transform = True
                    self.pipeline_inverse.append(instantiate(cfg))
        # Reverse the order of the inverse transformations
        self.pipeline_inverse.reverse()

    def _load(self) -> torch.Tensor:
        """ Load data from file without transformations.

            Returns
            -------
            torch.Tensor. Raw data tensor in shared memory.
        """

        # Load data using the instantiated load config
        arr = instantiate(self.load_cfg)

        # Convert to contiguous array for efficient memory layout
        arr = np.ascontiguousarray(arr)

        # Create torch tensor from numpy array
        t = torch.from_numpy(arr)

        # Enable sharing across processes
        t.share_memory_()

        return t

    def _load_and_transform(self) -> torch.Tensor:
        """ Load data from file and apply transformation pipeline.

            Returns
            -------
            torch.Tensor. Transformed data tensor in shared memory.
        """

        # Load raw data
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

    def _inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """ Apply inverse transformation pipeline.

            Parameters
            ----------
            data: torch.Tensor. Transformed data tensor.

            Returns
            -------
            torch.Tensor. Inverse transformed data tensor.
        """

        # Apply inverse transformations in sequence
        for transform in self.pipeline_inverse:
            data = transform(data)

        return data

    def __len__(self) -> int:
        """ Return length of the dataset. Must be implemented by subclass.

            Returns
            -------
            int: Number of samples.
        """
        raise NotImplementedError("Subclass must implement __len__")

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Get item from dataset. Must be implemented by subclass.

            Parameters
            ----------
            idx : int. Index to retrieve.

            Returns
            -------
            torch.Tensor. Data at the given index.
        """
        raise NotImplementedError("Subclass must implement __getitem__")


class EagerDataset(UnivariateDataset):
    """ Dataset for single variables with per-sample indexed data.

        Inherits from UnivariateDataset. Loads data from a file, applies a transformation
        pipeline, and stores the result in shared memory. Each __getitem__ call returns
        data at the specified index for efficient indexed access across samples.
    """

    def __init__(
        self,
        load: DictConfig,
        transformations: DictConfig | None = None,
        **kwargs,
    ) -> None:
        """ Initialize EagerDataset.

            Parameters
            ----------
            load        : DictConfig. Loading function config (e.g. data.io.load_npy).
            transformations: DictConfig or None. Transformation pipeline config.
            **kwargs    : Additional fields from config passed but not used.

            Returns
            -------
            None.
        """

        # Initialize base class
        super().__init__(load=load, transformations=transformations, **kwargs)

        # Load and transform data once at initialization (will be stored in subclass)
        self.data = self._load_and_transform()

    def __len__(self) -> int:
        """ Return length of the dataset.

            Returns
            -------
            int: Number of samples (first dimension of data tensor).
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Get item from dataset by index.

            Parameters
            ----------
            idx : int. Index to retrieve.

            Returns
            -------
            torch.Tensor. Data at the given index.
        """
        return self.data[idx]


class ConstantDataset(UnivariateDataset):
    """ Dataset for single variables with constant (non-indexed) values.

        Inherits from UnivariateDataset. Loads data from a file, applies a transformation
        pipeline, and stores the result as a single constant tensor in shared memory.
        Each __getitem__ call returns the same value regardless of index.

        Ideal for variables that have a fixed value shared across all samples.
    """

    def __init__(
        self,
        load: DictConfig,
        transformations: DictConfig | None = None,
        squeeze: bool = True,
        **kwargs,
    ) -> None:
        """ Initialize ConstantDataset.

            Parameters
            ----------
            load            : DictConfig. Loading function config with _target_ specifying the loader.
            transformations : DictConfig or None. Transformation pipeline config.
            squeeze         : bool. If True and data has shape[0]==1, squeeze the first dimension.
                              Default True.
            **kwargs        : Additional fields from config passed but not used.

            Returns
            -------
            None.
        """

        # Whether to squeeze the data or not
        self.squeeze = squeeze

        # Initialize base class
        super().__init__(load=load, transformations=transformations, **kwargs)

        # Load and transform data once at initialization (will be stored in subclass)
        self.data = self._load_and_transform()

        # Apply squeeze if requested
        if self.squeeze and self.data.shape[0] == 1:
            self.constant = self.data.squeeze(0)

    def __len__(self) -> int:
        """ Return length of the dataset.

            For a constant dataset, this always returns 1 since there is only one value.

            Returns
            -------
            int: Always 1 (single constant value).
        """
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Get constant value (index is ignored).

            Parameters
            ----------
            idx : int. Index (ignored for constants).

            Returns
            -------
            torch.Tensor. The constant tensor (same for all indices).
        """
        return self.data


class MultivariateDataset(Dataset):
    """ Dataset combining multiple univariate datasets for multi-branch inputs/outputs.

        Each entry in `input` (and optionally `target` and `context`) is a dataset config
        with its own `_target_` pointing to a dataset class and nested configs
        (load, transformations, etc.). MultivariateDataset instantiates each one independently
        and combines them into aligned samples, supporting mixed dataset types
        (e.g., both EagerDataset and ConstantDataset).
    """

    def __init__(
        self,
        input: DictConfig,
        target: DictConfig | None = None,
        context: DictConfig | None = None,
        results: DictConfig | None = None,
    ) -> None:
        """ Initialize MultivariateDataset.

            Parameters
            ----------
            input   : DictConfig. Ordered mapping of variable name → dataset config.
                      Each entry should carry _target_ and _recursive_: false so nested
                      configs (load, transformations, path, etc.) are passed as-is and
                      resolved inside the dataset's __init__.
            target  : DictConfig or None. Same structure as input for target variables.
                      Must contain the same number of entries in the same order as input.
            context : DictConfig or None. Same structure as input for context variables.
                      Must contain the same number of entries in the same order as input.
            results : DictConfig or None. Results configuration for saving outputs.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Instantiate input datasets; preserve order via keys list
        self.input_keys = list(input.keys())
        self.input_datasets = {
            key: instantiate(cfg)
            for key, cfg in input.items()
        }

        # Instantiate target datasets if provided; aligned 1:1 with input
        self.target_keys = list(target.keys()) if target is not None else None
        self.target_datasets = (
            {key: instantiate(cfg) for key, cfg in target.items()}
            if target is not None else None
        )

        # Instantiate context datasets if provided; aligned 1:1 with input
        self.context_keys = list(context.keys()) if context is not None else None
        self.context_datasets = (
            {key: instantiate(cfg) for key, cfg in context.items()}
            if context is not None else None
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
        if self.target_datasets is not None and self.target_keys is not None:
            target_dict = {
                key: self.target_datasets[key][idx]
                for key in self.target_keys
            }
            out['target'] = target_dict

        # Add context dictionary if context is provided
        if self.context_datasets is not None and self.context_keys is not None:
            context_dict = {
                key: self.context_datasets[key][idx]
                for key in self.context_keys
            }
            out['context'] = context_dict

        return out
