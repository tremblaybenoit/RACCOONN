from torch.utils.data import Dataset
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from utilities.tensors import to_tensor
from code.data.transformations import compose_transformations
import os
import numpy as np
import torch
import torch.multiprocessing as mp
# Set this BEFORE any dataloader starts
mp.set_sharing_strategy('file_system')
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class UnivariateDataset(Dataset):
    """ Base class for single-variable datasets with transformation pipelines.

        Provides common infrastructure for building transformation pipelines
        (preprocessing, normalization, etc.) and applying them in order.
        Subclasses control how data is loaded (eager vs. lazy) and how indexing works.
    """

    def __init__(
        self,
        transformations: DictConfig | None = None,
        as_tensor: bool = True,
        apply_transformations: bool = False,
        **kwargs,
    ) -> None:
        """ Initialize UnivariateDataset.

            Parameters
            ----------
            transformations: DictConfig or None. Contains transformation steps for this variable:
                             - preprocessing: initial transformations (if any)
                             - normalization: statistical normalization (if any)
                             - ... additional steps as needed, all with _partial_: true
            as_tensor      : bool. If True, return torch.Tensor; if False, return np.ndarray.
                             Default True (tensor).
            apply_transformations : bool. If True, apply transformations to data.
                                    Default False (no transformations).
            **kwargs       : Additional fields from config passed but not used by base class.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Store output format preference
        self.as_tensor = as_tensor
        self.apply_transformations = apply_transformations

        # Create transformation functions using transform
        self.transform_fn = compose_transformations(transformations, inverse_transform=False)
        self.inverse_transform_fn = compose_transformations(transformations, inverse_transform=True)

    def __len__(self) -> int:
        """ Return length of the dataset. Must be implemented by subclass.

            Returns
            -------
            int: Number of samples.
        """
        raise NotImplementedError("Subclass must implement __len__")

    def __getitem__(self, index: int | slice) -> np.ndarray | torch.Tensor:
        """ Get item from dataset. Must be implemented by subclass.

            Parameters
            ----------
            index : int or slice. Index or slice to retrieve.

            Returns
            -------
            np.ndarray or torch.Tensor. Data at the given index, format determined by as_tensor.
        """
        raise NotImplementedError("Subclass must implement __getitem__")


class EagerDataset(UnivariateDataset):
    """ Base class for eager-loaded single-variable datasets.

        Inherits from UnivariateDataset. Loads data from a file at initialization,
        applies a transformation pipeline, and stores the result as numpy array.
        Designed to be subclassed or used directly for indexed per-sample access.
        Output format (numpy or tensor) is controlled by as_tensor parameter.
    """

    def __init__(
        self,
        load: DictConfig,
        transformations: DictConfig | None = None,
        type: str | list[str] | None = None,
        as_tensor: bool = True,
        apply_transformations: bool = False,
        **kwargs,
    ) -> None:
        """ Initialize EagerDataset.

            Parameters
            ----------
            load            : DictConfig. Loading function config (e.g. data.io.load_npy).
            transformations : DictConfig or None. Transformation pipeline config.
            type            : str or list[str] or None. Variable(s) string(s).
            as_tensor       : bool. If True, return tensors; if False, return numpy arrays.
                              Default True.
            apply_transformations : bool. If True, apply transformations to data.
                                    Default False (no transformations).
            **kwargs        : Additional fields from config passed but not used.

            Returns
            -------
            None.
        """

        # Initialize base class
        super().__init__(transformations=transformations, as_tensor=as_tensor, 
                        apply_transformations=apply_transformations, **kwargs)

        # Load and transform data once at initialization
        self.load_cfg = load
        self.data = self.load()
        self.type = type

    def load(self) -> np.ndarray | torch.Tensor:
        """ Load data from file and apply transformation pipeline if enabled.

            Returns
            -------
            np.ndarray or torch.Tensor. Transformed data in requested format.
                - If as_tensor=True: torch.Tensor with shared memory (for multiprocessing efficiency).
                - If as_tensor=False: np.ndarray (contiguous).
        """
        # Load raw data
        arr = instantiate(self.load_cfg)

        # Apply transformations only if enabled
        if self.apply_transformations:
            arr = self.transform_fn(arr)

        # Convert to requested output format (only once, at init time)
        if self.as_tensor:
            # Tensor with shared memory for multiprocessing
            return to_tensor(arr, shared=True)
        else:
            # Keep as contiguous numpy array
            return np.ascontiguousarray(arr)

    def __len__(self) -> int:
        """ Return length of the dataset.

            Returns
            -------
            int: Number of samples (first dimension of data array).
        """
        return len(self.data)

    def __getitem__(self, index: int | slice) -> np.ndarray | torch.Tensor:
        """ Get item from dataset by index.

            Data format (numpy or tensor) is determined by as_tensor parameter set at init time.

            Parameters
            ----------
            index : int or slice. Index or slice to retrieve.

            Returns
            -------
            np.ndarray or torch.Tensor. Data at the given index (already in requested format).
        """
        return self.data[index]


class ConstantDataset(EagerDataset):
    """ Dataset for single variables with constant (non-indexed) values.

        Inherits from EagerDataset. Loads a constant value at initialization and
        returns the same value regardless of index. Each __getitem__ call returns
        the same constant, with output format controlled by as_tensor parameter.

        Ideal for variables that have a fixed value shared across all samples.
    """

    def __init__(
        self,
        load: DictConfig,
        transformations: DictConfig | None = None,
        type: str | list[str] | None = None,
        squeeze: bool = True,
        as_tensor: bool = True,
        apply_transformations: bool = False,
        **kwargs,
    ) -> None:
        """ Initialize ConstantDataset.

            Parameters
            ----------
            load            : DictConfig. Loading function config with _target_ specifying the loader.
            transformations : DictConfig or None. Transformation pipeline config.
            type:           : str or list[str] or None. Variable(s) string(s).
            squeeze         : bool. If True and data has shape[0]==1, squeeze the first dimension.
                              Default True.
            as_tensor       : bool. If True, return tensors; if False, return numpy arrays.
                              Default True.
            apply_transformations : bool. If True, apply transformations to data.
                                    Default False (no transformations).
            **kwargs        : Additional fields from config passed but not used.

            Returns
            -------
            None.
        """

        # Whether to squeeze the data or not
        self.squeeze = squeeze

        # Initialize parent class (loads and transforms data)
        super().__init__(load=load, transformations=transformations, as_tensor=as_tensor, 
                        apply_transformations=apply_transformations, type=type, **kwargs)

        # Apply squeeze if requested
        if self.squeeze and self.data.shape[0] == 1:
            self.data = self.data.squeeze(0)

    def __len__(self) -> int:
        """ Return length of the dataset.

            For a constant dataset, this always returns 1 since there is only one value.

            Returns
            -------
            int: Always 1 (single constant value).
        """
        return 1

    def __getitem__(self, index: int | slice) -> np.ndarray | torch.Tensor:
        """ Get constant value (index is ignored).

            Data format (numpy or tensor) is determined by as_tensor parameter set at init time.

            Parameters
            ----------
            index : int or slice. Index (ignored for constants).

            Returns
            -------
            np.ndarray or torch.Tensor. The constant value (already in requested format).
        """
        return self.data


class LazyDataset(UnivariateDataset):
    """ Dataset for single variables with on-demand file loading.

        Inherits from UnivariateDataset. Instead of loading all data at initialization,
        LazyDataset discovers a list of file paths and loads them on-demand in __getitem__.
        Each sample is loaded from disk, transformed via the pipeline, and returned.
        Output format (numpy or tensor) is controlled by as_tensor parameter.

        Ideal for large datasets where eager loading would consume too much memory.
        Transforms are applied identically to EagerDataset, but only for the requested sample.
    """

    def __init__(
        self,
        path: DictConfig,
        load: DictConfig,
        transformations: DictConfig | None = None,
        as_tensor: bool = True,
        apply_transformations: bool = False,
        **kwargs,
    ) -> None:
        """ Initialize LazyDataset.

            Parameters
            ----------
            path            : DictConfig. Config with _target_ pointing to a file discovery
                              function (e.g. data.filters.filter_files) that returns a list
                              of file paths. File discovery and split boundaries are fully
                              embedded via Hydra interpolation.
            load            : DictConfig. Loading function config (e.g. data.io.load_npy)
                              with _target_ specifying the loader function. At runtime,
                              this will be instantiated with path=files[idx].
            transformations : DictConfig or None. Transformation pipeline config.
            as_tensor       : bool. If True, return tensors; if False, return numpy arrays.
                              Default True.
            apply_transformations : bool. If True, apply transformations to data.
                                    Default False (no transformations).
            **kwargs        : Additional fields from config passed but not used.

            Returns
            -------
            None.
        """

        # Initialize base class (builds transformation pipelines)
        super().__init__(transformations=transformations, as_tensor=as_tensor, 
                        apply_transformations=apply_transformations, **kwargs)

        # Resolve file list at construction time; split is baked-in via Hydra interpolation
        self.files = instantiate(path)
        self.load_fn = instantiate(load)
        self.data = None

    def load(self) -> np.ndarray | torch.Tensor:
        """ Load all files and apply transformation pipeline if enabled.

            Concatenates all files into a single array, applies transformations (if enabled),
            and converts to requested output format. Useful for materializing
            lazy datasets into memory when needed (e.g., for compatibility with
            eager-only code).

            Returns
            -------
            np.ndarray or torch.Tensor. Concatenated transformed data in requested format.
                - If as_tensor=True: torch.Tensor with shared memory (for multiprocessing).
                - If as_tensor=False: np.ndarray (contiguous).
        """
        # Load all files and concatenate
        arrays = [np.ascontiguousarray(self.load_fn(path=f)) for f in self.files]
        arr = np.concatenate(arrays, axis=0)

        # Apply transformations only if enabled
        if self.apply_transformations:
            arr = self.transform_fn(arr)

        # Convert to requested output format
        if self.as_tensor:
            # Tensor with shared memory for multiprocessing
            return to_tensor(arr, shared=True)
        else:
            # Keep as contiguous numpy array
            return np.ascontiguousarray(arr)

    def __len__(self) -> int:
        """ Return length of the dataset.

            Returns
            -------
            int: Number of files in the discovered file list.
        """
        return len(self.files)

    def __getitem__(self, index: int | slice) -> np.ndarray | torch.Tensor:
        """ Load one file on-demand, apply transformations if enabled, and convert to requested format.

            Parameters
            ----------
            index : int or slice. Index or slice into the file list.

            Returns
            -------
            np.ndarray or torch.Tensor. Transformed data from the loaded file,
                                         format determined by as_tensor.
        """

        # Load the file at this index using the instantiated load config
        arr = self.load_fn(path=self.files[index])

        # Ensure contiguous array
        arr = np.ascontiguousarray(arr)

        # Apply transformations only if enabled
        if self.apply_transformations:
            arr = self.transform_fn(arr)

        # Convert to requested output format (per-item, no shared memory for ephemeral tensors)
        if self.as_tensor:
            return to_tensor(arr, shared=False)
        else:
            return arr


class MultivariateDataset(Dataset):
    """ Dataset combining multiple univariate datasets for multi-branch inputs/outputs.

        Each entry in `input` (and optionally `target` and `context`) is a dataset config
        with its own `_target_` pointing to a dataset class and nested configs
        (load, transformations, etc.). MultivariateDataset instantiates each one independently
        and combines them into aligned samples, supporting mixed dataset types
        (e.g., both EagerDataset and ConstantDataset).

        The as_tensor parameter is imposed on all sub-datasets, overriding their individual
        settings for consistency.
    """

    def __init__(
        self,
        variables: DictConfig,
        as_tensor: bool = True,
        apply_transformations: bool = False,
    ) -> None:
        """ Initialize MultivariateDataset.

            Parameters
            ----------
            variables   : DictConfig. Ordered mapping of variable name → dataset config.
                      Each entry should carry _target_ and _recursive_: false so nested
                      configs (load, transformations, path, etc.) are passed as-is and
                      resolved inside the dataset's __init__.
            as_tensor : bool. If True, all sub-datasets return tensors; if False, numpy arrays.
                        This value is imposed on all sub-datasets, overriding their configs.
                        Default True.
            apply_transformations : bool. If True, apply transformations to sub-datasets.
                                    Default False (no transformations).

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Impose as_tensor and apply_transformations on input datasets (one per input variable)
        self.keys = list(variables.keys())
        self.datasets = {
            key: instantiate(cfg, as_tensor=as_tensor, apply_transformations=apply_transformations)
            for key, cfg in variables.items()
        }

        # Infer dataset length from first input variable
        self._len = len(next(iter(self.datasets.values())))

    def __len__(self) -> int:
        """ Return length of the dataset.

            Returns
            -------
            int: Number of samples.
        """
        return self._len

    def __getitem__(self, index: int | slice) -> dict:
        """ Get sample at index, assembling input and target dictionaries.

            Parameters
            ----------
            index : int. Index to retrieve.

            Returns
            -------
            dict with keys:
                - 'input': dict mapping variable name → tensor
                - 'target': dict mapping variable name → tensor (if targets provided)
        """

        # Build input dictionary by fetching from each input variable dataset
        out = {
            key: self.datasets[key][index]
            for key in self.keys
        }

        return out


class MultivariateDatasets(Dataset):
    """ Dataset combining multiple univariate datasets for multi-branch inputs/outputs.

        Each entry in `input` (and optionally `target` and `context`) is a dataset config
        with its own `_target_` pointing to a dataset class and nested configs
        (load, transformations, etc.). MultivariateDataset instantiates each one independently
        and combines them into aligned samples, supporting mixed dataset types
        (e.g., both EagerDataset and ConstantDataset).

        The as_tensor parameter is imposed on all sub-datasets, overriding their individual
        settings for consistency.
    """

    def __init__(
        self,
        input: DictConfig,
        target: DictConfig | None = None,
        context: DictConfig | None = None,
        output: DictConfig | None = None,
        as_tensor: bool = True,
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
            output  : DictConfig or None. Results configuration for saving outputs.
            as_tensor : bool. If True, all sub-datasets return tensors; if False, numpy arrays.
                        This value is imposed on all sub-datasets, overriding their configs.
                        Default True.

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Impose as_tensor on input datasets (one per input variable)
        self.input = instantiate(input, as_tensor=as_tensor)

        # Impose as_tensor on target datasets if provided (one per target variable)
        if target is not None:
            self.target = instantiate(target, as_tensor=as_tensor)
        else:
            self.target = None

        # Impose as_tensor on context datasets if provided (one per context variable)
        if context is not None:
            self.context = instantiate(context, as_tensor=as_tensor)
        else:
            self.context = None

        # Store output config
        self.output = output

        # Infer dataset length from first input variable
        self._len = len(self.input)

    def __len__(self) -> int:
        """ Return length of the dataset.

            Returns
            -------
            int: Number of samples.
        """
        return self._len

    def __getitem__(self, index: int | slice) -> dict:
        """ Get sample at index, assembling input and target dictionaries.

            Parameters
            ----------
            index : int. Index to retrieve.

            Returns
            -------
            dict with keys:
                - 'input': dict mapping variable name → tensor
                - 'target': dict mapping variable name → tensor (if targets provided)
        """

        # Build input dictionary by fetching from each input variable dataset
        input_dict = self.input[index]
        out = {'input': input_dict}

        # Add target dictionary if targets are provided
        if self.target is not None:
            target_dict = self.target[index]
            out['target'] = target_dict

        # Add context dictionary if context is provided
        if self.context is not None:
            context_dict = self.context[index]
            out['context'] = context_dict

        return out


class MultivariateDataset0(Dataset):
    """ Dataset combining multiple univariate datasets for multi-branch inputs/outputs.

        Each entry in `input` (and optionally `target` and `context`) is a dataset config
        with its own `_target_` pointing to a dataset class and nested configs
        (load, transformations, etc.). MultivariateDataset instantiates each one independently
        and combines them into aligned samples, supporting mixed dataset types
        (e.g., both EagerDataset and ConstantDataset).

        The as_tensor parameter is imposed on all sub-datasets, overriding their individual
        settings for consistency.
    """

    def __init__(
        self,
        input: DictConfig,
        target: DictConfig | None = None,
        context: DictConfig | None = None,
        output: DictConfig | None = None,
        as_tensor: bool = True,
        apply_transformations: bool = False,
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
            output  : DictConfig or None. Results configuration for saving outputs.
            as_tensor : bool. If True, all sub-datasets return tensors; if False, numpy arrays.
                        This value is imposed on all sub-datasets, overriding their configs.
                        Default True.
            apply_transformations : bool. If True, apply transformations to sub-datasets.
                                    Default False (no transformations).

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Impose as_tensor and apply_transformations on input datasets (one per input variable)
        self.input_keys = list(input.keys())
        self.input_datasets = {
            key: instantiate(cfg, as_tensor=as_tensor, apply_transformations=apply_transformations)
            for key, cfg in input.items()
        }

        # Impose as_tensor and apply_transformations on target datasets if provided (one per target variable)
        self.target_keys = list(target.keys()) if target is not None else None
        if target is not None:
            self.target_datasets = {
                key: instantiate(cfg, as_tensor=as_tensor, apply_transformations=apply_transformations)
                for key, cfg in target.items()
            }
        else:
            self.target_datasets = None

        # Impose as_tensor and apply_transformations on context datasets if provided (one per context variable)
        self.context_keys = list(context.keys()) if context is not None else None
        if context is not None:
            self.context_datasets = {
                key: instantiate(cfg, as_tensor=as_tensor, apply_transformations=apply_transformations)
                for key, cfg in context.items()
            }
        else:
            self.context_datasets = None

        # Store output config
        self.output = output

        # Infer dataset length from first input variable
        self._len = len(next(iter(self.input_datasets.values())))

    def __len__(self) -> int:
        """ Return length of the dataset.

            Returns
            -------
            int: Number of samples.
        """
        return self._len

    def __getitem__(self, index: int | slice) -> dict:
        """ Get sample at index, assembling input and target dictionaries.

            Parameters
            ----------
            index : int. Index to retrieve.

            Returns
            -------
            dict with keys:
                - 'input': dict mapping variable name → tensor
                - 'target': dict mapping variable name → tensor (if targets provided)
        """

        # Build input dictionary by fetching from each input variable dataset
        input_dict = {
            key: self.input_datasets[key][index]
            for key in self.input_keys
        }

        out = {'input': input_dict}

        # Add target dictionary if targets are provided
        if self.target_datasets is not None and self.target_keys is not None:
            target_dict = {
                key: self.target_datasets[key][index]
                for key in self.target_keys
            }
            out['target'] = target_dict

        # Add context dictionary if context is provided
        if self.context_datasets is not None and self.context_keys is not None:
            context_dict = {
                key: self.context_datasets[key][index]
                for key in self.context_keys
            }
            out['context'] = context_dict

        return out
