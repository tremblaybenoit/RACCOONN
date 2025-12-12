import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from data.io import load_var_and_normalize
from utilities.instantiators import instantiate
from typing import Union, Optional
from utilities.tensors import to_torch
import multiprocessing
import os
from tqdm import tqdm
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BaseDataloader(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = None,
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
    def __init__(self, stage: DictConfig, batch_size: int = 32, num_workers: int = None,
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
            self.ds_train, self.ds_valid = instantiate(self.stage.train), instantiate(self.stage.valid)
        elif stage == 'test':
            # Test/prediction data
            self.ds_test = instantiate(self.stage.test)
        elif stage == 'pred':
            # Prediction data
            self.ds_pred = instantiate(self.stage.predict)


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


class CRTMDataset(BaseDataset):
    """CRTMDataset supporting either numpy.memmap (disk-backed) or torch shared tensors.

    Args:
      input: DictConfig for input vars.
      target: optional DictConfig for target vars.
      results: optional results config.
      memmap_dir: directory for memmap files (required for memmap mode).
      share_memory: 'memmap' | 'torch' -- storage strategy. Use 'torch' on Windows to avoid WinError 8.
    """
    def __init__(self, input: DictConfig, target: DictConfig = None, results: DictConfig = None,
                 memmap_dir: str = '../mm', share_memory: str = 'torch') -> None:
        if share_memory not in ('memmap', 'torch'):
            raise ValueError("share_memory must be 'memmap' or 'torch'")
        if share_memory == 'memmap' and memmap_dir is None:
            raise ValueError("memmap_dir must be provided for memmap mode")

        self.results = results
        self.memmap_dir = memmap_dir
        self.share_memory = share_memory
        self._raw_x = {'input': {}}
        self._memmaps_opened = False

        if self.share_memory == 'memmap':
            os.makedirs(self.memmap_dir, exist_ok=True)

        # load and store input vars
        for var, config in tqdm(input.items()):
            arr = load_var_and_normalize(config)
            if self.share_memory == 'torch':
                t = torch.from_numpy(np.ascontiguousarray(arr))
                t.share_memory_()
                self._raw_x['input'][var] = t
                del arr
            else:  # memmap
                shape = arr.shape
                dtype = arr.dtype
                fname = os.path.join(self.memmap_dir, f"{var}.dat")
                mm = np.memmap(fname, dtype=dtype, mode='w+', shape=shape)
                mm[:] = arr[:]
                mm.flush()
                del arr
                self._raw_x['input'][var] = {'_memmap_path': fname, 'dtype': str(dtype), 'shape': shape}

        # load and store target vars if provided
        if target is not None:
            self._raw_x['target'] = {}
            for var, config in target.items():
                arr = load_var_and_normalize(config)
                if self.share_memory == 'torch':
                    t = torch.from_numpy(np.ascontiguousarray(arr))
                    t.share_memory_()
                    self._raw_x['target'][var] = t
                    del arr
                else:
                    shape = arr.shape
                    dtype = arr.dtype
                    fname = os.path.join(self.memmap_dir, f"target_{var}.dat")
                    mm = np.memmap(fname, dtype=dtype, mode='w+', shape=shape)
                    mm[:] = arr[:]
                    mm.flush()
                    del arr
                    self._raw_x['target'][var] = {'_memmap_path': fname, 'dtype': str(dtype), 'shape': shape}

        # initialize BaseDataset with descriptor; memmaps opened lazily
        super().__init__(self._raw_x)

    def _ensure_memmaps_opened(self):
        """Open memmaps in current process (only used in memmap mode)."""
        if self._memmaps_opened or self.share_memory == 'torch':
            return

        for grouping in ('input', 'target'):
            if grouping not in self._raw_x:
                continue
            for var, val in list(self._raw_x[grouping].items()):
                if isinstance(val, dict) and '_memmap_path' in val:
                    path = val['_memmap_path']
                    dtype = np.dtype(val['dtype'])
                    shape = tuple(val['shape'])
                    mm = np.memmap(path, dtype=dtype, mode='r', shape=shape)
                    self._raw_x[grouping][var] = mm

        self.x = self._raw_x
        self._memmaps_opened = True

    def __len__(self) -> int:
        if self.share_memory == 'memmap' and not self._memmaps_opened:
            self._ensure_memmaps_opened()
        return super().__len__()

    def __getitem__(self, idx: int) -> dict:
        if self.share_memory == 'memmap' and not self._memmaps_opened:
            self._ensure_memmaps_opened()
        return super().__getitem__(idx)


class TorchDataset(BaseDataset):
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

        def _load(var_dict: DictConfig) -> dict:
            """ Load and normalize variables from configuration.

                Parameters
                ----------
                var_dict : DictConfig. Configuration for variables.

                Returns
                -------
                dict. Dictionary of loaded and normalized variables as torch tensors.
            """
            loaded_vars = {}
            for var, config in tqdm(var_dict.items()):
                # Load and normalize variable
                arr = load_var_and_normalize(config)
                arr = np.ascontiguousarray(arr)
                # Create torch tensor and share memory
                tnsr = torch.from_numpy(arr)
                tnsr.share_memory_()
                loaded_vars[var] = tnsr
                # Free numpy array memory
                del arr
            return loaded_vars

        # Store results config
        self.results = results
        # Load input variables
        x = {'input': _load(input)}
        # Load target variables if provided
        if target is not None:
            x['target'] = _load(target)

        # Class inheritance
        super().__init__(x)


class MemmapDataset(Dataset):
    """ Dataset using numpy memmaps stored on disk with lazy loading and per-process handling. """
    def __init__(self,
                 input: DictConfig, target: Optional[DictConfig] = None,
                 split: Union[slice, np.ndarray, int] = None) -> None:
        """ Initialize the dataset.

            Parameters
            ----------
            input : DictConfig. Configuration for input variables.
            target : DictConfig. Configuration for target variables (optional).
            split : slice | np.ndarray | int. Global split to apply to all variables (optional).

            Returns
            -------
            None.
        """

        # Initialize the configs
        self.input_config = input
        self.target_config = target
        self.split = split

        # Detect multiprocessing method (fork means inherited memmaps are usable)
        try:
            start = multiprocessing.get_start_method(allow_none=True)
        except (RuntimeError, ValueError):
            start = None
        self._is_fork = (start == 'fork')

        # Storage for opened arrays (opened per-process in _ensure_open)
        self.x = {'input': {}}
        if target is not None:
            self.x['target'] = {}
        # Process ID
        self._opened_pid: Optional[int] = None

        # Compute dataset length once (without applying heavy copies) by probing first input var
        first_config = next(iter(self.input_config.values()))
        probe = load_var_and_normalize(first_config, split=None)
        total = probe.shape[0]
        # Compute length depending on global split
        if self.split is None:
            self._len = total
        elif isinstance(self.split, slice):
            start, stop, step = self.split.start, self.split.stop, self.split.step
            rng = range(*slice(start, stop, step).indices(total))
            self._len = len(rng)
        else:
            arr = np.asarray(self.split)
            if arr.dtype == np.bool_:
                arr = np.flatnonzero(arr)
            self._len = arr.shape[0]

    def _ensure_open(self) -> None:
        """ Ensure memmaps are opened in the current process with split applied.

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """

        # Current process ID
        cur_pid = os.getpid()

        # If already opened in this PID, skip
        if self._opened_pid == cur_pid:
            return
        # If we use the fork start method, and it is opened in parent, inherited memmaps are valid.
        # Skip reopen if already opened
        if self._opened_pid is not None and self._is_fork and self._opened_pid != cur_pid:
            # Inherited handle is valid; just mark as opened in this PID
            self._opened_pid = cur_pid
            return

        # If it's the first time doing open or spawn/forkserver with different PID, open per-config with split applied
        for group_name, cfg_map in (('input', self.input_config), ('target', self.target_config or {})):
            # Load and normalize each variable with split applied
            opened = {}
            for var_name, cfg in cfg_map.items():
                arr = load_var_and_normalize(cfg, split=self.split)
                opened[var_name] = arr
            # Store opened group
            self.x[group_name] = opened
        # Mark as opened in current PID
        self._opened_pid = cur_pid

    def __len__(self) -> int:
        """ Get the length of the dataset.

            Parameters
            ----------
            None.

            Returns
            -------
            int. Length of the dataset.
        """
        return int(self._len)

    def __getitem__(self, idx: int) -> dict:
        """ Get item from data

            Parameters
            ----------
            idx : int. Index of the item to retrieve.

            Returns
            -------
            Dataset object.
        """

        # Ensure memmaps are opened in current process
        self._ensure_open()
        out = {'input': {}, 'target': {}} if 'target' in self.x else {'input': {}}

        def fetch(group_dict: dict) -> dict:
            """ Fetch data at index from group dict.

                Parameters
                ----------
                group_dict : dict. Dictionary of variables in the group.

                Returns
                -------
                dict. Dictionary of fetched variables at index.
            """

            # Fetch variables at index
            res = {}
            for k, v in group_dict.items():
                # if variable length equals dataset length, index by idx; else leave as-is (broadcasted)
                if getattr(v, 'shape', None) and v.shape[0] == self._len:
                    val = v[idx]
                else:
                    val = v
                # Convert to torch tensor
                res[k] = to_torch(val)
            return res

        # Fetch input and target data
        out['input'] = fetch(self.x.get('input', {}))
        if 'target' in self.x:
            out['target'] = fetch(self.x.get('target', {}))
        return out
