import os
import numpy as np
from typing import Union
from omegaconf import DictConfig, ListConfig
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from forward.utilities.instantiators import instantiate
from forward.data.transformations import identity


class BaseDataloader(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = None,
                 pin_memory: bool = True, shuffle: bool = True) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
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
                          pin_memory=self.pin_memory, persistent_workers=True, shuffle=self.shuffle)

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
                          pin_memory=self.pin_memory, persistent_workers=True)

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
                          pin_memory=self.pin_memory, persistent_workers=True)

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
                          pin_memory=self.pin_memory, persistent_workers=True)


class CRTMDataloader(BaseDataloader):
    def __init__(self, dir: str, sets: DictConfig, inputs: Union[ListConfig, list, str],
                 outputs: Union[ListConfig, list, str] = None, batch_size: int = 32, num_workers: int = None,
                 pin_memory: bool = True) -> None:
        """ Dataloader for the CRTM dataset.

        Parameters
        ----------
        dir : str. Path to the directory containing the dataset files.
        sets: DictConfig. Configuration object for the dataset sets.
        inputs : Union[list, str]. List of input data types (e.g., ['prof', 'surf', 'meta']).
        outputs : Union[list, str]. List of output data types (e.g., 'hofx'). If None, no outputs are loaded.
        batch_size : int. Batch size for the dataloader.
        num_workers : int. Number of workers for the dataloader.
        pin_memory : bool. If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

        Returns
        -------
        None.
        """

        #  Class inheritance
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        # Path to data
        self.ds_path = dir
        # Inputs and outputs
        self.inputs = inputs if isinstance(inputs, list) or isinstance(inputs, ListConfig) else [inputs]
        self.outputs = outputs if isinstance(outputs, list) or isinstance(inputs, ListConfig) else [outputs] if outputs is not None else None
        # Data sets
        self.sets = sets

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

        # Inputs
        x = {}
        for input in inputs:
            f_norm = instantiate(self.sets[stage][input]['normalization']) \
                if hasattr(self.sets[stage][input], 'normalization') else identity
            x[input] = f_norm(np.array(instantiate(self.sets[stage][input]['load']), dtype=np.float32)
                              [self.sets[stage]['split']['start']:self.sets[stage]['split']['end']])

        # Outputs
        if outputs is not None:
            y = {}
            for output in outputs:
                f_norm = instantiate(self.sets[stage][output]['normalization']) \
                    if hasattr(self.sets[stage][output], 'normalization') else identity
                y[output] = f_norm(np.array(instantiate(self.sets[stage][output]['load']), dtype=np.float32)
                                   [self.sets[stage]['split']['start']:self.sets[stage]['split']['end']])
            return CRTMDataset(x, y)
        return CRTMDataset(x)

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
            # Training data
            self.ds_train = self._make_dataset('train', inputs=self.inputs, outputs=self.outputs)
            # Validation data
            self.ds_valid = self._make_dataset('valid', inputs=self.inputs, outputs=self.outputs)
        elif stage == 'test':
            # Test data
            self.ds_test = self._make_dataset(stage, inputs=self.inputs, outputs=self.outputs)
        elif stage == 'pred':
            # Prediction data
            self.ds_pred = self._make_dataset(stage, inputs=self.inputs, outputs=None)


class CRTMDataset(Dataset):
    """Lazy loader for the CRTM dataset."""

    def __init__(self, x: dict, y: dict=None) -> None:
        """Initialize the lazy loader.

        Parameters
        ----------
        x : tuple. Tuple containing input tensors (profiles, surface, meta).
        y : np.ndarray. Target tensor (BT) (optional).

        Returns
        -------
        None.
        """

        # Store data
        self.x = x
        self.y = y

    def __len__(self) -> int:
        """ Get the length of the dataset.

        Returns
        -------
        int. Length of the dataset.
        """
        # Return the length of the first input tensor
        return len(next(iter(self.x.values())))

    def __getitem__(self, idx: int) -> Union[tuple, dict]:
        """ Get data

        Returns
        -------
        Dataset object.
        """

        # Get the data at the specified index
        x_out = {k: v[idx] for k, v in self.x.items()}
        if self.y is not None:
            y_out = {k: v[idx] for k, v in self.y.items()}
            return x_out, y_out
        else:
            return x_out
