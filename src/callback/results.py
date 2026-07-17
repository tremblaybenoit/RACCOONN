from pytorch_lightning.callbacks import Callback
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class ResultsLogger(Callback):
    """
    Callback to accumulate test and predict results in memory.

    Accumulates predictions and targets batch-by-batch during test/predict stages,
    storing them as concatenated numpy arrays in memory for easy access.
    """

    def __init__(
        self,
        store_targets: bool = True,
        store_inputs: bool = False,
    ) -> None:
        """
        Initialize ResultsLogger callback.

        Parameters
        ----------
        store_targets : bool, default True
            Whether to store target data from batch.
        store_inputs : bool, default False
            Whether to store input data (can be large).
        """

        super().__init__()

        self.store_targets = store_targets
        self.store_inputs = store_inputs

        # Accumulated data per stage
        self.results = {}  # {stage: {'outputs': {...}, 'targets': {...}, ...}}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Process each test batch and accumulate results.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        outputs : dict
            Model's test_step output with predictions.
        batch : dict
            Input batch with inputs, targets, and context.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        self._accumulate_batch(outputs, batch, batch_idx, stage="test")

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Process each predict batch and accumulate results.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        outputs : dict
            Model's predict_step output with predictions.
        batch : dict
            Input batch with inputs and context.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        self._accumulate_batch(outputs, batch, batch_idx, stage="predict")

    def on_test_epoch_end(self, trainer, pl_module):
        """
        Finalize test results at end of test epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """
        self._compile_results(stage="test")

    def on_predict_epoch_end(self, trainer, pl_module):
        """
        Finalize predict results at end of predict epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The model instance.
        """

        self._compile_results(stage="predict")

    def _accumulate_batch(self, outputs: dict, batch: dict, batch_idx: int, stage: str):
        """
        Extract and accumulate results from a single batch.

        Parameters
        ----------
        outputs : dict
            Model output with predictions (numpy arrays).
        batch : dict
            Batch from DataLoader with tensors.
        batch_idx : int
            Batch index (unused).
        stage : str
            Current stage: "test" or "predict".
        """
        if stage not in self.results:
            self.results[stage] = {
                "output": {},
                "target": {},
                "input": {},
            }

        # Accumulate outputs
        self._accumulate_results(outputs.get("output", {}), self.results[stage]["output"])

        # Accumulate targets
        if self.store_targets and "target" in batch:
            self._accumulate_results(batch["target"], self.results[stage]["target"])

        # Accumulate inputs
        if self.store_inputs and "input" in batch:
            self._accumulate_results(batch["input"], self.results[stage]["input"])

    @staticmethod
    def _accumulate_results(data: dict, accumulator: dict):
        """
        Generic accumulation for any category (output, target, input).

        Parameters
        ----------
        data : dict
            Data dict with keys and values (tensors or arrays).
        accumulator : dict
            The accumulator dict to store lists of values.
        """
        for key, value in data.items():
            if value is None:
                continue

            # Convert to numpy if needed
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()

            # Initialize list if needed
            if key not in accumulator:
                accumulator[key] = []

            # Append batch data
            accumulator[key].append(value)

    def _compile_results(self, stage: str):
        """
        Concatenate accumulated batches into single arrays.

        Parameters
        ----------
        stage : str
            Current stage: "test" or "predict".
        """
        if stage not in self.results or not self.results[stage]["output"]:
            logger.warning(f"No results to finalize for stage '{stage}'")
            return

        # Concatenate batch lists into single arrays
        stage_results = self.results[stage]

        # Outputs, targets, and inputs
        for category in stage_results:
            for key, batches in stage_results[category].items():
                stage_results[category][key] = np.concatenate(batches, axis=0)

        # Show summary
        logger.info(f"Finalized {stage} results: {self._format_results_summary(stage_results)}")

    def get_results(self, stage: str = "test") -> dict:
        """
        Get accumulated results for a stage.

        Parameters
        ----------
        stage : str, default "test"
            Stage to retrieve results from: "test" or "predict".

        Returns
        -------
        dict
            Accumulated results with structure:
            {
                'output': {key: array, ...},
                'target': {key: array, ...},  # if store_targets=True
                'input': {key: array, ...},   # if store_inputs=True
            }
        """
        # If results are not found
        if stage not in self.results:
            logger.warning(f"No results found for stage '{stage}'")
            return {}
        # Return stage-specific results
        return self.results[stage]

    def clear_results(self, stage: str | None = None):
        """
        Clear accumulated results.

        Parameters
        ----------
        stage : str, optional
            Stage to clear. If None, clears all stages.
        """
        if stage is None:
            self.results.clear()
        else:
            self.results.pop(stage, None)

    @staticmethod
    def _format_results_summary(result: dict) -> str:
        """
        Format a summary of results structure.

        Parameters
        ----------
        result : dict
            Results dictionary.

        Returns
        -------
        str
            Summary string.
        """
        summary = []
        if "output" in result:
            for key, arr in result["output"].items():
                summary.append(f"output_{key}:{arr.shape}")
        if "target" in result and result["target"]:
            for key, arr in result["target"].items():
                summary.append(f"target_{key}:{arr.shape}")
        if "input" in result and result["input"]:
            for key, arr in result["input"].items():
                summary.append(f"input_{key}:{arr.shape}")
        return ", ".join(summary)




