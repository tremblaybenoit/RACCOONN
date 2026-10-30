import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from typing import Any


class ProgressBar(TQDMProgressBar):
    """
    Custom progress bar callback for PyTorch Lightning training.

    Extends TQDMProgressBar to clean up output cosmetics by removing the default
    version number ('v_num') entry and formatting all metric values to use
    scientific notation.
    """

    def __init__(self, format_spec: str = ".2e", *args: Any, **kwargs: Any) -> None:
        """
        Initialize modified progress bar callback.

        Parameters
        ----------
        format_spec : str
            Format specification for scientific notation (default: '.2e').
        *args : Any
            Variable length argument list passed to parent TQDMProgressBar.
        **kwargs : Any
            Arbitrary keyword arguments passed to parent TQDMProgressBar.
        """
        super().__init__(*args, **kwargs)
        self.format_spec = format_spec

    def _format_val(self, val: Any) -> Any:
        """
        Convert numeric metric values into scientific notation strings.

        Parameters
        ----------
        val : Any
            The metric value to format.

        Returns
        -------
        Any
            The formatted scientific notation string if float/int/tensor,
            otherwise the original value.
        """
        if isinstance(val, (float, int)):
            return float(format(float(val), self.format_spec))
        elif hasattr(val, "item") and val.numel() == 1:
            return float(format(val.item(), self.format_spec))
        return val

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict:
        """
        Get and format progress bar metrics.

        Removes the redundant version number ('v_num') from the progress bar
        display and converts all remaining numeric metric values into scientific
        notation.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            The model.

        Returns
        -------
        dict
            Dictionary of filtered and formatted metrics for the progress bar.
        """
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)

        # Apply scientific notation formatting to each metric value
        formatted_items = {key: self._format_val(value) for key, value in items.items()}
        return formatted_items