"""Callback module - Lightning callbacks for metrics collection and figure logging."""

from src.callback.metrics import (
    LossLogger,
    MetricsLogger,
    ForwardMetricsLogger,
    InverseMetricsLogger,
    MetricsCollector,  # Backward compatibility alias
)
from src.callback.figure import FigureLogger, ForwardLogger, InverseLogger

__all__ = [
    'LossLogger',
    'MetricsLogger',
    'ForwardMetricsLogger',
    'InverseMetricsLogger',
    'MetricsCollector',  # Backward compatibility
    'FigureLogger',
    'ForwardLogger',
    'InverseLogger',
]


