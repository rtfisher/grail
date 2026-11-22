"""APNN emulator: feed-forward NN replacing the rate-equation solve. the technical reference §4."""

from .apnn import APNN
from .jvp import jvp, make_cycle_adapter
from .twolevel.training import TrainingData, generate_training_data, train, train_val_split
from .twolevel.validation import compute_residuals, validation_mae

__all__ = [
    "APNN",
    "TrainingData",
    "compute_residuals",
    "generate_training_data",
    "jvp",
    "make_cycle_adapter",
    "train",
    "train_val_split",
    "validation_mae",
]
