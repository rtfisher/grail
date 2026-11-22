"""Two-level emulator modules. SPEC_threelevel §M3a.

Exposes training data generation, training loop, validation metrics, and
the cycle adapter for the 2-level APNN emulator.
"""

from .adapter import make_cycle_adapter
from .training import TrainingData, generate_training_data, train, train_val_split
from .validation import compute_residuals, validation_mae

__all__ = [
    "TrainingData",
    "compute_residuals",
    "generate_training_data",
    "make_cycle_adapter",
    "train",
    "train_val_split",
    "validation_mae",
]
