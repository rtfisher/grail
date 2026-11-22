"""5-level (multilevel) emulator modules. SPEC_nlte_scaleup §M.4.

Exposes the APNN architecture, training pipeline, validation metrics,
cycle adapter, and FD-JVP helper for the 5-level Balmer hydrogen APNN.
"""

from .apnn import APNN, make_apnn_5level
from .jvp import _apnn_fd_jvp_multilevel, make_cycle_adapter_5level
from .training import (
    TrainingDataMultilevel,
    generate_training_data_multilevel,
    train_multilevel,
    train_val_split_multilevel,
)
from .validation import compute_residuals_multilevel, validation_mae_multilevel
from .positive_apnn import (
    PositivePopulationAPNN,
    RateRelaxedAPNN,
    load_rate_relaxed_checkpoint,
    make_positive_cycle_adapter,
)

__all__ = [
    "APNN",
    "TrainingDataMultilevel",
    "_apnn_fd_jvp_multilevel",
    "compute_residuals_multilevel",
    "generate_training_data_multilevel",
    "make_apnn_5level",
    "make_cycle_adapter_5level",
    "train_multilevel",
    "train_val_split_multilevel",
    "validation_mae_multilevel",
    "PositivePopulationAPNN",
    "RateRelaxedAPNN",
    "load_rate_relaxed_checkpoint",
    "make_positive_cycle_adapter",
]
