"""Back-compat shim: re-exports from emulator.twolevel.training.

The canonical implementation has moved to emulator/twolevel/training.py
as part of SPEC_threelevel §M3a.  All existing imports of the form
``from emulator.training import ...`` continue to work unchanged via
this re-export.
"""

from .twolevel.training import (  # noqa: F401
    TrainingData,
    generate_training_data,
    train,
    train_val_split,
)
