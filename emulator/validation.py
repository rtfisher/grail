"""Back-compat shim: re-exports from emulator.twolevel.validation.

The canonical implementation has moved to emulator/twolevel/validation.py
as part of SPEC_threelevel §M3a.  All existing imports of the form
``from emulator.validation import ...`` continue to work unchanged via
this re-export.
"""

from .twolevel.validation import (  # noqa: F401
    compute_residuals,
    validation_mae,
)
