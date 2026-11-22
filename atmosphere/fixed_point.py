"""Back-compat shim: re-exports from atmosphere.twolevel.fixed_point.

The canonical implementation has moved to atmosphere/twolevel/fixed_point.py
as part of SPEC_threelevel §M3a.  All existing imports of the form
``from atmosphere.fixed_point import ...`` continue to work unchanged via
this re-export.

Private helpers (_source_from_populations, _make_reference_se_solver) are
also re-exported because decisions/ and tests/ import them directly.
"""

from .twolevel.fixed_point import (  # noqa: F401
    AliResult,
    _make_reference_se_solver,
    _source_from_populations,
    ali_loop,
    cycle_emulator,
    cycle_truth,
)
