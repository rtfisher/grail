"""Two-level NLTE solver modules. SPEC_threelevel §M3a.

Exposes the 2-level rate-equation solver and ALI fixed-point driver.
All public and private symbols used by decisions/ and tests/ are re-exported
here for convenient import from ``atmosphere.twolevel``.
"""

from .fixed_point import (
    AliResult,
    _make_reference_se_solver,
    _source_from_populations,
    ali_loop,
    cycle_emulator,
    cycle_truth,
)
from .rate_equations import rate_equation_solve

__all__ = [
    "AliResult",
    "_make_reference_se_solver",
    "_source_from_populations",
    "ali_loop",
    "cycle_emulator",
    "cycle_truth",
    "rate_equation_solve",
]
