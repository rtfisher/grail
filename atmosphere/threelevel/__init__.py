"""3-level NLTE atmosphere module. SPEC_threelevel §M3b.

This subpackage implements the 3-level extension of the 2-level prototype,
including the rate-equation solver, cycle functions, and ALI driver for
K-transition systems.  See SPEC_threelevel §6.1–6.2 for the public API spec.
"""

from .rate_equations import rate_equation_solve_3level
from .fixed_point import (
    cycle_truth_3level,
    cycle_emulator_3level,
    ali_loop_3level,
    AliResult3Level,
)

__all__ = [
    "rate_equation_solve_3level",
    "cycle_truth_3level",
    "cycle_emulator_3level",
    "ali_loop_3level",
    "AliResult3Level",
]
