"""Host RT code: 1-D plane-parallel NLTE atmosphere. the technical reference §3; SPEC_threelevel §4."""

from .atmosphere import Atmosphere
from .atomic_model import AtomicModel, Transition
from .formal_solver import formal_solve
from .grid import AngleQuadrature, DepthGrid, FrequencyGrid
from .twolevel.fixed_point import (
    AliResult,
    _make_reference_se_solver,
    ali_loop,
    cycle_emulator,
    cycle_truth,
)
from .twolevel.rate_equations import rate_equation_solve
# 3-level public API (SPEC_threelevel §M3b TASK-014)
from .threelevel import (
    AliResult3Level,
    ali_loop_3level,
    cycle_emulator_3level,
    cycle_truth_3level,
    rate_equation_solve_3level,
)

__all__ = [
    # shared / 2-level
    "AngleQuadrature",
    "AliResult",
    "AtomicModel",
    "Atmosphere",
    "DepthGrid",
    "FrequencyGrid",
    "Transition",
    "_make_reference_se_solver",
    "ali_loop",
    "cycle_emulator",
    "cycle_truth",
    "formal_solve",
    "rate_equation_solve",
    # 3-level
    "AliResult3Level",
    "ali_loop_3level",
    "cycle_emulator_3level",
    "cycle_truth_3level",
    "rate_equation_solve_3level",
]
