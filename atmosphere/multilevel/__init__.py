"""Generic N-level NLTE atmosphere module for the GRAIL Modest-tier scale-up.

Provides the host-agnostic N-level statistical-equilibrium solver and
ALI / Picard drivers that generalize the 2-level and 3-level implementations
to arbitrary level count L with K = L*(L-1)/2 bound-bound transitions.

Public API
----------
rate_equation_solve_nlevel                  : N-level SE solver (batched over depth).
cycle_truth_nlevel                          : One T_RT^K o T_SE cycle step (truth path).
cycle_emulator_nlevel                       : One T_RT^K o T_SE cycle step (emulator path).
ali_loop_nlevel                             : ALI outer loop (direct or iterative mode).
picard_iterate_nlevel                       : Picard iteration with optional Anderson AA(m).
AliResultMultilevel                         : Converged output from ali_loop_nlevel.
PicardResultMultilevel                      : Output from picard_iterate_nlevel.
make_lambda_star_preconditioner_multilevel  : Diagonal Λ*-preconditioner factory (M.6).
make_selective_lambda_preconditioner_multilevel : Selective Λ*/Λ hybrid preconditioner (M.6 option 1b).

SPEC_nlte_scaleup Phase M tasks M.2, M.3, and M.6.
"""

from .rate_equations import rate_equation_solve_nlevel
from .fixed_point import (
    AliResultMultilevel,
    PicardResultMultilevel,
    ali_loop_nlevel,
    cycle_emulator_nlevel,
    cycle_truth_nlevel,
    picard_iterate_nlevel,
    solve_equilibrium_nlevel,
    radiation_from_populations_nlevel,
)
from .precond_lambda_star import (
    make_lambda_star_preconditioner_multilevel,
    make_selective_lambda_preconditioner_multilevel,
)

__all__ = [
    "rate_equation_solve_nlevel",
    "AliResultMultilevel",
    "PicardResultMultilevel",
    "ali_loop_nlevel",
    "cycle_emulator_nlevel",
    "cycle_truth_nlevel",
    "picard_iterate_nlevel",
    "solve_equilibrium_nlevel",
    "radiation_from_populations_nlevel",
    "make_lambda_star_preconditioner_multilevel",
    "make_selective_lambda_preconditioner_multilevel",
]
