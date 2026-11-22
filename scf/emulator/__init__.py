"""Neural exchange–correlation surrogate for the SCF testbed.  SPEC_scf §3.2, §6 P3, §8.10 R3.

Phase R3 deliverables (UHF/UKS; supersedes the obsolete P3 closed-shell path):
- :class:`XCNet` — feed-forward NN mapping doubled-seam packed-flat density
  pairs ``(P_α, P_β)`` to doubled-seam packed-flat ``(V_xc^α, V_xc^β)``,
  used as a drop-in for B3LYP's XC quadrature inside the UHF/UKS Fock build.
- :func:`generate_training_data_uhf` / :func:`train_xc_emulator_uhf` —
  LHS sampling over UHF density perturbations around the converged
  broken-symmetry ``(P_α*, P_β*)`` at varying bond lengths R, with
  spin-resolved B3LYP truth labels.
- :func:`validation_rel_mae_uhf` — gate (< 3 %; SPEC_scf §8.10).
- :func:`save_xc_emulator` / :func:`load_xc_emulator` — HDF5 round-trip so
  P4 (GRAIL integration) can pick up a frozen emulator artifact.

Mirrors ``emulator/twolevel/`` at a smaller scale.  Per SPEC_scf §6 P3 the
emulator target is ``V_xc`` only (NOT the full ``G[P] = J[P] + V_xc[P]``);
the analytic Hartree-Coulomb piece is exactly computable at negligible cost,
so emulating it adds no scientific value.  For UHF the Hartree piece is
built from the total density ``P_α + P_β`` and shared across spins; the
emulator learns the spin-resolved XC contributions only.

This module is intentionally PySCF-free at module-import time: the truth
``V_xc`` evaluation in :func:`generate_training_data_uhf` imports PySCF
lazily.  Loading a trained ``XCNet`` from HDF5 (P4 path) therefore does
not pay the PySCF startup or OpenMP triple-init cost (SPEC_scf §8.5).
"""

from .io import load_xc_emulator, save_xc_emulator
from .network import XCNet
from .training import generate_training_data_uhf, train_xc_emulator_uhf
from .validation import validation_rel_mae_uhf

__all__ = [
    "XCNet",
    "generate_training_data_uhf",
    "load_xc_emulator",
    "save_xc_emulator",
    "train_xc_emulator_uhf",
    "validation_rel_mae_uhf",
]
