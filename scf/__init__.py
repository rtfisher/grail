"""SCF testbed package: stretched-H₂ closed-shell RKS/RHF SCF on PySCF.

SPEC_scf §3.  This is GRAIL's second application testbed — a Roothaan–Hall
fixed-point iteration on the density matrix, in which the neural surrogate
replaces the exchange–correlation matrix `V_xc[P]` inside the Fock build.
The bond length `R` provides the single-knob amplification dial (HOMO–LUMO
gap shrinks as R stretches → spectral radius of D𝒯 approaches 1).

Phase P1 deliverables (this commit): truth SCF host package
(MoleculeState, pack/unpack, build_fock_truth, picard_iterate_scf) plus the
Szabo–Ostlund H₂ analytic anchor.  P2–P6 deferred (see SPEC_scf §6).

The package is structured to mirror ``atmosphere/twolevel/`` and
``emulator/twolevel/``:
- ``state.py``          : MoleculeState dataclass + pack/unpack seam.
- ``fock_build.py``     : truth Fock construction h_core + J[P] + V_xc[P].
- ``host.py``           : Roothaan–Hall picard_iterate_scf with optional DIIS.
- ``_make_state.py``    : H₂ convenience constructor.

PySCF imports are kept inside functions (not at module top) to avoid OpenMP
load-order surprises at package import time (SPEC_scf §8.5).
"""

# Public symbols.  We re-export the cheap, PySCF-free pieces eagerly
# (``MoleculeState``, the pack/unpack seam, the convenience constructor),
# and expose the PySCF-dependent entry points lazily via module-level
# ``__getattr__`` so that consumers who only need state-vector utilities
# (e.g. the GRAIL adapter that does not call ``build_fock_truth`` directly)
# do not pay the PySCF import cost — and tests that don't need PySCF can
# still collect when PySCF is not installed (using ``pytest.importorskip``
# on the relevant modules).
from ._make_state import make_h2_state
from .state import (
    MoleculeState,
    n_unique,
    pack_density_matrix,
    unpack_density_matrix,
)

# ─── P2 — analytic CPKS Jacobian + A_op(R) sweep ─────────
from .jacobian import (
    A_op_analytic,
    cpks_jacobian_analytic,
    cpks_jacobian_fd,
    spectral_radius,
)
# ─────────────────────────────────────────────────────────

# ─── R2 — UHF CPKS Jacobian (extends P2) ─────────────
from scf.jacobian import (
    cpks_jacobian_analytic_uhf,
    cpks_jacobian_fd_uhf,
)
# ─────────────────────────────────────────────────────────

# ─── R3 — UHF neural XC emulator (supersedes P3) ──────
from scf.emulator import (
    XCNet,
    generate_training_data_uhf,
    train_xc_emulator_uhf,
    validation_rel_mae_uhf,
    save_xc_emulator,
    load_xc_emulator,
)
# ─────────────────────────────────────────────────────────

# ─── P4 — JVP wrapper + GRAIL integration ─────────────
from scf.jvp import (
    make_cycle_jvps_scf,
    make_fock_builder_with_nn,
    make_host_jvp_scf,
    make_emulator_jvp_scf,
)
# ─────────────────────────────────────────────────────────

# ─── P6 — Split-conformal coverage on δP* ─────────────
from scf.conformal import (
    build_scf_residual_ensemble,
    evaluate_coverage_scf,
    fit_split_conformal_scf,
)
# ─────────────────────────────────────────────────────────

__all__ = [
    "A_op_analytic",
    "MoleculeState",
    "SCFResult",
    "XCNet",
    "build_fock_truth",
    "build_scf_residual_ensemble",
    "cpks_jacobian_analytic",
    "cpks_jacobian_analytic_uhf",
    "cpks_jacobian_fd",
    "cpks_jacobian_fd_uhf",
    "evaluate_coverage_scf",
    "fit_split_conformal_scf",
    "generate_training_data_uhf",
    "load_xc_emulator",
    "make_emulator_jvp_scf",
    "make_cycle_jvps_scf",
    "make_fock_builder_with_nn",
    "make_h2_state",
    "make_host_jvp_scf",
    "n_unique",
    "pack_density_matrix",
    "picard_iterate_scf",
    "save_xc_emulator",
    "spectral_radius",
    "train_xc_emulator_uhf",
    "unpack_density_matrix",
    "validation_rel_mae_uhf",
]


def __getattr__(name: str):
    """Lazy import of PySCF-dependent entry points (SPEC_scf §8.5)."""
    if name == "build_fock_truth":
        from .fock_build import build_fock_truth
        return build_fock_truth
    if name == "picard_iterate_scf":
        from .host import picard_iterate_scf
        return picard_iterate_scf
    if name == "SCFResult":
        from .host import SCFResult
        return SCFResult
    raise AttributeError(f"module 'scf' has no attribute {name!r}")
