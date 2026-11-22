"""Tier A — A-scf-szabo-ostlund-h2.  SPEC_scf §6 P1, §8.6.

RHF/STO-3G H₂ at R = 0.74 Å (close to the experimental equilibrium and the
textbook S–O reference geometry).  Total energy should be approximately
-1.117 Ha at textbook precision.  The numerical check uses a separate
PySCF RHF solve at the matching 0.74 Å geometry and a 1e-10 Ha tolerance;
the rounded textbook value cannot support a microhartree comparison.

This anchor uses RHF mode because the Szabo–Ostlund textbook number is bare
Hartree–Fock — RKS+LDA (or any DFT XC) does not reproduce it.  ``MoleculeState
.method = "rhf"`` sets ``V_xc = 0`` in PySCF and the Fock build reduces to
``h_core + J[P] + K[P]``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.host import picard_iterate_scf


@pytest.mark.analytic
def test_A_scf_szabo_ostlund_h2() -> None:
    """A-scf-szabo-ostlund-h2: RHF/STO-3G H₂ total energy ≈ -1.117 Ha."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="rhf")
    result = picard_iterate_scf(state, accelerator="diis", tol=1e-9, max_iter=100)

    assert result.converged is True, (
        f"RHF H₂ did not converge (iter_count={result.iter_count}, "
        f"last residual={result.residual_history[-1] if result.residual_history else None})."
    )

    # Independent pyscf.scf.RHF.kernel(conv_tol=1e-13), R=0.74 Å;
    # recorded in the adversarial audit independent_scf.json.  Bare RHF
    # has no numerical XC quadrature grid.
    expected = -1.1167593073964255
    assert abs(result.energy_total - expected) < 1e-10, (
        f"RHF/STO-3G H₂ total energy = {result.energy_total:.6f} Ha; "
        f"expected {expected} Ha (independent matching-geometry RHF solve)."
    )

    # Sanity check on the converged density:
    # (i) symmetric;
    # (ii) positive semi-definite (all eigenvalues ≥ 0 to within float slop);
    # (iii) tr(P S) ≈ n_electrons = 2.
    P = result.P_star
    assert np.allclose(P, P.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(P)
    assert eigs.min() > -1e-8, (
        f"P not PSD; smallest eigenvalue = {eigs.min()}."
    )

    # tr(P S) should equal n_electrons for a converged closed-shell density.
    from scf.fock_build import _build_mf, _build_mol
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    trPS = float(np.trace(P @ S))
    assert abs(trPS - 2.0) < 1e-8, (
        f"tr(P S) = {trPS}; expected 2.0 (n_electrons)."
    )
