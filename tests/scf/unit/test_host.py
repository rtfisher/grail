"""Tier U tests for scf/host.py.  SPEC_scf §6 P1.

Covers:
- ``picard_iterate_scf`` converges and returns iter_count <= 30 at R_eq with
  the production setup (RKS+B3LYP, STO-3G, DIIS).

These tests require PySCF.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.host import SCFResult, picard_iterate_scf


@pytest.mark.unit
def test_picard_converges_at_R_eq() -> None:
    """RKS/B3LYP/STO-3G H₂ at R_eq=0.74 Å converges in ≤ 30 iterates."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="rks")
    result = picard_iterate_scf(state)
    assert isinstance(result, SCFResult)
    assert result.converged is True, (
        f"SCF did not converge at R_eq; iter_count={result.iter_count}, "
        f"last residual={result.residual_history[-1] if result.residual_history else None}"
    )
    assert result.iter_count <= 30, (
        f"R_eq convergence required > 30 iterates ({result.iter_count}); "
        "SPEC says ~5–10."
    )


@pytest.mark.unit
def test_picard_returns_consistent_shapes() -> None:
    """SCFResult fields have the expected shapes for H₂/STO-3G (M=2)."""
    state = make_h2_state(R=0.74, basis="sto-3g")
    result = picard_iterate_scf(state)
    assert result.P_star.shape == (2, 2)
    assert result.mo_energy.shape == (2,)
    assert result.mo_coeff.shape == (2, 2)
    assert isinstance(result.iter_count, int)
    assert isinstance(result.converged, bool)
    assert isinstance(result.energy_total, float)


@pytest.mark.unit
def test_picard_P_star_is_symmetric_and_PSD() -> None:
    """Converged P is symmetric and (approximately) positive semi-definite."""
    state = make_h2_state(R=0.74, basis="sto-3g")
    result = picard_iterate_scf(state)
    P = result.P_star
    assert P.shape == (2, 2)
    # Symmetry to float precision.
    import numpy as np
    assert np.allclose(P, P.T, atol=1e-12)
    # PSD: all eigenvalues non-negative (allow small numerical slop).
    eigs = np.linalg.eigvalsh(P)
    assert eigs.min() > -1e-8, f"P not PSD; smallest eig = {eigs.min()}"


@pytest.mark.unit
def test_picard_rejects_bad_accelerator() -> None:
    """Unknown accelerator string raises ValueError."""
    state = make_h2_state(R=0.74, basis="sto-3g")
    with pytest.raises(ValueError, match="accelerator must be"):
        picard_iterate_scf(state, accelerator="anderson")


@pytest.mark.unit
def test_picard_no_diis_also_converges_at_R_eq() -> None:
    """Plain Roothaan–Hall (no DIIS) converges at R_eq, just slower."""
    state = make_h2_state(R=0.74, basis="sto-3g")
    result = picard_iterate_scf(state, accelerator="none", max_iter=200)
    assert result.converged is True
    # SPEC P1 §6 Tier U: ≤ 15 iters without DIIS at R_eq.
    assert result.iter_count <= 30, (
        f"plain Picard at R_eq required > 30 iters ({result.iter_count})."
    )
    assert result.diis_engaged is False
