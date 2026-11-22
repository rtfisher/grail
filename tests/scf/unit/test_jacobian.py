"""Tier U tests for scf/jacobian.py.  SPEC_scf §6 P2.

Covers:
- ``cpks_jacobian_analytic`` returns the expected shape on H₂/STO-3G.
- ``cpks_jacobian_fd`` returns the expected shape on H₂/STO-3G.
- Analytic and FD Jacobians agree at ``R = R_eq`` (RKS+B3LYP) to 6 sig
  figs on the dominant matrix element (1e-5 absolute on the small entries
  — the small entries are dominated by FD discretization noise at the
  ``h = 1e-4`` step that TASK-035 / SPEC v1.6 mandates).
- ``A_op_analytic`` returns a positive finite scalar.
- ``spectral_radius`` returns ``< 1`` at R_eq for STO-3G (the local cycle
  is contractive at equilibrium; this would fail at strongly stretched
  bonds where the Picard cycle becomes locally non-contractive — see
  the Tier A amplification-sweep test for the full ``R`` sweep).

These tests require PySCF.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.host import picard_iterate_scf
from scf.jacobian import (
    A_op_analytic,
    cpks_jacobian_analytic,
    cpks_jacobian_fd,
    spectral_radius,
)
from scf.state import n_unique


# ---------------------------------------------------------------------------
# Helper: converge an SCF and return the pieces P2 needs
# ---------------------------------------------------------------------------

def _converge_h2(R: float = 0.74, basis: str = "sto-3g") -> tuple:
    """Run truth SCF at ``R`` and return (state, P_star, mo_energy, mo_coeff)."""
    state = make_h2_state(R=R, basis=basis, method="rks")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200)
    assert result.converged, (
        f"SCF did not converge at R={R} basis={basis}: "
        f"iter_count={result.iter_count}, last residual="
        f"{result.residual_history[-1] if result.residual_history else None}"
    )
    return state, result.P_star, result.mo_energy, result.mo_coeff


# ---------------------------------------------------------------------------
# Shape and dtype checks
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cpks_analytic_returns_square_n_unique_matrix() -> None:
    """For H₂/STO-3G (M=2, n_unique=3), analytic D𝒯 is (3, 3) float64."""
    state, P_star, mo_e, mo_c = _converge_h2()
    DT = cpks_jacobian_analytic(state, P_star, mo_e, mo_c)
    assert DT.shape == (n_unique(2), n_unique(2))  # (3, 3)
    assert DT.dtype == np.float64
    assert np.all(np.isfinite(DT))


@pytest.mark.unit
def test_cpks_analytic_symmetric_for_h2_sto3g() -> None:
    """For H₂/STO-3G the Jacobian happens to be symmetric (rank-1 in a
    1-occ × 1-virt subspace); the test exercises that property even though
    a CPKS Jacobian is in general non-symmetric.

    Tolerance: 1e-10 on the absolute symmetry deficit.  This is a
    same-basis double-check that the orbital-rotation transform is being
    applied consistently.
    """
    state, P_star, mo_e, mo_c = _converge_h2()
    DT = cpks_jacobian_analytic(state, P_star, mo_e, mo_c)
    assert np.allclose(DT, DT.T, atol=1e-10), (
        f"D𝒯 not symmetric to 1e-10 for H₂/STO-3G; "
        f"max asymmetry = {np.max(np.abs(DT - DT.T))}."
    )


@pytest.mark.unit
def test_cpks_fd_returns_square_n_unique_matrix() -> None:
    """For H₂/STO-3G, FD D𝒯 is (3, 3) float64."""
    state, P_star, _mo_e, _mo_c = _converge_h2()
    DT_fd = cpks_jacobian_fd(state, P_star, h=1e-4)
    assert DT_fd.shape == (n_unique(2), n_unique(2))
    assert DT_fd.dtype == np.float64
    assert np.all(np.isfinite(DT_fd))


# ---------------------------------------------------------------------------
# FD vs analytic agreement (the load-bearing cross-check)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cpks_fd_matches_analytic_at_R_eq() -> None:
    """At R_eq, FD and analytic Jacobians agree to 6 sig figs.

    SPEC_scf §6 P2 Tier U: ``cpks_jacobian_fd(state, P*) ≈
    cpks_jacobian_analytic(...)`` at ``R = R_eq`` to relative tolerance
    1e-4 per matrix element (6 sig figs for max abs element; 1e-4
    relative covers small-value entries near machine precision).

    Empirically (R=0.74, STO-3G, RKS+B3LYP, h=1e-4):
      max abs diff ≈ 5e-10
      max abs DT entry ≈ 1.9e-1
      ⇒ relative agreement ≈ 3e-9 on dominant entries.
    """
    state, P_star, mo_e, mo_c = _converge_h2()
    DT_an = cpks_jacobian_analytic(state, P_star, mo_e, mo_c)
    DT_fd = cpks_jacobian_fd(state, P_star, h=1e-4)

    # Dominant-entry agreement: at R_eq this should be at the FD floor
    # (~1e-9 relative).
    max_an = np.max(np.abs(DT_an))
    diff = np.max(np.abs(DT_an - DT_fd))
    assert diff / max_an < 1e-5, (
        f"FD-vs-analytic Jacobian agreement at R_eq: max abs diff = {diff:.3e}, "
        f"max abs analytic entry = {max_an:.3e}, ratio = {diff/max_an:.3e}; "
        f"required < 1e-5 (6 sig figs)."
    )


# ---------------------------------------------------------------------------
# A_op_analytic and spectral_radius sanity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_A_op_analytic_positive_at_R_eq() -> None:
    """``A_op_analytic`` returns a positive, finite scalar at R_eq."""
    state, P_star, mo_e, mo_c = _converge_h2()
    DT = cpks_jacobian_analytic(state, P_star, mo_e, mo_c)
    A = A_op_analytic(DT)
    assert isinstance(A, float)
    assert np.isfinite(A)
    assert A > 0.0


@pytest.mark.unit
def test_spectral_radius_lt_one_at_R_eq() -> None:
    """``spectral_radius(D𝒯) < 1`` at R_eq — the local cycle is
    contractive at equilibrium geometry.

    Empirically ρ ≈ 0.378 for H₂/STO-3G/B3LYP at R=0.74 Å.  This is what
    makes plain Roothaan–Hall converge in a handful of iterations
    without acceleration (SPEC_scf §6 P1 A-scf-fast-converge-eq).
    """
    state, P_star, mo_e, mo_c = _converge_h2()
    DT = cpks_jacobian_analytic(state, P_star, mo_e, mo_c)
    rho = spectral_radius(DT)
    assert isinstance(rho, float)
    assert rho < 1.0, (
        f"Local cycle Jacobian spectral radius ρ = {rho:.4f} at R=R_eq; "
        f"expected < 1 (contractive)."
    )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cpks_analytic_rejects_nonsquare_P() -> None:
    """``cpks_jacobian_analytic`` raises ValueError on non-square P_star."""
    state = make_h2_state(R=0.74)
    bad_P = np.zeros((2, 3), dtype=np.float64)
    mo_e = np.zeros(2)
    mo_c = np.eye(2)
    with pytest.raises(ValueError, match="square"):
        cpks_jacobian_analytic(state, bad_P, mo_e, mo_c)


@pytest.mark.unit
def test_cpks_analytic_rejects_shape_mismatch() -> None:
    """``cpks_jacobian_analytic`` raises if mo_coeff/mo_energy don't match P_star."""
    state = make_h2_state(R=0.74)
    P = np.eye(2, dtype=np.float64)
    bad_mo_e = np.zeros(3)
    mo_c = np.eye(2)
    with pytest.raises(ValueError, match="mo_coeff must be"):
        cpks_jacobian_analytic(state, P, bad_mo_e, mo_c)


@pytest.mark.unit
def test_cpks_fd_rejects_nonpositive_h() -> None:
    """``cpks_jacobian_fd`` raises on non-positive FD step."""
    state, P_star, _mo_e, _mo_c = _converge_h2()
    with pytest.raises(ValueError, match="h must be positive"):
        cpks_jacobian_fd(state, P_star, h=0.0)


@pytest.mark.unit
def test_A_op_rejects_nonsquare_input() -> None:
    """``A_op_analytic`` raises on non-square input."""
    with pytest.raises(ValueError, match="square"):
        A_op_analytic(np.zeros((2, 3), dtype=np.float64))


@pytest.mark.unit
def test_spectral_radius_rejects_nonsquare_input() -> None:
    """``spectral_radius`` raises on non-square input."""
    with pytest.raises(ValueError, match="square"):
        spectral_radius(np.zeros((2, 3), dtype=np.float64))
