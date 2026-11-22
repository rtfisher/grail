"""Tier U tests for scf/fock_build.py.  SPEC_scf §6 P1.

Covers:
- ``build_fock_truth`` returns symmetric (M, M) of the expected shape.
- ``F = h_core`` when ``P = 0`` (Coulomb J[0] and V_xc[0] vanish — though
  V_xc[0] returns NaN/Inf in some XC quadrature codes if the density is
  literally zero, so we use a very small density instead; the check is
  ``F ≈ h_core`` to leading order).  We document the exact form.
- ``F`` is finite (no NaN) on a real-looking H₂ ground-state density.

These tests require PySCF.  They run only if PySCF is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module gracefully if PySCF is not yet installed.
pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.fock_build import _build_mf, _build_mol, build_fock_truth


# ---------------------------------------------------------------------------
# Shape and symmetry
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fock_truth_returns_symmetric_2x2_for_sto3g_h2() -> None:
    """For H₂/STO-3G, build_fock_truth returns a symmetric (2, 2) ndarray."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="rks")
    P = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    F = build_fock_truth(state, P)
    assert F.shape == (2, 2)
    assert F.dtype == np.float64
    assert np.allclose(F, F.T, atol=1e-12)


@pytest.mark.unit
def test_fock_truth_returns_symmetric_4x4_for_631g_h2() -> None:
    """For H₂/6-31G, build_fock_truth returns a symmetric (4, 4) ndarray."""
    state = make_h2_state(R=0.74, basis="6-31g", method="rks")
    # Build a plausible symmetric density: 4x4 random + transpose.
    rng = np.random.default_rng(42)
    A = rng.standard_normal((4, 4))
    P = 0.1 * (A + A.T)  # small entries for stability
    F = build_fock_truth(state, P)
    assert F.shape == (4, 4)
    assert F.dtype == np.float64
    assert np.allclose(F, F.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Limiting cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fock_truth_recovers_h_core_when_P_is_zero() -> None:
    """At P = 0, J[P] = 0 and V_xc[P] = 0, so F = h_core.

    We compute h_core directly from PySCF for the reference, then compare
    against ``build_fock_truth(state, P=0)``.  Some DFT XC quadrature codes
    return NaN at exactly zero density; PySCF handles that gracefully, but
    the test would still catch any future regression.
    """
    state = make_h2_state(R=0.74, basis="sto-3g", method="rks")

    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    h_core = np.asarray(mf.get_hcore(), dtype=np.float64)

    P_zero = np.zeros((2, 2), dtype=np.float64)
    F = build_fock_truth(state, P_zero)

    # Coulomb J[0] = 0 trivially.  V_xc[0]: B3LYP at zero density returns
    # zero to within float-precision (PySCF returns 0.0 exactly when the
    # density on the grid is 0 — both LDA exchange and the gradient
    # corrections multiply by the density and vanish).
    assert np.allclose(F, h_core, atol=1e-10)


@pytest.mark.unit
def test_fock_truth_finite_on_h2_ground_state() -> None:
    """build_fock_truth returns finite (no NaN/Inf) values on a real density.

    We construct a plausible H₂ ground-state density by hand: in the
    STO-3G basis a symmetric ~1.5 charge per orbital is in the right ballpark
    of the converged P (the actual converged trace is 2 = n_electrons).
    """
    state = make_h2_state(R=0.74, basis="sto-3g", method="rks")
    # Density with trace ~2 (closed-shell H₂); typical converged P has both
    # diagonals ~0.6 and off-diagonal ~0.6 due to bonding-orbital overlap.
    P = np.array([[0.6, 0.55], [0.55, 0.6]], dtype=np.float64)
    F = build_fock_truth(state, P)
    assert np.all(np.isfinite(F))


@pytest.mark.unit
def test_fock_truth_rejects_nonsquare_P() -> None:
    """build_fock_truth raises ValueError on a non-square P."""
    state = make_h2_state(R=0.74)
    with pytest.raises(ValueError, match="square"):
        build_fock_truth(state, np.zeros((2, 3), dtype=np.float64))
