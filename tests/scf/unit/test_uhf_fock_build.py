"""Tier U tests for the UHF/UKS extension of scf/fock_build.py.  SPEC_scf §8.10 R1.

Covers:
- ``build_fock_truth(state_uhf, (P_alpha, P_beta))`` returns a tuple of two
  symmetric (M, M) matrices.
- The UHF Fock build is internally consistent with the closed-shell RHF
  build when α = β = P/2 (since UHF then reduces to RHF).
- ``_build_mf`` returns a PySCF UHF / UKS object for the matching methods.
- The closed-shell entry point is unchanged.

These tests require PySCF.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.fock_build import _build_mf, _build_mol, build_fock_truth


# ---------------------------------------------------------------------------
# Shape and symmetry
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fock_truth_uhf_returns_tuple_of_symmetric_2x2() -> None:
    """For H₂/STO-3G UHF, build_fock_truth returns (F_α, F_β), each (2, 2) symmetric."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="uhf")
    P_alpha = np.array([[0.7, 0.4], [0.4, 0.3]], dtype=np.float64)
    P_beta = np.array([[0.3, 0.4], [0.4, 0.7]], dtype=np.float64)
    F_pair = build_fock_truth(state, (P_alpha, P_beta))
    assert isinstance(F_pair, tuple)
    assert len(F_pair) == 2
    F_alpha, F_beta = F_pair
    assert F_alpha.shape == (2, 2)
    assert F_beta.shape == (2, 2)
    assert F_alpha.dtype == np.float64
    assert F_beta.dtype == np.float64
    assert np.allclose(F_alpha, F_alpha.T, atol=1e-12)
    assert np.allclose(F_beta, F_beta.T, atol=1e-12)


@pytest.mark.unit
def test_fock_truth_uks_returns_tuple_of_symmetric_2x2() -> None:
    """For H₂/STO-3G UKS B3LYP, build_fock_truth returns (F_α, F_β) (2, 2) symmetric."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="uks", xc="B3LYP")
    P_alpha = np.array([[0.6, 0.35], [0.35, 0.4]], dtype=np.float64)
    P_beta = np.array([[0.4, 0.35], [0.35, 0.6]], dtype=np.float64)
    F_pair = build_fock_truth(state, (P_alpha, P_beta))
    assert isinstance(F_pair, tuple)
    F_alpha, F_beta = F_pair
    assert F_alpha.shape == (2, 2)
    assert F_beta.shape == (2, 2)
    # XC quadrature noise is larger than for HF (multiple grid points), but
    # the symmetrization step should still bring the result to ~1e-12.
    assert np.allclose(F_alpha, F_alpha.T, atol=1e-10)
    assert np.allclose(F_beta, F_beta.T, atol=1e-10)


# ---------------------------------------------------------------------------
# UHF reduces to RHF when α = β
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fock_truth_uhf_reduces_to_rhf_when_alpha_eq_beta() -> None:
    """When P_α = P_β = P_total/2, UHF F_α = F_β = F_RHF.

    This is the textbook reduction: a closed-shell density P (total trace 2)
    has α and β blocks each carrying trace 1.  UHF with these identical
    spin densities produces F_σ = F_RHF for both spins.
    """
    R = 0.74
    P_total = np.array([[0.6, 0.55], [0.55, 0.6]], dtype=np.float64)
    P_half = 0.5 * P_total  # P_α = P_β = P_total / 2

    state_rhf = make_h2_state(R=R, basis="sto-3g", method="rhf")
    state_uhf = make_h2_state(R=R, basis="sto-3g", method="uhf")

    F_rhf = build_fock_truth(state_rhf, P_total)
    F_alpha, F_beta = build_fock_truth(state_uhf, (P_half, P_half))

    # Both UHF spin Fock matrices should coincide with the RHF Fock matrix.
    assert np.allclose(F_alpha, F_beta, atol=1e-12), (
        "F_alpha and F_beta differ for symmetric input."
    )
    assert np.allclose(F_alpha, F_rhf, atol=1e-10), (
        f"UHF F_α does not match RHF F at α=β: "
        f"max|ΔF|={np.max(np.abs(F_alpha - F_rhf)):.3e}."
    )


@pytest.mark.unit
def test_fock_truth_uks_reduces_to_rks_when_alpha_eq_beta() -> None:
    """When P_α = P_β = P_total/2, UKS F_α = F_β = F_RKS (B3LYP)."""
    R = 0.74
    P_total = np.array([[0.6, 0.55], [0.55, 0.6]], dtype=np.float64)
    P_half = 0.5 * P_total

    state_rks = make_h2_state(R=R, basis="sto-3g", method="rks", xc="B3LYP")
    state_uks = make_h2_state(R=R, basis="sto-3g", method="uks", xc="B3LYP")

    F_rks = build_fock_truth(state_rks, P_total)
    F_alpha, F_beta = build_fock_truth(state_uks, (P_half, P_half))

    # XC quadrature noise floor: ~1e-10 is robust for the symmetric input.
    assert np.allclose(F_alpha, F_beta, atol=1e-10)
    assert np.allclose(F_alpha, F_rks, atol=1e-8), (
        f"UKS F_α does not match RKS F at α=β: "
        f"max|ΔF|={np.max(np.abs(F_alpha - F_rks)):.3e}."
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fock_truth_uhf_rejects_single_matrix_input() -> None:
    """UHF build_fock_truth with a single matrix (not a tuple) raises."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="uhf")
    P = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError, match="2-tuple"):
        build_fock_truth(state, P)


@pytest.mark.unit
def test_fock_truth_uhf_rejects_nonsquare_spin_block() -> None:
    """UHF build_fock_truth with a non-square spin block raises."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="uhf")
    P_alpha = np.zeros((2, 3), dtype=np.float64)
    P_beta = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="square"):
        build_fock_truth(state, (P_alpha, P_beta))


# ---------------------------------------------------------------------------
# _build_mf dispatch
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_mf_returns_uhf_for_method_uhf() -> None:
    """_build_mf returns a PySCF UHF mean-field for method='uhf'."""
    from pyscf import scf as pyscf_scf

    state = make_h2_state(R=0.74, basis="sto-3g", method="uhf")
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    assert isinstance(mf, pyscf_scf.uhf.UHF)


@pytest.mark.unit
def test_build_mf_returns_uks_for_method_uks() -> None:
    """_build_mf returns a PySCF UKS mean-field for method='uks'."""
    from pyscf.dft import uks as pyscf_uks

    state = make_h2_state(R=0.74, basis="sto-3g", method="uks", xc="B3LYP")
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    # PySCF's dft.UKS is a factory function; the actual class is dft.uks.UKS.
    assert isinstance(mf, pyscf_uks.UKS)
    assert mf.xc == "B3LYP"


# ---------------------------------------------------------------------------
# Finite output on a plausible H₂ density
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fock_truth_uhf_finite_on_h2_density() -> None:
    """UHF build_fock_truth produces finite output on a plausible H₂ density."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="uhf")
    P_alpha = np.array([[0.6, 0.55], [0.55, 0.6]], dtype=np.float64) * 0.5
    P_beta = np.array([[0.6, 0.55], [0.55, 0.6]], dtype=np.float64) * 0.5
    F_alpha, F_beta = build_fock_truth(state, (P_alpha, P_beta))
    assert np.all(np.isfinite(F_alpha))
    assert np.all(np.isfinite(F_beta))
