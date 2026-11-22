"""Tier U tests for the UHF/UKS extension of scf/host.py.  SPEC_scf §8.10 R1.

Covers:
- ``picard_iterate_scf(method="uhf")`` on H₂/STO-3G at R=1.30 Å (post-CF):
  converges and reports non-trivial ⟨S²⟩.
- At R=1.00 Å (pre-CF) UHF collapses back to RHF: ⟨S²⟩ ≈ 0.
- Iteration count ≤ 30 with DIIS.
- ``SCFResult`` field shapes match the documented UHF semantics
  (P_star, mo_energy, mo_coeff are 2-tuples).

Reference values from ``examples/uhf_amplification_diagnostic.md`` (run
2026-05-11): R=1.00 Å → ⟨S²⟩=0, iter=12; R=1.18 Å → ⟨S²⟩=0.087, iter=17;
R=1.20 Å → ⟨S²⟩=0.147, iter=14; R=1.30 Å → ⟨S²⟩=0.394, iter=6.

These tests require PySCF.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.host import SCFResult, picard_iterate_scf


# Reference ⟨S²⟩ values from the diagnostic (2026-05-11).  The Picard driver
# may produce values within ~1% of these due to a different convergence
# path than PySCF's mf.kernel — both flow to the same fixed point.
_REF_S2_AT_R = {
    1.00: 0.000,
    1.18: 0.087,
    1.20: 0.147,
    1.30: 0.394,
}


# ---------------------------------------------------------------------------
# Convergence and ⟨S²⟩ at post-CF stretched bond
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_picard_uhf_converges_at_R_130_with_nontrivial_S2() -> None:
    """UHF on H₂/STO-3G at R=1.30 Å converges to a broken-symmetry state.

    Reference: ⟨S²⟩ ≈ 0.394 from the diagnostic; we accept any value > 0.1
    (the "broken-symmetry takes" gate) and check the diagnostic value
    within ~1%.
    """
    state = make_h2_state(R=1.30, basis="sto-3g", method="uhf")
    assert state.init_guess == "breaksym"
    result = picard_iterate_scf(state, accelerator="diis", tol=1e-9, max_iter=200)
    assert result.converged is True, (
        f"UHF did not converge at R=1.30 Å; iter_count={result.iter_count}, "
        f"last residual={result.residual_history[-1] if result.residual_history else None}."
    )
    assert result.s_squared is not None
    assert result.s_squared > 0.1, (
        f"⟨S²⟩={result.s_squared:.4f} at R=1.30 Å; expected > 0.1 (broken-"
        f"symmetry regime). UHF appears to have collapsed to RHF."
    )
    # Diagnostic value 0.394; accept ±1% to absorb path-to-fixed-point
    # variation between our explicit Picard driver and PySCF's mf.kernel.
    expected = _REF_S2_AT_R[1.30]
    assert abs(result.s_squared - expected) < 0.01 * max(expected, 0.1), (
        f"⟨S²⟩={result.s_squared:.6f} differs from diagnostic "
        f"{expected:.6f} by more than 1%."
    )


@pytest.mark.unit
def test_picard_uhf_collapses_to_rhf_at_R_100() -> None:
    """UHF on H₂/STO-3G at R=1.00 Å collapses to RHF: ⟨S²⟩ ≈ 0.

    Below the Coulson–Fischer point the broken-symmetry initial guess is
    not a stable solution; SCF descends to the RHF fixed point and ⟨S²⟩
    vanishes to numerical zero.
    """
    state = make_h2_state(R=1.00, basis="sto-3g", method="uhf")
    result = picard_iterate_scf(state, accelerator="diis", tol=1e-9, max_iter=200)
    assert result.converged is True
    assert result.s_squared is not None
    # |⟨S²⟩| < 1e-3 is "essentially zero" — PySCF's spin_square returns
    # tiny negative values (~ -1e-15) at perfectly closed-shell UHF.
    assert abs(result.s_squared) < 1e-3, (
        f"⟨S²⟩={result.s_squared} at R=1.00 Å (pre-CF); expected ≈ 0."
    )


@pytest.mark.unit
def test_picard_uhf_iter_count_bounded_at_R_130() -> None:
    """UHF with DIIS at R=1.30 Å converges in ≤ 30 iterates."""
    state = make_h2_state(R=1.30, basis="sto-3g", method="uhf")
    result = picard_iterate_scf(state, accelerator="diis", tol=1e-9, max_iter=200)
    assert result.converged is True
    assert result.iter_count <= 30, (
        f"UHF at R=1.30 Å took {result.iter_count} iters; SPEC says ≤ 30 with DIIS."
    )


# ---------------------------------------------------------------------------
# SCFResult shape & semantics for UHF
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_picard_uhf_result_has_tuple_shapes() -> None:
    """SCFResult fields are 2-tuples for UHF; s_squared is populated."""
    state = make_h2_state(R=1.30, basis="sto-3g", method="uhf")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200)
    assert isinstance(result, SCFResult)
    # P_star, mo_energy, mo_coeff are all 2-tuples.
    assert isinstance(result.P_star, tuple) and len(result.P_star) == 2
    assert isinstance(result.mo_energy, tuple) and len(result.mo_energy) == 2
    assert isinstance(result.mo_coeff, tuple) and len(result.mo_coeff) == 2
    P_alpha, P_beta = result.P_star
    assert P_alpha.shape == (2, 2)
    assert P_beta.shape == (2, 2)
    eps_a, eps_b = result.mo_energy
    assert eps_a.shape == (2,)
    assert eps_b.shape == (2,)
    C_a, C_b = result.mo_coeff
    assert C_a.shape == (2, 2)
    assert C_b.shape == (2, 2)
    # s_squared populated for UHF, not None.
    assert result.s_squared is not None


@pytest.mark.unit
def test_picard_uhf_spin_blocks_symmetric_and_PSD() -> None:
    """Converged UHF P_α and P_β are each symmetric and PSD."""
    state = make_h2_state(R=1.30, basis="sto-3g", method="uhf")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200)
    P_alpha, P_beta = result.P_star
    for label, P in (("alpha", P_alpha), ("beta", P_beta)):
        assert np.allclose(P, P.T, atol=1e-12), f"P_{label} not symmetric."
        eigs = np.linalg.eigvalsh(P)
        assert eigs.min() > -1e-8, (
            f"P_{label} not PSD; smallest eig = {eigs.min()}."
        )


@pytest.mark.unit
def test_picard_uhf_total_charge_conserved() -> None:
    """tr(P_α S) + tr(P_β S) = n_electrons at the converged density."""
    from scf.fock_build import _build_mf, _build_mol

    state = make_h2_state(R=1.30, basis="sto-3g", method="uhf")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200)

    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    P_alpha, P_beta = result.P_star
    tr_PS = float(np.trace(P_alpha @ S) + np.trace(P_beta @ S))
    assert abs(tr_PS - 2.0) < 1e-8, (
        f"tr(P_α S) + tr(P_β S) = {tr_PS}; expected 2.0 (n_electrons)."
    )


# ---------------------------------------------------------------------------
# UKS smoke test
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_picard_uks_converges_at_R_130() -> None:
    """UKS/B3LYP on H₂/STO-3G at R=1.30 Å converges; ⟨S²⟩ tracked.

    UKS may or may not break symmetry at R=1.30 Å depending on the B3LYP
    Coulson–Fischer point (close to but not identical to the UHF value).
    We assert only convergence and that s_squared is populated; the
    UHF/STO-3G CF transition is the load-bearing test for the
    amplification claim, not the UKS one.
    """
    state = make_h2_state(R=1.30, basis="sto-3g", method="uks", xc="B3LYP")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200)
    assert result.converged is True
    assert result.s_squared is not None
    assert result.iter_count <= 50


# ---------------------------------------------------------------------------
# Closed-shell paths are unchanged
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_picard_rhf_path_unchanged_returns_array_not_tuple() -> None:
    """The RHF/RKS path continues to return single-matrix SCFResult fields."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="rhf")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200)
    assert isinstance(result.P_star, np.ndarray)
    assert result.P_star.shape == (2, 2)
    assert isinstance(result.mo_energy, np.ndarray)
    assert result.mo_energy.shape == (2,)
    assert result.s_squared is None
