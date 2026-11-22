"""Tier A — UHF amplification-factor sweep across the Coulson–Fischer point.

SPEC_scf §8.10 R2.  Sweeps ``R ∈ {1.00, 1.10, 1.15, 1.18, 1.30} Å`` for
H₂ / STO-3G / UHF with the broken-symmetry initial guess.  At each R:
converge UHF, build the analytic CPKS Jacobian via the doubled state-
vector seam, and characterize the dominant POSITIVE eigenvalue and the
operator-norm amplification.

The diagnostic ``examples/diagnose_uhf_amplification.py`` (2026-05-11)
established the expected behavior:

  R (Å) | λ_max+   | A_op    | ⟨S²⟩
  ------|----------|---------|------
  1.00  | 0.8689   |   7.63  |  ≈ 0
  1.10  | 0.9526   |  21.08  |  ≈ 0
  1.15  | 0.9969   | 320.72  |  ≈ 0   ← peak (singlet side of R_CF)
  1.18  | 0.9535   |  21.90  |  0.087  (broken-symmetry takes)
  1.30  | 0.7804   |   4.77  |  0.394

λ_max+ approaches +1 from below as R → R_CF^- (singlet side), peaks just
before the broken-symmetry branch takes, then retreats.  This is the
textbook Cizek–Paldus 1967 RHF→UHF triplet-instability signature
(SPEC_scf §8.10): the cycle's spin-localizing mode becomes the unstable
direction that triggers the bifurcation.

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
    cpks_jacobian_analytic_uhf,
    cpks_jacobian_fd_uhf,
    spectral_radius,
)


# Sweep grid (Å) from the R2 task brief — frames the CF transition.
_R_SWEEP = (1.00, 1.10, 1.15, 1.18, 1.30)


def _measure_one(R: float) -> dict:
    """Converge UHF and compute (A_op, ρ, λ_max+) at one bond length.

    Returns NaN observables if SCF fails to converge (should not occur on
    this 5-point grid; the diagnostic converged all 8 points cleanly).
    """
    state = make_h2_state(R=R, basis="sto-3g", method="uhf")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200, accelerator="diis")
    if not result.converged:
        return {
            "R": R,
            "A_op": float("nan"),
            "spectral_radius": float("nan"),
            "lambda_max_positive": float("nan"),
            "s_squared": float("nan"),
        }
    DT = cpks_jacobian_analytic_uhf(
        state,
        P_star_alpha=result.P_star[0], P_star_beta=result.P_star[1],
        mo_energy_alpha=result.mo_energy[0], mo_energy_beta=result.mo_energy[1],
        mo_coeff_alpha=result.mo_coeff[0], mo_coeff_beta=result.mo_coeff[1],
    )
    eigs = np.linalg.eigvals(DT)
    # Most-positive real part — the Cizek–Paldus instability mode.
    lam_pos = float(np.max(eigs.real))
    return {
        "R": R,
        "A_op": A_op_analytic(DT),
        "spectral_radius": spectral_radius(DT),
        "lambda_max_positive": lam_pos,
        "s_squared": float(result.s_squared) if result.s_squared is not None
                    else float("nan"),
    }


# ---------------------------------------------------------------------------
# A-scf-uhf-amplification-cf : peak + monotone rise/fall structure
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_A_scf_uhf_amplification_cf() -> None:
    """A-scf-uhf-amplification-cf: λ_max+ peaks near R_CF and A_op tracks it.

    Diagnostic observed (2026-05-11):
      R=1.00 Å  λ_max+=0.8689   A_op≈ 7.6
      R=1.10 Å  λ_max+=0.9526   A_op≈21.1
      R=1.15 Å  λ_max+=0.9969   A_op≈320.7   (peak)
      R=1.18 Å  λ_max+=0.9535   A_op≈21.9
      R=1.30 Å  λ_max+=0.7804   A_op≈ 4.8

    Assertions:
      (a) λ_max+ at R=1.15 Å peaks with value ≥ 0.99.
      (b) A_op at R=1.15 Å is ≥ 100 (the diagnostic measured 320).
      (c) Monotone rise from R=1.00 Å to the peak at R=1.15 Å.
      (d) Monotone drop from R=1.18 Å to R=1.30 Å (post-CF retreat).
      (e) Each measured row reproduces the diagnostic λ_max+ to ≤ 1%.
    """
    rows = [_measure_one(R) for R in _R_SWEEP]
    for r in rows:
        assert np.isfinite(r["lambda_max_positive"]), (
            f"λ_max+ not finite at R={r['R']}: UHF likely failed to converge."
        )
        assert np.isfinite(r["A_op"]), f"A_op not finite at R={r['R']}."

    # Build a {R: row} index for clarity below.
    by_R = {r["R"]: r for r in rows}

    # (a) Peak λ_max+ at R=1.15 Å reaches the near-singular value.
    lam_at_peak = by_R[1.15]["lambda_max_positive"]
    assert lam_at_peak >= 0.99, (
        f"λ_max+ at R=1.15 Å peak = {lam_at_peak:.4f}; expected ≥ 0.99 "
        f"(diagnostic measured 0.9969)."
    )

    # (b) A_op at the peak meets the diagnostic floor.
    A_at_peak = by_R[1.15]["A_op"]
    assert A_at_peak >= 100.0, (
        f"A_op at R=1.15 Å peak = {A_at_peak:.2f}; expected ≥ 100 "
        f"(diagnostic measured 320)."
    )

    # (c) Monotone rise: R=1.00 → R=1.10 → R=1.15.
    lam_R100 = by_R[1.00]["lambda_max_positive"]
    lam_R110 = by_R[1.10]["lambda_max_positive"]
    lam_R115 = by_R[1.15]["lambda_max_positive"]
    assert lam_R100 < lam_R110 < lam_R115, (
        f"λ_max+ not monotonically rising approaching the CF peak: "
        f"R=1.00→{lam_R100:.4f}, R=1.10→{lam_R110:.4f}, R=1.15→{lam_R115:.4f}."
    )
    # Sanity: R=1.00 Å pre-CF should be roughly the diagnostic value 0.869.
    assert lam_R100 > 0.85, (
        f"λ_max+ at R=1.00 Å = {lam_R100:.4f}; expected > 0.85 (diagnostic 0.869)."
    )

    # (d) Monotone drop post-CF: R=1.18 → R=1.30.
    lam_R118 = by_R[1.18]["lambda_max_positive"]
    lam_R130 = by_R[1.30]["lambda_max_positive"]
    assert lam_R118 > lam_R130, (
        f"λ_max+ did not drop from R=1.18 to R=1.30 (post-CF retreat): "
        f"R=1.18→{lam_R118:.4f}, R=1.30→{lam_R130:.4f}."
    )
    # The peak at R=1.15 should be higher than both R=1.18 and R=1.30 too.
    assert lam_R115 > lam_R118, (
        f"λ_max+ peak at R=1.15 ({lam_R115:.4f}) should exceed R=1.18 "
        f"({lam_R118:.4f}); the peak is supposed to sit on the singlet side "
        f"just before broken-symmetry takes."
    )

    # (e) Diagnostic-value reproduction to ≤ 1% per row.  Reference values
    # come straight from examples/plot_data/uhf_amplification_diagnostic.npz.
    _DIAGNOSTIC_LAMBDA_POS = {
        1.00: 0.86894516,
        1.10: 0.95257227,
        1.15: 0.99688203,
        1.18: 0.95353387,
        1.30: 0.78038835,
    }
    for R, lam_expected in _DIAGNOSTIC_LAMBDA_POS.items():
        lam_observed = by_R[R]["lambda_max_positive"]
        rel_diff = abs(lam_observed - lam_expected) / abs(lam_expected)
        assert rel_diff <= 0.01, (
            f"λ_max+ at R={R:.2f} Å = {lam_observed:.6f} differs from "
            f"diagnostic {lam_expected:.6f} by {rel_diff*100:.2f}% > 1%."
        )


# ---------------------------------------------------------------------------
# A-scf-uhf-fd-agreement : FD vs analytic UHF Jacobian at R=1.18 Å
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_A_scf_uhf_fd_agreement() -> None:
    """A-scf-uhf-fd-agreement: FD-vs-analytic at R=1.18 Å passes to 4 sig figs.

    Looser tolerance than the Tier-U R=1.18 Å test (which asserts 6 sig
    figs).  R=1.18 Å sits just past R_CF; the Jacobian entries are O(1)
    and central-FD discretization noise at ``h=1e-4`` is ~1e-9 absolute.
    The Tier-A gate keeps this assertion separate from the unit test so
    a future small-magnitude perturbation regression surfaces at the
    correct test tier.
    """
    R = 1.18
    state = make_h2_state(R=R, basis="sto-3g", method="uhf")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200, accelerator="diis")
    assert result.converged, f"UHF SCF did not converge at R={R}."

    DT_an = cpks_jacobian_analytic_uhf(
        state,
        P_star_alpha=result.P_star[0], P_star_beta=result.P_star[1],
        mo_energy_alpha=result.mo_energy[0], mo_energy_beta=result.mo_energy[1],
        mo_coeff_alpha=result.mo_coeff[0], mo_coeff_beta=result.mo_coeff[1],
    )
    DT_fd = cpks_jacobian_fd_uhf(state, result.P_star, h=1e-4)

    max_an = np.max(np.abs(DT_an))
    diff = np.max(np.abs(DT_an - DT_fd))
    rel = diff / max_an
    assert rel < 1e-4, (
        f"FD-vs-analytic UHF Jacobian agreement at R={R}: rel = {rel:.3e} "
        f"> 1e-4; max abs diff = {diff:.3e}, max abs analytic = {max_an:.3e}."
    )
