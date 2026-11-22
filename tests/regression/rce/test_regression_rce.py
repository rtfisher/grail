"""Tier R regression tests for the climate RCE testbed.

These tests load committed HDF5 references from `tests/regression/rce/` and
assert that the documented physical invariants hold on the stored values.
They are deliberately data-integrity checks (load + assert invariants),
not re-run tests --- the re-run path is exercised in the integration
tests at `tests/climate_rce/integration/test_science_rce_mip.py`, which
also stage the proposed references.  This file guards the reference data
itself against accidental corruption.

Activated 2026-05-21 (PI sign-off; see `examples/proposed_regression/
rce_tier_r_review.md`).

Test IDs:
  R-rce-baseline   — baseline RCE column-mean T and 2×CO₂ sensitivity
                     lie inside the Wing-2018 simple-column-model band.
                     ρ(K_eff_T) < 1 (strict contraction gate).
  R-rce-systematic — Born-remainder ratio ≤ C=1.25 at every converged
                     dial level across the smooth-surrogate dt sweep
                     (5 levels from dt=2e5 s down to dt=500 s).
  R-rce-alignment  — at the near-critical-f operating point
                     (dt=500 s, ρ(K) ≈ 0.9994, A_op ≈ 1604), the
                     dominant resolvent eigenvector aligns with the
                     dominant K eigenvector to < 20°.  Confirms the
                     amplification is the recognised water-vapor /
                     cloud-feedback mode, not a numerical artifact.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

_REG_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# R-rce-baseline
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_R_rce_baseline() -> None:
    """R-rce-baseline: Wing-2018 SCM-envelope plausibility floor.

    Column-mean T inside [200, 310] K; 2×CO₂ sensitivity inside [0.5, 12] K;
    ρ(K_eff_T) < 1 strictly (no destabilisation at the baseline).
    """
    ref_path = _REG_DIR / "rce_baseline.h5"
    assert ref_path.exists(), f"Reference file not found: {ref_path}"

    with h5py.File(ref_path, "r") as f:
        T_mean = float(np.asarray(f["T_mean_K"])[0])
        rce_sens = float(np.asarray(f["rce_sensitivity_K"])[0])
        rho_K = float(np.asarray(f["rho_K"])[0])
        T_profile = np.asarray(f["T_profile_K"])
        T_mean_lo = float(np.asarray(f["metadata/T_mean_band_lo"]))
        T_mean_hi = float(np.asarray(f["metadata/T_mean_band_hi"]))
        sens_lo = float(np.asarray(f["metadata/sens_band_lo"]))
        sens_hi = float(np.asarray(f["metadata/sens_band_hi"]))

    # Wing-2018 SCM-envelope plausibility floor (NOT a tight RCE-MIP gate).
    assert T_mean_lo <= T_mean <= T_mean_hi, (
        f"R-rce-baseline: column-mean T={T_mean:.2f} K outside the "
        f"Wing-2018 SCM band [{T_mean_lo}, {T_mean_hi}] K."
    )
    assert sens_lo <= rce_sens <= sens_hi, (
        f"R-rce-baseline: 2×CO₂ sensitivity {rce_sens:.2f} K outside the "
        f"Wing-2018 SCM band [{sens_lo}, {sens_hi}] K."
    )

    # Strict contraction: ρ(K_eff_T) < 1.
    assert 0.0 < rho_K < 1.0, (
        f"R-rce-baseline: ρ(K_eff_T)={rho_K:.6f} not in (0, 1) — "
        f"contraction gate failed."
    )

    # Sanity: profile shape.
    assert T_profile.shape == (10,), (
        f"R-rce-baseline: T_profile shape {T_profile.shape} != (10,)."
    )
    assert np.all(np.isfinite(T_profile)), (
        "R-rce-baseline: T_profile contains non-finite values."
    )


# ---------------------------------------------------------------------------
# R-rce-systematic
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_R_rce_systematic() -> None:
    """R-rce-systematic: Born-remainder ratio ≤ C across the dt sweep.

    Five dial levels from dt=2e5 s (well-conditioned) to dt=500 s
    (near-critical-f, A_op ≈ 1604).  All converged levels must pass the
    Born-remainder bound (ratio ≤ C + tol_floor / floor) and not be
    flagged born_nonlinear.
    """
    ref_path = _REG_DIR / "rce_systematic.h5"
    assert ref_path.exists(), f"Reference file not found: {ref_path}"

    with h5py.File(ref_path, "r") as f:
        ratio = np.asarray(f["ratio"])
        bound = np.asarray(f["bound"])
        converged = np.asarray(f["converged"]).astype(bool)
        passes = np.asarray(f["passes"]).astype(bool)
        born_nonlinear = np.asarray(f["born_nonlinear"]).astype(bool)
        dt_seconds = np.asarray(f["dt_seconds"])
        A_emp = np.asarray(f["A_emp"])
        C_const = float(np.asarray(f["metadata/C"]))

    # Five dial levels expected.
    assert ratio.shape == (5,), (
        f"R-rce-systematic: ratio shape {ratio.shape} != (5,)."
    )

    # Sweep spans from well-conditioned to near-critical.
    assert A_emp[0] < 100.0, (
        f"R-rce-systematic: well-conditioned level A_emp={A_emp[0]:.2f} "
        "should be << 100."
    )
    assert A_emp[-1] > 1000.0, (
        f"R-rce-systematic: near-critical level A_emp={A_emp[-1]:.2f} "
        "should be > 1000."
    )

    # All converged levels must pass the Born-remainder bound.
    for i, (r, b, conv, pass_, nl, dt) in enumerate(
        zip(ratio, bound, converged, passes, born_nonlinear, dt_seconds)
    ):
        if not conv:
            continue
        assert pass_, (
            f"R-rce-systematic [level {i}, dt={dt:.0f} s]: "
            f"ratio={r:.4f} > bound={b:.4f}; gate failed."
        )
        assert not nl, (
            f"R-rce-systematic [level {i}, dt={dt:.0f} s]: "
            f"flagged born_nonlinear (ratio={r:.4f})."
        )
        assert r <= b, (
            f"R-rce-systematic [level {i}, dt={dt:.0f} s]: "
            f"ratio={r:.4f} > bound={b:.4f}."
        )

    # C bound is the SCF-tied value 1.25.
    assert abs(C_const - 1.25) < 1e-12, (
        f"R-rce-systematic: stored C={C_const} != 1.25."
    )


# ---------------------------------------------------------------------------
# R-rce-alignment
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_R_rce_alignment() -> None:
    """R-rce-alignment: dominant G-mode aligns with dominant K-mode.

    At the near-critical-f operating point (dt=500 s), the dominant mode
    of G = (I - K)^{-1} is the dominant mode of K (the slowest feedback
    mode = water-vapor / cloud-feedback mode under the fixed-RH closure).
    The angle gate is < 20°; central-FD floor ≈ 0.2°.
    """
    ref_path = _REG_DIR / "rce_alignment.h5"
    assert ref_path.exists(), f"Reference file not found: {ref_path}"

    with h5py.File(ref_path, "r") as f:
        angle_deg = float(np.asarray(f["angle_deg"])[0])
        rho_K = float(np.asarray(f["rho_K"])[0])
        A_op = float(np.asarray(f["A_op_approx"])[0])
        gate_angle_max = float(np.asarray(f["metadata/gate_angle_max_deg"]))
        gate_A_op_min = float(np.asarray(f["metadata/gate_A_op_min"]))
        dom_eigvec_G = np.asarray(f["dom_eigvec_G"])
        dom_eigvec_K = np.asarray(f["dom_eigvec_K"])

    # Strict contraction at the near-critical operating point.
    assert 0.0 < rho_K < 1.0, (
        f"R-rce-alignment: ρ(K)={rho_K:.6f} not in (0, 1) — "
        f"contraction gate failed at the near-critical config."
    )

    # Near-critical: A_op meets the documented floor (≥ 100).
    assert A_op >= gate_A_op_min, (
        f"R-rce-alignment: A_op={A_op:.2f} < gate {gate_A_op_min:.0f}; "
        f"near-critical condition not reached."
    )

    # Alignment gate: dominant G-mode within 20° of dominant K-mode.
    assert angle_deg < gate_angle_max, (
        f"R-rce-alignment: dominant G-mode vs K-mode angle "
        f"{angle_deg:.2f}° ≥ gate {gate_angle_max:.0f}°."
    )

    # Cross-check the alignment angle against the stored eigenvectors.
    e_G = dom_eigvec_G / max(np.linalg.norm(dom_eigvec_G), 1e-300)
    e_K = dom_eigvec_K / max(np.linalg.norm(dom_eigvec_K), 1e-300)
    cos_theta = abs(float(np.dot(e_G, e_K)))
    cos_theta = min(cos_theta, 1.0)
    angle_recomputed_deg = float(np.degrees(np.arccos(cos_theta)))
    assert abs(angle_recomputed_deg - angle_deg) < 1.0, (
        f"R-rce-alignment: stored angle {angle_deg:.2f}° disagrees with "
        f"recomputed {angle_recomputed_deg:.2f}° by > 1°."
    )
