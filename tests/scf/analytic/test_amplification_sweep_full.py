"""Tier A sanity test for the P5 full amplification-sweep pipeline.

SPEC_scf §6 P5 / §8.10 R3.  The "full" suffix distinguishes this test
from R2's analytic-only sweep (``test_uhf_amplification_sweep.py``,
``test_amplification_sweep.py``): this exercises the **full** emulator-
coupled pipeline — truth SCF → NN-substituted emulator SCF → analytic
CPKS Jacobian → GRAIL resolvent → Born-remainder identity — at a coarse
3-point sub-grid (``R ∈ {1.10, 1.18, 1.25}`` Å) drawn from the P5 R sweep
(``np.linspace(1.10, 1.25, 8)`` Å).  The pipeline itself is the same as
``examples/measure_scf_amplification.py --method uhf --emulator-sweep``;
this Tier-A test budget-runs three R points in-pytest as the standing
acceptance for the measurement script's monotone-A_emp + Born-ratio
behavior.

The load-bearing acceptance test for the underlying machinery is the
P4 integration test ``test_resolvent_predicts_scf.py``.  This Tier A
test adds two structural properties of the P5 sweep that are NOT
asserted in the P4 single-R integration test:

  (i)  ``A_emp`` is monotone across the 3-point grid in the trained-NN
       regime.  Empirically (2026-05-11 first measurement) the trace is
       monotone-decreasing R = 1.10 → 1.18 → 1.25 Å because the CF peak
       at R ≈ 1.15 Å sits just below the 1.10 endpoint on the
       singlet/broken-symmetry boundary.  The test accepts either
       monotone-rising or monotone-falling — non-monotone indicates a
       regression.
  (ii) The Born-remainder ratio stays below the SPEC §7 acceptance bound
       (C = 1.25) at every sampled R inside the trained-NN window
       [1.10, 1.25] Å.

Wall-clock budget: ~30 s on the conda env (3 R points × ~10 s per R; the
peak R≈1.15 region is the slowest because of long Picard chains, but
this test's 3-point grid skips the peak interior and uses three quick
endpoints / mid-point).

Skipped when the trained UHF NN artifact is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pyscf")


# Path mirroring examples/measure_scf_amplification.py's _XC_NET_PATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_XC_NET_PATH = _PROJECT_ROOT / "examples" / "scf_xc_emulator_uhf_sto3g.h5"

# Coarse 3-point sub-grid of the P5 sweep.  Chosen for budget:
#   R = 1.10 Å — lowest in-range R; A_op ~ 21, low amplification anchor.
#   R = 1.18 Å — just past R_CF on the broken-symmetry branch (S² > 0).
#   R = 1.25 Å — top of R3 training range, broken-symmetry well-developed.
# Skips the R ≈ 1.15 Å peak interior (where Picard can take ~60 iter) so
# the in-pytest wall-clock stays under the 30-s budget.
_R_SWEEP_TIERA = (1.10, 1.18, 1.25)

# Born-remainder constants (SPEC_scf §7) — mirrored from the P4 integration
# test ``test_resolvent_predicts_scf.py``.
_C = 1.25
_TOL_FLOOR = 0.02


@pytest.mark.analytic
def test_A_scf_p5_sweep_full() -> None:
    """A-scf-p5-sweep-full: P5 emulator-coupled pipeline is well-behaved.

    Asserts:
      (1) Truth + emulator SCF both converge at every R in the 3-point grid.
      (2) ``A_emp`` is monotone across the 3-point grid (either rising or
          falling — empirically falling, per the 2026-05-11 first
          measurement, because the CF peak at R ≈ 1.15 Å lies just below
          the R = 1.10 lower endpoint).
      (3) The Born-remainder ratio stays below the SPEC §7 acceptance bound
          (``C = 1.25``, ``tol_floor = 0.02``) at every R sample.
      (4) GMRES converges within the diagnostic cap (50 iter).
    """
    if not _XC_NET_PATH.exists():
        pytest.skip(
            f"trained XCNet artifact not found at {_XC_NET_PATH}; "
            "run examples/train_xc_emulator_uhf_sto3g.py to produce it."
        )

    # Import inside the test so module import doesn't pay the PySCF cost when
    # the artifact is missing and this whole test would just skip anyway.
    from examples.measure_scf_amplification import _measure_one_p5
    from scf.emulator import load_xc_emulator

    xc_net = load_xc_emulator(_XC_NET_PATH, dtype=torch.float64)
    rows = [_measure_one_p5(R, xc_net) for R in _R_SWEEP_TIERA]

    # (1) Convergence on every R.
    for r in rows:
        assert r["converged_truth"], (
            f"Truth SCF failed to converge at R={r['R']:.4f} Å "
            f"(iter={r['truth_picard_iter']}).  This Tier-A test cannot "
            "exercise the GRAIL pipeline without truth convergence; if "
            "this is a real regression, check scf/host.py / scf/fock_build.py."
        )
        assert r["converged_emul"], (
            f"Emulator SCF failed to converge at R={r['R']:.4f} Å "
            f"(iter={r['emul_picard_iter']}).  Re-check the R3 NN artifact "
            "(examples/scf_xc_emulator_uhf_sto3g.h5) and the R3 rel_mae gate."
        )

    A_emp_by_R = {r["R"]: r["A_emp"] for r in rows}
    ratio_by_R = {r["R"]: r["born_ratio"] for r in rows}
    bound_by_R = {r["R"]: r["born_bound"] for r in rows}

    # (2) Monotone A_emp across the 3-point grid.  Empirically (the
    # measurement-script first run, 2026-05-11) the 3-point trace is
    # monotone-DECREASING from R = 1.10 → 1.18 → 1.25 Å — the singlet-side
    # CF peak at R ≈ 1.15 Å is bracketed between R = 1.10 (just inside the
    # trained NN range on the singlet side) and R = 1.18 (past R_CF on the
    # broken-symmetry side), so A_op falls steeply after the bifurcation,
    # which in turn pulls A_emp down.  The structural property the spec
    # asks for is "monotone A_emp in the trained-NN regime"; either
    # monotone-decreasing or monotone-increasing on the 3-point coarse
    # grid satisfies that.
    is_monotone_dec = (A_emp_by_R[1.10] > A_emp_by_R[1.18] > A_emp_by_R[1.25])
    is_monotone_inc = (A_emp_by_R[1.10] < A_emp_by_R[1.18] < A_emp_by_R[1.25])
    assert is_monotone_dec or is_monotone_inc, (
        f"A_emp values across the 3-point sweep are not monotone "
        f"(R=1.10 → {A_emp_by_R[1.10]:.4f}, "
        f"R=1.18 → {A_emp_by_R[1.18]:.4f}, "
        f"R=1.25 → {A_emp_by_R[1.25]:.4f}).  "
        "P5 sweep is expected to be monotone (rising or falling) across "
        "the trained-NN R range; non-monotone indicates a regression in "
        "the Picard/GRAIL pipeline or a structurally different topological "
        "regime than expected."
    )

    # (3) Born-remainder ratio inside the SPEC §7 bound at every R.
    for r in rows:
        R = r["R"]
        A_emp = r["A_emp"]
        rel_err = r["rel_err"]
        born_ratio = ratio_by_R[R]
        born_bound = bound_by_R[R]
        # Flat 5% check at A_emp <= 2 (Born series trivially short).
        # Same pattern as the P4 integration test.
        if A_emp <= 2.0:
            assert rel_err < 0.05, (
                f"P5 sweep flat 5% check failed at R={R} Å "
                f"(A_emp={A_emp:.4f} <= 2): rel_err={rel_err:.4f} >= 0.05."
            )
        else:
            assert born_ratio <= born_bound, (
                f"P5 sweep Born-remainder VIOLATION at R={R} Å: "
                f"ratio={born_ratio:.4f} > bound={born_bound:.4f}  "
                f"(rel_err={rel_err:.4f}, A_emp={A_emp:.3f}).  "
                "Check SPEC §6 P4 step 8 failure-disambiguation block."
            )
        # Defense against ratio > C (SPEC §7 envelope before tol_floor):
        # the tol_floor adjustment must not be the only thing keeping us
        # under bound on the Tier-A grid.  This is a stricter version of
        # the integration test's loose bound — the Tier-A grid avoids
        # the peak interior so we expect headroom against bare C.
        assert born_ratio <= _C, (
            f"P5 sweep ratio exceeds bare C at R={R} Å: "
            f"ratio={born_ratio:.4f} > C={_C}.  Tier-A grid avoids the "
            "R≈1.15 Å peak interior so this should never trigger; if it "
            "does, check the FD-JVP / chain-rule-labeling pipeline."
        )

    # (4) GMRES convergence cap.
    for r in rows:
        assert 0 <= r["gmres_iter"] <= 50, (
            f"GMRES iter count at R={r['R']:.4f} Å is "
            f"{r['gmres_iter']} (cap=50).  Either the cap is being hit "
            "(GMRES did not converge) or the value is negative (sentinel "
            "for non-convergence — would have been caught by (1) above)."
        )
