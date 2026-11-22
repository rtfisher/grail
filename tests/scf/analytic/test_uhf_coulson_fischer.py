"""Tier A — A-scf-coulson-fischer.  SPEC_scf §8.10 R1.

UHF on H₂/STO-3G across a bond-length sweep that straddles the Coulson–
Fischer point (R_CF ≈ 1.16 Å for STO-3G).  The textbook signature of the
Čížek–Paldus (1967) RHF→UHF instability is:

- Below R_CF: ⟨S²⟩ = 0 (UHF collapses to RHF; the broken-symmetry initial
  guess is not a stable solution).
- Above R_CF: ⟨S²⟩ > 0 grows monotonically with R; the spin-localizing
  orbital-rotation mode becomes the SCF descent direction.

We sweep R ∈ {1.00, 1.15, 1.18, 1.30} Å and verify:
- R=1.00 Å: ⟨S²⟩ ≈ 0.
- R=1.30 Å: ⟨S²⟩ > 0.3 (deep into broken-symmetry regime).
- The CF transition is captured between R=1.15 and R=1.18 Å: ∂⟨S²⟩/∂R
  becomes non-zero in that interval (⟨S²⟩(1.15) = 0, ⟨S²⟩(1.18) > 0).
- Total energies match the diagnostic's recorded values to 1e-6 Ha.

Reference values from ``examples/uhf_amplification_diagnostic.md``
(run 2026-05-11, PySCF 2.11):

| R (Å) | ⟨S²⟩    | E_total    |
|-------|---------|------------|
| 1.00  | 0.000   | -1.066109  |
| 1.15  | 0.000   | -1.020964  |
| 1.18  | 0.086690| -1.011897  |
| 1.30  | 0.394041| -0.984028  |

This test exercises the load-bearing physics of the SCF testbed pivot
(SPEC_scf §8.10).  Failure means the broken-symmetry initial guess is no
longer selecting the UHF branch — which would invalidate the R2 / R3
amplification measurements downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.host import picard_iterate_scf


# Diagnostic reference values (2026-05-11).
_R_SWEEP = (1.00, 1.15, 1.18, 1.30)
_REF = {
    1.00: {"S2": 0.000000, "E_total": -1.066109},
    1.15: {"S2": 0.000000, "E_total": -1.020964},
    1.18: {"S2": 0.086690, "E_total": -1.011897},
    1.30: {"S2": 0.394041, "E_total": -0.984028},
}


def _converge_uhf(R: float) -> tuple[float, float, int]:
    """Run UHF at one bond length and return (⟨S²⟩, E_total, iter_count)."""
    state = make_h2_state(R=R, basis="sto-3g", method="uhf")
    result = picard_iterate_scf(state, accelerator="diis", tol=1e-9, max_iter=200)
    assert result.converged is True, (
        f"UHF did not converge at R={R:.2f} Å (iter_count={result.iter_count})."
    )
    return float(result.s_squared), float(result.energy_total), int(result.iter_count)


@pytest.mark.analytic
def test_A_scf_coulson_fischer_sweep() -> None:
    """A-scf-coulson-fischer: UHF sweep straddling R_CF on H₂/STO-3G.

    The test fixes four bond lengths spanning R_CF ≈ 1.16 Å and verifies
    the textbook Čížek–Paldus instability signature: ⟨S²⟩ stays at zero
    on the singlet side of R_CF and grows monotonically above.
    """
    out: dict[float, tuple[float, float, int]] = {}
    for R in _R_SWEEP:
        s2, e_total, n_iter = _converge_uhf(R)
        out[R] = (s2, e_total, n_iter)

    # ---- (1) Total energies match the diagnostic to 1e-6 Ha. ----------
    for R, (s2, e_total, _) in out.items():
        ref_e = _REF[R]["E_total"]
        assert abs(e_total - ref_e) < 1e-6, (
            f"E_total at R={R:.2f} Å = {e_total:.7f}; diagnostic value "
            f"{ref_e:.7f}; deviation {e_total - ref_e:+.2e} Ha exceeds 1e-6."
        )

    # ---- (2) ⟨S²⟩ ≈ 0 at R=1.00 Å (RHF regime). ----------------------
    s2_100 = out[1.00][0]
    assert abs(s2_100) < 1e-3, (
        f"⟨S²⟩ at R=1.00 Å = {s2_100}; expected ≈ 0 (pre-CF, RHF regime)."
    )

    # ---- (3) ⟨S²⟩ > 0.3 at R=1.30 Å (broken-symmetry regime). --------
    s2_130 = out[1.30][0]
    assert s2_130 > 0.3, (
        f"⟨S²⟩ at R=1.30 Å = {s2_130:.4f}; expected > 0.3 (broken-symmetry "
        "regime).  UHF appears to have collapsed back to RHF — the broken-"
        "symmetry initial guess is no longer selecting the UHF branch."
    )

    # ---- (4) CF transition captured between R=1.15 and R=1.18 Å. -----
    # At R=1.15 ⟨S²⟩=0 (pre-CF / singlet branch); at R=1.18 ⟨S²⟩>0.05 (post-CF).
    # ∂⟨S²⟩/∂R becomes non-zero in (1.15, 1.18) Å.
    s2_115 = out[1.15][0]
    s2_118 = out[1.18][0]
    assert abs(s2_115) < 1e-3, (
        f"⟨S²⟩ at R=1.15 Å = {s2_115:.4f}; expected ≈ 0 (just below R_CF). "
        "If this fires, R_CF for STO-3G has shifted from its diagnostic value."
    )
    assert s2_118 > 0.05, (
        f"⟨S²⟩ at R=1.18 Å = {s2_118:.4f}; expected > 0.05 (just above R_CF). "
        "If this fires, the broken-symmetry initial guess is failing to "
        "select the post-CF branch."
    )
    # ⟨S²⟩ values must match the diagnostic within ~1% of the reference.
    for R in (1.18, 1.30):
        ref_s2 = _REF[R]["S2"]
        got_s2 = out[R][0]
        # Tolerance: 1% relative or 1e-3 absolute, whichever is larger.
        tol_abs = max(1e-3, 0.01 * ref_s2)
        assert abs(got_s2 - ref_s2) < tol_abs, (
            f"⟨S²⟩ at R={R:.2f} Å = {got_s2:.6f}; diagnostic value "
            f"{ref_s2:.6f}; deviation exceeds tol={tol_abs:.2e}."
        )

    # ---- (5) Monotonicity check on the post-CF branch. -----------------
    # Above R_CF, ⟨S²⟩ grows strictly with R.
    assert out[1.18][0] < out[1.30][0], (
        f"⟨S²⟩ does not grow monotonically post-CF: "
        f"S²(1.18)={out[1.18][0]:.4f} not < S²(1.30)={out[1.30][0]:.4f}."
    )


@pytest.mark.analytic
def test_A_scf_coulson_fischer_post_cf_energy_lower() -> None:
    """Post-CF (R=1.30 Å) the broken-symmetry UHF energy is below the RHF energy.

    This is the textbook quantitative signature of the CF transition: above
    R_CF the broken-symmetry solution has lower total energy than the
    singlet RHF reference, confirming that the UHF branch is variationally
    preferred (Čížek–Paldus 1967).
    """
    R = 1.30
    state_uhf = make_h2_state(R=R, basis="sto-3g", method="uhf")
    state_rhf = make_h2_state(R=R, basis="sto-3g", method="rhf")
    res_uhf = picard_iterate_scf(state_uhf, tol=1e-9, max_iter=200)
    res_rhf = picard_iterate_scf(state_rhf, tol=1e-9, max_iter=200)
    assert res_uhf.converged and res_rhf.converged

    # UHF energy should be strictly below RHF at R > R_CF.
    delta_E = res_uhf.energy_total - res_rhf.energy_total
    assert delta_E < -1e-4, (
        f"UHF energy = {res_uhf.energy_total:.6f} Ha is not below "
        f"RHF energy = {res_rhf.energy_total:.6f} Ha at R={R:.2f} Å; "
        f"delta_E = {delta_E:+.6f} Ha.  Expected delta_E < -1e-4."
    )
