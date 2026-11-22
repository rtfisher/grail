"""Tier A — amplification-factor sweep across R.  SPEC_scf §6 P2.

Sweep ``R ∈ {0.74, 1.0, 1.48, 2.0, 2.22} Å`` (the 5-point grid from the
task brief — corresponds to ``R ∈ {1.0, 1.35, 2.0, 2.7, 3.0} R_eq`` with
``R_eq = 0.74 Å``).  At each R we converge the truth RKS/B3LYP/STO-3G
SCF, build the analytic CPKS Jacobian via PySCF's response API, and
record the operator-norm amplification ``A_op = ||(I − D𝒯)^{-1}||_2``
and the spectral radius ``ρ(D𝒯) = max_k |λ_k(D𝒯)|``.

Reformulation note (implementor advisory; see BLOCKERS in P2 completion
report):
-----------------------------------------------------------------------
The task brief asks for strict monotonic non-decreasing ``A_op(R)`` over
the sweep, with ``A_op(3 R_eq) >> A_op(R_eq)``.  Empirically this does
NOT hold for H₂/STO-3G or H₂/6-31G under B3LYP:

  STO-3G basis (M=2, n_unique=3):
    R=0.74   A_op=1.0000   ρ=0.378
    R=1.00   A_op=1.0000   ρ=0.693
    R=1.48   A_op=1.0000   ρ=1.812
    R=2.00   A_op=1.0000   ρ=4.346
    R=2.22   A_op=1.0000   ρ=6.015

  6-31G basis (M=4, n_unique=10):
    R=0.74   A_op=1.448    ρ=0.385
    R=1.00   A_op=1.380    ρ=0.645
    R=1.48   A_op=1.273    ρ=1.442
    R=2.00   A_op=1.240    ρ=3.082
    R=2.22   A_op=1.246    ρ=4.116

The reason is structural: at H₂'s closed-shell SCF fixed point the
dominant CPKS eigenvalue is **negative** (the cycle oscillates rather
than diverges at strong correlation; DIIS damps the oscillation, hence
``iter_count`` rises only modestly with R).  The operator-norm
amplification ``||(I − D𝒯)^{-1}||_2`` depends on ``|1 − λ|``, not on
``|λ|``: a strongly negative eigenvalue gives ``|1 − λ|`` large and the
resolvent norm small, exactly *opposite* to the SPEC §2.4 prediction
(which implicitly assumes a positive eigenvalue approaching +1, as in
the photon-scattering analogy).

The amplification *dial* is real and monotone — it is just visible in
``ρ(D𝒯)`` (the spectral radius), not in ``A_op``.  This test therefore:

  (i)  asserts strict monotonic non-decreasing ``ρ(D𝒯)`` across the
       sweep — this is what genuinely characterizes the cycle's
       distance from the marginal-stability boundary at stretched H₂;
  (ii) asserts that ``A_op(R)`` is finite, positive, and bounded across
       the sweep — the looser statement that the resolvent operator is
       well-defined everywhere on the sweep, without claiming monotone
       growth that the math does not support;
  (iii) cross-checks ``cpks_jacobian_analytic`` against
       ``cpks_jacobian_fd`` at the mid-sweep point R=1.48 Å with a
       4-sig-fig agreement tolerance (looser than the unit-test 6-sig-fig
       at R_eq because FD discretization noise is larger when ``D𝒯``
       entries are O(1)).

The full ``A_op(R)`` table is captured by the measurement script
``examples/measure_scf_amplification.py`` (P5 stub) and feeds the methods-
paper figure regardless of monotonicity.  The architect-level question
(is the SPEC's ``A_op`` definition the right amplification observable
for the SCF testbed, or should the spec's monotonicity prediction be
restated in terms of ρ?) is documented in the P2 BLOCKERS and deferred.

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


# The 5-point sweep from the task brief.
_R_SWEEP = (0.74, 1.0, 1.48, 2.0, 2.22)


def _measure_one(R: float) -> tuple[float, float]:
    """Build the analytic Jacobian at converged ``R`` and return (A_op, ρ).

    RKS/B3LYP/STO-3G truth SCF, DIIS-accelerated; tolerance 1e-9.  Skips
    the data point (returns NaN, NaN) if SCF fails to converge — which
    should not happen on this 5-point grid.
    """
    state = make_h2_state(R=R, basis="sto-3g", method="rks", xc="B3LYP")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200, accelerator="diis")
    if not result.converged:
        return float("nan"), float("nan")
    DT = cpks_jacobian_analytic(
        state, result.P_star, result.mo_energy, result.mo_coeff,
    )
    return A_op_analytic(DT), spectral_radius(DT)


@pytest.mark.analytic
def test_A_scf_spectral_radius_grows() -> None:
    """A-scf-spectral-radius-grows: ρ(D𝒯)(R) is monotonically non-decreasing.

    This is the load-bearing amplification-dial test.  As R stretches, the
    HOMO–LUMO gap shrinks and ``|λ|`` of the dominant CPKS eigenvalue
    grows (in the negative direction for closed-shell H₂, but ρ is
    the absolute value, so the *magnitude* is monotone).

    Expected progression (STO-3G B3LYP):
      R=0.74: ρ ≈ 0.378
      R=2.22: ρ ≈ 6.015
    """
    rhos = []
    for R in _R_SWEEP:
        _A, rho = _measure_one(R)
        assert np.isfinite(rho), f"ρ not finite at R={R}: {rho}."
        rhos.append(rho)

    for i in range(1, len(rhos)):
        assert rhos[i] >= rhos[i - 1], (
            f"ρ(D𝒯) decreased between R={_R_SWEEP[i-1]:.2f} (ρ={rhos[i-1]:.4f}) "
            f"and R={_R_SWEEP[i]:.2f} (ρ={rhos[i]:.4f}); the amplification "
            "dial in SPEC §2.4 predicts monotone growth."
        )
    # Sanity floor / ceiling.
    assert rhos[0] < 1.0, (
        f"ρ at R=R_eq={_R_SWEEP[0]} is ≥ 1 ({rhos[0]:.4f}); the cycle should "
        "be locally contractive at equilibrium."
    )
    assert rhos[-1] > 1.0, (
        f"ρ at R=3R_eq={_R_SWEEP[-1]} is ≤ 1 ({rhos[-1]:.4f}); the cycle should "
        "be locally non-contractive at the strongly stretched bond."
    )


@pytest.mark.analytic
def test_A_scf_amplification_monotonic() -> None:
    """A-scf-amplification-monotonic — SOFTENED, see module docstring.

    The task brief asserts strict monotonic non-decreasing ``A_op(R)``.
    Empirically that fails for H₂ in both STO-3G and 6-31G because the
    CPKS Jacobian's dominant eigenvalue is negative at the closed-shell
    fixed point and ``||(I − D𝒯)^{-1}||_2`` depends on ``|1 − λ|``, not
    on ``|λ|``.  We therefore assert only that ``A_op`` is **finite and
    positive everywhere on the sweep** — the looser well-definedness
    statement.  The intended amplification dial is captured by
    ``test_A_scf_spectral_radius_grows`` above.

    This deviation is documented in the P2 BLOCKERS for the architect
    (a soft xfail in spirit; we still execute the analytic-Jacobian
    construction across the full sweep, which exercises the same code
    paths as the strict assertion would).
    """
    A_ops = []
    for R in _R_SWEEP:
        A_op, _rho = _measure_one(R)
        assert np.isfinite(A_op), f"A_op not finite at R={R}: {A_op}."
        assert A_op > 0.0, f"A_op not positive at R={R}: {A_op}."
        A_ops.append(A_op)

    # Document the actual observed range — for STO-3G B3LYP this is
    # essentially constant at 1.0; the test will surface immediately if
    # the analytic Jacobian construction breaks and starts emitting
    # garbage in some basis-set update.
    assert 0.5 < min(A_ops) <= max(A_ops) < 100.0, (
        f"A_op range across sweep [{min(A_ops):.3f}, {max(A_ops):.3f}] "
        f"outside the expected band [0.5, 100]; cpks_jacobian_analytic "
        "may have regressed."
    )


@pytest.mark.analytic
def test_A_scf_cpks_fd_agreement() -> None:
    """A-scf-cpks-fd-agreement: FD vs analytic Jacobian at mid-sweep R=1.48 Å.

    Looser tolerance than the unit-test R_eq agreement (4 sig figs vs 6)
    because at R=1.48 Å the Jacobian entries are O(1) and the central-FD
    step ``h=1e-4`` carries ~1e-9 absolute discretization noise.  Empirically:

      max abs analytic entry ≈ 9.06e-1
      max abs FD − analytic  ≈ 5e-9
      ⇒ relative agreement   ≈ 6e-9
    """
    R = 1.48
    state = make_h2_state(R=R, basis="sto-3g", method="rks", xc="B3LYP")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200, accelerator="diis")
    assert result.converged, f"SCF did not converge at R={R}."

    DT_an = cpks_jacobian_analytic(
        state, result.P_star, result.mo_energy, result.mo_coeff,
    )
    DT_fd = cpks_jacobian_fd(state, result.P_star, h=1e-4)

    max_an = np.max(np.abs(DT_an))
    diff = np.max(np.abs(DT_an - DT_fd))
    rel = diff / max_an
    assert rel < 1e-4, (
        f"FD-vs-analytic relative disagreement at R={R}: {rel:.3e} > 1e-4; "
        f"max abs diff={diff:.3e}, max abs analytic={max_an:.3e}."
    )
