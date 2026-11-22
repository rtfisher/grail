"""Tests for atmosphere/multilevel/fixed_point.py. M.3 acceptance gates.

Test IDs: I-cycle-5level-converges, I-cycle-5level-lte-recovery,
          I-picard-5level-deep-eps.

SPEC_nlte_scaleup §M.3 acceptance criteria.
PLAN_nlte_scaleup_architect.md §M.3 test list.
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.atmosphere import Atmosphere
from atmosphere.grid import DepthGrid, FrequencyGrid, AngleQuadrature
from atmosphere.atomic_model import AtomicModel
from atmosphere.multilevel.fixed_point import (
    AliResultMultilevel,
    PicardResultMultilevel,
    ali_loop_nlevel,
    cycle_truth_nlevel,
    picard_iterate_nlevel,
    _make_reference_se_solver_nlevel,
)
from atmosphere.multilevel.rate_equations import _H, _K as _KB, _C


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

def _make_isothermal_5level(
    T0: float = 8000.0,
    n_e0: float = 1e12,
    epsilon: float = 1e-3,
    tau_min: float = 1e-4,
    tau_max: float = 1e4,
    N_tau: int = 80,
    N_nu: int = 11,
    N_mu: int = 3,
) -> Atmosphere:
    """Build a standard isothermal 5-level Balmer atmosphere for testing."""
    depth = DepthGrid.log_spaced(tau_min=tau_min, tau_max=tau_max, n_tau=N_tau)
    frequency = FrequencyGrid.doppler(n_nu=N_nu, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=N_mu)
    return Atmosphere.balmer_isothermal(
        T0=T0,
        n_e0=n_e0,
        epsilon=epsilon,
        depth=depth,
        frequency=frequency,
        angle=angle,
    )


# ---------------------------------------------------------------------------
# I-cycle-5level-converges: ali_loop_nlevel direct mode convergences
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_cycle_5level_converges() -> None:
    """I-cycle-5level-converges: ali_loop_nlevel direct mode converges rapidly.

    The full Balmer transition graph has closed loops. Direct mode must
    solve the coupled population cycle and report its actual defect.
    Checks that populations are valid (positive, sum to 1) and that the
    AliResultMultilevel has the expected shape.

    SPEC_nlte_scaleup §M.3: converges on isothermal atmosphere at
    eps=1e-3 within 1000 iterations; coupled Newton is substantially faster.
    """
    # Use a moderate optical depth and epsilon suitable for Balmer 5-level.
    atm = _make_isothermal_5level(
        T0=8000.0,
        n_e0=1e12,
        epsilon=1e-3,
        tau_max=1e4,
        N_tau=80,
    )

    result = ali_loop_nlevel(atm, tol=1e-6, max_iter=1000, mode="direct")

    assert isinstance(result, AliResultMultilevel)
    assert result.populations.shape == (80, 5)
    assert result.S_lines.shape == (80, 10)
    assert result.J_bar_lines.shape == (80, 10)
    assert result.converged, result.failure_reason
    assert 1 < result.iter_count <= 40
    actual = cycle_truth_nlevel(result.populations, atm)
    assert float(((actual-result.populations)/result.populations).abs().max()) < 1e-6

    # Populations must be positive.
    assert result.populations.min().item() > 0, (
        f"Negative population: min={result.populations.min().item():.3e}"
    )

    # Populations must sum to 1 at every depth.
    row_sums = result.populations.sum(dim=-1)
    max_conservation_err = (row_sums - 1.0).abs().max().item()
    assert max_conservation_err < 1e-10, (
        f"Conservation violated: max |sum n_i - 1| = {max_conservation_err:.2e}"
    )


# ---------------------------------------------------------------------------
# I-cycle-5level-lte-recovery: eps=1 -> Boltzmann populations
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_cycle_5level_lte_recovery() -> None:
    """I-cycle-5level-lte-recovery: at eps=1, converged populations match Boltzmann.

    At eps_k = 1 for all lines, the source function S_k = B(T, nu_k) for
    any J_bar value.  The SE solver then returns Boltzmann populations
    regardless of the radiation field.  The converged populations should
    match n_u/n_l = (g_u/g_l) * exp(-h nu / k T) to < 1e-4 relative.

    SPEC_nlte_scaleup §M.3 acceptance criterion.
    """
    T0 = 8000.0
    atm = _make_isothermal_5level(
        T0=T0,
        n_e0=1e12,
        epsilon=1.0,  # LTE: all lines
        tau_max=1e4,
        N_tau=80,
    )

    result = ali_loop_nlevel(atm, tol=1e-6, max_iter=200, mode="direct")

    pops = result.populations  # (N_tau, 5)
    model = atm.atomic_model

    # Check Boltzmann ratio for each transition.
    for k, tr in enumerate(model.transitions):
        l_k = tr.lower
        u_k = tr.upper
        nu_k = tr.nu_0
        g_l = model.statistical_weights[l_k]
        g_u = model.statistical_weights[u_k]

        ratio_expected = (g_u / g_l) * math.exp(-_H * nu_k / (_KB * T0))
        ratio_actual = (
            pops[:, u_k] / pops[:, l_k].clamp(min=1e-300)
        ).mean().item()

        rel_err = abs(ratio_actual - ratio_expected) / (ratio_expected + 1e-300)
        assert rel_err < 1e-4, (
            f"LTE recovery transition k={k} ({l_k}->{u_k}, {tr.label!r}): "
            f"ratio_actual={ratio_actual:.6e}, ratio_expected={ratio_expected:.6e}, "
            f"rel_err={rel_err:.2e} (limit 1e-4)"
        )


# ---------------------------------------------------------------------------
# I-picard-5level-deep-eps: Picard+Anderson converges at deep eps
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_picard_deep_eps_reports_cap_and_coupled_solver_converges() -> None:
    """A capped Picard run must report failure; coupled Newton closes physics.

    After normalized quadrature and positive-domain repair, the historical
    2000-step deep-epsilon Picard claim does not meet its true residual gate.
    Test the honest failure contract with a short fixed cap and separately
    require the actual positive coupled equilibrium, without raising tolerance.
    """
    from atmosphere.multilevel.fixed_point import solve_equilibrium_nlevel
    atm = _make_isothermal_5level(T0=8000., epsilon=1e-5, tau_max=1e4, N_tau=80)
    se = _make_reference_se_solver_nlevel(atm)
    result = picard_iterate_nlevel(atm, se, tol=1e-7, max_iter=40,
                                   accelerator="anderson", anderson_history=5)
    assert not result.converged
    assert result.failure_reason == "maximum iterations reached"
    assert result.residual_history[-1] > 1e-7
    solved = solve_equilibrium_nlevel(atm, se, tol=1e-9)
    assert solved.converged, solved.failure_reason
    y = cycle_truth_nlevel(solved.populations, atm)
    assert float(((y-solved.populations)/solved.populations).abs().max()) < 1e-9


# ---------------------------------------------------------------------------
# U-multilevel-atm-constructors: Atmosphere.balmer_isothermal/power_law
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_balmer_isothermal_constructor() -> None:
    """U-multilevel-atm-constructors: balmer_isothermal carries correct atomic model.

    PLAN_nlte_scaleup_architect.md §M.3 test list (U-multilevel-atm-constructors).
    """
    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=50)
    frequency = FrequencyGrid.doppler(n_nu=11, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=3)

    atm = Atmosphere.balmer_isothermal(
        T0=8000.0,
        n_e0=1e12,
        epsilon=1e-4,
        depth=depth,
        frequency=frequency,
        angle=angle,
    )

    assert atm.atomic_model is not None
    assert atm.atomic_model.n_levels == 5, (
        f"Expected n_levels=5, got {atm.atomic_model.n_levels}"
    )
    assert atm.atomic_model.K == 10, (
        f"Expected K=10, got {atm.atomic_model.K}"
    )
    assert atm.atomic_model.label == "balmer_5level"
    assert atm.T.shape == (50,)
    assert (atm.T == 8000.0).all()


# ---------------------------------------------------------------------------
# U-multilevel-3level-regression: cycle_truth_nlevel matches cycle_truth_3level
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_nlevel_cycle_3level_regression() -> None:
    """U-multilevel-3level-regression: cycle_truth_nlevel on 3-level model matches cycle_truth_3level.

    LOAD-BEARING: guarantees no regression between the generalised and
    specialised cycle functions.

    PLAN_nlte_scaleup_architect.md §M.3 test list (U-multilevel-3level-regression).
    """
    from atmosphere.threelevel.fixed_point import cycle_truth_3level

    # Build a 3-level atmosphere using the existing 3-level constructor.
    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e3, n_tau=40)
    frequency = FrequencyGrid.doppler(n_nu=11, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=3)

    atm = Atmosphere.three_level_hydrogen_isothermal(
        T0=8000.0,
        n_e0=1e12,
        epsilon_lya=1e-2,
        epsilon_ha=1e-3,
        depth=depth,
        frequency=frequency,
        angle=angle,
    )

    N_tau = depth.tau.shape[0]
    L = atm.atomic_model.n_levels   # = 3

    # Use LTE initial populations as the starting point.
    T = atm.T.to(torch.float64)
    n_e = atm.n_e.to(torch.float64)
    se_ref = _make_reference_se_solver_nlevel(atm)

    # Build J_bar_lte for initial guess.
    from atmosphere.twolevel.fixed_point import _planck_nu
    J_bar_lte_list = []
    for k in range(atm.atomic_model.K):
        tr = atm.atomic_model.transitions[k]
        B_k = _planck_nu(tr.nu_0, T)
        J_bar_lte_list.append(B_k)
    J_bar_lte = torch.stack(J_bar_lte_list, dim=1)
    x0 = se_ref(J_bar_lte, T, n_e).detach()  # (N_tau, 3)

    # Run one cycle with both functions.
    x_nlevel = cycle_truth_nlevel(x0, atm)
    x_3level = cycle_truth_3level(x0, atm)

    max_abs_diff = (x_nlevel - x_3level).abs().max().item()
    assert max_abs_diff < 1e-12, (
        f"cycle_truth_nlevel vs cycle_truth_3level: "
        f"max |diff| = {max_abs_diff:.2e} (limit 1e-12)"
    )
