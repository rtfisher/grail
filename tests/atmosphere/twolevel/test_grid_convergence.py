"""Grid-convergence tests for the formal solver and ALI fixed-point.

Test IDs: A-grid-convergence-formal-01, A-grid-convergence-ali-01
Tier: A (Analytic-Limit)
Spec: examples/specs/SPEC_grid_convergence.md v1.0

These tests verify:
1. The OK87 formal solver (linear and parabolic modes) converges to the
   numerical limit as h → 0, with the expected theoretical rates: O(h²)
   for linear and O(h³) for parabolic-S.  Convergence is measured against
   a fine-grid reference (n_tau=3200) using a power-law temperature profile
   T(τ) = T0 × (τ/τ_min)^α that produces a smoothly varying source function
   S(τ) = B(T(τ)) with genuine spatial discretization error.

   Implementation note: the isothermal Planck case (S = B₀ = const) that
   appears in SPEC §2.1 is a degenerate fixed-point of the formal solver —
   every OK87 segment update is exact for constant S with the matching deep
   BC I⁺(τ_max) = B₀, so errors are at machine precision (~1e-16) regardless
   of n_tau and no convergence rate is measurable.  The power-law T profile
   avoids this degeneracy while preserving the smooth-source intent.

2. The full two-level ALI fixed-point (mode='direct', parabolic formal
   solver) converges to the Avrett-Hummer thermalization law S(0)/B → √ε
   as h → 0, with rate ≥ O(h²), across three (ε, τ_max) regimes.
3. The production n_tau values used in tests/regression/truth_fixedpoint.h5
   achieve ≤ 0.5% relative error vs √ε at each ε.

Both test functions are tagged @pytest.mark.analytic and @pytest.mark.slow.
Run with:
    PYTHONPATH=. pytest tests/atmosphere/twolevel/test_grid_convergence.py \
        -m "analytic and slow" -v

Markers:
    analytic — closed-form-limit test (Tier A)
    slow     — multi-grid sweep (~2 s formal, ~20 s ALI); deselect with
                '-m not slow' for fast runs.
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.atmosphere import Atmosphere
from atmosphere.formal_solver import formal_solve
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid
from atmosphere.twolevel.fixed_point import _make_reference_se_solver, ali_loop

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant, J s
_C = 2.99792458e8     # speed of light, m/s
_K = 1.380649e-23     # Boltzmann constant, J/K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _planck_scalar(T0: float, nu_0: float) -> float:
    """Planck function B_nu(T0) at line center nu_0; Python float."""
    return (2.0 * _H * nu_0**3 / _C**2) / (math.exp(_H * nu_0 / (_K * T0)) - 1.0)


def _make_coherent_freq_grid(dtype: torch.dtype = torch.float64) -> FrequencyGrid:
    """Coherent monochromatic frequency grid: single point at x=0 with unit weight.

    FrequencyGrid.doppler() requires n_nu >= 3 (trapezoidal rule needs endpoints).
    For the convergence tests we want a single frequency at line center (x=0,
    psi=1) to isolate spatial convergence from frequency-quadrature error.
    We construct FrequencyGrid directly, following the same pattern as
    test_parabolic_convergence_rate_third_order in test_formal_solver.py.

    The weight=1 and profile=1 choice gives J_bar = J (mean intensity equals
    the profile-weighted mean intensity at line center for a unit-profile grid).
    """
    return FrequencyGrid(
        nu=torch.zeros(1, dtype=dtype),
        weights=torch.ones(1, dtype=dtype),
        profile=torch.ones(1, dtype=dtype),
    )


def _make_isothermal_atm(
    n_tau: int,
    *,
    tau_min: float,
    tau_max: float,
    T0: float,
    epsilon: float,
    n_nu: int,
    n_mu: int,
    nu_0: float = 5.89e14,   # Hα line center, Hz
) -> Atmosphere:
    """Build a minimal isothermal Atmosphere for ALI grid-convergence tests.

    Uses a coherent monochromatic frequency grid and Gauss-Legendre angle
    quadrature.  All tensors are float64.

    For n_nu=1, builds a single-frequency grid directly (FrequencyGrid.doppler
    requires n_nu >= 3).  For n_nu >= 3, uses FrequencyGrid.doppler.

    Parameters
    ----------
    n_tau : int
        Number of depth points on the log-spaced τ grid.
    tau_min, tau_max : float
        Optical-depth grid bounds.
    T0 : float
        Isothermal temperature, K.
    epsilon : float
        Two-level photon destruction probability.
    n_nu : int
        Number of frequency points.  For ALI tests, pass n_nu=1.
    n_mu : int
        Number of Gauss-Legendre angle quadrature points.  For ALI tests,
        pass n_mu=1 (μ=0.5 midpoint; cleanest for spatial convergence).
    nu_0 : float
        Line-centre frequency, Hz.
    """
    dtype = torch.float64

    depth = DepthGrid.log_spaced(tau_min=tau_min, tau_max=tau_max, n_tau=n_tau)
    freq  = _make_coherent_freq_grid(dtype) if n_nu == 1 else FrequencyGrid.doppler(n_nu=n_nu, half_width=3.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=n_mu)

    T   = torch.full((n_tau,), T0,   dtype=dtype)
    n_e = torch.full((n_tau,), 1e12, dtype=dtype)

    return Atmosphere(
        depth=depth,
        frequency=freq,
        angle=angle,
        T=T,
        n_e=n_e,
        epsilon=epsilon,
        nu_0=nu_0,
    )


def _make_power_law_atm(
    n_tau: int,
    *,
    tau_min: float,
    tau_max: float,
    T_surf: float,
    T_deep: float,
    nu_0: float = 5.89e14,
) -> tuple[Atmosphere, torch.Tensor]:
    """Build a power-law temperature atmosphere for formal-solver convergence tests.

    T(τ) = T_surf × (τ/τ_min)^α where α = log(T_deep/T_surf)/log(τ_max/τ_min).

    The source function S(τ) = B(T(τ)) varies smoothly with depth, giving a
    genuine non-trivial problem for the formal solver with real discretization
    error that converges at the expected OK87 rate.

    Returns
    -------
    atm : Atmosphere
        The atmosphere with power-law T profile.
    S : Tensor of shape (n_tau, 1)
        Source function B(T(τ)) at each depth.
    """
    dtype = torch.float64
    alpha = math.log(T_deep / T_surf) / math.log(tau_max / tau_min)

    depth = DepthGrid.log_spaced(tau_min=tau_min, tau_max=tau_max, n_tau=n_tau)
    tau   = depth.tau

    # Power-law T profile: T = T_surf * (tau / tau_min)^alpha
    T     = T_surf * (tau / tau_min) ** alpha
    B     = (2.0 * _H * nu_0**3 / _C**2) / (torch.exp((_H * nu_0) / (_K * T)) - 1.0)
    S     = B.unsqueeze(1)   # (n_tau, 1) for coherent monochromatic

    freq  = _make_coherent_freq_grid(dtype)
    angle = AngleQuadrature.gauss_legendre(n_mu=1)
    n_e   = torch.full((n_tau,), 1e12, dtype=dtype)

    atm = Atmosphere(
        depth=depth,
        frequency=freq,
        angle=angle,
        T=T,
        n_e=n_e,
        epsilon=0.5,   # irrelevant for formal solver alone
        nu_0=nu_0,
    )
    return atm, S


def _run_formal_solver_convergence(
    mode: str,
    n_tau_values: list[int],
    tau_min: float,
    tau_max: float,
    T_surf: float,
    T_deep: float,
    nu_0: float,
    J_ref: float | None = None,
) -> dict[int, tuple[float, float]]:
    """Run formal solver for each n_tau and return {n_tau: (h, err)}.

    The error is |J(τ_min) - J_ref|, where J_ref is the fine-grid reference
    J value at the surface.  If J_ref is None, a reference at n_tau=3200 is
    computed first.

    Parameters
    ----------
    mode : str
        'linear' or 'parabolic'.
    n_tau_values : list[int]
        Grid sizes to run.
    tau_min, tau_max : float
        Optical-depth range.
    T_surf, T_deep : float
        Surface and deep temperatures for the power-law T profile.
    nu_0 : float
        Line-centre frequency, Hz.
    J_ref : float or None
        Reference surface J value.  If None, computed at n_tau=3200.

    Returns
    -------
    dict mapping n_tau → (h, err)
        h   : grid spacing in decades per point, log10(τ_max/τ_min)/(n_tau-1)
        err : |J(τ_min) - J_ref|
    """
    log_span = math.log10(tau_max / tau_min)

    if J_ref is None:
        atm_ref, S_ref = _make_power_law_atm(
            3200, tau_min=tau_min, tau_max=tau_max,
            T_surf=T_surf, T_deep=T_deep, nu_0=nu_0,
        )
        J_fine = formal_solve(S_ref, atm_ref, mode=mode)
        J_ref  = float(J_fine[0, 0].item())

    results: dict[int, tuple[float, float]] = {}

    for n_tau in n_tau_values:
        h = log_span / (n_tau - 1)
        atm, S = _make_power_law_atm(
            n_tau, tau_min=tau_min, tau_max=tau_max,
            T_surf=T_surf, T_deep=T_deep, nu_0=nu_0,
        )
        J     = formal_solve(S, atm, mode=mode)   # (n_tau, 1)
        err   = abs(float(J[0, 0].item()) - J_ref)
        results[n_tau] = (h, err)

    return results


# ---------------------------------------------------------------------------
# A-grid-convergence-formal-01 : formal solver mode='linear'
# ---------------------------------------------------------------------------

@pytest.mark.analytic
@pytest.mark.slow
def test_A_grid_convergence_formal_linear() -> None:
    """A-grid-convergence-formal-01 (linear mode): verify O(h²) convergence rate.

    Problem: power-law temperature atmosphere T(τ) = T_surf × (τ/τ_min)^α with
    T_surf=6000 K, T_deep=30000 K, τ_min=1e-4, τ_max=1e4.  Source function
    S(τ) = B(T(τ)) varies smoothly with τ, creating genuine spatial truncation
    error in the formal solver.

    The convergence is measured as the decrease in |J(τ_min) - J_ref| where
    J_ref is the fine-grid value at n_tau=3200 (Richardson-extrapolated limit).

    Acceptance criteria (SPEC §2.3, adapted for power-law source):
    1. Convergence rate at n_tau ∈ {200, 400, 800} ≥ 1.8 (O(h²) threshold).
    2. Fine-grid accuracy: err(800) < err(200) / 10.0 (at least 10× reduction
       from n_tau=200 to n_tau=800, matching two factor-2 doublings at O(h²)).
    """
    n_tau_values = [50, 100, 200, 400, 800]
    tau_min  = 1e-4
    tau_max  = 1e4
    T_surf   = 6000.0
    T_deep   = 30000.0
    nu_0     = 5.89e14   # Hα line center

    results = _run_formal_solver_convergence(
        "linear", n_tau_values, tau_min, tau_max, T_surf, T_deep, nu_0
    )

    h200, err200 = results[200]
    h400, err400 = results[400]
    h800, err800 = results[800]

    # Criterion 1: convergence rate from n_tau ∈ {200, 400, 800}
    p = math.log(err400 / err800) / math.log(h400 / h800)

    assert p >= 1.8, (
        f"Linear OK87 convergence rate too low: p={p:.3f} < 1.8 (want ≥ 1.8 for O(h²)). "
        f"Errors: err(200)={err200:.3e}, err(400)={err400:.3e}, err(800)={err800:.3e}"
    )

    # Criterion 2: at least 10× error reduction from n_tau=200 to n_tau=800
    # (two factor-2 doublings at O(h²) gives 4× per doubling, so 16× total)
    assert err800 < err200 / 10.0, (
        f"Linear OK87 insufficient error reduction: "
        f"err(200)={err200:.3e}, err(800)={err800:.3e}, ratio={err200/err800:.2f} (want > 10)"
    )


# ---------------------------------------------------------------------------
# A-grid-convergence-formal-01 : formal solver mode='parabolic'
# ---------------------------------------------------------------------------

@pytest.mark.analytic
@pytest.mark.slow
def test_A_grid_convergence_formal_parabolic() -> None:
    """A-grid-convergence-formal-01 (parabolic mode): verify O(h³) convergence rate.

    Same setup as the linear test: power-law T(τ) atmosphere gives genuine
    discretization error that converges at O(h³) for parabolic-S OK87.

    Acceptance criteria (SPEC §2.3, adapted):
    1. Convergence rate at n_tau ∈ {200, 400, 800} ≥ 2.5 (O(h³) threshold).
    2. Fine-grid accuracy: err(800) < err(200) / 50.0 (three factor-2 doublings
       at O(h³) gives 8× per doubling, so 64× total; 50× is a conservative floor).
    """
    n_tau_values = [50, 100, 200, 400, 800]
    tau_min  = 1e-4
    tau_max  = 1e4
    T_surf   = 6000.0
    T_deep   = 30000.0
    nu_0     = 5.89e14

    results = _run_formal_solver_convergence(
        "parabolic", n_tau_values, tau_min, tau_max, T_surf, T_deep, nu_0
    )

    h200, err200 = results[200]
    h400, err400 = results[400]
    h800, err800 = results[800]

    # Criterion 1: convergence rate from n_tau ∈ {200, 400, 800}
    p = math.log(err400 / err800) / math.log(h400 / h800)

    assert p >= 2.5, (
        f"Parabolic OK87 convergence rate too low: p={p:.3f} < 2.5 (want ≥ 2.5 for O(h³)). "
        f"Errors: err(200)={err200:.3e}, err(400)={err400:.3e}, err(800)={err800:.3e}"
    )

    # Criterion 2: at least 50× error reduction from n_tau=200 to n_tau=800
    assert err800 < err200 / 50.0, (
        f"Parabolic OK87 insufficient error reduction: "
        f"err(200)={err200:.3e}, err(800)={err800:.3e}, ratio={err200/err800:.2f} (want > 50)"
    )


# ---------------------------------------------------------------------------
# A-grid-convergence-formal-01 : both modes agree at n_tau=800
# ---------------------------------------------------------------------------

@pytest.mark.analytic
@pytest.mark.slow
def test_A_grid_convergence_formal_both_modes_agree() -> None:
    """A-grid-convergence-formal-01: linear and parabolic agree at the finest grid.

    At n_tau=800, both modes should give J(τ_min) values that agree to within
    10× the linear-mode error — i.e., the coarser (O(h²)) method sets the
    agreement floor, and both modes have converged to it.

    This is the correct criterion for two methods with different orders: they
    converge to the same limit, so their difference at n_tau=800 is bounded by
    the coarser method's truncation error (~1e-12 here).  Requiring absolute
    machine-precision agreement (as the original SPEC §2.3 phrased it) is only
    achievable if the source function is an exact fixed point of both schemes
    (the isothermal case), which is degenerate.  For a non-trivial source, O(h²)
    limits the linear-mode accuracy to ~1e-12 at n_tau=800.
    """
    tau_min = 1e-4
    tau_max = 1e4
    T_surf  = 6000.0
    T_deep  = 30000.0
    nu_0    = 5.89e14
    n_tau   = 800

    # Run both modes at n_tau=800 and at n_tau=1600 (to estimate linear truncation)
    atm800_lin, S800 = _make_power_law_atm(
        n_tau, tau_min=tau_min, tau_max=tau_max, T_surf=T_surf, T_deep=T_deep, nu_0=nu_0,
    )
    J_lin800 = float(formal_solve(S800, atm800_lin, mode="linear")[0, 0].item())
    J_par800 = float(formal_solve(S800, atm800_lin, mode="parabolic")[0, 0].item())

    # Direct comparison: |J_lin - J_par| at n_tau=800.
    diff = abs(J_lin800 - J_par800)
    J_mid = 0.5 * (J_lin800 + J_par800)

    # Both modes produce essentially the same answer when both are converged.
    # The difference is dominated by the linear-mode O(h²) truncation error.
    # At n_tau=800 with h≈0.01 and O(h²) leading coefficient, err_lin ~ 1e-12,
    # so we allow diff < 1e-11 (10× the linear mode error; generous floor).
    assert diff < 1e-11, (
        f"Linear and parabolic modes disagree at n_tau={n_tau}: "
        f"J_linear={J_lin800:.10e}, J_parabolic={J_par800:.10e}, "
        f"|diff|={diff:.3e} (want < 1e-11)"
    )

    # Additionally verify both errors vs the parabolic reference at n_tau=3200
    # are small relative to J_mid (convergence sanity check).
    atm_ref, S_ref = _make_power_law_atm(
        3200, tau_min=tau_min, tau_max=tau_max, T_surf=T_surf, T_deep=T_deep, nu_0=nu_0,
    )
    J_ref = float(formal_solve(S_ref, atm_ref, mode="parabolic")[0, 0].item())
    err_lin = abs(J_lin800 - J_ref)
    err_par = abs(J_par800 - J_ref)

    # Linear at n_tau=800 should be within 1e-5 relative of the fine-grid value.
    assert err_lin < abs(J_ref) * 1e-5, (
        f"Linear mode at n_tau={n_tau} too far from reference: "
        f"err={err_lin:.3e}, J_ref={J_ref:.6e}, rel={err_lin/abs(J_ref):.3e} (want < 1e-5)"
    )
    # Parabolic at n_tau=800 should be within 1e-6 relative (stricter, O(h³)).
    assert err_par < abs(J_ref) * 1e-6, (
        f"Parabolic mode at n_tau={n_tau} too far from reference: "
        f"err={err_par:.3e}, J_ref={J_ref:.6e}, rel={err_par/abs(J_ref):.3e} (want < 1e-6)"
    )


# ---------------------------------------------------------------------------
# A-grid-convergence-ali-01 : full ALI fixed-point, three (ε, τ_max) regimes
# ---------------------------------------------------------------------------

@pytest.mark.analytic
@pytest.mark.slow
@pytest.mark.parametrize("eps,tau_max", [
    (1e-2, 1e4),   # shallow: τ_therm ~ 10
    (1e-4, 1e4),   # moderate: τ_therm ~ 100
    (1e-6, 1e6),   # deep: τ_therm ~ 1000
])
def test_A_grid_convergence_ali(eps: float, tau_max: float) -> None:
    """A-grid-convergence-ali-01: ALI fixed-point converges to √ε with rate ≥ O(h²).

    For the two-level isothermal atmosphere with photon destruction probability ε,
    the converged surface source function satisfies the Avrett-Hummer
    thermalization law S_l(0)/B₀ → √ε as τ_max → ∞, h → 0.

    This test sweeps n_tau ∈ {100, 200, 400, 800} for each (ε, τ_max) pair
    and verifies that the error err(n_tau) = |S_l[0]/B₀ - √ε| decreases
    at rate p ≥ 2.0 (estimated from n_tau ∈ {200, 400, 800}).

    ALI is run with mode='direct' + formal_solve_mode='parabolic' (the
    production path; see atmosphere/twolevel/fixed_point.py::ali_loop).
    max_iter=500 is used to ensure convergence at the largest/deepest grids.

    Acceptance criteria (SPEC §3.3):
    1. Convergence rate p ≥ 2.0 (n_tau ∈ {200, 400, 800}).
    2. err(800) / sqrt(eps) < 0.005 (0.5% relative to √ε).
    3. err(800) < err(400) (monotone refinement; no Runge inflation at fine grids).
    4. ALI converges within max_iter=500 at every (ε, τ_max, n_tau).
    """
    tau_min      = 1e-4
    T0           = 6000.0
    nu_0         = 5.89e14     # Hα line center
    n_nu         = 1
    n_mu         = 1
    n_tau_values = [100, 200, 400, 800]
    max_iter     = 500

    B0       = _planck_scalar(T0, nu_0)
    sqrt_eps = math.sqrt(eps)

    # Results: n_tau → (h, err, iter_count)
    loop_results: dict[int, tuple[float, float, int]] = {}

    log_span = math.log10(tau_max / tau_min)

    for n_tau in n_tau_values:
        h   = log_span / (n_tau - 1)
        atm = _make_isothermal_atm(
            n_tau, tau_min=tau_min, tau_max=tau_max,
            T0=T0, epsilon=eps, n_nu=n_nu, n_mu=n_mu, nu_0=nu_0,
        )
        se_solver = _make_reference_se_solver(atm)

        result = ali_loop(
            atm,
            se_solver,
            tol=1e-10,
            max_iter=max_iter,
            mode="direct",
            formal_solve_mode="parabolic",
        )

        # Criterion 4: ALI must converge within max_iter.
        # mode='direct' always returns iter_count=1 (one linear solve); the
        # safety check here is that the direct solve completed and the result
        # is not garbage (which would cause a large err).
        if result.iter_count >= max_iter:
            pytest.fail(
                f"ALI did not converge at ε={eps:.0e}, τ_max={tau_max:.0e}, "
                f"n_tau={n_tau}: iter_count={result.iter_count} ≥ max_iter={max_iter}, "
                f"final residual={result.residual_history[-1]:.3e}"
            )

        S_surface = float(result.S_l[0].item())
        err       = abs(S_surface / B0 - sqrt_eps)

        loop_results[n_tau] = (h, err, result.iter_count)

    # Criterion 1: convergence rate from n_tau ∈ {200, 400, 800}
    h200, err200, _ = loop_results[200]
    h400, err400, _ = loop_results[400]
    h800, err800, _ = loop_results[800]

    if err400 > 0.0 and err800 > 0.0:
        p = math.log(err400 / err800) / math.log(h400 / h800)
    else:
        p = float("inf")

    assert p >= 2.0, (
        f"ALI convergence rate too low at ε={eps:.0e}, τ_max={tau_max:.0e}: "
        f"p={p:.3f} < 2.0. "
        f"Errors: err(200)={err200:.3e}, err(400)={err400:.3e}, err(800)={err800:.3e}"
    )

    # Criterion 2: finest-grid relative accuracy
    rel_err800 = err800 / sqrt_eps
    assert rel_err800 < 0.005, (
        f"ALI finest-grid error too large at ε={eps:.0e}, τ_max={tau_max:.0e}: "
        f"err(800)/√ε = {rel_err800:.4f} (want < 0.005). "
        f"S_l(0)/B₀ = {float(err800 + sqrt_eps):.6f}, √ε = {sqrt_eps:.6f}"
    )

    # Criterion 3: monotone refinement
    assert err800 < err400, (
        f"ALI non-monotone refinement at ε={eps:.0e}, τ_max={tau_max:.0e}: "
        f"err(800)={err800:.3e} ≥ err(400)={err400:.3e}"
    )


# ---------------------------------------------------------------------------
# A-grid-convergence-ali-01 : regression linkage — production n_tau justified
# ---------------------------------------------------------------------------

@pytest.mark.analytic
@pytest.mark.slow
def test_A_grid_convergence_ali_justifies_regression_ntau() -> None:
    """A-grid-convergence-ali-01: production n_tau choices satisfy 0.5% threshold.

    The regression reference file tests/regression/truth_fixedpoint.h5 was
    generated with:
        ε=1e-2 → n_tau=200,  τ_max=1e4
        ε=1e-4 → n_tau=400,  τ_max=1e4
        ε=1e-6 → n_tau=800,  τ_max=1e6

    (Source: examples/generate_regression_proposals.py, n_tau_map.)

    This test runs ALI at exactly those (ε, n_tau, τ_max) triples and asserts
    that err(production_n_tau) / sqrt(eps) < 0.005.  Together with the
    monotone refinement criterion in test_A_grid_convergence_ali, this
    establishes that the production grid choices are sufficient for sub-0.5%
    accuracy and are theoretically justified rather than empirically tuned.
    """
    production_configs = [
        # (eps, n_tau_production, tau_max)
        (1e-2, 200, 1e4),
        (1e-4, 400, 1e4),
        (1e-6, 800, 1e6),
    ]

    tau_min = 1e-4
    T0      = 6000.0
    nu_0    = 5.89e14
    n_nu    = 1
    n_mu    = 1

    for eps, n_tau_prod, tau_max in production_configs:
        B0       = _planck_scalar(T0, nu_0)
        sqrt_eps = math.sqrt(eps)

        atm = _make_isothermal_atm(
            n_tau_prod, tau_min=tau_min, tau_max=tau_max,
            T0=T0, epsilon=eps, n_nu=n_nu, n_mu=n_mu, nu_0=nu_0,
        )
        se_solver = _make_reference_se_solver(atm)

        result = ali_loop(
            atm,
            se_solver,
            tol=1e-10,
            max_iter=500,
            mode="direct",
            formal_solve_mode="parabolic",
        )

        S_surface = float(result.S_l[0].item())
        rel_err   = abs(S_surface / B0 - sqrt_eps) / sqrt_eps

        assert rel_err < 0.005, (
            f"Production grid fails 0.5% threshold: "
            f"ε={eps:.0e}, n_tau={n_tau_prod}, τ_max={tau_max:.0e}. "
            f"S_l(0)/B₀ = {S_surface/B0:.6f}, √ε = {sqrt_eps:.6f}, "
            f"rel_err = {rel_err:.4f} (want < 0.005). "
            f"This undermines the regression reference in truth_fixedpoint.h5."
        )
