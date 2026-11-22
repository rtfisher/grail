"""Tests for the atmosphere/ package. Test IDs match docs/testing.md."""

from __future__ import annotations

import pytest
import torch

from atmosphere import AngleQuadrature, DepthGrid, FrequencyGrid


@pytest.mark.unit
def test_imports() -> None:
    """M0 acceptance: package imports resolve and public surface is exposed."""
    from atmosphere import (  # noqa: F401
        AngleQuadrature,
        Atmosphere,
        DepthGrid,
        FrequencyGrid,
        ali_loop,
        cycle_emulator,
        cycle_truth,
        formal_solve,
        rate_equation_solve,
    )


# --- Tier U: grid -------------------------------------------------------------

@pytest.mark.unit
def test_U_grid_01_depth_monotonic_log_spaced() -> None:
    """U-grid-01: depth grid monotonic, log-spaced, respects tau_min/tau_max."""
    tau_min, tau_max, n_tau = 1e-4, 1e4, 80
    g = DepthGrid.log_spaced(tau_min, tau_max, n_tau)

    assert g.tau.shape == (n_tau,)
    assert g.tau.dtype == torch.float64
    assert g.tau[0].item() == pytest.approx(tau_min, rel=1e-12)
    assert g.tau[-1].item() == pytest.approx(tau_max, rel=1e-12)

    diffs = g.tau[1:] - g.tau[:-1]
    assert torch.all(diffs > 0).item(), "depth grid must be strictly monotonic"

    ratios = g.tau[1:] / g.tau[:-1]
    assert torch.allclose(ratios, ratios[0].expand_as(ratios), rtol=1e-12), (
        "log-spaced grid must have constant tau[i+1]/tau[i] ratio"
    )


@pytest.mark.unit
def test_U_grid_02_doppler_profile_normalized() -> None:
    """U-grid-02: frequency grid symmetric around line center; integration of
    the Doppler profile against the grid weights returns 1 to 1e-10.
    """
    g = FrequencyGrid.doppler(n_nu=21, half_width=5.0)

    assert g.nu.shape == (21,)
    assert g.weights.shape == (21,)
    assert g.profile.shape == (21,)

    assert torch.allclose(
        g.nu + g.nu.flip(0), torch.zeros_like(g.nu), atol=1e-14
    ), "frequency grid must be symmetric around line center"

    integral = (g.weights * g.profile).sum().item()
    assert abs(integral - 1.0) < 1e-10, (
        f"int phi(x) dx via grid weights should be 1, got {integral} "
        f"(error {integral - 1.0:.2e})"
    )


@pytest.mark.unit
def test_U_grid_03_gauss_legendre_exact() -> None:
    """U-grid-03: Gauss-Legendre integrates int_0^1 mu^k dmu for k in {0,1,2,3}
    exactly to machine precision.

    n_mu-point Gauss-Legendre is exact for polynomials of degree <= 2*n_mu - 1;
    n_mu=5 covers up to degree 9, well past k=3.
    """
    g = AngleQuadrature.gauss_legendre(n_mu=5)
    expected = [1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0]
    for k, exp_val in enumerate(expected):
        integral = (g.weights * g.mu**k).sum().item()
        assert abs(integral - exp_val) < 1e-14, (
            f"k={k}: int_0^1 mu^k dmu should be {exp_val}, got {integral}"
        )


# Note: Tier U/A tests for formal_solver, rate_equations, and fixed_point
# live in their per-module test files (test_formal_solver.py,
# test_rate_equations.py, test_fixed_point.py).  This file owns only the
# U-grid tests above.


# Note: Tier R regression tests (R-truth-fixedpoint, R-formal-solver,
# R-resolvent-output) are active in tests/regression/test_regression_twolevel.py
# against committed HDF5 reference files.  The M0 stubs that formerly lived
# here were removed after M1d when the real tests were promoted.
