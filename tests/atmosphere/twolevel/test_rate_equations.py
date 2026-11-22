"""Unit tests for atmosphere/twolevel/rate_equations.py.

Moved here from tests/atmosphere/test_rate_equations.py as part of
SPEC_threelevel §M3a.  Imports are unchanged because atmosphere/rate_equations.py
re-exports from atmosphere/twolevel/rate_equations.py.

Test IDs match docs/testing.md Tier U (U-rateeq-01, U-rateeq-02) and add two
additional coverage tests (positivity, dtype).
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.rate_equations import _C, _H, _K, rate_equation_solve

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

# H-alpha line center frequency (approximately 4.57e14 Hz)
NU_0_HALPHA = 4.568e14   # Hz

# Temperatures spanning cool star to hot star
TEMPERATURES_K = [3000.0, 5800.0, 10000.0]

# Default statistical weights (hydrogen-like 1s->2p: g_l=2, g_u=6; or generic)
G_LOWER = 1
G_UPPER = 3


# ---------------------------------------------------------------------------
# U-rateeq-01 — LTE limit recovers Boltzmann ratio
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_U_rateeq_01_lte_boltzmann() -> None:
    """U-rateeq-01: epsilon=1 (LTE) returns Boltzmann ratios to absolute 1e-10.

    At epsilon=1 the line source function S_l = B(T) regardless of J_bar,
    and the population ratio must reduce to the Boltzmann factor:

        n_2 / n_1 = (g_upper / g_lower) * exp(-h*nu_0 / (k*T))

    This is a fundamental requirement of any SE solver (HM14 §13.2;
    Hubeny & Mihalas 2014, "Theory of Stellar Atmospheres", §13.2).

    Tested across T in {3000, 5800, 10000} K and H-alpha (nu_0 ~ 4.57e14 Hz).
    J_bar is set to 0 to confirm it has no effect at epsilon=1.

    Cites: HM14 §13.2 (Boltzmann excitation equation in LTE).
    """
    N_tau = 10
    device = torch.device("cpu")

    for T_val in TEMPERATURES_K:
        T = torch.full((N_tau,), T_val, dtype=torch.float64, device=device)
        # J_bar = 0: in LTE it should be irrelevant
        J_bar = torch.zeros(N_tau, dtype=torch.float64, device=device)
        n_e = torch.zeros(N_tau, dtype=torch.float64, device=device)

        pops = rate_equation_solve(
            J_bar,
            T,
            n_e,
            epsilon=1.0,
            nu_0=NU_0_HALPHA,
            g_lower=G_LOWER,
            g_upper=G_UPPER,
            n_tot=1.0,
        )

        assert pops.shape == (N_tau, 2), f"wrong shape at T={T_val}"
        assert pops.dtype == torch.float64, "must be float64"

        n_1 = pops[:, 0]
        n_2 = pops[:, 1]
        ratio_computed = (n_2 / n_1).double()

        # Analytic Boltzmann ratio
        x = _H * NU_0_HALPHA / (_K * T_val)
        ratio_boltzmann = (G_UPPER / G_LOWER) * math.exp(-x)

        max_abs_err = (ratio_computed - ratio_boltzmann).abs().max().item()
        assert max_abs_err < 1e-10, (
            f"T={T_val} K: Boltzmann ratio error {max_abs_err:.3e} exceeds 1e-10. "
            f"computed={ratio_computed[0].item():.15e}, "
            f"expected={ratio_boltzmann:.15e}"
        )

    # Also test with non-zero J_bar to confirm LTE is J_bar-independent
    for T_val in TEMPERATURES_K:
        T = torch.full((N_tau,), T_val, dtype=torch.float64, device=device)
        # Large J_bar that would dominate if epsilon were < 1
        J_bar = torch.full((N_tau,), 1e-5, dtype=torch.float64, device=device)
        n_e = torch.zeros(N_tau, dtype=torch.float64, device=device)

        pops = rate_equation_solve(
            J_bar,
            T,
            n_e,
            epsilon=1.0,
            nu_0=NU_0_HALPHA,
            g_lower=G_LOWER,
            g_upper=G_UPPER,
            n_tot=1.0,
        )

        n_1 = pops[:, 0]
        n_2 = pops[:, 1]
        ratio_computed = n_2 / n_1

        x = _H * NU_0_HALPHA / (_K * T_val)
        ratio_boltzmann = (G_UPPER / G_LOWER) * math.exp(-x)

        max_abs_err = (ratio_computed - ratio_boltzmann).abs().max().item()
        assert max_abs_err < 1e-10, (
            f"T={T_val} K, non-zero J_bar: Boltzmann ratio error {max_abs_err:.3e}"
        )


# ---------------------------------------------------------------------------
# U-rateeq-02 — population conservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_U_rateeq_02_population_conservation() -> None:
    """U-rateeq-02: n_1 + n_2 = n_tot to relative 1e-12 across random inputs.

    Verifies the closure condition is enforced exactly (within float64
    rounding) for arbitrary valid inputs, including variable n_tot.
    """
    torch.manual_seed(42)
    N_tau = 50
    device = torch.device("cpu")

    # Random physically plausible inputs
    T = torch.empty(N_tau, dtype=torch.float64).uniform_(2000.0, 15000.0)
    # J_bar drawn from log-uniform [1e-20, 1e-5] (SI: W/m^2/Hz/sr)
    log_J = torch.empty(N_tau, dtype=torch.float64).uniform_(-20.0, -5.0)
    J_bar = 10.0 ** log_J
    n_e = torch.empty(N_tau, dtype=torch.float64).uniform_(1e8, 1e14)

    for epsilon in [1e-6, 1e-4, 1e-2, 0.5, 1.0]:
        # Scalar n_tot
        pops = rate_equation_solve(
            J_bar,
            T,
            n_e,
            epsilon=epsilon,
            nu_0=NU_0_HALPHA,
            g_lower=G_LOWER,
            g_upper=G_UPPER,
            n_tot=1.0,
        )

        n_sum = pops[:, 0] + pops[:, 1]
        # Relative error: |sum - 1| / 1
        max_rel_err = (n_sum - 1.0).abs().max().item()
        assert max_rel_err < 1e-12, (
            f"epsilon={epsilon}: conservation violated, max rel err = {max_rel_err:.3e}"
        )

    # Tensor n_tot (depth-dependent density)
    n_tot_tensor = torch.empty(N_tau, dtype=torch.float64).uniform_(1e10, 1e15)
    pops = rate_equation_solve(
        J_bar,
        T,
        n_e,
        epsilon=0.01,
        nu_0=NU_0_HALPHA,
        g_lower=G_LOWER,
        g_upper=G_UPPER,
        n_tot=n_tot_tensor,
    )

    n_sum = pops[:, 0] + pops[:, 1]
    rel_err = ((n_sum - n_tot_tensor) / n_tot_tensor).abs().max().item()
    assert rel_err < 1e-12, (
        f"Tensor n_tot: conservation violated, max rel err = {rel_err:.3e}"
    )


# ---------------------------------------------------------------------------
# test_population_positivity — n_1 > 0 and n_2 > 0
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_population_positivity() -> None:
    """Both populations are strictly positive for all valid inputs.

    n_2 = 0 would imply S_l = 0, which requires J_bar = 0 AND B(T) = 0
    (i.e., T = 0).  For any physical J_bar >= 0 and T > 0, both levels
    must be populated.
    """
    torch.manual_seed(7)
    N_tau = 80
    device = torch.device("cpu")

    T = torch.empty(N_tau, dtype=torch.float64).uniform_(500.0, 50000.0)
    J_bar = torch.empty(N_tau, dtype=torch.float64).uniform_(0.0, 1e-3)
    n_e = torch.zeros(N_tau, dtype=torch.float64)

    for epsilon in [1e-8, 1e-4, 0.1, 1.0]:
        pops = rate_equation_solve(
            J_bar,
            T,
            n_e,
            epsilon=epsilon,
            nu_0=NU_0_HALPHA,
            g_lower=G_LOWER,
            g_upper=G_UPPER,
            n_tot=1.0,
        )

        assert (pops[:, 0] > 0.0).all(), (
            f"epsilon={epsilon}: n_1 <= 0 found: min={pops[:, 0].min().item()}"
        )
        assert (pops[:, 1] > 0.0).all(), (
            f"epsilon={epsilon}: n_2 <= 0 found: min={pops[:, 1].min().item()}"
        )

    # Edge case: J_bar = 0 exactly (pure thermal emission drives the excitation)
    J_bar_zero = torch.zeros(N_tau, dtype=torch.float64)
    T_warm = torch.full((N_tau,), 5000.0, dtype=torch.float64)
    pops_zero_J = rate_equation_solve(
        J_bar_zero,
        T_warm,
        n_e,
        epsilon=0.01,
        nu_0=NU_0_HALPHA,
        g_lower=G_LOWER,
        g_upper=G_UPPER,
        n_tot=1.0,
    )
    assert (pops_zero_J[:, 0] > 0.0).all(), "n_1 <= 0 at J_bar=0"
    assert (pops_zero_J[:, 1] > 0.0).all(), "n_2 <= 0 at J_bar=0"


@pytest.mark.unit
def test_tiny_excited_fraction_avoids_subtractive_cancellation():
    """The Wien-tail population retains relative precision below 1e-10."""
    from atmosphere.twolevel.rate_equations import rate_equation_solve
    import math
    T = torch.tensor([3000.0], dtype=torch.float64)
    nu = 2.467381e15
    h, k, c = 6.62607015e-34, 1.380649e-23, 2.99792458e8
    Bstar = 2*h*nu**3/c**2
    B = Bstar/math.expm1(h*nu/(k*3000))
    result = rate_equation_solve(torch.tensor([B],dtype=torch.float64),T,T*0+1e10,
                                 epsilon=1.,nu_0=nu)
    expected = 1/(1+math.exp(h*nu/(k*3000)))
    assert result[0,1].item() == pytest.approx(expected,rel=1e-13,abs=0)
    assert result.sum().item() == 1.0
