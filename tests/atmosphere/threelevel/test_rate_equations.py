"""Unit tests for atmosphere/threelevel/rate_equations.py.

Test IDs: U-rateeq3-01, U-rateeq3-02 per SPEC_threelevel §G and §M3b TASK-008.

References
----------
- Hubeny & Mihalas (2014), §13.2 — Boltzmann excitation in LTE.
- SPEC_threelevel §6.1 — rate_equation_solve_3level algorithm.
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.atomic_model import AtomicModel
from atmosphere.threelevel.rate_equations import _H, _K, rate_equation_solve_3level

# ---------------------------------------------------------------------------
# Shared test atoms
# ---------------------------------------------------------------------------

# Ly-alpha and H-alpha frequencies from AtomicModel.three_level_hydrogen_chained
_NU_LYA = 2.467381e15   # Hz
_NU_HA = 4.569225e14    # Hz

# A standard test temperature
_T_TEST = 6000.0        # K


def _hydrogen_3level(eps_lya: float = 1.0, eps_ha: float = 1.0) -> AtomicModel:
    """Build the toy hydrogen 3-level atomic model."""
    return AtomicModel.three_level_hydrogen_chained(
        epsilon_lya=eps_lya,
        epsilon_ha=eps_ha,
    )


# ---------------------------------------------------------------------------
# U-rateeq3-01 — LTE Boltzmann at epsilon=1
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_rateeq3_01_lte_boltzmann() -> None:
    """U-rateeq3-01: at eps_k=1 for all k, populations match 3-level Boltzmann to 1e-10.

    At eps_k=1 the source functions S_k = B(T, nu_k) regardless of J_bar,
    so the rate matrix forces each transition pair into Boltzmann equilibrium:
        n_u / n_l = (g_u/g_l) * exp(-h*nu_k / (k*T))

    Combined with sum n_l = 1, this gives the 3-level LTE distribution.

    J_bar is set to zero to confirm it has no effect at eps=1.

    Cites: HM14 §13.2 (Boltzmann excitation in LTE).
    """
    N_tau = 20
    device = torch.device("cpu")

    model = _hydrogen_3level(eps_lya=1.0, eps_ha=1.0)
    # Statistical weights all 1 (minimum viable atom convention).
    g0, g1, g2 = model.statistical_weights

    temperatures = [3000.0, _T_TEST, 10000.0]

    for T_val in temperatures:
        T = torch.full((N_tau,), T_val, dtype=torch.float64, device=device)
        # J_bar = 0 for both lines: must be irrelevant at eps=1.
        J_bar_lines = torch.zeros(N_tau, model.K, dtype=torch.float64, device=device)
        n_e = torch.zeros(N_tau, dtype=torch.float64, device=device)

        pops = rate_equation_solve_3level(
            J_bar_lines, T, n_e, atomic_model=model, n_tot=1.0
        )

        assert pops.shape == (N_tau, 3), (
            f"T={T_val}: expected shape ({N_tau}, 3), got {pops.shape}"
        )
        assert pops.dtype == torch.float64, "populations must be float64"

        n0 = pops[:, 0]  # level 0 (ground)
        n1 = pops[:, 1]  # level 1
        n2 = pops[:, 2]  # level 2

        # Boltzmann ratio for Ly-alpha (0->1): n1/n0 = (g1/g0)*exp(-h*nu_lya/(k*T))
        x_lya = _H * _NU_LYA / (_K * T_val)
        ratio_01_boltzmann = (g1 / g0) * math.exp(-x_lya)
        ratio_01_computed = (n1 / n0).mean().item()
        err_01 = abs(ratio_01_computed - ratio_01_boltzmann)
        assert err_01 < 1e-10, (
            f"T={T_val}: n1/n0 Boltzmann error {err_01:.3e} > 1e-10. "
            f"computed={ratio_01_computed:.15e}, expected={ratio_01_boltzmann:.15e}"
        )

        # Boltzmann ratio for H-alpha (1->2): n2/n1 = (g2/g1)*exp(-h*nu_ha/(k*T))
        x_ha = _H * _NU_HA / (_K * T_val)
        ratio_12_boltzmann = (g2 / g1) * math.exp(-x_ha)
        ratio_12_computed = (n2 / n1).mean().item()
        err_12 = abs(ratio_12_computed - ratio_12_boltzmann)
        assert err_12 < 1e-10, (
            f"T={T_val}: n2/n1 Boltzmann error {err_12:.3e} > 1e-10. "
            f"computed={ratio_12_computed:.15e}, expected={ratio_12_boltzmann:.15e}"
        )

    # Repeat with non-zero J_bar to confirm LTE is J_bar-independent.
    T = torch.full((N_tau,), _T_TEST, dtype=torch.float64)
    J_bar_lines = torch.full((N_tau, model.K), 1e-15, dtype=torch.float64)
    n_e = torch.zeros(N_tau, dtype=torch.float64)

    pops = rate_equation_solve_3level(
        J_bar_lines, T, n_e, atomic_model=model, n_tot=1.0
    )
    n0, n1, n2 = pops[:, 0], pops[:, 1], pops[:, 2]
    x_lya = _H * _NU_LYA / (_K * _T_TEST)
    ratio_01_boltzmann = (g1 / g0) * math.exp(-x_lya)
    err = abs((n1 / n0).mean().item() - ratio_01_boltzmann)
    assert err < 1e-10, (
        f"Non-zero J_bar, T={_T_TEST}: n1/n0 Boltzmann error {err:.3e}"
    )


# ---------------------------------------------------------------------------
# U-rateeq3-02 — population conservation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_rateeq3_02_population_conservation() -> None:
    """U-rateeq3-02: sum n_l = 1 over 3 levels to 1e-12 for arbitrary inputs.

    Verifies that the normalisation constraint (last-row replacement) is
    enforced exactly (within float64 rounding) for random physically
    plausible inputs across a range of epsilon values.

    Also tests with depth-varying n_tot.
    """
    torch.manual_seed(42)
    N_tau = 50
    device = torch.device("cpu")

    model = _hydrogen_3level(eps_lya=1e-2, eps_ha=1e-4)

    # Random physically plausible inputs.
    T = torch.empty(N_tau, dtype=torch.float64).uniform_(2000.0, 15000.0)
    # J_bar_lines drawn log-uniform in [1e-30, 1e-10] (SI units).
    log_J = torch.empty(N_tau, model.K, dtype=torch.float64).uniform_(-30.0, -10.0)
    J_bar_lines = 10.0 ** log_J
    n_e = torch.empty(N_tau, dtype=torch.float64).uniform_(1e8, 1e14)

    for eps_lya, eps_ha in [(1e-4, 1e-4), (1e-2, 1e-6), (1.0, 1.0), (0.5, 0.1)]:
        m = _hydrogen_3level(eps_lya=eps_lya, eps_ha=eps_ha)
        pops = rate_equation_solve_3level(
            J_bar_lines, T, n_e, atomic_model=m, n_tot=1.0
        )
        n_sum = pops.sum(dim=-1)  # (N_tau,)
        max_err = (n_sum - 1.0).abs().max().item()
        assert max_err < 1e-12, (
            f"eps=({eps_lya}, {eps_ha}): conservation violated, "
            f"max |sum-1| = {max_err:.3e}"
        )

    # Depth-varying n_tot.
    n_tot_tensor = torch.empty(N_tau, dtype=torch.float64).uniform_(1e10, 1e15)
    pops = rate_equation_solve_3level(
        J_bar_lines, T, n_e, atomic_model=model, n_tot=n_tot_tensor
    )
    n_sum = pops.sum(dim=-1)
    rel_err = ((n_sum - n_tot_tensor) / n_tot_tensor).abs().max().item()
    assert rel_err < 1e-12, (
        f"Tensor n_tot: conservation violated, max rel err = {rel_err:.3e}"
    )


# ---------------------------------------------------------------------------
# Additional: ValueError for wrong n_levels
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rateeq3_wrong_n_levels_raises() -> None:
    """rate_equation_solve_3level raises ValueError for non-3-level model."""
    from atmosphere.atomic_model import AtomicModel

    model_2lev = AtomicModel.two_level_master_epsilon(nu_0=2.467e15, epsilon=0.01)
    N_tau = 5
    J_bar = torch.zeros(N_tau, 1, dtype=torch.float64)
    T = torch.full((N_tau,), 6000.0, dtype=torch.float64)
    n_e = torch.zeros(N_tau, dtype=torch.float64)

    with pytest.raises(ValueError, match="n_levels"):
        rate_equation_solve_3level(J_bar, T, n_e, atomic_model=model_2lev)
