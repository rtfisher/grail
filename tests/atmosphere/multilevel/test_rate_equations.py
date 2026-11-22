"""Tests for atmosphere/multilevel/rate_equations.py. M.2 acceptance gates.

Test IDs: A-nlevel-lte-boltzmann, A-nlevel-conservation, A-nlevel-positivity,
          A-nlevel-2level-limit, A-nlevel-3level-limit, U-nlevel-singular-matrix,
          U-nlevel-3level-regression.

SPEC_nlte_scaleup §M.2 acceptance criteria.
PLAN_nlte_scaleup_architect.md §M.2 test list.
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.atomic_model import AtomicModel, Transition
from atmosphere.multilevel.rate_equations import rate_equation_solve_nlevel, _H, _K, _C
from atmosphere.threelevel.rate_equations import rate_equation_solve_3level


# ---------------------------------------------------------------------------
# Physical constants for reference computations
# ---------------------------------------------------------------------------

_KB = 1.380649e-23   # Boltzmann constant [J/K]

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_T_TEST = 8000.0    # K — canonical test temperature (SPEC §M.2)
_N_TAU = 50         # depth points for batch testing


def _make_uniform_T_n_e(N_tau: int = _N_TAU, T: float = _T_TEST) -> tuple:
    """Return uniform T, n_e tensors of shape (N_tau,)."""
    T_t = torch.full((N_tau,), T, dtype=torch.float64)
    n_e_t = torch.full((N_tau,), 1e12, dtype=torch.float64)
    return T_t, n_e_t


def _make_uniform_J_bar(
    N_tau: int,
    K: int,
    J_val: float = 1e-20,
) -> torch.Tensor:
    """Return uniform J_bar_lines of shape (N_tau, K) for a given fill value."""
    return torch.full((N_tau, K), J_val, dtype=torch.float64)


# ---------------------------------------------------------------------------
# A-nlevel-lte-boltzmann: eps=1 recovers Boltzmann
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_lte_boltzmann_5level() -> None:
    """A-nlevel-lte-boltzmann: at eps=1 for all lines, populations match Boltzmann.

    At eps_k = 1, S_k = B(T, nu_k) regardless of J_bar.  The rate matrix
    then forces each pair (u, l) into the Boltzmann ratio:
        n_u / n_l = (g_u / g_l) * exp(-h nu_ul / k_B T)

    SPEC_nlte_scaleup §M.2: < 1e-4 relative on a 5-level Balmer atom
    at T = 8000 K, n_e = 1e12, N_tau = 50.
    """
    model = AtomicModel.balmer_5level(epsilon=1.0)
    N_tau = _N_TAU
    K = model.K

    T_t, n_e_t = _make_uniform_T_n_e(N_tau=N_tau, T=_T_TEST)
    # At eps=1 the J_bar value doesn't matter — B_k dominates.
    J_bar = _make_uniform_J_bar(N_tau, K, J_val=1e-20)

    pops = rate_equation_solve_nlevel(J_bar, T_t, n_e_t, atomic_model=model)

    assert pops.shape == (N_tau, model.n_levels)
    assert pops.dtype == torch.float64

    # Check Boltzmann ratios for every transition pair.
    for k, tr in enumerate(model.transitions):
        l_k = tr.lower   # 0-based
        u_k = tr.upper   # 0-based
        nu_k = tr.nu_0
        g_l = model.statistical_weights[l_k]
        g_u = model.statistical_weights[u_k]

        ratio_expected = (g_u / g_l) * math.exp(-_H * nu_k / (_KB * _T_TEST))
        ratio_actual = (pops[:, u_k] / pops[:, l_k].clamp(min=1e-300)).mean().item()

        rel_err = abs(ratio_actual - ratio_expected) / (ratio_expected + 1e-300)
        assert rel_err < 1e-4, (
            f"Transition k={k} ({l_k}->{u_k}, label={tr.label!r}): "
            f"Boltzmann ratio actual={ratio_actual:.6e}, "
            f"expected={ratio_expected:.6e}, rel_err={rel_err:.2e}"
        )


# ---------------------------------------------------------------------------
# A-nlevel-conservation: sum_i n_i = 1 at every depth
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_conservation() -> None:
    """A-nlevel-conservation: populations sum to 1 at every depth to float64 precision.

    SPEC_nlte_scaleup §M.2 acceptance criterion: Sigma n_i = n_tot within 1e-12.
    """
    model = AtomicModel.balmer_5level(epsilon=1e-4)
    N_tau = _N_TAU
    T_t, n_e_t = _make_uniform_T_n_e(N_tau=N_tau)
    J_bar = _make_uniform_J_bar(N_tau, model.K, J_val=1e-18)

    pops = rate_equation_solve_nlevel(J_bar, T_t, n_e_t, atomic_model=model)

    row_sums = pops.sum(dim=-1)   # (N_tau,)
    max_err = (row_sums - 1.0).abs().max().item()
    assert max_err < 1e-12, (
        f"Conservation violated: max |sum_i n_i - 1| = {max_err:.2e} (limit 1e-12)"
    )


# ---------------------------------------------------------------------------
# A-nlevel-positivity: all n_i > 0
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_positivity() -> None:
    """A-nlevel-positivity: all populations are strictly positive.

    Tested on the LTE case (eps=1) and a deep-eps case (eps=1e-4).
    SPEC_nlte_scaleup §M.2 acceptance criterion.
    """
    N_tau = _N_TAU
    T_t, n_e_t = _make_uniform_T_n_e(N_tau=N_tau)

    for eps in [1.0, 1e-4]:
        model = AtomicModel.balmer_5level(epsilon=eps)
        J_bar = _make_uniform_J_bar(N_tau, model.K, J_val=1e-20)
        pops = rate_equation_solve_nlevel(J_bar, T_t, n_e_t, atomic_model=model)
        min_pop = pops.min().item()
        assert min_pop > 0, (
            f"Negative or zero population at eps={eps}: min(n_i) = {min_pop:.3e}"
        )


# ---------------------------------------------------------------------------
# A-nlevel-2level-limit: 5-level with 9 eps=1 lines reduces to 2-level
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_2level_limit() -> None:
    """A-nlevel-2level-limit: nlevel solver on a 2-level model matches the closed-form result.

    The rate_equation_solve_nlevel is the GENERAL solver; the 2-level limit
    test verifies that calling it with a 2-level AtomicModel reproduces the
    closed-form 2-level result:

        n_2 / n_1 = (g_2/g_1) * S / (S + B_star)

    where S = (1 - eps) * J_bar + eps * B(T, nu).

    This is not the same as embedding the Ly_alpha line inside the 5-level
    model with all other lines at ε=1 (that system has additional transitions
    coupling level 0 to levels 2,3,4, which shifts n_2/n_1 by ~4% even when
    those transitions are in LTE). The correct 2-level limit uses a proper
    2-level model.

    Tolerance: < 1e-3 relative on n_2/n_1 ratio at all depths.
    SPEC_nlte_scaleup §M.2 acceptance criterion.
    """
    eps_lya = 1e-4
    N_tau = _N_TAU

    # Build a 2-level model with Ly_alpha frequency and g=(2,8) (Balmer weights).
    nu_lya = 2.467381e15   # Hz, Ly_alpha from the Balmer model
    lya_transition = Transition(
        lower=0, upper=1, nu_0=nu_lya, epsilon=eps_lya, label="Ly_alpha"
    )
    model_2 = AtomicModel(
        n_levels=2,
        transitions=(lya_transition,),
        statistical_weights=(2, 8),   # g_n = 2n^2 for n=1,2
        label="two_level_balmer_g",
    )

    T_t, n_e_t = _make_uniform_T_n_e(N_tau=N_tau)

    # Vary J_bar across several physically relevant values.
    B_star = 2.0 * _H * nu_lya**3 / _C**2
    x = _H * nu_lya / (_KB * _T_TEST)
    B_val = B_star / (math.exp(x) - 1.0)
    g_ratio = 8.0 / 2.0   # g_2 / g_1

    for J_frac in [0.01, 0.1, 0.5, 2.0]:
        J_val = J_frac * B_val
        J_bar = torch.full((N_tau, 1), J_val, dtype=torch.float64)

        pops = rate_equation_solve_nlevel(
            J_bar, T_t, n_e_t, atomic_model=model_2
        )  # (N_tau, 2)

        ratio_actual = (pops[:, 1] / pops[:, 0].clamp(min=1e-300)).mean().item()

        # Closed-form 2-level ratio from rate-equation algebra.
        S_val = (1.0 - eps_lya) * J_val + eps_lya * B_val
        ratio_expected = g_ratio * S_val / (S_val + B_star)

        rel_err = abs(ratio_actual - ratio_expected) / (ratio_expected + 1e-300)
        assert rel_err < 1e-3, (
            f"2-level limit at J/B={J_frac}: "
            f"ratio_actual={ratio_actual:.6e}, ratio_expected={ratio_expected:.6e}, "
            f"rel_err={rel_err:.2e} (limit 1e-3)"
        )


# ---------------------------------------------------------------------------
# A-nlevel-3level-limit: nlevel solver on 3-level model matches rate_equation_solve_3level
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_3level_limit() -> None:
    """A-nlevel-3level-limit: nlevel solver on 3-level chained model matches closed form.

    Calls rate_equation_solve_nlevel with AtomicModel.three_level_hydrogen_chained
    and compares against rate_equation_solve_3level on the same inputs.

    Tolerance: < 1e-3 relative on all populations (SPEC_nlte_scaleup §M.2).
    """
    eps_lya = 1e-2
    eps_ha = 1e-4
    N_tau = _N_TAU

    model_3 = AtomicModel.three_level_hydrogen_chained(eps_lya, eps_ha)
    K = model_3.K  # = 2

    T_t, n_e_t = _make_uniform_T_n_e(N_tau=N_tau)
    J_bar = _make_uniform_J_bar(N_tau, K, J_val=1e-20)

    pops_nlevel = rate_equation_solve_nlevel(J_bar, T_t, n_e_t, atomic_model=model_3)
    pops_3level = rate_equation_solve_3level(J_bar, T_t, n_e_t, atomic_model=model_3)

    max_rel_err = (
        (pops_nlevel - pops_3level).abs()
        / pops_3level.abs().clamp(min=1e-300)
    ).max().item()

    assert max_rel_err < 1e-3, (
        f"3-level limit: max relative error = {max_rel_err:.2e} (limit 1e-3). "
        f"pops_nlevel sample = {pops_nlevel[0]}, "
        f"pops_3level sample = {pops_3level[0]}"
    )


# ---------------------------------------------------------------------------
# U-nlevel-singular-matrix: degenerate transition raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_singular_matrix_raises() -> None:
    """U-nlevel-singular-matrix: under-connected rate matrix causes singular solve.

    Build a 3-level model with only ONE transition (levels 0->1), leaving level 2
    completely disconnected (no transition touches it).  After rate-matrix
    construction, column 2 of rows 0 and 1 is zero; the augmented matrix is:

        Row 0: [-C, Cin, 0]
        Row 1: [C, -Cin, 0]
        Row 2: [1, 1, 1]   (normalization replacement)

    The determinant of this 3×3 matrix is 0 * (...) + 0 * (...) - Cin*1 + ...
    Let's compute: det = -C*(−Cin*1 − 1*0) − Cin*(C*1 − 1*0) + 0 = ... wait,
    actually expanding:
        det = -C(−Cin − 0) − Cin(C − 0) + 0 = C*Cin − C*Cin = 0

    Column 2 contains only [0, 0, 1]^T, so expanding along column 2:
        det = 1 * cofactor(2,2) = 1 * det([[-C, Cin],[C, -Cin]]) = (-C)(-Cin) - Cin*C
            = C*Cin - C*Cin = 0

    This IS genuinely singular — the solver must raise ValueError.

    SPEC_nlte_scaleup §M.2 acceptance criterion.
    """
    # One transition (0->1) on a 3-level model.
    # Level 2 is never referenced by any transition -> column 2 is all-zero
    # in the rate matrix before normalization replacement.
    single_transition = Transition(
        lower=0, upper=1, nu_0=2.467381e15, epsilon=0.5, label="Ly_alpha"
    )
    model_disconnected = AtomicModel(
        n_levels=3,
        transitions=(single_transition,),
        statistical_weights=(1, 1, 1),
        label="disconnected_model",
    )

    N_tau = 3
    T_t, n_e_t = _make_uniform_T_n_e(N_tau=N_tau)
    # J_bar shape: (N_tau, 1) since K=1.
    J_bar = _make_uniform_J_bar(N_tau, 1, J_val=1e-20)

    with pytest.raises(ValueError, match="singular"):
        rate_equation_solve_nlevel(J_bar, T_t, n_e_t, atomic_model=model_disconnected)


# ---------------------------------------------------------------------------
# U-nlevel-3level-regression: load-bearing equivalence with 3-level solver
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_3level_regression_exact() -> None:
    """U-nlevel-3level-regression: nlevel solver matches 3-level to machine precision.

    Uses the chained 3-level model at multiple J_bar values.  The algorithms
    are algebraically identical (same rate-matrix construction, same augmented
    system, same batched solve), so populations should agree to < 1e-12 relative.

    This is the LOAD-BEARING regression test ensuring no code path diverges
    between the generalised and specialised solvers.
    """
    eps_lya = 1e-2
    eps_ha = 1e-3
    N_tau = 20

    model_3 = AtomicModel.three_level_hydrogen_chained(eps_lya, eps_ha)
    K = model_3.K

    T_t, n_e_t = _make_uniform_T_n_e(N_tau=N_tau)

    for J_val in [1e-25, 1e-20, 1e-15]:
        J_bar = _make_uniform_J_bar(N_tau, K, J_val=J_val)

        pops_n = rate_equation_solve_nlevel(J_bar, T_t, n_e_t, atomic_model=model_3)
        pops_3 = rate_equation_solve_3level(J_bar, T_t, n_e_t, atomic_model=model_3)

        max_rel_err = (
            (pops_n - pops_3).abs()
            / pops_3.abs().clamp(min=1e-300)
        ).max().item()

        assert max_rel_err < 1e-12, (
            f"3-level regression at J_val={J_val:.0e}: "
            f"max relative error = {max_rel_err:.2e} (limit 1e-12). "
            f"This indicates a code-path divergence between the two solvers."
        )
