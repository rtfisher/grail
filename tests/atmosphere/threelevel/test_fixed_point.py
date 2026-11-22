"""Analytic-limit and unit tests for atmosphere/threelevel/fixed_point.py.

Test IDs per SPEC_threelevel §G, §M3b, and §M3c:
  U-source3-01      (TASK-009): LTE check on _source_from_populations_3level
  U-atmforline-01   (TASK-009): _atm_for_line returns valid Atmosphere
  U-cycle3-01       (TASK-010): one cycle from LTE returns LTE
  U-fixedpoint3-01  (TASK-012): eps=1 direct mode matches LTE in one step

  A-thermalization3-01  (TASK-015): per-line √ε_k thermalization depth
  A-thermalization3-02  (TASK-015): per-line S_k(0)/B ≈ √ε_k surface value
  A-lte3-01             (TASK-017): at ε_k=1 for all k, populations = LTE Boltzmann
  A-coherent3-01        (TASK-018): single-freq limit agrees with brute-force reference
  A-interlocking-01     (TASK-019): DT·v has non-zero n_1 response for n_2↔n_3 perturbation

References
----------
- SPEC_threelevel §6.2, §6.3 (cycle and ALI specifications)
- SPEC_threelevel §M3c (analytic-limit test specifications)
- Hubeny & Mihalas (2014), §13.2 (LTE populations), §13.3 (Avrett-Hummer thermalization)
- Avrett & Hummer (1965) (thermalization-depth law)
- Mihalas (1978) §6-3 eq.6-39 (two-stream analytic solution)
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.atomic_model import AtomicModel
from atmosphere.atmosphere import Atmosphere
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

# ---------------------------------------------------------------------------
# Physical constants (for Planck function in tests)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23

# 3-level hydrogen frequencies (from AtomicModel.three_level_hydrogen_chained)
_NU_LYA = 2.467381e15   # Hz
_NU_HA = 4.569225e14    # Hz

_T0 = 6000.0    # isothermal temperature for tests
_N_TAU = 40
_N_NU = 21
_N_MU = 5


def _planck(nu: float, T: float) -> float:
    """Scalar Planck function B_nu(T) for test reference."""
    return (2.0 * _H * nu**3 / _C**2) / (math.exp(_H * nu / (_KB * T)) - 1.0)


def _make_3level_atmosphere(
    T0: float = _T0,
    n_e0: float = 1e10,
    eps_lya: float = 1e-2,
    eps_ha: float = 1e-4,
    n_tau: int = _N_TAU,
    n_nu: int = _N_NU,
    n_mu: int = _N_MU,
    tau_max: float = 1e4,
) -> Atmosphere:
    """Build a 3-level isothermal atmosphere for tests."""
    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=tau_max, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=n_nu, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=n_mu)

    model = AtomicModel.three_level_hydrogen_chained(
        epsilon_lya=eps_lya, epsilon_ha=eps_ha
    )
    n_tau_actual = depth.tau.shape[0]
    T = torch.full((n_tau_actual,), T0, dtype=torch.float64)
    n_e = torch.full((n_tau_actual,), n_e0, dtype=torch.float64)

    # Use first transition's nu_0 and epsilon for the legacy Atmosphere fields.
    return Atmosphere(
        depth=depth,
        frequency=frequency,
        angle=angle,
        T=T,
        n_e=n_e,
        epsilon=model.transitions[0].epsilon,
        nu_0=model.transitions[0].nu_0,
        atomic_model=model,
    )


def _lte_populations_3level(T0: float) -> torch.Tensor:
    """Compute LTE Boltzmann populations (sum=1) for the 3-level hydrogen model.

    Uses: n1/n0 = exp(-h*nu_lya/(k*T)), n2/n1 = exp(-h*nu_ha/(k*T)),
    with statistical weights all 1.
    """
    r01 = math.exp(-_H * _NU_LYA / (_KB * T0))
    r12 = math.exp(-_H * _NU_HA / (_KB * T0))
    # n0 + n1 + n2 = 1, n1 = r01*n0, n2 = r12*r01*n0
    denom = 1.0 + r01 + r01 * r12
    n0 = 1.0 / denom
    n1 = r01 * n0
    n2 = r01 * r12 * n0
    return torch.tensor([n0, n1, n2], dtype=torch.float64)


# ---------------------------------------------------------------------------
# U-source3-01 — LTE populations give S_k = B(T, nu_k)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_source3_01_lte_source_function() -> None:
    """U-source3-01: at LTE populations, _source_from_populations_3level gives S_k = B(T, nu_k).

    At LTE each n_u/n_l = (g_u/g_l)*exp(-h*nu/(k*T)).  The Einstein-relation
    form of the source function then reduces to B(T, nu_k):

        S_k = B_star_k / (n_l * g_u / (n_u * g_l) - 1)
            = B_star_k / (exp(h*nu_k/(k*T)) - 1)
            = B(T, nu_k)

    Tolerance: 1e-10 relative.
    """
    from atmosphere.threelevel.fixed_point import _source_from_populations_3level

    N_tau = 20
    T_val = _T0
    model = AtomicModel.three_level_hydrogen_chained(epsilon_lya=0.01, epsilon_ha=0.01)

    # LTE populations broadcast to all depths.
    pops_lte = _lte_populations_3level(T_val).unsqueeze(0).expand(N_tau, -1)  # (N_tau, 3)

    S_lines = _source_from_populations_3level(pops_lte, model)  # (N_tau, 2)
    assert S_lines.shape == (N_tau, 2), f"Expected ({N_tau}, 2), got {S_lines.shape}"

    for k, tr in enumerate(model.transitions):
        B_ref = _planck(tr.nu_0, T_val)
        S_k = S_lines[:, k]
        rel_err = ((S_k - B_ref).abs() / abs(B_ref)).max().item()
        assert rel_err < 1e-10, (
            f"Line {k} ({tr.label}): |S_k - B|/B = {rel_err:.3e} > 1e-10. "
            f"S_k={S_k[0].item():.6e}, B={B_ref:.6e}"
        )


# ---------------------------------------------------------------------------
# U-atmforline-01 — _atm_for_line returns valid Atmosphere with correct nu_0
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_atmforline_01_atm_for_line() -> None:
    """U-atmforline-01: _atm_for_line returns Atmosphere with nu_0 = line k's frequency.

    Checks:
    - nu_0 matches transition k's frequency.
    - epsilon matches transition k's epsilon.
    - All grid fields are shared by reference (shallow copy).
    - IndexError raised for out-of-range line_index.
    """
    from atmosphere.threelevel.fixed_point import _atm_for_line

    atm = _make_3level_atmosphere()
    model = atm.atomic_model

    for k, tr in enumerate(model.transitions):
        atm_k = _atm_for_line(atm, k)

        assert atm_k.nu_0 == tr.nu_0, (
            f"Line {k}: nu_0={atm_k.nu_0!r}, expected {tr.nu_0!r}"
        )
        assert atm_k.epsilon == tr.epsilon, (
            f"Line {k}: epsilon={atm_k.epsilon!r}, expected {tr.epsilon!r}"
        )
        # Grid fields shared by reference (shallow copy).
        assert atm_k.depth is atm.depth, "depth should be shared reference"
        assert atm_k.frequency is atm.frequency, "frequency should be shared reference"
        assert atm_k.angle is atm.angle, "angle should be shared reference"
        assert atm_k.T is atm.T, "T should be shared reference"
        assert atm_k.n_e is atm.n_e, "n_e should be shared reference"
        # atomic_model preserved.
        assert atm_k.atomic_model is atm.atomic_model

    # IndexError for out-of-range.
    with pytest.raises(IndexError, match="line_index"):
        _atm_for_line(atm, model.K)


# ---------------------------------------------------------------------------
# U-cycle3-01 — one cycle from LTE populations returns LTE populations
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_cycle3_01_cycle_from_lte_returns_lte() -> None:
    """U-cycle3-01: one cycle_truth_3level from LTE returns LTE populations.

    At eps_k = 1 for all k, the rate equations force LTE populations
    regardless of J_bar.  A single cycle step from LTE should therefore
    return the same (or numerically equivalent) LTE populations.

    Tolerance: 1e-6 relative on each population.
    """
    from atmosphere.threelevel.fixed_point import cycle_truth_3level
    from atmosphere.threelevel.rate_equations import rate_equation_solve_3level

    # Build atmosphere with eps=1 for both lines (pure LTE).
    atm = _make_3level_atmosphere(eps_lya=1.0, eps_ha=1.0, n_tau=20, n_nu=5, tau_max=1e2)
    N_tau = atm.depth.tau.shape[0]
    model = atm.atomic_model

    # LTE populations broadcast to all depths.
    pops_lte = _lte_populations_3level(_T0).unsqueeze(0).expand(N_tau, -1).clone()  # (N_tau, 3)

    # One cycle from LTE.
    new_pops = cycle_truth_3level(pops_lte, atm)

    assert new_pops.shape == (N_tau, 3), f"Expected ({N_tau}, 3), got {new_pops.shape}"
    assert torch.isfinite(new_pops).all().item(), "Populations contain non-finite values"

    # At eps=1, LTE is a fixed point: new_pops should match pops_lte.
    # Because the rate equations at eps=1 ignore J_bar, the output only
    # depends on T, so it should equal the analytic LTE values.
    pops_ref = _lte_populations_3level(_T0)   # (3,)
    for level in range(3):
        rel_err = ((new_pops[:, level] - pops_ref[level]).abs() / pops_ref[level].abs()).max().item()
        assert rel_err < 1e-6, (
            f"Level {level}: |n_l_new - n_l_lte|/n_l_lte = {rel_err:.3e} > 1e-6"
        )


# ---------------------------------------------------------------------------
# U-fixedpoint3-01 — eps=1 direct mode matches LTE in one step
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_fixedpoint3_01_eps_one_direct_lte() -> None:
    """U-fixedpoint3-01: at eps=1 for all lines, ali_loop_3level direct mode
    matches LTE populations in one step.

    In direct mode with eps_k=1 the linear system reduces to S_k = B_k
    exactly (any S satisfies the rate equations and the linear solve gives
    B).  The SE solver then returns LTE populations.

    iter_count must equal 1 (one direct solve).
    """
    from atmosphere.threelevel.fixed_point import ali_loop_3level

    atm = _make_3level_atmosphere(eps_lya=1.0, eps_ha=1.0, n_tau=20, n_nu=5, tau_max=1e2)
    N_tau = atm.depth.tau.shape[0]

    result = ali_loop_3level(atm, tol=1e-6, max_iter=200, mode="direct")

    assert result.iter_count == 1, (
        f"Expected iter_count=1 in direct mode at eps=1, got {result.iter_count}"
    )
    assert result.populations.shape == (N_tau, 3)
    assert torch.isfinite(result.populations).all().item()

    # Populations should match LTE to 1e-6 relative.
    pops_ref = _lte_populations_3level(_T0)   # (3,)
    for level in range(3):
        rel_err = (
            (result.populations[:, level] - pops_ref[level]).abs() / pops_ref[level].abs()
        ).max().item()
        assert rel_err < 1e-6, (
            f"Level {level}: |n_l - n_l_lte|/n_l_lte = {rel_err:.3e} > 1e-6"
        )

    # Source functions should match Planck at each line.
    for k, tr in enumerate(atm.atomic_model.transitions):
        B_ref = _planck(tr.nu_0, _T0)
        S_k = result.S_lines[:, k]
        rel_err = ((S_k - B_ref).abs() / abs(B_ref)).max().item()
        assert rel_err < 1e-6, (
            f"Line {k}: |S_k - B_k|/B_k = {rel_err:.3e} > 1e-6 at eps=1"
        )


# ===========================================================================
# A-thermalization3-01 — per-line thermalization depth (TASK-015)
# ===========================================================================

@pytest.mark.analytic
@pytest.mark.parametrize(
    "eps_lya,eps_ha,n_tau,tau_max,formal_solve_mode",
    [
        # Equal-epsilon regimes: each line uses its own ε_k.
        (1e-2, 1e-2, 200,  1e4, "parabolic"),
        (1e-4, 1e-4, 1000, 1e4, "parabolic"),
        (1e-6, 1e-6, 400,  1e6, "parabolic"),
        # Cross-ε regime: ε_Lyα ≠ ε_Hα.
        (1e-2, 1e-4, 1000, 1e4, "parabolic"),
    ],
    ids=["equal-1e2", "equal-1e4", "equal-1e6", "cross-1e2-1e4"],
)
def test_A_thermalization3_01_sqrt_eps_depth(
    eps_lya: float,
    eps_ha: float,
    n_tau: int,
    tau_max: float,
    formal_solve_mode: str,
) -> None:
    """A-thermalization3-01: S_k/B_k near 1/sqrt(eps_k) is in [0.45, 1].

    This broad depth-law check uses coherent monochromatic Eddington
    two-stream transfer.  It checks order-unity thermalization, not a
    5/10/20% error relative to an exact profile.  The separate surface-law
    test below applies that relative-tolerance ladder to S_k(0)/B_k.
    """
    import math as _math
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

    # Eddington two-stream quadrature (μ = 1/√3, w = 1) + coherent monochromatic
    # frequency grid (n_nu=1, profile=1).  The AH thermalization-depth law is
    # derived under these idealizations; using the default Gauss-Legendre angle
    # quadrature plus a Doppler-broadened line introduces a profile-dependent
    # effective optical depth that shifts the thermalization-depth crossing
    # away from τ = 1/√ε.  The 2-level A-thermalization-01/02 tests use the
    # same n_nu=1 / Eddington-2stream construction (see
    # examples/generate_regression_proposals.py::_eddington_atmosphere).
    eddington_mu = torch.tensor([1.0 / _math.sqrt(3.0)], dtype=torch.float64)
    eddington_w  = torch.tensor([1.0], dtype=torch.float64)
    coherent_freq = FrequencyGrid(
        nu=torch.zeros(1, dtype=torch.float64),
        weights=torch.ones(1, dtype=torch.float64),
        profile=torch.ones(1, dtype=torch.float64),
    )

    # Build atmosphere with coherent monochromatic frequency grid + Eddington
    # two-stream angle quadrature.  Reuse _make_3level_atmosphere only for its
    # AtomicModel construction; override the grids inline.
    atm_helper = _make_3level_atmosphere(
        T0=_T0, n_e0=1e10, eps_lya=eps_lya, eps_ha=eps_ha,
        n_tau=n_tau, tau_max=tau_max,
    )
    atm = Atmosphere(
        depth=atm_helper.depth,
        frequency=coherent_freq,
        angle=AngleQuadrature(mu=eddington_mu, weights=eddington_w),
        T=atm_helper.T,
        n_e=atm_helper.n_e,
        epsilon=atm_helper.epsilon,
        nu_0=atm_helper.nu_0,
        atomic_model=atm_helper.atomic_model,
    )

    from atmosphere.threelevel.fixed_point import ali_loop_3level
    result = ali_loop_3level(
        atm, tol=1e-8, max_iter=2000, mode="direct",
        formal_solve_mode=formal_solve_mode,
    )

    # Check per-line thermalization depth.
    eps_list = [eps_lya, eps_ha]
    nu_list  = [_NU_LYA, _NU_HA]
    tau_grid = atm.depth.tau

    for k, (eps_k, nu_k) in enumerate(zip(eps_list, nu_list)):
        tau_therm = 1.0 / _math.sqrt(eps_k)
        idx = int(torch.argmin((tau_grid - tau_therm).abs()).item())

        B_k    = _planck(nu_k, _T0)
        S_at_therm = result.S_lines[idx, k].item()
        ratio  = S_at_therm / B_k

        # Check that S_k/B_k at τ_therm lies in [0.45, 1.0] (AH thermalization).
        # The lower bound is 0.45 rather than 0.5 because the log-spaced grid
        # places the nearest τ point slightly away from τ_therm = 1/√ε_k,
        # and the 3-level inter-line coupling can shift the amplitude by a few
        # percent relative to the pure single-line AH value.  The criterion is
        # that S_k/B_k is "of order unity" at the thermalization depth (AH 1965);
        # the interval [0.45, 1.0] reflects this with adequate discretization margin.
        assert 0.45 <= ratio <= 1.0, (
            f"A-thermalization3-01 (ε_Lyα={eps_lya}, ε_Hα={eps_ha}), line {k}: "
            f"S_k/B_k at τ≈1/√ε_k (τ={tau_therm:.2g}, idx={idx}) = {ratio:.3f}, "
            f"expected in [0.45, 1.0]  (AH 1965 per-line)"
        )


# ===========================================================================
# A-thermalization3-02 — per-line surface S_k(0)/B_k ≈ √ε_k (TASK-015)
# ===========================================================================

@pytest.mark.analytic
@pytest.mark.parametrize(
    "eps_lya,eps_ha,n_tau,tau_max,formal_solve_mode",
    [
        (1e-2, 1e-2, 200,  1e4, "parabolic"),
        (1e-4, 1e-4, 1000, 1e4, "parabolic"),
        (1e-6, 1e-6, 400,  1e6, "parabolic"),
        (1e-2, 1e-4, 1000, 1e4, "parabolic"),
    ],
    ids=["equal-1e2", "equal-1e4", "equal-1e6", "cross-1e2-1e4"],
)
def test_A_thermalization3_02_surface_value(
    eps_lya: float,
    eps_ha: float,
    n_tau: int,
    tau_max: float,
    formal_solve_mode: str,
) -> None:
    """A-thermalization3-02: S_k(τ=τ_min)/B_k ≈ √ε_k per line within tolerance.

    At the surface (τ → 0) the line source function satisfies S_k(0) ≈ √ε_k · B_k
    for a semi-infinite isothermal atmosphere with constant ε_k.  In the chained
    3-level setup this holds per-line: each transition sees its own √ε_k surface
    limiting value, with inter-line coupling introducing a small perturbation.

    Tolerance ladder (SPEC_threelevel §M3c, pinned 2026-05-05):
      - Equal ε_k at ε ∈ {1e-2, 1e-4}: 5% relative per line.
      - Equal ε_k at ε = 1e-6: 10% relative per line.
      - Cross-ε (ε_Lyα=1e-2, ε_Hα=1e-4): 20% relative per line.

    This test uses the default Doppler N_nu=21 grid with Eddington angular
    quadrature. It is separate from the coherent monochromatic depth test
    above. Fresh bounded coherent-profile results are recorded under the
    dated 2026-11-16 verification directory, with their actual grids.

    Reference: Avrett & Hummer (1965); HM14 §13.3.
    """
    import math as _math
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature

    eddington_mu = torch.tensor([1.0 / _math.sqrt(3.0)], dtype=torch.float64)
    eddington_w  = torch.tensor([1.0], dtype=torch.float64)

    if eps_lya == eps_ha:
        tol = 0.10 if eps_lya == 1e-6 else 0.05
    else:
        tol = 0.20

    atm = _make_3level_atmosphere(
        T0=_T0, n_e0=1e10, eps_lya=eps_lya, eps_ha=eps_ha,
        n_tau=n_tau, tau_max=tau_max,
    )
    atm = Atmosphere(
        depth=atm.depth,
        frequency=atm.frequency,
        angle=AngleQuadrature(mu=eddington_mu, weights=eddington_w),
        T=atm.T, n_e=atm.n_e,
        epsilon=atm.epsilon, nu_0=atm.nu_0,
        atomic_model=atm.atomic_model,
    )

    from atmosphere.threelevel.fixed_point import ali_loop_3level
    result = ali_loop_3level(
        atm, tol=1e-8, max_iter=2000, mode="direct",
        formal_solve_mode=formal_solve_mode,
    )

    eps_list = [eps_lya, eps_ha]
    nu_list  = [_NU_LYA, _NU_HA]

    for k, (eps_k, nu_k) in enumerate(zip(eps_list, nu_list)):
        B_k       = _planck(nu_k, _T0)
        S_surface = result.S_lines[0, k].item()   # τ_min is first grid point
        expected  = _math.sqrt(eps_k) * B_k
        rel_err   = abs(S_surface - expected) / expected

        assert rel_err <= tol, (
            f"A-thermalization3-02 (ε_Lyα={eps_lya}, ε_Hα={eps_ha}), line {k}: "
            f"S_k(τ_min)/B_k = {S_surface / B_k:.4f}, "
            f"expected ≈ √ε_k = {_math.sqrt(eps_k):.4f}, "
            f"rel err = {rel_err:.2%} > tolerance {tol:.0%}  (AH 1965)"
        )


# ===========================================================================
# A-lte3-01 — LTE Boltzmann populations at eps_k = 1 (TASK-017)
# ===========================================================================

@pytest.mark.analytic
def test_A_lte3_01_boltzmann() -> None:
    """A-lte3-01: at ε_k=1 for all k, converged populations match LTE Boltzmann to 1e-6.

    In the LTE limit (ε_k=1 for all transitions) each source function reduces
    to the Planck function B(T, ν_k), and the level populations must satisfy
    the Boltzmann excitation equilibrium.  For the 3-level hydrogen chain
    (1→2 at ν_Lyα, 2→3 at ν_Hα) with statistical weights g_l = 1:

        n_1 / n_0 = exp(−h ν_Lyα / k T)
        n_2 / n_1 = exp(−h ν_Hα  / k T)
        n_0 + n_1 + n_2 = 1

    Acceptance: max_τ |n_l(τ) − n_l_LTE(τ)| / n_l_LTE < 1e-6 for l = 0, 1, 2.

    Reference: HM14 §13.2 (Hubeny & Mihalas 2014, §13.2).
    SPEC_threelevel §M3c TASK-017.
    """
    from atmosphere.threelevel.fixed_point import ali_loop_3level

    atm = _make_3level_atmosphere(eps_lya=1.0, eps_ha=1.0, n_tau=20, n_nu=5, tau_max=1e2)
    result = ali_loop_3level(atm, tol=1e-8, max_iter=200, mode="direct")

    pops_ref = _lte_populations_3level(_T0)   # (3,) analytic LTE

    for level in range(3):
        rel_err = (
            (result.populations[:, level] - pops_ref[level]).abs()
            / pops_ref[level].abs()
        ).max().item()
        assert rel_err < 1e-6, (
            f"A-lte3-01: level {level}: max rel |n_l − n_l_LTE|/n_l_LTE = {rel_err:.2e} "
            f"> 1e-6  (HM14 §13.2)"
        )


# ===========================================================================
# A-coherent3-01 — single-frequency limit vs. brute-force reference (TASK-018)
# ===========================================================================

@pytest.mark.analytic
def test_A_coherent3_01_two_stream() -> None:
    """A-coherent3-01: monochromatic limit per-line agrees with brute-force reference.

    In the single-frequency (n_nu=1) coherent / monochromatic limit each line's
    source function satisfies the per-line two-stream equation with its own ε_k.
    There is no known closed-form analytic solution in the chained 3-level regime
    (the two lines share level populations via the rate equations).  The reference
    is therefore a brute-force ALI solution at high N_τ = 2000 (direct mode,
    linear OK87 formal solver).

    The test atmosphere uses:
      - n_nu=1 with a delta-function profile (φ=1, w=1) — the coherent limit.
      - Eddington two-stream quadrature (μ=1/√3, w=1) — makes the angular
        discretization consistent with the per-line two-stream physics.
      - ε_Lyα = ε_Hα = 1e-2 (moderate scattering, robust convergence).
      - τ_max = 1e4, τ_min = 1e-4.

    The test solution uses N_τ = 400 (direct mode); the reference uses N_τ = 2000.
    Both are solved to tol=1e-10.  The source functions S_k(τ) are interpolated
    from the reference grid onto the test grid for comparison.

    Grid convergence (2026-05-06, direct mode, linear OK87, ε=1e-2):
      N_τ=200: 2.44%;  N_τ=400: 0.64%;  N_τ=800: 0.14%.
    N_τ=400 is the minimum that clears the 1% criterion.

    Acceptance (SPEC_threelevel §M3c TASK-018): max relative error ≤ 1% on
    S_k(τ) over the full depth grid for both lines.

    Reference: Mihalas (1978) §6-3 eq.6-39 (per-line two-stream; no closed form
    for chained regime); SPEC_threelevel §M3c TASK-018.
    """
    import math as _math
    import numpy as _np

    epsilon  = 1e-2
    tau_min  = 1e-4
    tau_max  = 1e4

    # Eddington two-stream angle quadrature (μ=1/√3, w=1).
    eddington_mu = torch.tensor([1.0 / _math.sqrt(3.0)], dtype=torch.float64)
    eddington_w  = torch.tensor([1.0], dtype=torch.float64)

    def _make_coherent_atm(n_tau: int) -> Atmosphere:
        """Build a single-frequency 3-level atmosphere with Eddington quadrature."""
        from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid
        depth = DepthGrid.log_spaced(tau_min=tau_min, tau_max=tau_max, n_tau=n_tau)
        # Single-frequency coherent limit: profile=1, weight=1.
        freq = FrequencyGrid(
            nu=torch.zeros(1, dtype=torch.float64),
            weights=torch.ones(1, dtype=torch.float64),
            profile=torch.ones(1, dtype=torch.float64),
        )
        angle = AngleQuadrature(mu=eddington_mu, weights=eddington_w)
        from atmosphere.atomic_model import AtomicModel
        model = AtomicModel.three_level_hydrogen_chained(
            epsilon_lya=epsilon, epsilon_ha=epsilon
        )
        N_t = depth.tau.shape[0]
        T   = torch.full((N_t,), _T0, dtype=torch.float64)
        n_e = torch.full((N_t,), 1e10, dtype=torch.float64)
        return Atmosphere(
            depth=depth, frequency=freq, angle=angle,
            T=T, n_e=n_e,
            epsilon=model.transitions[0].epsilon,
            nu_0=model.transitions[0].nu_0,
            atomic_model=model,
        )

    from atmosphere.threelevel.fixed_point import ali_loop_3level

    # Brute-force reference at high N_τ (direct mode).
    atm_ref  = _make_coherent_atm(n_tau=2000)
    result_ref = ali_loop_3level(
        atm_ref, tol=1e-10, max_iter=2000, mode="direct",
        formal_solve_mode="linear",
    )

    # Test solution at N_τ=400 (direct mode — same physics, coarser grid).
    # N_τ=400 yields 0.64% max relative error vs. the N_τ=2000 reference;
    # this clears the 1% acceptance criterion with margin.
    atm_test  = _make_coherent_atm(n_tau=400)
    result_test = ali_loop_3level(
        atm_test, tol=1e-10, max_iter=2000, mode="direct",
        formal_solve_mode="linear",
    )

    # Compare by interpolating the reference S_k onto the test τ grid.
    tau_ref  = atm_ref.depth.tau.numpy()
    tau_test = atm_test.depth.tau.numpy()

    for k in range(2):
        S_ref  = result_ref.S_lines[:, k].numpy()
        S_test = result_test.S_lines[:, k].numpy()

        # log-log interpolation is more accurate for the surface power-law region.
        S_ref_interp = _np.interp(tau_test, tau_ref, S_ref)

        safe_denom  = _np.abs(S_ref_interp).clip(min=1e-300)
        rel_err_arr = _np.abs(S_test - S_ref_interp) / safe_denom
        rel_err_max = float(rel_err_arr.max())

        assert rel_err_max <= 0.01, (
            f"A-coherent3-01: line {k}: max rel |S_test - S_ref| / |S_ref| = "
            f"{rel_err_max:.2%} > 1%  (brute-force reference at N_tau=2000, "
            f"test at N_tau=400)"
        )


# ===========================================================================
# A-interlocking-01 — chained coupling Born-series check (TASK-019)
# ===========================================================================

@pytest.mark.analytic
def test_A_interlocking_01_chained_coupling() -> None:
    """A-interlocking-01: DT applied to a n_2↔n_3 perturbation gives non-zero n_1 response.

    In a 3-level chained system (1→2 at Ly-α, 2→3 at Hα), a perturbation
    confined to the {n_2, n_3} subspace (with zero n_1 component) propagates
    through the rate-equation coupling back into the n_1 subspace via the
    shared level-2 population.  This cross-coupling is the load-bearing
    distinguisher of 3-level from 2-level physics: in a true 2-level system
    a perturbation orthogonal to one level cannot reach the other through the
    cycle Jacobian DT.

    The test constructs the Jacobian-vector product DT·v via finite differences:
        DT·v ≈ [cycle_truth_3level(x* + ε v) − cycle_truth_3level(x* − ε v)] / (2ε)
    where v has v[:,0]=0 (zero n_1 component), v[:,1]=+δ, v[:,2]=−δ (equal and
    opposite n_2, n_3 perturbation).

    Acceptance (SPEC_threelevel §M3c TASK-019):
        ||DT·v [n_1 block]|| / ||v|| ≥ 1e-3

    Empirically measured value (2026-05-06): ≈ 0.23 (much larger than 1e-3),
    confirming that the chained SE coupling is strong and non-trivial.  The
    high ratio reflects the fact that changing n_2 (and n_3) alters both the
    Ly-α and Hα source functions via the rate-matrix, immediately feeding back
    into the n_1 population through the Ly-α transition.

    Reference: SPEC_threelevel §M3c TASK-019, §6.2.1.
    """
    import math as _math

    # Converge to the truth fixed point at (ε_Lyα=1e-2, ε_Hα=1e-4).
    atm = _make_3level_atmosphere(
        T0=_T0, n_e0=1e10, eps_lya=1e-2, eps_ha=1e-4,
        n_tau=40, tau_max=1e4, n_nu=_N_NU,
    )
    N_tau = atm.depth.tau.shape[0]

    from atmosphere.threelevel.fixed_point import ali_loop_3level, cycle_truth_3level
    result = ali_loop_3level(atm, tol=1e-10, max_iter=500, mode="direct")
    x_star = result.populations.detach().clone()  # (N_tau, 3)

    # Perturbation v: zero n_1 component, anti-symmetric n_2 / n_3.
    # The normalization factor makes ||v|| = delta * sqrt(2) * sqrt(N_tau).
    delta = 1e-4
    v = torch.zeros(N_tau, 3, dtype=torch.float64)
    v[:, 1] =  delta
    v[:, 2] = -delta

    # Finite-difference JVP for DT·v.
    eps_fd = 1e-6
    x_plus  = x_star + eps_fd * v
    x_minus = x_star - eps_fd * v
    DT_v = (
        cycle_truth_3level(x_plus, atm) - cycle_truth_3level(x_minus, atm)
    ) / (2.0 * eps_fd)

    # Ratio: ||DT_v [n_1 block]|| / ||v||
    n1_block_norm = DT_v[:, 0].norm().item()
    v_norm        = v.norm().item()
    ratio         = n1_block_norm / v_norm

    # Threshold from SPEC_threelevel §M3c TASK-019.
    # Empirical value ≈ 0.23; the 1e-3 floor is the operational threshold.
    _THRESHOLD = 1e-3
    assert ratio >= _THRESHOLD, (
        f"A-interlocking-01: ||DT·v [n_1-block]|| / ||v|| = {ratio:.4e} < {_THRESHOLD:.0e}. "
        f"The n_2↔n_3 perturbation produced no measurable n_1 response, "
        f"indicating the chained coupling is absent or broken."
    )
