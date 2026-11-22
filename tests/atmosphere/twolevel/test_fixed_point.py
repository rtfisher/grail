"""Tests for atmosphere/twolevel/fixed_point.py.

Moved here from tests/atmosphere/test_fixed_point.py as part of
SPEC_threelevel §M3a.  Imports are unchanged because atmosphere/fixed_point.py
re-exports from atmosphere/twolevel/fixed_point.py.

Test IDs match docs/testing.md.

References
----------
- HM14 §13.3: Hubeny & Mihalas (2014), Theory of Stellar Atmospheres,
  §13.3 (Avrett & Hummer 1965 thermalization-depth law).
- M78 §6-3 eq.6-39: Mihalas (1978), Stellar Atmospheres, 2nd ed.,
  §6-3, eq. 6-39 (coherent two-stream analytic solution).
- OAB86: Olson, Auer & Buchler (1986), JQSRT 35, 431.
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Physical constants (duplicated from fixed_point.py for test self-containment)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant [J s]
_C = 2.99792458e8     # speed of light [m/s]
_KB = 1.380649e-23    # Boltzmann constant [J/K]

# Hydrogen Lyman-alpha frequency as a representative line (used only to
# set a physically plausible nu_0 in test atmospheres; the exact value does
# not affect the dimensionless analytic limits we test against).
_NU0 = 2.47e15  # Hz (Ly-α)


# ---------------------------------------------------------------------------
# Helpers — build a minimal Atmosphere-like object for tests.
# These are constructed as plain dataclass instances so the tests do not
# depend on Atmosphere.isothermal() (which is also a M0 stub).
# ---------------------------------------------------------------------------

def _make_isothermal_atmosphere(
    T0: float,
    n_e0: float,
    epsilon: float,
    tau_max: float = 1e4,
    n_tau: int = 80,
    n_nu: int = 1,
    n_mu: int = 5,
    nu_0: float = _NU0,
):
    """Build a minimal isothermal Atmosphere instance for tests.

    Uses real grid constructors on main.  For n_nu == 1 (the coherent /
    monochromatic limit) we override the FrequencyGrid manually with a
    delta-function-like representation (single point, profile=1, weight=1)
    rather than going through .doppler(); for n_nu > 1 we use .doppler().
    """
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=tau_max, n_tau=n_tau)

    if n_nu == 1:
        # Single-frequency / coherent monochromatic limit: psi(x) = 1, phi = 1,
        # quadrature weight = 1 so int phi(x) dx -> 1.
        frequency = FrequencyGrid(
            nu=torch.zeros(1, dtype=torch.float64),
            weights=torch.ones(1, dtype=torch.float64),
            profile=torch.ones(1, dtype=torch.float64),
        )
    else:
        frequency = FrequencyGrid.doppler(n_nu=n_nu, half_width=5.0)

    angle = AngleQuadrature.gauss_legendre(n_mu=n_mu)

    T = torch.full((n_tau,), T0, dtype=torch.float64)
    n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)

    return Atmosphere(
        depth=depth,
        frequency=frequency,
        angle=angle,
        T=T,
        n_e=n_e,
        epsilon=epsilon,
        nu_0=nu_0,
    )


# ---------------------------------------------------------------------------
# U-fixedpoint-01
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_fixedpoint_01_eps_one_one_iter() -> None:
    """U-fixedpoint-01: isothermal ε=1 ALI converges in one step to S=B.

    At ε=1 the line source function equals the Planck function at every
    depth regardless of the radiation field (pure thermal emission, no
    scattering).  The ALI update with the LTE initial guess S^(0)=B must
    therefore converge in a single iteration with zero residual.

    docs/testing.md: U-fixedpoint-01.  the technical reference §3.4.
    """
    from atmosphere.fixed_point import AliResult, ali_loop
    from atmosphere.rate_equations import rate_equation_solve

    atm = _make_isothermal_atmosphere(T0=6000.0, n_e0=1e10, epsilon=1.0)

    result: AliResult = ali_loop(atm, tol=1e-6, max_iter=200)

    # At ε=1 the LTE initial guess is already the fixed point;
    # one formal-solve iteration should satisfy the convergence criterion.
    assert result.iter_count == 1, (
        f"Expected 1 iteration at ε=1, got {result.iter_count}"
    )

    # S_l should equal B(T) at every depth.
    B = (
        2.0 * _H * _NU0**3 / _C**2
        / (torch.exp(torch.tensor(_H * _NU0 / (_KB * 6000.0))) - 1.0)
    )
    rel_err = ((result.S_l - B).abs() / B.abs()).max().item()
    assert rel_err < 1e-6, f"max |S-B|/B = {rel_err:.2e} at ε=1"


# ---------------------------------------------------------------------------
# A-thermalization-01
# ---------------------------------------------------------------------------

@pytest.mark.analytic
@pytest.mark.parametrize("epsilon,tau_max,n_tau,formal_solve_mode", [
    (1e-2, 1e4, 200, "linear"),
    (1e-4, 1e4, 1000, "linear"),
    (1e-6, 1e6, 400, "parabolic"),   # was xfail; parabolic at N_tau=400 passes
])
def test_A_thermalization_01_sqrt_eps_depth(
    epsilon: float, tau_max: float, n_tau: int, formal_solve_mode: str
) -> None:
    """A-thermalization-01: S_l/B at τ = 1/√ε lies in [0.5, 1.0].

    The Avrett-Hummer (1965) thermalization-depth law states that the line
    source function thermalizes to the Planck function at a depth
    τ_therm ~ 1/√ε (measured in line optical depth units).  At that depth
    S_l/B should be of order unity; we verify it falls in [0.5, 1.0] to
    allow for discretization and quadrature effects.

    The short-characteristic source interpolation and depth resolution are
    explicit: N_τ=200 and
    1000 with linear-S at ε=1e-2 and 1e-4, and N_τ=400 with parabolic-S
    at ε=1e-6.  The asserted depth-law interval is separate from the
    surface-law relative tolerance in the following test.

    Reference: HM14 §13.3 (Hubeny & Mihalas 2014, Theory of Stellar
    Atmospheres, §13.3; Avrett & Hummer 1965).

    docs/testing.md: A-thermalization-01.
    """
    import math
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.fixed_point import ali_loop
    from atmosphere.grid import AngleQuadrature
    from atmosphere.rate_equations import rate_equation_solve  # noqa: F401

    tau_therm = 1.0 / math.sqrt(epsilon)

    atm = _make_isothermal_atmosphere(
        T0=6000.0,
        n_e0=1e10,
        epsilon=epsilon,
        tau_max=tau_max,
        n_tau=n_tau,
    )
    # Use the Eddington two-stream angular quadrature (μ=1/√3, w=1).  The
    # analytic Avrett-Hummer surface law and S(τ) profile are derived under
    # the two-stream Eddington approximation (Mihalas 1978 §6-3); using the
    # default 5-point Gauss-Legendre quadrature would introduce a separate
    # ~5% angular discrepancy on top of the depth discretization error.
    eddington_mu = torch.tensor([1.0 / math.sqrt(3.0)], dtype=torch.float64)
    eddington_w = torch.tensor([1.0], dtype=torch.float64)
    atm = Atmosphere(
        depth=atm.depth,
        frequency=atm.frequency,
        angle=AngleQuadrature(mu=eddington_mu, weights=eddington_w),
        T=atm.T,
        n_e=atm.n_e,
        epsilon=atm.epsilon,
        nu_0=atm.nu_0,
    )
    result = ali_loop(atm, tol=1e-6, max_iter=200, formal_solve_mode=formal_solve_mode)

    # Find the depth index closest to tau_therm.
    tau_grid = atm.depth.tau
    idx = int(torch.argmin((tau_grid - tau_therm).abs()).item())

    # Planck function at T0 = 6000 K.
    B = (
        2.0 * _H * _NU0**3 / _C**2
        / (torch.exp(torch.tensor(_H * _NU0 / (_KB * 6000.0))) - 1.0)
    ).item()

    S_at_therm = result.S_l[idx].item()
    ratio = S_at_therm / B

    assert 0.5 <= ratio <= 1.0, (
        f"ε={epsilon}: S/B at τ≈1/√ε (τ={tau_therm:.2g}) = {ratio:.3f}, "
        f"expected in [0.5, 1.0]  (HM14 §13.3)"
    )


# ---------------------------------------------------------------------------
# A-thermalization-02
# ---------------------------------------------------------------------------

@pytest.mark.analytic
@pytest.mark.parametrize("epsilon,tau_max,n_tau,formal_solve_mode", [
    (1e-2, 1e4, 200, "linear"),
    (1e-4, 1e4, 1000, "linear"),
    (1e-6, 1e6, 400, "parabolic"),   # was xfail; parabolic at N_tau=400 passes
])
def test_A_thermalization_02_surface_value(
    epsilon: float, tau_max: float, n_tau: int, formal_solve_mode: str
) -> None:
    """A-thermalization-02: S_l(τ=τ_min)/B ≈ √ε within 10%.

    At the surface (τ → 0) the source function satisfies S_l(0) ≈ √ε · B
    for a semi-infinite isothermal atmosphere with constant ε.  This is the
    classical result of Avrett & Hummer (1965).

    See the docstring of test_A_thermalization_01_sqrt_eps_depth for
    the discretization-error analysis that motivates the ε-dependent n_tau
    selection here.

    Tolerance: 10% relative error, i.e. |S/B - √ε| / √ε ≤ 0.10.

    Reference: HM14 §13.3 (Avrett & Hummer 1965).

    docs/testing.md: A-thermalization-02.
    """
    import math
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.fixed_point import ali_loop
    from atmosphere.grid import AngleQuadrature
    from atmosphere.rate_equations import rate_equation_solve  # noqa: F401

    atm = _make_isothermal_atmosphere(
        T0=6000.0,
        n_e0=1e10,
        epsilon=epsilon,
        tau_max=tau_max,
        n_tau=n_tau,
    )
    # Use the Eddington two-stream angular quadrature (μ=1/√3, w=1).  The
    # AH surface law S(0)=√ε·B is the τ→0 limit of the two-stream Eddington
    # solution; using the default 5-point Gauss-Legendre quadrature would
    # introduce a separate ~5% angular discrepancy.
    eddington_mu = torch.tensor([1.0 / math.sqrt(3.0)], dtype=torch.float64)
    eddington_w = torch.tensor([1.0], dtype=torch.float64)
    atm = Atmosphere(
        depth=atm.depth,
        frequency=atm.frequency,
        angle=AngleQuadrature(mu=eddington_mu, weights=eddington_w),
        T=atm.T,
        n_e=atm.n_e,
        epsilon=atm.epsilon,
        nu_0=atm.nu_0,
    )
    result = ali_loop(atm, tol=1e-6, max_iter=200, formal_solve_mode=formal_solve_mode)

    # Surface is the first (smallest τ) depth point.
    S_surface = result.S_l[0].item()

    B = (
        2.0 * _H * _NU0**3 / _C**2
        / (torch.exp(torch.tensor(_H * _NU0 / (_KB * 6000.0))) - 1.0)
    ).item()

    expected = math.sqrt(epsilon) * B
    rel_err = abs(S_surface - expected) / expected

    assert rel_err <= 0.10, (
        f"ε={epsilon}: S(τ_min)/B = {S_surface/B:.4f}, "
        f"expected ≈ √ε = {math.sqrt(epsilon):.4f}, "
        f"relative error {rel_err:.2%}  (HM14 §13.3)"
    )


# ---------------------------------------------------------------------------
# A-coherent-01
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_A_coherent_01_two_stream() -> None:
    """A-coherent-01: monochromatic limit reproduces analytic two-stream solution.

    In the coherent (no redistribution), single-frequency limit (N_ν=1,
    φ(ν)=1) the analytic solution to the two-stream radiative-transfer
    equation with the ε-formulation source function is (Mihalas 1978, §6-3,
    eq. 6-39):

        S_l(τ) = B · [1 − (1 − √ε) · exp(−√(3ε) · τ)]

    The √(3ε) decay constant is the signature of the *two-stream Eddington
    approximation*, which corresponds to a 1-ray angular quadrature with
    μ = 1/√3 (Mihalas 1978 §6-3, Hubeny & Mihalas 2014 §11.2).  To reproduce
    the analytic limit faithfully the formal solver must therefore use the
    same single-stream quadrature; the default 5-point Gauss-Legendre
    quadrature is more accurate than the two-stream approximation and
    differs by ~8% at intermediate depths.  This test therefore overrides
    the default angle quadrature with the Eddington two-stream value.

    We verify agreement to within 5% at each depth grid point.

    Reference: M78 §6-3 eq.6-39 (Mihalas 1978, Stellar Atmospheres, 2nd ed.).

    docs/testing.md: A-coherent-01.
    """
    import math
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.fixed_point import ali_loop
    from atmosphere.grid import AngleQuadrature
    from atmosphere.rate_equations import rate_equation_solve  # noqa: F401

    epsilon = 1e-2
    T0 = 6000.0

    atm = _make_isothermal_atmosphere(
        T0=T0,
        n_e0=1e10,
        epsilon=epsilon,
        tau_max=1e4,
        n_tau=200,  # finer grid: with n_tau=80 the max rel err is ~10% even
                    # with the correct two-stream μ; n_tau=200 brings it under
                    # 5% (we observe 2.47% at n_tau=200, μ=1/√3).
        n_nu=1,  # single-frequency: coherent / monochromatic limit
    )
    # Override the angle quadrature with the Eddington two-stream value
    # μ = 1/√3, weight = 1.  This matches the analytic solution which is
    # derived under the two-stream Eddington approximation.
    eddington_mu = torch.tensor([1.0 / math.sqrt(3.0)], dtype=torch.float64)
    eddington_w = torch.tensor([1.0], dtype=torch.float64)
    atm = Atmosphere(
        depth=atm.depth,
        frequency=atm.frequency,
        angle=AngleQuadrature(mu=eddington_mu, weights=eddington_w),
        T=atm.T,
        n_e=atm.n_e,
        epsilon=atm.epsilon,
        nu_0=atm.nu_0,
    )
    result = ali_loop(atm, tol=1e-8, max_iter=500)

    B = (
        2.0 * _H * _NU0**3 / _C**2
        / (torch.exp(torch.tensor(_H * _NU0 / (_KB * T0))) - 1.0)
    ).item()

    sqrt_eps = math.sqrt(epsilon)
    tau_grid = atm.depth.tau  # (N_tau,)

    # Analytic solution: S(τ) = B[1 − (1 − √ε) exp(−√(3ε) τ)]
    S_analytic = B * (1.0 - (1.0 - sqrt_eps) * torch.exp(-math.sqrt(3.0 * epsilon) * tau_grid))

    # Relative error at each depth (avoid division by near-zero at surface).
    safe_S_analytic = S_analytic.abs().clamp(min=1e-300)
    rel_err = ((result.S_l - S_analytic).abs() / safe_S_analytic).max().item()

    assert rel_err <= 0.05, (
        f"max |S_ALI - S_analytic| / |S_analytic| = {rel_err:.2%} > 5%  "
        f"(M78 §6-3 eq.6-39)"
    )


# ---------------------------------------------------------------------------
# A-lte-01
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_A_lte_01_boltzmann() -> None:
    """A-lte-01: at ε=1 isothermal, populations match Boltzmann ratio to 1e-6.

    In the LTE limit (ε=1) the source function equals the Planck function at
    every depth, and the upper/lower level populations must satisfy the
    Boltzmann excitation ratio:

        n_2 / n_1 = (g_2 / g_1) · exp(−hν₀ / kT)

    Relative tolerance: 1e-6.

    Reference: the technical reference §3.3; HM14 §2 (LTE definition).

    docs/testing.md: A-lte-01.
    """
    import math
    from atmosphere.fixed_point import ali_loop
    from atmosphere.rate_equations import rate_equation_solve

    T0 = 6000.0
    atm = _make_isothermal_atmosphere(T0=T0, n_e0=1e10, epsilon=1.0)
    result = ali_loop(atm, tol=1e-6, max_iter=200)

    n1 = result.populations[:, 0]
    n2 = result.populations[:, 1]

    # Boltzmann ratio g_2/g_1 * exp(-hν/kT); g_ratio defaults to 1.
    g_ratio = getattr(atm, "g_ratio", 1.0)
    boltzmann = g_ratio * math.exp(-_H * _NU0 / (_KB * T0))

    # Check at every depth point.
    ratio_computed = (n2 / n1.clamp(min=1e-300)).double()
    rel_err = ((ratio_computed - boltzmann) / boltzmann).abs().max().item()

    assert rel_err < 1e-6, (
        f"Boltzmann ratio error at ε=1: max rel err = {rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# I-truthcycle-01
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_I_truthcycle_01_convergence() -> None:
    """I-truthcycle-01: ALI on isothermal ε=1e-4, τ_max=1e4 converges in <200 iters.

    This is the canonical integration test for the truth cycle.  It verifies
    that the ALI iteration (with diagonal Λ* preconditioner) converges within
    the 200-iteration budget for a moderately scattering atmosphere.

    Records iteration count and final residual; if the count exceeds a 20%
    drift from a committed baseline (60 iterations) this test will flag it.
    The strict bound is <200 per docs/testing.md I-truthcycle-01; the drift bound
    is advisory and printed as a warning.

    docs/testing.md: I-truthcycle-01.  the technical reference §3.4.
    """
    from atmosphere.fixed_point import AliResult, ali_loop
    from atmosphere.rate_equations import rate_equation_solve  # noqa: F401

    epsilon = 1e-4
    tau_max = 1e4

    atm = _make_isothermal_atmosphere(
        T0=6000.0,
        n_e0=1e10,
        epsilon=epsilon,
        tau_max=tau_max,
        n_tau=80,
    )

    # Pass se_solver=None so ali_loop uses the reference adapter that binds
    # epsilon and nu_0 from the atmosphere (rate_equation_solve has those as
    # keyword-only args; the (J_bar, T, n_e) contract requires the adapter).
    result: AliResult = ali_loop(atm, tol=1e-6, max_iter=200)

    # Hard bound from docs/testing.md.
    assert result.iter_count < 200, (
        f"ALI did not converge in <200 iterations (ε={epsilon}, τ_max={tau_max}): "
        f"iter_count={result.iter_count}, final residual={result.residual_history[-1]:.2e}"
    )

    # Final residual must be below tolerance.
    final_res = result.residual_history[-1]
    assert final_res < 1e-6, (
        f"Final residual {final_res:.2e} exceeds tol=1e-6 after "
        f"{result.iter_count} iterations"
    )

    # Advisory: warn if count drifts >20% from a rough baseline of 60 iters.
    _BASELINE = 60
    if result.iter_count > _BASELINE * 1.20:
        import warnings
        warnings.warn(
            f"I-truthcycle-01: iter_count={result.iter_count} drifted "
            f">20% above baseline={_BASELINE}",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# U-picard2-01 — smoke test for picard_iterate_2level (TASK-034, §M3g)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_picard2_01_isothermal_lte() -> None:
    """U-picard2-01: plain Picard with the truth SE solver converges to LTE at ε=1.

    At ε=1.0, the rate equation solution is fully collisional (LTE):
    the emitter population is proportional to exp(-hν/kT).  The LTE
    fixed point is also the initial guess (we start from B(T)), so
    Picard should converge in ≤ 5 iterations.  This smoke test confirms:
      (a) picard_iterate_2level is callable with the reference se_solver
          (via _make_reference_se_solver — the same adapter that ali_loop uses).
      (b) The driver correctly wires the 2-level cycle _cycle(x, atm, se_solver).
      (c) Convergence is reported correctly in PicardResult2Level.

    At ε=1, the truth fixed point IS the LTE initial guess so the Picard
    residual after the first step should already be < tol.

    SPEC_threelevel §M3g TASK-034, acceptance U-picard2-01.
    """
    from atmosphere.twolevel.fixed_point import (
        PicardResult2Level,
        _make_reference_se_solver,
        picard_iterate_2level,
    )

    # Build a small isothermal atmosphere at ε=1 (fully thermalized, LTE).
    atm = _make_isothermal_atmosphere(
        T0=6000.0,
        n_e0=1e10,
        epsilon=1.0,
        tau_max=1e3,
        n_tau=40,
        n_nu=5,
        n_mu=3,
    )

    se_solver = _make_reference_se_solver(atm)

    result = picard_iterate_2level(
        atm,
        se_solver,
        tol=1e-6,
        max_iter=10,
        accelerator="none",
    )

    print(f"\nU-picard2-01 (ε=1.0, LTE):")
    print(f"  iter_count={result.iter_count}, converged={result.converged}")
    print(f"  anderson_engaged={result.anderson_engaged}")
    if result.residual_history:
        print(f"  final residual={result.residual_history[-1]:.3e}")

    assert isinstance(result, PicardResult2Level), (
        "picard_iterate_2level must return a PicardResult2Level instance."
    )
    assert result.converged, (
        f"U-picard2-01: Picard did NOT converge at ε=1.0 (LTE). "
        f"iter_count={result.iter_count}.  "
        "At ε=1.0 the LTE initial guess IS the fixed point; one iteration "
        "should produce a residual well below tol=1e-6."
    )
    assert result.iter_count <= 5, (
        f"U-picard2-01: Too many iterations at ε=1.0 (LTE): "
        f"iter_count={result.iter_count} > 5.  "
        "The LTE initial guess is the fixed point; Picard should converge in ≤ 5 steps."
    )
    # Populations must be non-negative and sum to 1 at every depth.
    pops = result.populations
    assert (pops >= -1e-8).all().item(), (
        "U-picard2-01: Negative populations in PicardResult2Level."
    )
    assert ((pops.sum(dim=-1) - 1.0).abs() < 1e-6).all().item(), (
        "U-picard2-01: Populations do not sum to 1 at every depth."
    )
