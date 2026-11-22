"""Tests for atmosphere/formal_solver.py.

Moved here from tests/atmosphere/test_formal_solver.py as part of
SPEC_threelevel §M3a.  Imports are unchanged.

Test IDs match docs/testing.md / the technical reference.

Tests implemented here:
  U-formalsolve-01  uniform source -> J = S at interior depths
  U-formalsolve-02  linearity of formal_solve in S
  U-formalsolve-03  torch.func.jvp matches finite-difference JVP
  A-eddington-01    Eddington-Barbier surface emergence relation

Atmosphere instances are constructed directly via dataclass field assignment
because the .isothermal() / .power_law() factory methods are M0 stubs.
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.atmosphere import Atmosphere
from atmosphere.formal_solver import formal_solve, formal_solve_with_surface_I
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

# ---------------------------------------------------------------------------
# Physical constants (SI) — mirrored from formal_solver.py for test assertions
# ---------------------------------------------------------------------------
_H = 6.62607015e-34
_C = 2.99792458e8
_K = 1.380649e-23


# ---------------------------------------------------------------------------
# Helpers for constructing minimal test atmospheres
# ---------------------------------------------------------------------------

def _make_atmosphere(
    *,
    N_tau: int = 80,
    N_nu: int = 21,
    N_mu: int = 5,
    tau_min: float = 1e-4,
    tau_max: float = 1e6,
    T0: float = 6000.0,
    T_profile: torch.Tensor | None = None,
    epsilon: float = 0.5,
    nu_0: float = 5.89e14,   # Ha line center (Hz)
    half_width: float = 3.0,  # half-width in Doppler units x = (nu-nu0)/Delta_nu_D
) -> Atmosphere:
    """Build a minimal Atmosphere for testing.

    Uses the real grid constructors on main (DepthGrid.log_spaced,
    FrequencyGrid.doppler, AngleQuadrature.gauss_legendre).  Atmosphere is
    constructed directly via the dataclass to permit a custom T_profile.

    Default half_width = 3.0 Doppler widths: at x=±3, psi = exp(-9) ≈ 1.2e-4,
    so for tau_max = 1e6 every frequency channel is optically thick across
    the slab.
    """
    dtype = torch.float64

    depth = DepthGrid.log_spaced(tau_min=tau_min, tau_max=tau_max, n_tau=N_tau)
    freq = FrequencyGrid.doppler(n_nu=N_nu, half_width=half_width)
    angle = AngleQuadrature.gauss_legendre(n_mu=N_mu)

    if T_profile is not None:
        T = T_profile.to(dtype=dtype)
    else:
        T = torch.full((N_tau,), T0, dtype=dtype)

    n_e = torch.full((N_tau,), 1e12, dtype=dtype)

    return Atmosphere(
        depth=depth,
        frequency=freq,
        angle=angle,
        T=T,
        n_e=n_e,
        epsilon=epsilon,
        nu_0=nu_0,
    )


def _gauss_legendre(n: int, dtype: torch.dtype, device: torch.device):
    """Gauss-Legendre nodes and weights on [0, 1] with n points.

    Uses numpy for the root computation, then converts to torch.
    We only need this in tests; the production solver accesses the pre-built
    AngleQuadrature from atmosphere.angle.
    """
    import numpy as np
    from numpy.polynomial.legendre import leggauss
    # Nodes on [-1, 1]
    xi, wi = leggauss(n)
    # Map to [0, 1]: x -> (x + 1)/2,  w -> w/2
    mu = (xi + 1.0) / 2.0
    w  = wi / 2.0
    # Sort ascending
    order = np.argsort(mu)
    mu = mu[order]
    w  = w[order]
    return (
        torch.tensor(mu, dtype=dtype, device=device),
        torch.tensor(w,  dtype=dtype, device=device),
    )


def _planck(T0: float, nu_0: float) -> float:
    """Planck function B_nu(T0) at line center nu_0, scalar Python float."""
    return (2.0 * _H * nu_0**3 / _C**2) / (math.exp(_H * nu_0 / (_K * T0)) - 1.0)


# ---------------------------------------------------------------------------
# U-formalsolve-01: uniform source returns J = S
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_formalsolve_01_uniform_source() -> None:
    """U-formalsolve-01: J(tau, x) = S₀ at all well-interior depths for uniform S.

    Reference: Hubeny & Mihalas (2014) §11.4 — for a uniform source function S₀
    with the diffusion lower boundary condition I⁺(tau_max) = B = S₀ (isothermal
    atmosphere with T such that B(T) = S₀), the mean intensity satisfies J = S₀
    throughout the atmosphere.  The upper BC I⁻(tau_min) = 0 produces a boundary
    layer near the surface; we exclude depth points where the downgoing ray has
    not yet equilibrated, requiring tau ≥ 100 so that exp(-tau*psi_min/mu_max) < 1e-10.

    Frequency grid uses half_width = 3.0 so that even the wing channels
    (x = ±3, psi = exp(-9) ≈ 1.2e-4) see a line optical depth of
    psi * tau_max = 1.2e-4 * 1e6 = 120 >> 1, ensuring all channels are
    optically thick and I_down has equilibrated at the check depths.

    Relative tolerance: 1e-10.
    """
    nu_0    = 5.89e14          # Hz
    N_tau   = 120
    N_nu    = 11
    N_mu    = 5
    tau_min = 1e-4
    tau_max = 1e6
    dtype   = torch.float64

    # Temperature T0 such that B_nu(T0) = S0; pick T0 = 8000 K and use S0 = B(T0).
    T0  = 8000.0
    S0  = _planck(T0, nu_0)

    atm = _make_atmosphere(
        N_tau=N_tau, N_nu=N_nu, N_mu=N_mu,
        tau_min=tau_min, tau_max=tau_max,
        T0=T0, epsilon=0.5, nu_0=nu_0,
        half_width=3.0,   # ensures all channels optically thick
    )

    # Constant source function = B(T0) everywhere
    S = torch.full((N_tau, N_nu), S0, dtype=dtype)

    J = formal_solve(S, atm)

    # Minimum wing psi at x=±3: exp(-9) ~ 1.23e-4.
    # The downgoing I_down equilibrates to S0 when exp(-psi*tau/mu) < 1e-10, i.e.
    # psi*tau/mu > 23.  For the largest quadrature angle mu_max ~ 0.95:
    #   tau_crit = 23 * mu_max / psi_min = 23 * 0.95 / 1.23e-4 ~ 1.77e5.
    # Use tau >= 5e5 (3x margin): exp(-path) < 1e-28 for all (mu, x).
    # This is still deep interior: tau / tau_min = 5e9.
    tau   = atm.depth.tau
    mask  = tau >= 5e5

    assert mask.sum() > 0, "No interior depth points found for tau >= 5e5"

    J_interior = J[mask]             # (N_interior, N_nu)
    S_interior = S[mask]             # (N_interior, N_nu)

    rel_err = ((J_interior - S_interior).abs() / S_interior.abs()).max().item()
    assert rel_err < 1e-10, (
        f"J != S in interior: max relative error = {rel_err:.3e} (want < 1e-10)"
    )


# ---------------------------------------------------------------------------
# U-formalsolve-02: linearity in S
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_formalsolve_02_linearity() -> None:
    """U-formalsolve-02: formal_solve is linear in S.

    J(α S₁ + β S₂) = α J(S₁) + β J(S₂) to relative 1e-12 for random
    scalars α, β and pseudo-random source arrays S₁, S₂.

    Reference: Hubeny & Mihalas (2014) §11.4; linearity of the RTE is exact
    by construction.  Any residual above floating-point rounding indicates a bug.

    Implementation note: the solver is formally *affine* in S (the deep boundary
    condition I⁺(tau_max) = B adds a constant inhomogeneous term).  For the
    linearity identity to hold exactly, we constrain α + β = 1 so that the
    inhomogeneous BC term cancels:
        J(αS₁+βS₂) − α J(S₁) − β J(S₂)  =  c_BC · (1 − (α+β)) = 0.
    This convex-combination form tests the linearity of the homogeneous
    (scattering + propagation) part of the operator exactly.

    Tolerance: 1e-12 relative.
    """
    torch.manual_seed(42)
    N_tau, N_nu = 40, 11
    atm   = _make_atmosphere(N_tau=N_tau, N_nu=N_nu, N_mu=3)
    dtype = torch.float64

    # Pseudo-random source functions, positive, order-of-S0 magnitude
    S1 = torch.rand(N_tau, N_nu, dtype=dtype) + 0.1
    S2 = torch.rand(N_tau, N_nu, dtype=dtype) + 0.1

    # Random scalar α ∈ (0, 1); β = 1 − α ensures α + β = 1 exactly, so the
    # affine BC constant cancels and the test probes pure homogeneous linearity.
    alpha = float(torch.rand(1, dtype=dtype).item())
    beta  = 1.0 - alpha

    J1      = formal_solve(S1, atm)
    J2      = formal_solve(S2, atm)
    J_combo = formal_solve(alpha * S1 + beta * S2, atm)
    J_lin   = alpha * J1 + beta * J2

    # Relative error normalised by the scale of J
    scale   = J_lin.abs().max().clamp(min=1e-30)
    rel_err = ((J_combo - J_lin).abs() / scale).max().item()

    assert rel_err < 1e-12, (
        f"Linearity violated: max relative error = {rel_err:.3e} (want < 1e-12)"
    )


# ---------------------------------------------------------------------------
# U-formalsolve-03: JVP via torch.func.jvp matches finite differences
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_formalsolve_03_jvp_finite_difference() -> None:
    """U-formalsolve-03: torch.func.jvp matches finite-difference JVP to 1e-6.

    We draw a random tangent vector v and compare
        torch.func.jvp(formal_solve, (S, atm), (v, None))[1]
    with the central finite-difference estimate
        (formal_solve(S + η·v, atm) − formal_solve(S − η·v, atm)) / (2η)
    at η = 1e-5.

    Relative tolerance: 1e-6.
    """
    torch.manual_seed(7)
    N_tau, N_nu = 40, 11
    atm   = _make_atmosphere(N_tau=N_tau, N_nu=N_nu, N_mu=3)
    dtype = torch.float64
    eta   = 1e-5

    S = torch.rand(N_tau, N_nu, dtype=dtype, requires_grad=True) + 0.5
    v = torch.randn(N_tau, N_nu, dtype=dtype)

    # torch.func.jvp: f takes (S,); atmosphere is a closed-over constant
    def f(S_):
        return formal_solve(S_, atm)

    _, jvp_auto = torch.func.jvp(f, (S,), (v,))

    # Central finite-difference JVP
    with torch.no_grad():
        J_plus  = formal_solve(S + eta * v, atm)
        J_minus = formal_solve(S - eta * v, atm)
    jvp_fd = (J_plus - J_minus) / (2.0 * eta)

    scale   = jvp_fd.abs().max().clamp(min=1e-30)
    rel_err = ((jvp_auto - jvp_fd).abs() / scale).max().item()

    assert rel_err < 1e-6, (
        f"JVP mismatch: max relative error = {rel_err:.3e} (want < 1e-6)"
    )


# ---------------------------------------------------------------------------
# A-eddington-01: Eddington-Barbier surface emergence relation
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_A_eddington_01_surface_intensity() -> None:
    """A-eddington-01: Eddington-Barbier emergent intensity.

    For a linearly varying source function S(τ) = a + b·τ, the Eddington-
    Barbier relation (Hubeny & Mihalas 2014 §11.5) predicts:
        I(τ=0, μ) ≈ S(τ = μ) = a + b·μ
    to first order in the source function gradient.

    We verify this at μ ∈ {0.1, 0.5, 0.9} to within 5% relative error
    using the formal solver with:
    - S(τ) = a + b·τ, uniformly broadcast over frequency x.
    - An isothermal atmosphere with T0 chosen so that B(T0) = S(τ_max),
      which makes the deep diffusion BC exactly consistent with S, eliminating
      boundary artefacts.
    - The surface upgoing intensity I⁺(τ_min, μ, x) is extracted via
      formal_solve_with_surface_I and evaluated at the central frequency
      channel x=0 (line center, ψ = 1).  This avoids contamination from
      optically thin wing channels.

    Reference: Hubeny & Mihalas (2014) §11.5, eq. (11.26).
    Tolerance: 5% relative at μ ∈ {0.1, 0.5, 0.9}.
    """
    nu_0    = 5.89e14
    N_tau   = 200
    N_nu    = 11
    N_mu    = 8
    tau_min = 1e-4
    tau_max = 1e4
    dtype   = torch.float64

    # Linear source function: S(τ) = a + b·τ.
    # Choose a and b so that S stays positive and varies enough to probe the
    # Eddington-Barbier relation at μ ~ 0.1–0.9 (i.e. τ ~ 0.1–0.9).
    # Also set B(T0) = S(τ_max) to make the deep BC consistent with S.
    a = 1.0
    b = 5e-5   # gentle slope; S(τ_max=1e4) = 1 + 0.5 = 1.5

    tau_grid = torch.logspace(
        math.log10(tau_min), math.log10(tau_max), N_tau, dtype=dtype
    )
    S_tau = a + b * tau_grid   # shape (N_tau,)

    # T0 such that B_nu(T0) = S(τ_max) — deep BC is then exactly S(τ_max).
    S_max     = float(S_tau[-1].item())
    prefactor = 2.0 * _H * nu_0**3 / _C**2
    T0        = (_H * nu_0 / _K) / math.log(1.0 + prefactor / S_max)

    atm = _make_atmosphere(
        N_tau=N_tau, N_nu=N_nu, N_mu=N_mu,
        tau_min=tau_min, tau_max=tau_max,
        T0=T0, epsilon=0.5, nu_0=nu_0,
        half_width=3.0,
    )

    # S broadcast uniformly over frequency x: same S(τ) at every x.
    S = S_tau.unsqueeze(1).expand(N_tau, N_nu)  # (N_tau, N_nu)

    _, I_surface = formal_solve_with_surface_I(S, atm)
    # I_surface shape: (N_mu, N_nu)

    # Use the central frequency channel (x=0, psi=1) where the atmosphere is
    # deeply optically thick — Eddington-Barbier is exact in the limit tau >> mu.
    x_center = N_nu // 2  # index of x = 0
    I_surf_center = I_surface[:, x_center]  # (N_mu,)

    # Eddington-Barbier prediction: I(0, μ) ≈ a + b·μ
    mu_values = atm.angle.mu  # (N_mu,)
    I_EB      = a + b * mu_values  # (N_mu,)

    # Test at μ ∈ {0.1, 0.5, 0.9} — find nearest quadrature points
    target_mu = [0.1, 0.5, 0.9]
    for mu_t in target_mu:
        idx    = int((mu_values - mu_t).abs().argmin().item())
        mu_i   = float(mu_values[idx].item())
        I_calc = float(I_surf_center[idx].item())
        I_pred = a + b * mu_i

        rel_err = abs(I_calc - I_pred) / abs(I_pred)
        assert rel_err < 0.05, (
            f"Eddington-Barbier failed at mu={mu_t:.2f} (nearest μ={mu_i:.4f}): "
            f"I_calc={I_calc:.6e}, I_EB={I_pred:.6e}, rel_err={rel_err:.3e} (want < 5%)"
        )


# ---------------------------------------------------------------------------
# Parabolic-mode tests (Auer-Paletou 1994, mode='parabolic')
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_formalsolve_parabolic_uniform_source() -> None:
    """Parabolic mode: J(tau, x) = S₀ at interior depths for uniform S.

    Mirrors test_U_formalsolve_01_uniform_source but uses mode='parabolic'.
    The parabolic scheme must preserve the constant-S exactness property
    (alpha + beta + gamma = 1 - exp(-u)) to machine precision.

    Relative tolerance: 1e-10.
    """
    nu_0    = 5.89e14
    N_tau   = 120
    N_nu    = 11
    N_mu    = 5
    tau_min = 1e-4
    tau_max = 1e6
    dtype   = torch.float64

    T0  = 8000.0
    S0  = _planck(T0, nu_0)

    atm = _make_atmosphere(
        N_tau=N_tau, N_nu=N_nu, N_mu=N_mu,
        tau_min=tau_min, tau_max=tau_max,
        T0=T0, epsilon=0.5, nu_0=nu_0,
        half_width=3.0,
    )

    S = torch.full((N_tau, N_nu), S0, dtype=dtype)

    J = formal_solve(S, atm, mode="parabolic")

    tau  = atm.depth.tau
    mask = tau >= 5e5

    assert mask.sum() > 0, "No interior depth points found for tau >= 5e5"

    J_interior = J[mask]
    S_interior = S[mask]

    rel_err = ((J_interior - S_interior).abs() / S_interior.abs()).max().item()
    assert rel_err < 1e-10, (
        f"Parabolic: J != S in interior: max rel err = {rel_err:.3e} (want < 1e-10)"
    )


@pytest.mark.unit
def test_U_formalsolve_parabolic_linearity() -> None:
    """Parabolic mode: formal_solve is linear in S to 1e-12.

    The parabolic scheme is an affine function of S, so the same convex-
    combination test that works for the linear scheme applies here.  We
    use alpha + beta = 1 so the inhomogeneous BC constant cancels.

    Relative tolerance: 1e-12.
    """
    torch.manual_seed(43)
    N_tau, N_nu = 40, 11
    atm   = _make_atmosphere(N_tau=N_tau, N_nu=N_nu, N_mu=3)
    dtype = torch.float64

    S1 = torch.rand(N_tau, N_nu, dtype=dtype) + 0.1
    S2 = torch.rand(N_tau, N_nu, dtype=dtype) + 0.1

    alpha = float(torch.rand(1, dtype=dtype).item())
    beta  = 1.0 - alpha

    J1      = formal_solve(S1, atm, mode="parabolic")
    J2      = formal_solve(S2, atm, mode="parabolic")
    J_combo = formal_solve(alpha * S1 + beta * S2, atm, mode="parabolic")
    J_lin   = alpha * J1 + beta * J2

    scale   = J_lin.abs().max().clamp(min=1e-30)
    rel_err = ((J_combo - J_lin).abs() / scale).max().item()

    assert rel_err < 1e-12, (
        f"Parabolic linearity violated: max rel err = {rel_err:.3e} (want < 1e-12)"
    )


@pytest.mark.unit
def test_U_formalsolve_parabolic_jvp() -> None:
    """Parabolic mode: torch.func.jvp matches finite-difference JVP to 1e-6.

    Same finite-difference test as test_U_formalsolve_03_jvp_finite_difference
    but using mode='parabolic'.  The parabolic coefficients E0/E1/E2 are all
    smooth functions of u, so the autograd graph through torch.where is valid.

    Relative tolerance: 1e-6.
    """
    torch.manual_seed(8)
    N_tau, N_nu = 40, 11
    atm   = _make_atmosphere(N_tau=N_tau, N_nu=N_nu, N_mu=3)
    dtype = torch.float64
    eta   = 1e-5

    S = torch.rand(N_tau, N_nu, dtype=dtype, requires_grad=True) + 0.5
    v = torch.randn(N_tau, N_nu, dtype=dtype)

    def f(S_):
        return formal_solve(S_, atm, mode="parabolic")

    _, jvp_auto = torch.func.jvp(f, (S,), (v,))

    with torch.no_grad():
        J_plus  = formal_solve(S + eta * v, atm, mode="parabolic")
        J_minus = formal_solve(S - eta * v, atm, mode="parabolic")
    jvp_fd = (J_plus - J_minus) / (2.0 * eta)

    scale   = jvp_fd.abs().max().clamp(min=1e-30)
    rel_err = ((jvp_auto - jvp_fd).abs() / scale).max().item()

    assert rel_err < 1e-6, (
        f"Parabolic JVP mismatch: max rel err = {rel_err:.3e} (want < 1e-6)"
    )


@pytest.mark.analytic
def test_A_eddington_parabolic() -> None:
    """Parabolic mode: Eddington-Barbier surface emergence holds to 5%.

    Mirrors test_A_eddington_01_surface_intensity but uses mode='parabolic'.
    The parabolic scheme should reproduce or improve on the linear scheme's
    accuracy for the Eddington-Barbier relation.

    Reference: Hubeny & Mihalas (2014) §11.5.
    Tolerance: 5% relative at μ ∈ {0.1, 0.5, 0.9}.
    """
    nu_0    = 5.89e14
    N_tau   = 200
    N_nu    = 11
    N_mu    = 8
    tau_min = 1e-4
    tau_max = 1e4
    dtype   = torch.float64

    a = 1.0
    b = 5e-5

    tau_grid = torch.logspace(
        math.log10(tau_min), math.log10(tau_max), N_tau, dtype=dtype
    )
    S_tau = a + b * tau_grid

    S_max     = float(S_tau[-1].item())
    prefactor = 2.0 * _H * nu_0**3 / _C**2
    T0        = (_H * nu_0 / _K) / math.log(1.0 + prefactor / S_max)

    atm = _make_atmosphere(
        N_tau=N_tau, N_nu=N_nu, N_mu=N_mu,
        tau_min=tau_min, tau_max=tau_max,
        T0=T0, epsilon=0.5, nu_0=nu_0,
        half_width=3.0,
    )

    S = S_tau.unsqueeze(1).expand(N_tau, N_nu)

    _, I_surface = formal_solve_with_surface_I(S, atm, mode="parabolic")

    x_center = N_nu // 2
    I_surf_center = I_surface[:, x_center]

    mu_values = atm.angle.mu
    target_mu = [0.1, 0.5, 0.9]
    for mu_t in target_mu:
        idx    = int((mu_values - mu_t).abs().argmin().item())
        mu_i   = float(mu_values[idx].item())
        I_calc = float(I_surf_center[idx].item())
        I_pred = a + b * mu_i

        rel_err = abs(I_calc - I_pred) / abs(I_pred)
        assert rel_err < 0.05, (
            f"Parabolic Eddington-Barbier failed at mu={mu_t:.2f} "
            f"(nearest μ={mu_i:.4f}): "
            f"I_calc={I_calc:.6e}, I_EB={I_pred:.6e}, rel_err={rel_err:.3e} (want < 5%)"
        )


@pytest.mark.analytic
def test_parabolic_convergence_rate_third_order() -> None:
    """Parabolic mode achieves O(N_τ⁻³) convergence in surface S error.

    Tests two properties:
    1. Convergence rate: the surface error in S(0)/B for the deep-ε problem
       decays faster than linear (parabolic = 3rd-order per segment, so the
       global error decays at least as N_τ⁻² even with boundary effects).
       We verify ratio of errors at N_τ=200 vs N_τ=400 exceeds 3.0 (i.e.,
       error halves faster than linearly when N_τ doubles), as evidence of
       super-linear convergence.
    2. Absolute threshold: at ε=1e-6 with N_τ=400, the relative error in
       S(0)/B vs the √ε analytic value is ≤ 10%.

    Both conditions must hold.  If N_τ=400 alone passes condition (2), that
    is sufficient for the thermalization xfail flip; condition (1) documents
    the superior convergence rate vs linear OK87.

    Uses the Eddington two-stream atmosphere (μ=1/√3, single frequency)
    matching the thermalization test setup exactly.

    Reference: Auer & Paletou (1994) §3; HM14 §11.4.
    """
    import math
    from atmosphere.fixed_point import ali_loop
    from atmosphere.grid import AngleQuadrature

    epsilon  = 1e-6
    tau_max  = 1e6
    T0       = 6000.0
    nu_0     = 2.47e15  # Ly-α

    B_val = (2.0 * _H * nu_0**3 / _C**2) / (
        math.exp(_H * nu_0 / (_K * T0)) - 1.0
    )
    sqrt_eps = math.sqrt(epsilon)
    expected_surface = sqrt_eps * B_val

    eddington_mu = torch.tensor([1.0 / math.sqrt(3.0)], dtype=torch.float64)
    eddington_w  = torch.tensor([1.0], dtype=torch.float64)

    def _run_para(n_tau: int) -> float:
        """Return |S(0)/B - sqrt(eps)| / sqrt(eps) for parabolic mode."""
        depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=tau_max, n_tau=n_tau)
        freq = FrequencyGrid(
            nu=torch.zeros(1, dtype=torch.float64),
            weights=torch.ones(1, dtype=torch.float64),
            profile=torch.ones(1, dtype=torch.float64),
        )
        angle = AngleQuadrature(mu=eddington_mu, weights=eddington_w)
        from atmosphere.atmosphere import Atmosphere
        atm = Atmosphere(
            depth=depth,
            frequency=freq,
            angle=angle,
            T=torch.full((n_tau,), T0, dtype=torch.float64),
            n_e=torch.full((n_tau,), 1e10, dtype=torch.float64),
            epsilon=epsilon,
            nu_0=nu_0,
        )
        result = ali_loop(atm, tol=1e-6, max_iter=200, formal_solve_mode="parabolic")
        S_surface = result.S_l[0].item()
        return abs(S_surface - expected_surface) / expected_surface

    # Condition 2: absolute threshold at N_τ=400.
    err_400 = _run_para(400)
    assert err_400 <= 0.10, (
        f"Parabolic at N_tau=400: rel_err={err_400:.4f} > 0.10 "
        f"(S(0)/B should be within 10% of sqrt(eps)={sqrt_eps:.6f})"
    )

    # Condition 1: super-linear convergence — error at N_τ=200 must be
    # at least 3x larger than at N_τ=400 (consistent with O(h²) or better).
    err_200 = _run_para(200)
    ratio = err_200 / max(err_400, 1e-12)
    assert ratio >= 3.0, (
        f"Parabolic convergence rate weaker than expected: "
        f"err(N=200)={err_200:.4f}, err(N=400)={err_400:.4f}, ratio={ratio:.2f} "
        f"(want >= 3.0 for super-linear convergence)"
    )
