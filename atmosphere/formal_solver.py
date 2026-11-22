"""Short-characteristic formal solver, autograd-native. the technical reference §3.2.

The solver is a pure function of source-function S and atmosphere geometry,
returning the mean intensity J_nu(tau).  Because PyTorch records the graph,
JVPs of the host RT step come for free via torch.func.jvp.

Method: Olson-Kunasz (1987) short-characteristics with linear interpolation
of S along each segment (default ``mode='linear'``).  A parabolic-source
variant (Auer & Paletou 1994, Kunasz & Auer 1988, HM14 §11.4) is available
via ``mode='parabolic'``; it uses a quadratic fit through three consecutive
depth points, giving O(Δτ³) per-segment accuracy.  the technical reference §3.2 designates
this as a "configurable upgrade (off by default for the prototype)."

Only the linear mode has nonnegative transport coefficients and a passive
maximum principle for nonnegative boundary data. The legacy parabolic mode
extrapolates upstream, has negative coefficients, and is retained for explicit
diagnostics; it must not be assumed positive or used as a passivity benchmark.

See docs/reference_guide/grail_reference_guide.tex §6 for the linear-scheme derivation.

Boundary conditions (the technical reference §5.4 / reference guide §5.4):
  - Surface (tau_min):  I⁻(tau_min, mu) = 0  (no incoming radiation)
  - Deep (tau_max):     I⁺(tau_max, mu) = B(T_max) + mu * dB/dtau|_max
    For an isothermal atmosphere this reduces to I⁺ = B exactly.

Constants (SI):
  h  = 6.62607015e-34 J s
  c  = 2.99792458e8   m/s
  k  = 1.380649e-23   J/K
"""

from __future__ import annotations

import torch

from .atmosphere import Atmosphere

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant, J s
_C = 2.99792458e8     # speed of light, m/s
_K = 1.380649e-23     # Boltzmann constant, J/K


# ---------------------------------------------------------------------------
# Planck function
# ---------------------------------------------------------------------------

def _planck(T: torch.Tensor, nu_0: float) -> torch.Tensor:
    """Planck function B_nu(T) at frequency nu_0 (line center).

    B_nu = (2 h nu^3 / c^2) / (exp(h nu / k T) - 1)

    Parameters
    ----------
    T : Tensor, shape (N_tau,)
        Temperature at each depth point, Kelvin.
    nu_0 : float
        Line-centre frequency, Hz.

    Returns
    -------
    B : Tensor, shape (N_tau,)
        Planck function in SI units (W m⁻² Hz⁻¹ sr⁻¹).
    """
    prefactor = (2.0 * _H * nu_0**3) / (_C**2)
    x = (_H * nu_0) / (_K * T)
    return prefactor / (torch.exp(x) - 1.0)


# ---------------------------------------------------------------------------
# OK87 segment-integration coefficients (Olson & Kunasz 1987, eq. 3)
# ---------------------------------------------------------------------------

def _ok87_coefficients(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Olson-Kunasz 1987 linear-interpolation coefficients alpha(u), beta(u).

    For a segment of dimensionless optical depth u = psi(x) * Delta_tau / mu:
        I_b = I_a * exp(-u) + alpha(u) * S_a + beta(u) * S_b

    Exact forms (reference guide §6):
        alpha(u) = (1 - (1 + u) * exp(-u)) / u
        beta(u)  = (u - 1 + exp(-u)) / u

    Near u = 0 both are numerically unstable in their naive form; we switch to
    the Taylor-series limit (accurate to machine eps for u < 1e-6):
        alpha(u) ~ u/2 - u^2/3 + u^3/8  ...
        beta(u)  ~ u/2 - u^2/6 + u^3/24 ...

    We use a smooth blend via torch.where so the computation stays in the
    autograd graph without any .item() branching.

    Parameters
    ----------
    u : Tensor (any shape)
        Dimensionless optical depths, >= 0.

    Returns
    -------
    alpha, beta : Tensors of same shape as u.
    """
    exp_neg_u = torch.exp(-u)

    # Exact expressions (numerically problematic for small u)
    removed = -torch.expm1(-u)
    safe_u = u.clamp(min=torch.finfo(u.dtype).tiny)
    alpha_exact = (removed - u * exp_neg_u) / safe_u
    beta_exact = removed - alpha_exact

    # Taylor series to O(u^4) — accurate to ~1e-15 for u < 0.01
    alpha_small = u * (0.5 - u * (1.0/3.0 - u * (1.0/8.0 - u / 30.0)))
    beta_small  = u * (0.5 - u * (1.0/6.0 - u * (1.0/24.0 - u / 120.0)))

    # Switch at u = 1e-3 (safe margin; exact formula is fine above ~1e-3)
    small = u < 1e-3
    alpha = torch.where(small, alpha_small, alpha_exact)
    beta  = torch.where(small, beta_small,  beta_exact)

    return alpha, beta


# ---------------------------------------------------------------------------
# Parabolic-source coefficients (Auer & Paletou 1994, HM14 §11.4)
# ---------------------------------------------------------------------------

def _parabolic_coefficients(
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parabolic-source coefficients for Auer-Paletou (1994) short characteristics.

    For a segment from point "M" (upstream/deeper for upgoing rays) to point
    "O" (current target), with one further upstream point "U" beyond M, the
    parabolic short-characteristic update is:

        I(O) = I(M) * exp(-u) + alpha * S_U + beta * S_M + gamma * S_O

    where u = slant optical depth from M to O along the ray, and
    v = slant optical depth from U to M along the ray.

    The coefficients are derived by Lagrange-interpolating the source function
    as a quadratic through (S_U, S_M, S_O) at optical-path positions
    (-v, 0, +u) relative to M, then integrating analytically against the
    exponential propagator.  In terms of the standard moment integrals:

        E0 = 1 - exp(-u)
        E1 = integral_0^u t exp(-t) dt = 1 - (1 + u) * exp(-u)
        E2 = integral_0^u t^2 exp(-t) dt = 2 - (u^2 + 2u + 2) * exp(-u)

    Here t is distance measured backwards from the target O. Thus the
    interpolation nodes in this integration coordinate are O=0, M=u,
    U=u+v. These are not forward-path moments int s exp(-(u-s)) ds.

    the coefficients are (derived from first principles; verified analytically):

        alpha = (E2 - u * E1) / (v * (u + v))
        beta  = ((u + v) * E1 - E2) / (u * v)
        gamma = ((u^2 + u*v) * E0 - (2*u + v) * E1 + E2) / (u * (u + v))

    Consistency: alpha + beta + gamma = E0 = 1 - exp(-u)  [uniform-S limit].

    Note: the spec's e_1 = u - e_0 and e_2 = u^2 - 2*e_1 are DIFFERENT from
    E1 and E2 above.  The formulas here are derived directly from the Lagrange
    integral, not from the AP94 paper's notation (which uses a different
    definition of the e-moments than the spec).

    Near u = 0 the exact forms suffer catastrophic cancellation in E1 and E2;
    Taylor series are used for u < 1e-3 to maintain full float64 precision.
    The blend is implemented via torch.where so the computation stays in the
    autograd graph without .item() branching.

    Parameters
    ----------
    u : Tensor (any shape)
        Slant optical depth of the current segment (M to O), >= 0.  This is
        psi(x) * dtau / mu for frequency x, depth-spacing dtau, angle mu.
    v : Tensor, same shape as u
        Slant optical depth of the previous segment (U to M), > 0.
        The caller is responsible for falling back to linear OK87 for the
        boundary segment where U does not exist (v = 0 is a precondition
        violation here).

    Returns
    -------
    alpha, beta, gamma : Tensors of same shape as u, float64.
        Coefficients for S_U, S_M, S_O respectively.
    """
    exp_neg_u = torch.exp(-u)

    # Standard moment integrals (exact, numerically unstable near u=0):
    #   E0 = int_0^u exp(-(u-s)) ds = 1 - exp(-u)
    #   E1 = int_0^u t exp(-t) dt = 1 - (1+u)*exp(-u)
    #   E2 = int_0^u t^2 exp(-t) dt = 2 - (u^2+2u+2)*exp(-u)
    E0_exact = -torch.expm1(-u)
    E1_exact = 1.0 - (1.0 + u) * exp_neg_u
    E2_exact = 2.0 - (u * u + 2.0 * u + 2.0) * exp_neg_u

    # E_n = integral_0^u t^n exp(-t) dt. The old series used the
    # coefficients of different moments and did not match the exact branch.
    # (E0 = u - u^2/2 + u^3/6 - ... has no cancellation issue)
    E1_small = u**2 * (0.5 - u * (1/3 - u * (1/8 - u * (1/30 - u * (1/144 - u/840)))))
    E2_small = u**3 * (1/3 - u * (1/4 - u * (1/10 - u * (1/36 - u * (1/168 - u/960)))))

    small_u = u < 1e-2
    E0 = E0_exact  # always stable
    E1 = torch.where(small_u, E1_small, E1_exact)
    E2 = torch.where(small_u, E2_small, E2_exact)

    # Safe denominators (v > 0 is a precondition; clamp for autograd safety).
    v_safe = v.clamp(min=1e-300)
    u_safe = u.clamp(min=1e-300)

    denom_alpha = v_safe * (u_safe + v_safe)    # v*(u+v)
    denom_beta  = u_safe * v_safe               # u*v
    denom_gamma = u_safe * (u_safe + v_safe)    # u*(u+v)

    alpha = (E2 - u * E1) / denom_alpha
    beta  = ((u + v) * E1 - E2) / denom_beta
    gamma = ((u * u + u * v) * E0 - (2.0 * u + v) * E1 + E2) / denom_gamma

    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# Core sweeps
# ---------------------------------------------------------------------------

def _sweep_upgoing(
    S: torch.Tensor,
    tau: torch.Tensor,
    psi: torch.Tensor,
    mu: torch.Tensor,
    I_bottom: torch.Tensor,
    mode: str = "linear",
) -> torch.Tensor:
    """Integrate I⁺ from tau_max (index 0 in reversed order) to tau_min.

    The depth array `tau` is ordered surface-to-deep: tau[0] < tau[1] < ...
    Upgoing rays travel from deep to surface, i.e. from index N-1 down to 0.
    We iterate in reverse: start at i = N_tau-1 with boundary I_bottom, then
    step to i = N_tau-2, ..., 0.

    Parameters
    ----------
    S : Tensor, shape (N_tau, N_nu)
        Source function.
    tau : Tensor, shape (N_tau,)
        Optical depth grid (monotonically increasing).
    psi : Tensor, shape (N_nu,)
        Line profile phi(x) evaluated at each frequency point.
    mu : Tensor, shape ()
        Angle cosine (scalar), in (0, 1].
    I_bottom : Tensor, shape (N_nu,)
        Incoming intensity at tau_max (deep boundary).
    mode : str, default 'linear'
        'linear' for OK87 linear interpolation; 'parabolic' for Auer-Paletou
        1994 quadratic interpolation (O(Δτ³) per segment).

    Returns
    -------
    I_up : Tensor, shape (N_tau, N_nu)
        Upgoing specific intensity at every depth.
    """
    N_tau, N_nu = S.shape

    # Delta tau for each segment: dtau[i] = tau[i+1] - tau[i], shape (N_tau-1,)
    dtau = tau[1:] - tau[:-1]  # (N_tau-1,)

    # Dimensionless optical depths per segment per frequency:
    # u[i, x] = psi[x] * dtau[i] / mu,  shape (N_tau-1, N_nu)
    u = dtau.unsqueeze(1) * psi.unsqueeze(0) / mu  # (N_tau-1, N_nu)

    alpha_lin, beta_lin = _ok87_coefficients(u)  # (N_tau-1, N_nu) each
    exp_neg_u           = torch.exp(-u)           # (N_tau-1, N_nu)

    # Build list of I values at each depth, iterating from deep to surface.
    # I_list[k] holds I at depth index (N_tau - 1 - k), so:
    #   k=0 -> depth N_tau-1 (deepest)
    #   k=1 -> depth N_tau-2
    #   ...
    #   k=N_tau-1 -> depth 0 (surface)
    I_list: list[torch.Tensor] = [I_bottom]

    if mode == "linear":
        for k in range(N_tau - 1):
            # Going from depth index (N_tau-1-k) to (N_tau-2-k)
            # Segment index i = N_tau-2-k  (the segment between depth i and i+1)
            i = N_tau - 2 - k
            I_a = I_list[-1]          # intensity at deeper point (a = tau_a > tau_b)
            S_a = S[i + 1]            # source at deeper point, shape (N_nu,)
            S_b = S[i]                # source at shallower point, shape (N_nu,)
            I_b = (I_a * exp_neg_u[i]
                   + alpha_lin[i] * S_a
                   + beta_lin[i]  * S_b)
            I_list.append(I_b)

    else:  # parabolic mode
        # Upgoing rays travel from deep (index N_tau-1) toward surface (index 0).
        # For step k we move from depth d_M = N_tau-1-k to d_O = N_tau-2-k.
        # The "upstream" point U (one step further into the ray origin) is at
        # d_U = N_tau-k (deeper still), which only exists for k < N_tau-1.
        #
        # Segment index in u[]: segment i covers depths [i, i+1],
        #   so the segment from d_M to d_O has index i_MO = N_tau-2-k,
        #   and the segment from d_U to d_M has index i_UM = N_tau-1-k.
        #
        # For the first step (k=0) there is no U because d_U = N_tau is
        # out of range — fall back to linear OK87 for that segment.
        for k in range(N_tau - 1):
            i_MO = N_tau - 2 - k   # segment O <- M (current)
            d_M  = N_tau - 1 - k   # depth index of M
            d_O  = N_tau - 2 - k   # depth index of O (= i_MO)
            I_M  = I_list[-1]      # I at M, shape (N_nu,)

            if k == 0:
                # Boundary segment: no U exists — use linear OK87.
                I_b = (I_M * exp_neg_u[i_MO]
                       + alpha_lin[i_MO] * S[d_M]
                       + beta_lin[i_MO]  * S[d_O])
            else:
                # Parabolic: U is one step deeper than M.
                d_U   = N_tau - k      # depth index of U
                i_UM  = N_tau - 1 - k  # segment M <- U
                # v = optical depth of U->M segment
                v = u[i_UM]  # shape (N_nu,)
                # u_curr = optical depth of M->O segment
                u_curr = u[i_MO]  # shape (N_nu,)
                alpha_p, beta_p, gamma_p = _parabolic_coefficients(u_curr, v)
                I_b = (I_M * exp_neg_u[i_MO]
                       + alpha_p * S[d_U]
                       + beta_p  * S[d_M]
                       + gamma_p * S[d_O])

            I_list.append(I_b)

    # I_list is ordered deep -> surface; reverse to get surface -> deep
    # then stack into (N_tau, N_nu)
    I_up = torch.stack(list(reversed(I_list)), dim=0)  # (N_tau, N_nu)
    return I_up


def _sweep_downgoing(
    S: torch.Tensor,
    tau: torch.Tensor,
    psi: torch.Tensor,
    mu: torch.Tensor,
    mode: str = "linear",
) -> torch.Tensor:
    """Integrate I⁻ from tau_min (surface) to tau_max (deep), BC = 0.

    Downgoing rays start at the surface (tau[0]) with I = 0 and propagate
    inward (increasing tau index).

    Parameters
    ----------
    S : Tensor, shape (N_tau, N_nu)
        Source function.
    tau : Tensor, shape (N_tau,)
        Optical depth grid (monotonically increasing).
    psi : Tensor, shape (N_nu,)
        Line profile phi(x) evaluated at each frequency point.
    mu : Tensor, shape ()
        Angle cosine (scalar), in (0, 1].  We use |mu| here (downgoing).
    mode : str, default 'linear'
        'linear' for OK87 linear interpolation; 'parabolic' for Auer-Paletou
        1994 quadratic interpolation (O(Δτ³) per segment).

    Returns
    -------
    I_down : Tensor, shape (N_tau, N_nu)
        Downgoing specific intensity at every depth.
    """
    N_tau, N_nu = S.shape

    dtau = tau[1:] - tau[:-1]  # (N_tau-1,)
    u    = dtau.unsqueeze(1) * psi.unsqueeze(0) / mu  # (N_tau-1, N_nu)

    alpha_lin, beta_lin = _ok87_coefficients(u)
    exp_neg_u           = torch.exp(-u)

    # Boundary condition: no incoming radiation from above
    I_surface = torch.zeros(N_nu, dtype=S.dtype, device=S.device)

    I_list: list[torch.Tensor] = [I_surface]

    if mode == "linear":
        for i in range(N_tau - 1):
            # Going from depth index i to i+1 (increasing tau, downgoing)
            I_a = I_list[-1]  # intensity at shallower point
            S_a = S[i]        # source at shallower point
            S_b = S[i + 1]    # source at deeper point
            I_b = (I_a * exp_neg_u[i]
                   + alpha_lin[i] * S_a
                   + beta_lin[i]  * S_b)
            I_list.append(I_b)

    else:  # parabolic mode
        # Downgoing rays travel from surface (index 0) toward deep (index N_tau-1).
        # For step i we move from depth d_M = i to d_O = i+1.
        # The "upstream" point U (one step closer to the surface) is at d_U = i-1,
        # which only exists for i >= 1.
        #
        # Segment index in u[]: segment i covers depths [i, i+1].
        #   Current segment (M->O): index i_MO = i.
        #   Previous segment (U->M): index i_UM = i-1.
        #
        # For the first step (i=0) there is no U — fall back to linear OK87.
        for i in range(N_tau - 1):
            I_M = I_list[-1]   # I at M (shallower end), shape (N_nu,)

            if i == 0:
                # Boundary segment: no U exists — use linear OK87.
                I_b = (I_M * exp_neg_u[i]
                       + alpha_lin[i] * S[i]
                       + beta_lin[i]  * S[i + 1])
            else:
                # Parabolic: U is one step shallower than M.
                u_curr = u[i]      # optical depth M->O
                v      = u[i - 1]  # optical depth U->M
                alpha_p, beta_p, gamma_p = _parabolic_coefficients(u_curr, v)
                I_b = (I_M * exp_neg_u[i]
                       + alpha_p * S[i - 1]   # S_U (shallower)
                       + beta_p  * S[i]        # S_M
                       + gamma_p * S[i + 1])   # S_O (deeper)

            I_list.append(I_b)

    I_down = torch.stack(I_list, dim=0)  # (N_tau, N_nu)
    return I_down


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def formal_solve(
    S: torch.Tensor,
    atmosphere: Atmosphere,
    *,
    mode: str = "linear",
) -> torch.Tensor:
    """Short-characteristic formal solve. the technical reference §3.2.

    Integrates the radiative-transfer equation
        mu dI/dtau = phi(x) * (I - S)
    using the Olson-Kunasz (1987) short-characteristic scheme.

    Parameters
    ----------
    S : Tensor of shape (N_tau, N_nu), float64
        Source function on the (depth, frequency) grid.  Per assumption A6
        (CRD), S_l does not vary across the line; the same S is broadcast
        over frequency inside this function.
    atmosphere : Atmosphere
        Provides depth grid (tau), frequency grid (phi, weights), angle
        quadrature (mu, w_mu), temperature profile T, and nu_0.
    mode : str, default 'linear'
        Source-function interpolation mode along each ray segment.
        - 'linear'    : Olson-Kunasz (1987) linear interpolation.  O(Δτ²)
                        per-segment accuracy.  Default; preserves backward
                        compatibility with all existing tests.
        - 'parabolic' : Auer-Paletou (1994) quadratic interpolation through
                        three consecutive depth points.  O(Δτ³) per-segment
                        accuracy.  Boundary segments (where the upstream-of-M
                        point does not exist) fall back to linear OK87;
                        see the technical reference §3.2 (configurable upgrade).

    Returns
    -------
    J_nu : Tensor of shape (N_tau, N_nu), float64
        Angle-averaged mean intensity on the same grid,
        J(tau, x) = 1/2 * sum_i w_i * [I⁺(tau, x, mu_i) + I⁻(tau, x, mu_i)].
    """
    if mode not in {"linear", "parabolic"}:
        raise ValueError(f"unknown source interpolation mode {mode!r}")
    tau  = atmosphere.depth.tau          # (N_tau,)
    mu   = atmosphere.angle.mu           # (N_mu,)
    w_mu = atmosphere.angle.weights      # (N_mu,)
    T    = atmosphere.T                  # (N_tau,)

    # Doppler opacity profile psi(x) = exp(-x²), the technical reference §3.2.
    # x = (nu - nu_0) / Delta_nu_D is the dimensionless frequency offset stored
    # in FrequencyGrid.nu.  Note: psi(x) is the *unnormalised* Gaussian shape;
    # the normalised profile phi(x) = psi(x)/sqrt(pi) is only used in
    # line_profile_average for computing J_bar — not here.
    freq = atmosphere.frequency
    x    = freq.nu.to(dtype=S.dtype, device=S.device)   # (N_nu,)
    psi  = torch.exp(-x * x)                             # (N_nu,), psi(x) = exp(-x²)

    # Planck function B(T) at line centre, shape (N_tau,)
    nu_0 = getattr(atmosphere, 'nu_0', 5.0e14)  # Hz; fallback to visible
    B    = _planck(T.to(dtype=S.dtype, device=S.device), nu_0)

    # Deep-boundary upgoing intensity: I⁺(tau_max, mu) = B(T_max) + mu * dB/dtau
    # For an isothermal atmosphere dB/dtau = 0, so I⁺ = B exactly.
    T_is_constant = bool((T == T[0]).all())

    if T_is_constant:
        # Isothermal: diffusion BC reduces to I⁺ = B(T_max) for all mu
        B_max   = B[-1]                          # scalar
        # Shape broadcast over mu and nu handled inside the sweep
        dB_dtau = torch.zeros_like(B[-1])        # (scalar)
    else:
        # One-sided finite difference dB/dtau at tau_max
        # Use the last two depth points: dB/dtau ~ (B[-1] - B[-2]) / (tau[-1] - tau[-2])
        dB_dtau = (B[-1] - B[-2]) / (tau[-1] - tau[-2])
        B_max   = B[-1]

    # Accumulate J over angles
    J = torch.zeros_like(S)  # (N_tau, N_nu)

    for i in range(len(mu)):
        mu_i = mu[i]   # scalar tensor
        w_i  = w_mu[i]

        # Deep boundary for upgoing ray: I⁺(tau_max) = B_max + mu_i * dB_dtau
        # shape (N_nu,) — broadcast B_max and dB_dtau (scalars) over frequency
        I_bottom = (B_max + mu_i * dB_dtau).expand(S.shape[1])

        I_up   = _sweep_upgoing(S, tau, psi, mu_i, I_bottom, mode=mode)  # (N_tau, N_nu)
        I_down = _sweep_downgoing(S, tau, psi, mu_i, mode=mode)          # (N_tau, N_nu)

        J = J + w_i * (I_up + I_down)

    # Factor of 1/2 from the symmetry of the quadrature (mu in [0,1] covers
    # half the full solid angle; the weights sum to 1 over [0,1], so the
    # trapezoidal-style 1/2 factor is already absorbed if weights sum to 1.
    # The standard convention: J = (1/2) sum_i w_i [I⁺ + I⁻].
    J = 0.5 * J

    return J


def formal_solve_with_surface_I(
    S: torch.Tensor,
    atmosphere: Atmosphere,
    *,
    mode: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like formal_solve but also returns the upgoing surface intensity.

    This variant is used by the Eddington-Barbier analytic test
    (test_A_eddington_01).  It returns the upgoing intensity I⁺ at the
    surface (tau[0] = tau_min) for each (mu, x), which is what an observer
    at the stellar surface would measure.

    Parameters
    ----------
    S : Tensor of shape (N_tau, N_nu), float64
    atmosphere : Atmosphere
    mode : str, default 'linear'
        Same as in formal_solve — 'linear' (OK87) or 'parabolic' (AP94).

    Returns
    -------
    J_nu : Tensor of shape (N_tau, N_nu)
        Mean intensity (same as formal_solve).
    I_surface : Tensor of shape (N_mu, N_nu)
        Upgoing intensity I⁺(tau_min, mu, x) for each angle and frequency.
    """
    tau  = atmosphere.depth.tau
    mu   = atmosphere.angle.mu
    w_mu = atmosphere.angle.weights
    T    = atmosphere.T

    freq = atmosphere.frequency
    x    = freq.nu.to(dtype=S.dtype, device=S.device)   # (N_nu,)
    psi  = torch.exp(-x * x)                             # psi(x) = exp(-x²)

    nu_0 = getattr(atmosphere, 'nu_0', 5.0e14)
    B    = _planck(T.to(dtype=S.dtype, device=S.device), nu_0)

    T_is_constant = bool((T == T[0]).all())
    if T_is_constant:
        B_max   = B[-1]
        dB_dtau = torch.zeros_like(B[-1])
    else:
        dB_dtau = (B[-1] - B[-2]) / (tau[-1] - tau[-2])
        B_max   = B[-1]

    J = torch.zeros_like(S)
    I_surface_list: list[torch.Tensor] = []

    for i in range(len(mu)):
        mu_i = mu[i]
        w_i  = w_mu[i]

        I_bottom = (B_max + mu_i * dB_dtau).expand(S.shape[1])

        I_up   = _sweep_upgoing(S, tau, psi, mu_i, I_bottom, mode=mode)
        I_down = _sweep_downgoing(S, tau, psi, mu_i, mode=mode)

        J = J + w_i * (I_up + I_down)

        # Surface upgoing intensity: I_up at depth index 0 (tau_min)
        I_surface_list.append(I_up[0])  # shape (N_nu,)

    J = 0.5 * J
    I_surface = torch.stack(I_surface_list, dim=0)  # (N_mu, N_nu)

    return J, I_surface


def line_profile_average(J_nu: torch.Tensor, atmosphere: Atmosphere) -> torch.Tensor:
    """Profile-weighted mean intensity. the technical reference §3.2 closure.

    J_bar(tau) = sum_i weights_i * profile_i * J_nu(tau, x_i)

    Parameters
    ----------
    J_nu : Tensor of shape (N_tau, N_nu), float64
        Frequency-resolved mean intensity.
    atmosphere : Atmosphere
        Must provide frequency.weights and frequency.profile tensors.

    Returns
    -------
    J_bar : Tensor of shape (N_tau,), float64
        Profile-weighted mean intensity at each depth.
    """
    weights = atmosphere.frequency.weights.to(dtype=J_nu.dtype, device=J_nu.device)

    if hasattr(atmosphere.frequency, 'profile'):
        profile = atmosphere.frequency.profile.to(dtype=J_nu.dtype, device=J_nu.device)
    else:
        # Fall back to Doppler profile normalised to unit integral
        x       = atmosphere.frequency.nu.to(dtype=J_nu.dtype, device=J_nu.device)
        profile = torch.exp(-x * x)
        # Normalise: profile / (sum of profile * weights) so it integrates to 1
        norm    = (profile * weights).sum()
        profile = profile / norm

    # J_bar[tau] = sum_x weights[x] * profile[x] * J_nu[tau, x]
    # (N_tau, N_nu) @ (N_nu,) -> (N_tau,)
    J_bar = (J_nu * profile.unsqueeze(0) * weights.unsqueeze(0)).sum(dim=1)
    return J_bar
