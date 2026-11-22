"""ALI outer loop and the cycle_truth / cycle_emulator step functions. the technical reference §3.4.

The state vector x = (n_1, n_2) stacked over depth has shape (N_tau, 2).
One cycle is T = T_RT ∘ T_SE: populations -> J_bar -> updated populations.

ALI (Accelerated Lambda Iteration) uses the diagonal Λ* preconditioner
following Olson, Auer & Buchler (1986, JQSRT 35, 431).  The iteration
operates on the line source function S_l and derives updated populations
from the converged S_l via the rate-equation solver at the end.

References
----------
- Olson, G. L., Auer, L. H., & Buchler, J. R. (1986), JQSRT 35, 431.
  Diagonal Λ* preconditioner.
- Hubeny, I. & Mihalas, D. (2014), Theory of Stellar Atmospheres, §13.
  Thermalization depth, ALI convergence.
- Mihalas, D. (1978), Stellar Atmospheres, 2nd ed., §6-3 eq.6-39.
  Coherent two-stream analytic limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..atmosphere import Atmosphere
from ..formal_solver import formal_solve, line_profile_average
from .rate_equations import rate_equation_solve

# ---------------------------------------------------------------------------
# Physical constants (SI; used for Planck function)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant [J s]
_C = 2.99792458e8     # speed of light [m/s]
_KB = 1.380649e-23    # Boltzmann constant [J/K]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AliResult:
    """Converged output from :func:`ali_loop`.

    Attributes
    ----------
    populations : Tensor of shape (N_tau, 2)
        Converged level populations (n_1, n_2) at each depth.
    S_l : Tensor of shape (N_tau,)
        Converged line source function.
    J_bar : Tensor of shape (N_tau,)
        Converged profile-weighted mean intensity.
    iter_count : int
        Number of ALI iterations performed before convergence (or max_iter).
    residual_history : list[float]
        max |ΔS/S| at each iteration; useful for diagnosing stalling.
    """

    populations: torch.Tensor
    S_l: torch.Tensor
    J_bar: torch.Tensor
    iter_count: int
    residual_history: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planck function helpers
# ---------------------------------------------------------------------------

def _planck_nu(nu_0: float, T: torch.Tensor) -> torch.Tensor:
    """Planck function B_nu(T) at a single frequency nu_0.

    B_nu = (2 h nu^3 / c^2) / (exp(h nu / kT) - 1)

    Parameters
    ----------
    nu_0 : float
        Line center frequency [Hz].
    T : Tensor of shape (N_tau,)
        Temperature at each depth [K].

    Returns
    -------
    B : Tensor of shape (N_tau,)
    """
    x = _H * nu_0 / (_KB * T)
    return (2.0 * _H * nu_0**3 / _C**2) / (torch.exp(x) - 1.0)


# ---------------------------------------------------------------------------
# Source function from populations (Einstein-relation form)
# ---------------------------------------------------------------------------

def _source_from_populations(
    populations: torch.Tensor,
    atmosphere: Atmosphere,
) -> torch.Tensor:
    """Compute line source function from level populations.

    For a two-level atom with complete redistribution the line source
    function is

        S_l(τ) = (n_2 A_21) / (n_1 B_12 - n_2 B_21)

    Using the Einstein relations (A_21 = (2hν³/c²) B_21 and
    B_12 = (g_2/g_1) B_21) this reduces to

        S_l(τ) = B_★ / (n_1 g_2 / (n_2 g_1) - 1)

    where B_★ = 2hν₀³/c².  In practice, the ε-formulation gives the
    equivalent result S_l = (1-ε) J_bar + ε B(T) at the fixed point,
    but to compute S_l *from* populations (as required inside the cycle
    before we have J_bar) we use the Einstein-relation form.

    Here we assume g_2/g_1 = 1 (ground-state degeneracy ratio unity is
    the conventional prototype default; atmosphere implementers may
    override via atmosphere.g_ratio if that attribute is present).

    Parameters
    ----------
    populations : Tensor of shape (N_tau, 2)
        (n_1, n_2) at each depth.
    atmosphere : Atmosphere
        Must expose atmosphere.nu_0 (line-center frequency [Hz]).

    Returns
    -------
    S_l : Tensor of shape (N_tau,)
    """
    n1 = populations[:, 0]  # lower level
    n2 = populations[:, 1]  # upper level

    # Degeneracy ratio g2/g1; default 1 if not stored on atmosphere.
    g_ratio = getattr(atmosphere, "g_ratio", 1.0)

    nu_0: float = atmosphere.nu_0

    # B_★ = 2hν₀³/c²
    B_star = 2.0 * _H * nu_0**3 / _C**2

    # The positive-opacity two-level solver has no population-inverted
    # transport branch. Reject invalid states instead of creating an enormous
    # artificial source by flooring a negative/zero denominator. An exactly
    # empty upper level with positive lower level correctly gives S=0.
    denominator = g_ratio*n1-n2
    if (not bool(torch.isfinite(populations).all()) or bool((populations<0).any())
            or bool((denominator<=0).any())):
        raise ValueError("source requires finite nonnegative populations with g_ratio*n1 > n2")
    return B_star*n2/denominator


# ---------------------------------------------------------------------------
# Private shared cycle helper (Option B per spec decision)
# ---------------------------------------------------------------------------

def _cycle(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    se_solver: Callable,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT ∘ T_SE step shared by cycle_truth and cycle_emulator.

    Steps:
    1. Compute S_l from populations x via the Einstein-relation form.
    2. Broadcast S_l to all frequencies (CRD): S(τ, ν) = S_l(τ).
    3. Formal-solve S -> J_nu(τ, ν).
    4. Profile-average J_nu -> J_bar(τ).
    5. Call se_solver(J_bar, T, n_e) -> updated populations (N_tau, 2).

    Parameters
    ----------
    x : Tensor of shape (N_tau, 2)
        Current populations (n_1, n_2).
    atmosphere : Atmosphere
    se_solver : callable
        Signature: (J_bar, T, n_e) -> Tensor of shape (N_tau, 2).
    formal_solve_mode : str, default 'linear'
        Passed to formal_solve as ``mode=``; 'linear' (OK87) or 'parabolic'.

    Returns
    -------
    new_x : Tensor of shape (N_tau, 2)
    """
    # Step 1: source function from populations
    S_l = _source_from_populations(x, atmosphere)  # (N_tau,)

    # Step 2: broadcast to frequency dimension (CRD: S independent of nu)
    # frequency grid has N_nu points.
    N_nu = atmosphere.frequency.nu.shape[0]
    S = S_l.unsqueeze(1).expand(-1, N_nu)  # (N_tau, N_nu)

    # Step 3: formal solve
    J_nu = formal_solve(S, atmosphere, mode=formal_solve_mode)  # (N_tau, N_nu)

    # Step 4: profile-weighted mean intensity
    J_bar = line_profile_average(J_nu, atmosphere)  # (N_tau,)

    # Step 5: updated populations from rate equations
    new_x = se_solver(J_bar, atmosphere.T, atmosphere.n_e)  # (N_tau, 2)
    return new_x


# ---------------------------------------------------------------------------
# Public cycle functions
# ---------------------------------------------------------------------------

def _make_reference_se_solver(atmosphere: Atmosphere) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]:
    """Build a 3-arg adapter that calls rate_equation_solve with epsilon and
    nu_0 captured from the atmosphere.

    The se_solver contract that _cycle and ali_loop consume is
    (J_bar, T, n_e) -> (N_tau, 2).  rate_equation_solve requires epsilon and
    nu_0 as keyword-only arguments; this closure binds them so the
    reference solver matches the contract.  The same adapter pattern is
    used at M1c to wrap the APNN's packed-batch interface into the same
    three-positional-arg shape.
    """
    eps = atmosphere.epsilon
    nu0 = atmosphere.nu_0

    def reference_solver(
        J_bar: torch.Tensor,
        T: torch.Tensor,
        n_e: torch.Tensor,
    ) -> torch.Tensor:
        return rate_equation_solve(J_bar, T, n_e, epsilon=eps, nu_0=nu0)

    return reference_solver


def cycle_truth(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT ∘ T_SE^ref cycle using the reference rate-equation solver.

    The fixed point x*_truth = cycle_truth(x*_truth, atmosphere) defines the
    ground-truth converged state. the technical reference §3.4.

    Parameters
    ----------
    x : Tensor of shape (N_tau, 2)
        Current populations (n_1, n_2) at each depth.
    atmosphere : Atmosphere
    formal_solve_mode : str, default 'linear'
        Passed to formal_solve; 'linear' (OK87) or 'parabolic' (AP94).

    Returns
    -------
    new_x : Tensor of shape (N_tau, 2)
    """
    return _cycle(
        x, atmosphere, _make_reference_se_solver(atmosphere),
        formal_solve_mode=formal_solve_mode,
    )


def cycle_emulator(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    emulator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT ∘ T_SE^APNN cycle using the APNN emulator in place of T_SE.

    The fixed point x*_emul = cycle_emulator(x*_emul, atmosphere, emulator)
    defines the emulator-converged state. the technical reference §3.4.

    Parameters
    ----------
    x : Tensor of shape (N_tau, 2)
        Current populations (n_1, n_2) at each depth.
    atmosphere : Atmosphere
    emulator : callable
        Signature: (J_bar: Tensor[N_tau], T: Tensor[N_tau], n_e: Tensor[N_tau])
        -> Tensor of shape (N_tau, 2).
        The caller is responsible for adapting the APNN's packed-batch input
        to this signature (see the technical reference §3.4 and §4.2).
    formal_solve_mode : str, default 'linear'
        Passed to formal_solve; 'linear' (OK87) or 'parabolic' (AP94).

    Returns
    -------
    new_x : Tensor of shape (N_tau, 2)
    """
    return _cycle(x, atmosphere, emulator, formal_solve_mode=formal_solve_mode)


# ---------------------------------------------------------------------------
# Diagonal Λ* computation (Olson-Auer-Buchler 1986)
# ---------------------------------------------------------------------------

def _compute_lambda_star_diagonal(
    atmosphere: Atmosphere,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """Compute the diagonal of the Λ operator (Λ* in ALI parlance).

    The diagonal Λ*[i] ≡ Λ[i,i] is computed by formal-solving a unit-pulse
    source function concentrated at depth i and extracting the profile-weighted
    mean intensity at that same depth.

    Cost: N_tau formal solves (one per depth point), each over the full
    frequency/angle quadrature.  This is called once per ali_loop invocation
    and the result is cached in the returned tensor.

    Following Olson, Auer & Buchler (1986, JQSRT 35, 431), §2.

    Parameters
    ----------
    atmosphere : Atmosphere
    formal_solve_mode : str, default 'linear'
        Passed to formal_solve; see formal_solver.py.

    Returns
    -------
    lambda_star : Tensor of shape (N_tau,)
        Diagonal elements Λ*[i] = Λ[i,i].
    """
    Lambda_full, _ = _compute_lambda_full(atmosphere, formal_solve_mode=formal_solve_mode)
    return torch.diagonal(Lambda_full).clone()


def _compute_lambda_full(
    atmosphere: Atmosphere,
    formal_solve_mode: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the full Λ matrix and the boundary-contribution vector.

    The formal solver is affine in S: J_bar = Λ S + b, where b is the
    contribution of the deep diffusion BC (which uses B(T_max), independent
    of S).  We extract Λ and b separately by:
      1. Running formal_solve with S=0 to get b (pure BC contribution).
      2. For each depth i, running formal_solve with the i-th unit pulse
         and subtracting b to get column i of Λ.

    Cost: (N_tau + 1) formal solves.

    Parameters
    ----------
    atmosphere : Atmosphere
    formal_solve_mode : str, default 'linear'
        Passed to formal_solve as ``mode=``; see formal_solver.py for options.

    Returns
    -------
    Lambda : Tensor of shape (N_tau, N_tau), float64
        Discretized line-profile-averaged Λ operator.
    b : Tensor of shape (N_tau,), float64
        Boundary-contribution vector: J_bar = Λ·S + b for any input S.
    """
    N_tau = atmosphere.depth.tau.shape[0]
    N_nu = atmosphere.frequency.nu.shape[0]

    # 1. Boundary contribution: J_bar(S=0).
    S_zero = torch.zeros(N_tau, N_nu, dtype=torch.float64)
    J_zero = formal_solve(S_zero, atmosphere, mode=formal_solve_mode)
    b = line_profile_average(J_zero, atmosphere)  # (N_tau,)

    # 2. Build Λ column-by-column.
    Lambda = torch.zeros(N_tau, N_tau, dtype=torch.float64)
    for j in range(N_tau):
        S_pulse = torch.zeros(N_tau, N_nu, dtype=torch.float64)
        S_pulse[j, :] = 1.0
        J_nu = formal_solve(S_pulse, atmosphere, mode=formal_solve_mode)
        J_bar = line_profile_average(J_nu, atmosphere)  # (N_tau,)
        Lambda[:, j] = J_bar - b
    return Lambda, b


# ---------------------------------------------------------------------------
# ALI driver
# ---------------------------------------------------------------------------

def _ng_acceleration(history: list[torch.Tensor]) -> torch.Tensor | None:
    """Ng (1974) acceleration for Picard-style fixed-point iteration.

    Given the four most recent iterates S^{n-3}, S^{n-2}, S^{n-1}, S^n,
    forms an optimal linear combination that minimizes the residual over
    the 2-dimensional Krylov subspace {Δ_2, Δ_3} where Δ_k = S^{n-k+1} - S^{n-k}.

    Reference: Auer (1991, Stellar Atmospheres: Beyond Classical Models, p. 9);
    Olson, Auer & Buchler (1986, JQSRT 35, 431) §3.3.

    Parameters
    ----------
    history : list of length 4
        Four most recent iterates ordered oldest-first.

    Returns
    -------
    S_acc : torch.Tensor or None
        Accelerated iterate, or None if the 2x2 system is singular.
    """
    S_nm3, S_nm2, S_nm1, S_n = history
    # First differences (residuals)
    d1 = S_n - S_nm1
    d2 = S_nm1 - S_nm2
    d3 = S_nm2 - S_nm3
    # Second differences  Δ²_k = d_k - d_{k+1}
    Q1 = d1 - d2
    Q2 = d2 - d3
    # Solve 2x2: [<Q1,Q1> <Q1,Q2>; <Q2,Q1> <Q2,Q2>] [a; b] = [<Q1,d1>; <Q2,d1>]
    A11 = (Q1 * Q1).sum()
    A12 = (Q1 * Q2).sum()
    A22 = (Q2 * Q2).sum()
    rhs1 = (Q1 * d1).sum()
    rhs2 = (Q2 * d1).sum()
    det = A11 * A22 - A12 * A12
    # Use relative threshold against A11*A22 because the absolute scale of Q
    # tracks the residual magnitude (which can be ~1e-7 of S ~ 1e-13 → Q ~ 1e-20).
    # Singular if Q1 and Q2 are parallel (det << A11*A22).
    if (A11 * A22).item() <= 0.0 or det.item() < 1e-12 * (A11 * A22).item():
        return None
    a = (A22 * rhs1 - A12 * rhs2) / det
    b = (A11 * rhs2 - A12 * rhs1) / det
    # Accelerated iterate
    S_acc = (1.0 - a - b) * S_n + a * S_nm1 + b * S_nm2
    return S_acc


def ali_loop(
    atmosphere: Atmosphere,
    se_solver: Callable | None = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 200,
    initial_S: torch.Tensor | None = None,
    use_ng: bool = True,
    ng_period: int = 4,
    ng_warmup: int = 4,
    mode: str = "direct",
    formal_solve_mode: str = "linear",
) -> AliResult:
    """Accelerated Lambda Iteration on the line source function.

    Implements the standard ALI scheme with diagonal Λ* preconditioner
    (Olson, Auer & Buchler 1986, JQSRT 35, 431) plus optional Ng (1974)
    acceleration on every ng_period-th iteration.

    The iteration update at step n is

        S^(n+1) = [I - (1-ε) Λ*]^{-1} [(1-ε)(Λ - Λ*) S^(n) + ε B]

    which is equivalent to

        S^(n+1) = S^(n) + [I - (1-ε) Λ*]^{-1} [J_bar^(n) - S^(n) + ε(B - S^(n))]

    where J_bar^(n) = Λ S^(n) is computed by formal solve.

    Convergence criterion: max_τ |S^(n+1) − S^(n)| / |S^(n)| < tol.

    Once converged, populations are derived from S_l via se_solver using
    J_bar at the final iteration.

    Parameters
    ----------
    atmosphere : Atmosphere
        Must expose: depth.tau, frequency, angle, T, n_e, epsilon, nu_0.
    se_solver : callable
        Signature: (J_bar, T, n_e) -> Tensor of shape (N_tau, 2).
    tol : float
        Convergence tolerance on max |ΔS/S|.  Default 1e-6 per the technical reference §3.4.
    max_iter : int
        Maximum number of ALI iterations.  Default 200.
    initial_S : Tensor of shape (N_tau,) or None
        Initial guess for the line source function.  If None, use B(T) (LTE
        guess), which is optimal for ε = 1 and a reasonable start elsewhere.
    use_ng : bool, default True
        Apply Ng acceleration once we have 4 iterates.  Disabling Ng falls
        back to plain ALI (useful for diagnostic comparison).  Only used in
        ``mode='ali'``; ignored in ``mode='direct'``.
    ng_period : int, default 4
        Apply Ng every ng_period iterations after warmup.
    ng_warmup : int, default 4
        Number of plain ALI iterations before the first Ng step (need at
        least 4 iterates for Ng).
    mode : {'direct', 'ali'}, default 'direct'
        How to solve the line-source-function fixed point:
          - 'direct' (default): build the full discrete Λ matrix once
            (N_tau+1 formal solves) and solve (I - (1-ε)Λ) S = (1-ε) b + ε B
            via numpy.linalg.solve.  Converges in one linear solve and is
            insensitive to ε; iter_count is reported as 1.  This is the
            preferred mode for the prototype because the two-level CRD
            problem at fixed T is exactly linear in S.
          - 'ali': legacy ALI iteration with diagonal Λ* and optional Ng
            acceleration.  Used only when the caller explicitly wants to
            exercise the iterative path.
    formal_solve_mode : str, default 'linear'
        Source-function interpolation mode for the formal solver:
          - 'linear'    : Olson-Kunasz 1987 linear interpolation (default).
          - 'parabolic' : Auer-Paletou 1994 quadratic interpolation.
            Needed for deep-ε (ε=1e-6) to achieve <10% error at N_τ=400.
            See the technical reference §3.2 (configurable upgrade) and formal_solver.py.

    Returns
    -------
    AliResult
    """
    # Default to the reference rate-equation solver when none is supplied.
    # Test code (and any caller without a custom emulator) gets the truth
    # solver wired up automatically.
    if se_solver is None:
        se_solver = _make_reference_se_solver(atmosphere)

    N_tau = atmosphere.depth.tau.shape[0]
    N_nu = atmosphere.frequency.nu.shape[0]
    epsilon: float = atmosphere.epsilon
    nu_0: float = atmosphere.nu_0

    # Ensure float64 throughout for numerical precision.
    T = atmosphere.T.to(torch.float64)
    n_e = atmosphere.n_e.to(torch.float64)

    # Planck function B(T) at line center.
    B = _planck_nu(nu_0, T)  # (N_tau,)

    # ------------------------------------------------------------------
    # Direct mode: build the full Λ matrix and solve the linear system
    # exactly.  The two-level CRD problem at fixed T is linear in S:
    #     J_bar = Λ S + b_BC,
    #     S    = (1-ε) J_bar + ε B
    # gives  (I - (1-ε)Λ) S = (1-ε) b_BC + ε B.
    # ------------------------------------------------------------------
    if mode == "direct":
        Lambda, b_BC = _compute_lambda_full(atmosphere, formal_solve_mode=formal_solve_mode)  # (N_tau,N_tau), (N_tau,)
        A = torch.eye(N_tau, dtype=torch.float64) - (1.0 - epsilon) * Lambda
        rhs = (1.0 - epsilon) * b_BC + epsilon * B
        S = torch.linalg.solve(A, rhs)  # (N_tau,)

        # Final J_bar with converged S (re-uses cached Λ, b_BC).
        J_bar_final = Lambda @ S + b_BC
        populations = se_solver(J_bar_final, T, n_e)

        # Compute residual for diagnostic compatibility: |S - rhs_final| / |S|.
        rhs_final = (1.0 - epsilon) * J_bar_final + epsilon * B
        residual = ((rhs_final - S).abs() / S.abs().clamp(min=1e-300)).max().item()

        return AliResult(
            populations=populations,
            S_l=S,
            J_bar=J_bar_final,
            iter_count=1,
            residual_history=[residual],
        )

    if mode != "ali":
        raise ValueError(f"mode must be 'direct' or 'ali', got mode={mode!r}")

    # ------------------------------------------------------------------
    # ALI iterative mode (legacy / diagnostic path).
    # ------------------------------------------------------------------

    # Initial source function guess.
    if initial_S is not None:
        S = initial_S.to(torch.float64).clone()
    else:
        # LTE guess: S^(0) = B(T).  Optimal for ε=1 (converges in one step).
        S = B.clone()

    # Compute diagonal Λ* once (costs N_tau+1 formal solves).
    lambda_star = _compute_lambda_star_diagonal(atmosphere, formal_solve_mode=formal_solve_mode)  # (N_tau,)

    # Precompute ALI denominator: [I - (1-ε) Λ*]^{-1} is scalar per depth.
    # denom[i] = 1 - (1-ε) λ*[i]
    denom = 1.0 - (1.0 - epsilon) * lambda_star  # (N_tau,)

    residual_history: list[float] = []

    # History buffer for Ng acceleration (need 4 iterates).
    history: list[torch.Tensor] = []

    for iteration in range(max_iter):
        # Formal solve with current S: J_bar = Λ S.
        S_full = S.unsqueeze(1).expand(-1, N_nu)  # (N_tau, N_nu); CRD
        J_nu = formal_solve(S_full, atmosphere, mode=formal_solve_mode)  # (N_tau, N_nu)
        J_bar = line_profile_average(J_nu, atmosphere)  # (N_tau,)

        # ALI update: S^(n+1) = S^(n) + [denom]^{-1} [J_bar - S + ε(B - S)]
        # This is the rearranged form of the standard ALI equation:
        #   S^(n+1) = [I - (1-ε)Λ*]^{-1} [(1-ε)(Λ-Λ*)S + εB]
        #           = [I - (1-ε)Λ*]^{-1} [(1-ε)(J_bar - λ*·S) + εB]
        # Simplest correct form (Hubeny & Mihalas 2014, eq. 13.43):
        #   ΔS = [(1-ε)·J_bar + ε·B - S] / denom
        delta_S = ((1.0 - epsilon) * J_bar + epsilon * B - S) / denom
        S_new = S + delta_S

        # Convergence check: max |ΔS / S|.
        safe_S = S.abs().clamp(min=1e-300)
        rel_change = (delta_S.abs() / safe_S).max().item()
        residual_history.append(rel_change)

        S = S_new

        # Maintain history of 4 most recent iterates for Ng.
        history.append(S.clone())
        if len(history) > 4:
            history.pop(0)

        # Ng acceleration: apply every ng_period iters after warmup.
        # Accept the Ng step only if it actually reduces the residual on the
        # next ALI iteration (probe-and-revert pattern).  Naive Ng often
        # destabilizes the iteration in the source-function problem.
        if (
            use_ng
            and len(history) == 4
            and iteration >= ng_warmup
            and (iteration - ng_warmup) % ng_period == 0
        ):
            S_acc = _ng_acceleration(history)
            if S_acc is not None and torch.isfinite(S_acc).all() and (S_acc > 0).all():
                # Probe: take one ALI step from S_acc and check if its
                # update magnitude is smaller than from plain S.
                S_full_acc = S_acc.unsqueeze(1).expand(-1, N_nu)
                J_nu_acc = formal_solve(S_full_acc, atmosphere, mode=formal_solve_mode)
                J_bar_acc = line_profile_average(J_nu_acc, atmosphere)
                delta_S_acc = (
                    (1.0 - epsilon) * J_bar_acc + epsilon * B - S_acc
                ) / denom
                ng_residual = (
                    delta_S_acc.abs() / S_acc.abs().clamp(min=1e-300)
                ).max().item()
                if ng_residual < rel_change:
                    S = S_acc + delta_S_acc  # Accept Ng + the probe step
                    history = [S.clone()]
                    residual_history.append(ng_residual)
                    if ng_residual < tol:
                        break

        if rel_change < tol:
            break

    # Final J_bar with converged S.
    S_full = S.unsqueeze(1).expand(-1, N_nu)
    J_nu_final = formal_solve(S_full, atmosphere, mode=formal_solve_mode)
    J_bar_final = line_profile_average(J_nu_final, atmosphere)

    # Derive converged populations from rate equations.
    populations = se_solver(J_bar_final, T, n_e)  # (N_tau, 2)

    return AliResult(
        populations=populations,
        S_l=S,
        J_bar=J_bar_final,
        iter_count=len(residual_history),
        residual_history=residual_history,
    )


# ---------------------------------------------------------------------------
# Picard iteration on populations (TASK-034, SPEC_threelevel §M3g)
# ---------------------------------------------------------------------------

@dataclass
class PicardResult2Level:
    """Output from :func:`picard_iterate_2level`.

    Parallel to PicardResult3Level (§M3d TASK-024) but on the
    2-level (N_tau, 2) state vector.

    Attributes
    ----------
    populations : Tensor of shape (N_tau, 2)
        Populations at the last Picard iterate (converged if converged=True).
    iter_count : int
        Number of Picard iterations performed.
    residual_history : list[float]
        Max |Δx| / |x| at each iteration (relative change in population vector).
    anderson_engaged : bool
        True if Anderson acceleration was triggered during the run.
    converged : bool
        True if the residual dropped below tol before max_iter.
    """

    populations: torch.Tensor
    iter_count: int
    residual_history: list[float] = field(default_factory=list)
    anderson_engaged: bool = False
    converged: bool = False


def _anderson_step_2level(
    x_history: list[np.ndarray],
    f_history: list[np.ndarray],
    m: int,
) -> np.ndarray:
    """Compute the Anderson AA(m) update for 2-level Picard iteration.

    Identical algorithm to picard_iterate_3level's _anderson_step in
    atmosphere/threelevel/fixed_point.py — Anderson acceleration is
    dimension-agnostic.  Reimplemented here to keep the 2-level module
    self-contained (no cross-package import).

    Parameters
    ----------
    x_history : list of np.ndarray
        The last min(k, m)+1 iterates (most recent last).
    f_history : list of np.ndarray
        The last min(k, m)+1 residuals f_i = T(x_i) - x_i.
    m : int
        History window size.

    Returns
    -------
    x_new : np.ndarray
        Anderson-accelerated next iterate.
    """
    m_actual = min(len(x_history), m + 1)
    xs = x_history[-m_actual:]
    fs = f_history[-m_actual:]

    if m_actual < 2:
        # Not enough history; fall back to plain Picard step.
        return xs[-1] + fs[-1]

    n = xs[0].shape[0]
    F = np.column_stack(fs)   # (n, m_actual)
    X = np.column_stack(xs)   # (n, m_actual)

    # Difference form: dF[:, i] = F[:, i+1] - F[:, i]
    dF = np.diff(F, axis=1)   # (n, m_actual - 1)
    dX = np.diff(X, axis=1)   # (n, m_actual - 1)

    f_k = F[:, -1]             # (n,)

    # Least-squares solve: dF @ gamma = -f_k.
    gamma, _, _, _ = np.linalg.lstsq(dF, -f_k, rcond=None)

    # Anderson update: x_new = x_k + f_k - (dX + dF) @ gamma.
    x_new = X[:, -1] + f_k - (dX + dF) @ gamma
    return x_new


def picard_iterate_2level(
    atmosphere: Atmosphere,
    se_solver: Callable,                    # SE2Solver (e.g. APNN adapter)
    *,
    initial_x: torch.Tensor | None = None,  # (N_tau, 2) or None (LTE init)
    tol: float = 1e-6,
    max_iter: int = 500,
    formal_solve_mode: str = "linear",
    accelerator: str = "none",              # "none" | "anderson"
    anderson_history: int = 5,
    anderson_trigger_window: int = 200,
    anderson_trigger_ratio: float = 0.1,
) -> PicardResult2Level:
    """Picard iteration x_{k+1} = T_e(x_k) on 2-level populations.

    Parallel to picard_iterate_3level (§M3d TASK-024).  Used by
    `I-resolvent-predicts-2level` (§M3g TASK-035, definition (ii)
    per §6.2.1) to obtain the genuine emulator-coupled fixed
    point for the 2-level Born-identity demonstration.

    The cycle step is T_e(x) = _cycle(x, atmosphere, se_solver,
    formal_solve_mode).  Picard residual is |Δx| / |x| (max relative
    change in the (N_tau, 2) population array).

    Anderson acceleration (numpy-only)
    ----------------------------------
    Anderson AA(m) with m = anderson_history.  Triggered automatically
    when accelerator="anderson" AND plain-Picard residual reduction
    is < anderson_trigger_ratio over the last anderson_trigger_window
    iterations — this catches deep-ε stalls where the cycle's spectral
    radius approaches 1.

    The 2-level state vector is (N_tau, 2); the flat blocked form used
    internally by Anderson is populations.T.flatten() → (2*N_tau,),
    mirroring the GRAIL seam convention.  The 2-level cycle has a
    scalar S_l and a scalar J_bar (vs. (N_tau, K) in 3-level); see
    _cycle() in this module.

    Anderson is implemented via numpy least-squares on the residual
    history; no scipy / no new dependencies (per NFR-001 and the
    M1b LHS sampler / TASK-024 precedent).

    Parameters
    ----------
    atmosphere : Atmosphere
        2-level atmosphere (must expose epsilon, nu_0, T, n_e, depth, etc.).
    se_solver : callable
        SE2Solver: (J_bar: Tensor[N_tau], T: Tensor[N_tau],
                    n_e: Tensor[N_tau]) -> Tensor[N_tau, 2].
        Typically the APNN adapter from make_cycle_adapter or the reference
        solver from _make_reference_se_solver.
    initial_x : Tensor of shape (N_tau, 2) or None
        Initial population guess.  If None, uses LTE populations from the
        reference SE solver at J_bar = B(T, nu_0) (LTE guess).
    tol : float, default 1e-6
        Convergence tolerance on max relative change ||Δx|| / ||x||_inf.
    max_iter : int, default 500
        Maximum number of Picard iterations.
    formal_solve_mode : str, default 'linear'
        Passed to _cycle (and hence to formal_solve).
    accelerator : str, default 'none'
        'none' for plain Picard; 'anderson' for Anderson AA(m) with
        automatic stall trigger.
    anderson_history : int, default 5
        History window m for Anderson AA(m).
    anderson_trigger_window : int, default 200
        Number of iterations over which to check for stall.
    anderson_trigger_ratio : float, default 0.1
        Residual reduction threshold: if residual_history[-1] >
        anderson_trigger_ratio * residual_history[-trigger_window],
        Anderson is triggered (residual dropped by less than 10× in window).

    Returns
    -------
    PicardResult2Level
        populations    : converged (or last) populations.
        iter_count     : total iterations performed.
        residual_history : per-iteration max relative change.
        anderson_engaged : whether Anderson acceleration was activated.
        converged       : True if tol was met before max_iter.

    SPEC_threelevel §M3g TASK-034.
    """
    if accelerator not in ("none", "anderson"):
        raise ValueError(
            f"accelerator must be 'none' or 'anderson', got {accelerator!r}."
        )

    N_tau = atmosphere.depth.tau.shape[0]
    T = atmosphere.T.to(torch.float64)
    n_e = atmosphere.n_e.to(torch.float64)

    # -----------------------------------------------------------------------
    # Initial guess: LTE populations if not provided.
    # LTE: call reference SE solver at J_bar = B(T, nu_0).
    # -----------------------------------------------------------------------
    if initial_x is None:
        J_bar_lte = _planck_nu(atmosphere.nu_0, T)   # (N_tau,)
        ref_solver = _make_reference_se_solver(atmosphere)
        x = ref_solver(J_bar_lte, T, n_e).detach().clone()  # (N_tau, 2)
    else:
        x = initial_x.to(torch.float64).detach().clone()

    residual_history: list[float] = []
    anderson_engaged = False
    converged = False

    # -----------------------------------------------------------------------
    # Anderson acceleration buffers (used only when anderson is triggered).
    # Stored as flat (2*N_tau,) numpy arrays for the AA computation.
    # -----------------------------------------------------------------------
    aa_x_history: list[np.ndarray] = []    # iterates
    aa_f_history: list[np.ndarray] = []    # residuals f(x) = T(x) - x
    aa_active = False

    for iteration in range(max_iter):
        # One cycle step: T_e(x).
        x_new = _cycle(
            x, atmosphere, se_solver,
            formal_solve_mode=formal_solve_mode,
        ).detach()  # (N_tau, 2)

        # Residual: f(x) = x_new - x (the fixed-point defect).
        f_x = x_new - x   # (N_tau, 2)

        # Convergence metric: max |Δx| / max(|x|, eps).
        safe_x = x.abs().clamp(min=1e-300)
        rel_change = (f_x.abs() / safe_x).max().item()
        residual_history.append(rel_change)

        # -----------------------------------------------------------------------
        # Anderson acceleration trigger check.
        # Trigger if residual reduction over the last anderson_trigger_window
        # iterations is < anderson_trigger_ratio (stall detected).
        # -----------------------------------------------------------------------
        if accelerator == "anderson" and not aa_active:
            w = anderson_trigger_window
            if len(residual_history) >= w + 1:
                r_old = residual_history[-(w + 1)]
                r_now = residual_history[-1]
                if r_old > 0 and r_now / r_old > anderson_trigger_ratio:
                    # Stall detected: activate Anderson acceleration.
                    aa_active = True
                    anderson_engaged = True

        # -----------------------------------------------------------------------
        # Anderson step or plain Picard update.
        # Use flat blocked (2*N_tau,) layout for Anderson (GRAIL seam convention).
        # -----------------------------------------------------------------------
        if aa_active:
            # Flatten to (2*N_tau,) for AA: populations.T.flatten().
            x_np = x.numpy().T.flatten()
            f_np = f_x.numpy().T.flatten()
            aa_x_history.append(x_np)
            aa_f_history.append(f_np)
            # Keep at most m+1 entries.
            if len(aa_x_history) > anderson_history + 1:
                aa_x_history = aa_x_history[-(anderson_history + 1):]
                aa_f_history = aa_f_history[-(anderson_history + 1):]

            x_aa_np = _anderson_step_2level(aa_x_history, aa_f_history, anderson_history)
            # Unflatten: (2*N_tau,) -> (N_tau, 2) via reshape(2, N_tau).T.
            x_next = torch.tensor(
                x_aa_np.reshape(2, N_tau).T, dtype=torch.float64
            )
            # Clamp to [0, 1] and renormalize to preserve population sum = 1.
            x_next = x_next.clamp(min=0.0, max=1.0)
            x_sum = x_next.sum(dim=-1, keepdim=True).clamp(min=1e-300)
            x_next = x_next / x_sum
        else:
            # Plain Picard: x_{k+1} = T_e(x_k).
            x_next = x_new

        x = x_next

        # -----------------------------------------------------------------------
        # Convergence check.
        # -----------------------------------------------------------------------
        if rel_change < tol:
            converged = True
            break

    return PicardResult2Level(
        populations=x,
        iter_count=len(residual_history),
        residual_history=residual_history,
        anderson_engaged=anderson_engaged,
        converged=converged,
    )
