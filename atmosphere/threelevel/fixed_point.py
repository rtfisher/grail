"""ALI outer loop and cycle functions for 3-level NLTE. SPEC_threelevel §6.2, §6.3.

Implements the K-transition generalisation of the 2-level ALI driver in
atmosphere/twolevel/fixed_point.py.  The state vector for 3-level is
x = populations of shape (N_tau, L=3); the source-function vector is
S_lines of shape (N_tau, K).

Design decisions (per SPEC_threelevel §4):
  - Direct mode: K independent linear solves (one per line), then one
    SE call at the end.  Cross-line coupling enters only through SE.
  - ALI mode: per-line Λ* iteration on each S_k independently, with
    convergence judged on the stacked (N_tau, K) vector.
  - _atm_for_line: builds a per-line Atmosphere view (nu_0 swapped);
    reuses _compute_lambda_full from twolevel unchanged (TASK-011).

TASK-024 additions (SPEC_threelevel §M3d):
  - PicardResult3Level dataclass for Picard iteration output.
  - picard_iterate_3level: Picard iteration on populations x_{k+1} = T_e(x_k),
    with optional Anderson acceleration (numpy-only AA(m)).

References
----------
- Olson, Auer & Buchler (1986), JQSRT 35, 431 — diagonal Λ* preconditioner.
- Walker & Ni (2011), SIAM J. Numer. Anal. 49, 1715 — Anderson acceleration.
- SPEC_threelevel §6.2, §6.3, §M3d TASK-024 — specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..atmosphere import Atmosphere
from ..formal_solver import formal_solve, line_profile_average
from ..atomic_model import AtomicModel
from .rate_equations import rate_equation_solve_3level
from ..twolevel.fixed_point import (
    _compute_lambda_full,
    _compute_lambda_star_diagonal,
    _planck_nu,
)

# ---------------------------------------------------------------------------
# Physical constants (SI; used for Planck function and source-function forms)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant [J s]
_C = 2.99792458e8     # speed of light [m/s]
_KB = 1.380649e-23    # Boltzmann constant [J/K]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AliResult3Level:
    """Converged output from :func:`ali_loop_3level`.

    Attributes
    ----------
    populations : Tensor of shape (N_tau, L)
        Converged level populations at each depth.
    S_lines : Tensor of shape (N_tau, K)
        Converged source functions per line.
    J_bar_lines : Tensor of shape (N_tau, K)
        Converged profile-weighted mean intensities per line.
    iter_count : int
        Number of iterations performed before convergence.
        In direct mode, reported as 1.
    residual_history : list[float]
        Max |ΔS/S| at each iteration.
    """

    populations: torch.Tensor
    S_lines: torch.Tensor
    J_bar_lines: torch.Tensor
    iter_count: int
    residual_history: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source function from populations (Einstein-relation form, multi-line)
# ---------------------------------------------------------------------------

def _source_from_populations_3level(
    populations: torch.Tensor,
    atomic_model: AtomicModel,
) -> torch.Tensor:
    """Compute K line source functions from level populations.

    For each transition k = (lower=l_k, upper=u_k):
        S_k = B_star_k / (n_{l_k} * g_{u_k} / (n_{u_k} * g_{l_k}) - 1)

    where B_star_k = 2 h nu_k^3 / c^2.

    This is the Einstein-relation form analogous to the 2-level
    _source_from_populations (reference guide §7 and twolevel/fixed_point.py).
    It is used inside _cycle_3level before J_bar is available.

    Parameters
    ----------
    populations : Tensor of shape (N_tau, L)
        Level populations at each depth.
    atomic_model : AtomicModel
        Provides transitions, frequencies, and statistical weights.

    Returns
    -------
    S_lines : Tensor of shape (N_tau, K), float64
        Source function per line at each depth.
    """
    N_tau = populations.shape[0]
    K = atomic_model.K
    S_lines = torch.zeros(N_tau, K, dtype=torch.float64, device=populations.device)

    for k, tr in enumerate(atomic_model.transitions):
        n_lower = populations[:, tr.lower]   # (N_tau,)
        n_upper = populations[:, tr.upper]   # (N_tau,)

        g_lower = float(atomic_model.statistical_weights[tr.lower])
        g_upper = float(atomic_model.statistical_weights[tr.upper])

        # Wien amplitude: B_star_k = 2 h nu_k^3 / c^2 (scalar float).
        B_star_k = 2.0 * _H * tr.nu_0**3 / _C**2

        # S_k = B_star_k / (ratio - 1), where ratio = n_l * g_u / (n_u * g_l).
        # Clamp denominator away from zero to avoid division-by-zero.
        ratio = (n_lower * g_upper) / (n_upper.clamp(min=1e-300) * g_lower)
        S_lines[:, k] = B_star_k / (ratio - 1.0).clamp(min=1e-300)

    return S_lines


# ---------------------------------------------------------------------------
# Per-line atmosphere view (nu_0 swap for formal solver)
# ---------------------------------------------------------------------------

def _atm_for_line(atm: Atmosphere, line_index: int) -> Atmosphere:
    """Build a per-line atmosphere view with nu_0 set to line k's frequency.

    The formal solver reads atm.nu_0 for the deep boundary condition
    (Planck function at tau_max).  For multi-line setups we create a shallow
    copy of the Atmosphere with only nu_0 (and epsilon) swapped to line k's
    values; all grid and thermodynamic fields are shared by reference.

    The resulting Atmosphere passes to formal_solve and _compute_lambda_full
    exactly as the 2-level atmosphere does.  atomic_model is preserved.

    Parameters
    ----------
    atm : Atmosphere
        Full multi-line atmosphere.
    line_index : int
        0-based index into atm.atomic_model.transitions.

    Returns
    -------
    Atmosphere
        Shallow copy with nu_0 and epsilon set to transition line_index.

    Raises
    ------
    IndexError
        If line_index >= K (number of transitions).
    """
    K = atm.atomic_model.K
    if line_index >= K:
        raise IndexError(
            f"line_index={line_index!r} out of range for K={K} transitions."
        )

    tr = atm.atomic_model.transitions[line_index]

    # Build a new Atmosphere sharing all grid/profile fields but with
    # nu_0 and epsilon swapped to line k.  dataclass does not have copy();
    # we construct a new instance with the same fields.
    return Atmosphere(
        depth=atm.depth,
        frequency=atm.frequency,
        angle=atm.angle,
        T=atm.T,
        n_e=atm.n_e,
        epsilon=tr.epsilon,
        nu_0=tr.nu_0,
        atomic_model=atm.atomic_model,
    )


# ---------------------------------------------------------------------------
# 3-level cycle step
# ---------------------------------------------------------------------------

def _cycle_3level(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    se_solver: Callable,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT^K ∘ T_SE step for K transitions.

    Steps:
      1. Compute S_lines = _source_from_populations_3level(x, atomic_model).
      2. For each line k:
           a. atm_k = _atm_for_line(atmosphere, k)
           b. Broadcast S_lines[:, k] to all frequencies (CRD).
           c. J_nu = formal_solve(S_full, atm_k, mode=formal_solve_mode).
           d. J_bar_k = line_profile_average(J_nu, atm_k).
      3. Stack J_bar_lines of shape (N_tau, K).
      4. new_x = se_solver(J_bar_lines, T, n_e).

    Parameters
    ----------
    x : Tensor of shape (N_tau, L=3)
        Current populations.
    atmosphere : Atmosphere
        Has atomic_model with K transitions.
    se_solver : callable
        Signature: (J_bar_lines: Tensor[N_tau, K], T: Tensor[N_tau],
                    n_e: Tensor[N_tau]) -> Tensor[N_tau, L].
    formal_solve_mode : str, default 'linear'
        Passed to formal_solve.

    Returns
    -------
    new_x : Tensor of shape (N_tau, L)
    """
    K = atmosphere.atomic_model.K
    N_nu = atmosphere.frequency.nu.shape[0]

    # Step 1: source functions from populations.
    S_lines = _source_from_populations_3level(x, atmosphere.atomic_model)  # (N_tau, K)

    # Step 2-3: formal solve per line -> J_bar per line.
    J_bar_list = []
    for k in range(K):
        atm_k = _atm_for_line(atmosphere, k)
        S_k = S_lines[:, k]          # (N_tau,)
        S_full = S_k.unsqueeze(1).expand(-1, N_nu)   # (N_tau, N_nu)
        J_nu = formal_solve(S_full, atm_k, mode=formal_solve_mode)  # (N_tau, N_nu)
        J_bar_k = line_profile_average(J_nu, atm_k)   # (N_tau,)
        J_bar_list.append(J_bar_k)

    J_bar_lines = torch.stack(J_bar_list, dim=1)   # (N_tau, K)

    # Step 4: updated populations from SE solver.
    T = atmosphere.T.to(torch.float64)
    n_e = atmosphere.n_e.to(torch.float64)
    new_x = se_solver(J_bar_lines, T, n_e)   # (N_tau, L)
    return new_x


# ---------------------------------------------------------------------------
# Reference SE solver builder (3-level analog of _make_reference_se_solver)
# ---------------------------------------------------------------------------

def _make_reference_se_solver_3level(atmosphere: Atmosphere) -> Callable:
    """Build a 3-arg adapter (J_bar_lines, T, n_e) -> populations (N_tau, L).

    Captures atomic_model from the atmosphere so the returned callable
    matches the SE3Solver contract without keyword arguments.
    """
    model = atmosphere.atomic_model

    def reference_solver(
        J_bar_lines: torch.Tensor,
        T: torch.Tensor,
        n_e: torch.Tensor,
    ) -> torch.Tensor:
        return rate_equation_solve_3level(
            J_bar_lines, T, n_e, atomic_model=model
        )

    return reference_solver


# ---------------------------------------------------------------------------
# Public cycle functions
# ---------------------------------------------------------------------------

def cycle_truth_3level(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT^K ∘ T_SE^ref cycle using the reference 3-level SE solver.

    Parameters
    ----------
    x : Tensor of shape (N_tau, L=3)
        Current populations.
    atmosphere : Atmosphere
        Must have atomic_model.n_levels == 3.
    formal_solve_mode : str, default 'linear'

    Returns
    -------
    new_x : Tensor of shape (N_tau, 3)
    """
    se_solver = _make_reference_se_solver_3level(atmosphere)
    return _cycle_3level(x, atmosphere, se_solver, formal_solve_mode=formal_solve_mode)


def cycle_emulator_3level(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    emulator: Callable,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT^K ∘ T_SE^APNN cycle with the APNN emulator replacing T_SE.

    Parameters
    ----------
    x : Tensor of shape (N_tau, L=3)
        Current populations.
    atmosphere : Atmosphere
    emulator : callable
        Signature: (J_bar_lines: Tensor[N_tau, K], T: Tensor[N_tau],
                    n_e: Tensor[N_tau]) -> Tensor[N_tau, L].
    formal_solve_mode : str, default 'linear'

    Returns
    -------
    new_x : Tensor of shape (N_tau, 3)
    """
    return _cycle_3level(x, atmosphere, emulator, formal_solve_mode=formal_solve_mode)


# ---------------------------------------------------------------------------
# Full Λ matrix per line (TASK-011)
# ---------------------------------------------------------------------------

def _compute_lambda_full_3level(
    atmosphere: Atmosphere,
    formal_solve_mode: str = "linear",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build the full Λ matrix and boundary vector for each line.

    Reuses the 2-level _compute_lambda_full function by passing
    _atm_for_line(atm, k) — the 2-level function only reads nu_0 and the
    grids, so the per-line view is a drop-in.

    Parameters
    ----------
    atmosphere : Atmosphere
        Multi-line 3-level atmosphere.
    formal_solve_mode : str, default 'linear'

    Returns
    -------
    lambda_results : list of length K
        Each element is (Lambda_k, b_k) where:
          Lambda_k : Tensor of shape (N_tau, N_tau) — discretized Λ operator.
          b_k      : Tensor of shape (N_tau,) — boundary contribution.
    """
    K = atmosphere.atomic_model.K
    results = []
    for k in range(K):
        atm_k = _atm_for_line(atmosphere, k)
        Lambda_k, b_k = _compute_lambda_full(atm_k, formal_solve_mode=formal_solve_mode)
        results.append((Lambda_k, b_k))
    return results


# ---------------------------------------------------------------------------
# ALI driver (TASK-012)
# ---------------------------------------------------------------------------

def ali_loop_3level(
    atmosphere: Atmosphere,
    se_solver: Callable | None = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 200,
    initial_S: torch.Tensor | None = None,
    mode: str = "direct",
    formal_solve_mode: str = "linear",
) -> AliResult3Level:
    """Stacked-source ALI for K transitions.

    Direct mode
    -----------
    Builds the per-line Λ matrix via _compute_lambda_full_3level (reuses 2-level
    _compute_lambda_full per TASK-011).  Solves each line's source function
    INDEPENDENTLY in a single linear solve (the lines are decoupled in CRD
    with master-eps):
        (I - (1 - eps_k) Λ_k) S_k = (1 - eps_k) b_k + eps_k B(T, nu_k)
    Then calls the SE solver once with all K J_bar values to label populations.
    iter_count is reported as 1.

    Rationale: in CRD with per-line master-eps, S_k depends only on J_bar_k,
    not on J_bar_{k' != k}.  The cross-line coupling enters only through the
    SE solver, which takes all K J_bar simultaneously.  This is exactly
    analogous to the 2-level direct mode.

    ALI mode
    --------
    Per-line Λ* preconditioned iteration on each S_k independently.
    Convergence criterion: max over (tau, k) of |ΔS_k / S_k| < tol.

    Parameters
    ----------
    atmosphere : Atmosphere
        Must have atomic_model.n_levels == 3 and K transitions.
    se_solver : callable or None
        Signature: (J_bar_lines, T, n_e) -> Tensor[N_tau, L].
        If None, uses the reference 3-level SE solver.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum ALI iterations (ALI mode).
    initial_S : Tensor of shape (N_tau, K) or None
        Initial S_lines guess.  If None, use B(T, nu_k) per line (LTE guess).
    mode : {'direct', 'ali'}
        Solver mode; default 'direct'.
    formal_solve_mode : str, default 'linear'
        Passed to formal_solve.

    Returns
    -------
    AliResult3Level
    """
    if se_solver is None:
        se_solver = _make_reference_se_solver_3level(atmosphere)

    if mode not in ("direct", "ali"):
        raise ValueError(f"mode must be 'direct' or 'ali', got mode={mode!r}")

    K = atmosphere.atomic_model.K
    N_tau = atmosphere.depth.tau.shape[0]
    N_nu = atmosphere.frequency.nu.shape[0]

    T = atmosphere.T.to(torch.float64)
    n_e = atmosphere.n_e.to(torch.float64)

    # Planck function per line: B_k of shape (N_tau,) for each k.
    B_lines = []
    eps_list = []
    for k in range(K):
        tr = atmosphere.atomic_model.transitions[k]
        B_k = _planck_nu(tr.nu_0, T)    # (N_tau,)
        B_lines.append(B_k)
        eps_list.append(tr.epsilon)

    # ------------------------------------------------------------------
    # Direct mode: K independent linear solves.
    # ------------------------------------------------------------------
    if mode == "direct":
        lambda_results = _compute_lambda_full_3level(
            atmosphere, formal_solve_mode=formal_solve_mode
        )
        S_list = []
        J_bar_list = []
        for k in range(K):
            Lambda_k, b_k = lambda_results[k]
            eps_k = eps_list[k]
            B_k = B_lines[k]

            # Linear system: (I - (1-eps_k) Λ_k) S_k = (1-eps_k) b_k + eps_k B_k
            A_k = torch.eye(N_tau, dtype=torch.float64) - (1.0 - eps_k) * Lambda_k
            rhs_k = (1.0 - eps_k) * b_k + eps_k * B_k
            S_k = torch.linalg.solve(A_k, rhs_k)  # (N_tau,)

            # J_bar_k from converged S_k (reuse cached Λ, b).
            J_bar_k = Lambda_k @ S_k + b_k
            S_list.append(S_k)
            J_bar_list.append(J_bar_k)

        S_lines = torch.stack(S_list, dim=1)        # (N_tau, K)
        J_bar_lines = torch.stack(J_bar_list, dim=1)   # (N_tau, K)

        # Final populations from SE solver.
        populations = se_solver(J_bar_lines, T, n_e)  # (N_tau, L)

        # Diagnostic residual: max |S - ((1-eps)J_bar + eps B)| / |S|.
        residuals = []
        for k in range(K):
            eps_k = eps_list[k]
            S_k = S_lines[:, k]
            J_bar_k = J_bar_lines[:, k]
            B_k = B_lines[k]
            S_expected = (1.0 - eps_k) * J_bar_k + eps_k * B_k
            res_k = ((S_expected - S_k).abs() / S_k.abs().clamp(min=1e-300)).max().item()
            residuals.append(res_k)
        max_residual = max(residuals)

        return AliResult3Level(
            populations=populations,
            S_lines=S_lines,
            J_bar_lines=J_bar_lines,
            iter_count=1,
            residual_history=[max_residual],
        )

    # ------------------------------------------------------------------
    # ALI iterative mode: per-line Λ* preconditioner.
    # ------------------------------------------------------------------

    # Compute diagonal Λ* per line (one call per line, each costs N_tau+1
    # formal solves; total 2*(N_tau+1) for K=2).
    lambda_star_list = []
    for k in range(K):
        atm_k = _atm_for_line(atmosphere, k)
        lstar_k = _compute_lambda_star_diagonal(
            atm_k, formal_solve_mode=formal_solve_mode
        )   # (N_tau,)
        lambda_star_list.append(lstar_k)

    # Initial guess: LTE (S_k = B_k) if not provided.
    if initial_S is not None:
        S_lines = initial_S.to(torch.float64).clone()  # (N_tau, K)
    else:
        S_lines = torch.stack(B_lines, dim=1).clone()  # (N_tau, K)

    # ALI denominator per line: denom_k[i] = 1 - (1-eps_k)*lambda_star_k[i].
    denom_list = [
        1.0 - (1.0 - eps_list[k]) * lambda_star_list[k]
        for k in range(K)
    ]   # list of (N_tau,) tensors

    residual_history: list[float] = []

    for iteration in range(max_iter):
        J_bar_list = []
        for k in range(K):
            atm_k = _atm_for_line(atmosphere, k)
            S_k = S_lines[:, k]   # (N_tau,)
            S_full = S_k.unsqueeze(1).expand(-1, N_nu)   # (N_tau, N_nu)
            J_nu = formal_solve(S_full, atm_k, mode=formal_solve_mode)
            J_bar_k = line_profile_average(J_nu, atm_k)  # (N_tau,)
            J_bar_list.append(J_bar_k)

        J_bar_lines = torch.stack(J_bar_list, dim=1)   # (N_tau, K)

        # Per-line ALI update: ΔS_k = [(1-eps_k)*J_bar_k + eps_k*B_k - S_k] / denom_k
        delta_S_list = []
        for k in range(K):
            eps_k = eps_list[k]
            B_k = B_lines[k]
            J_bar_k = J_bar_lines[:, k]
            S_k = S_lines[:, k]
            denom_k = denom_list[k]
            delta_S_k = ((1.0 - eps_k) * J_bar_k + eps_k * B_k - S_k) / denom_k
            delta_S_list.append(delta_S_k)

        delta_S = torch.stack(delta_S_list, dim=1)   # (N_tau, K)
        S_new = S_lines + delta_S

        # Convergence: max over (tau, k) of |ΔS/S|.
        safe_S = S_lines.abs().clamp(min=1e-300)
        rel_change = (delta_S.abs() / safe_S).max().item()
        residual_history.append(rel_change)

        S_lines = S_new

        if rel_change < tol:
            break

    # Final J_bar with converged S_lines.
    J_bar_list_final = []
    for k in range(K):
        atm_k = _atm_for_line(atmosphere, k)
        S_k = S_lines[:, k]
        S_full = S_k.unsqueeze(1).expand(-1, N_nu)
        J_nu = formal_solve(S_full, atm_k, mode=formal_solve_mode)
        J_bar_k = line_profile_average(J_nu, atm_k)
        J_bar_list_final.append(J_bar_k)

    J_bar_lines_final = torch.stack(J_bar_list_final, dim=1)   # (N_tau, K)
    populations = se_solver(J_bar_lines_final, T, n_e)   # (N_tau, L)

    return AliResult3Level(
        populations=populations,
        S_lines=S_lines,
        J_bar_lines=J_bar_lines_final,
        iter_count=len(residual_history),
        residual_history=residual_history,
    )


# ---------------------------------------------------------------------------
# Picard iteration on populations (TASK-024)
# ---------------------------------------------------------------------------

@dataclass
class PicardResult3Level:
    """Output from :func:`picard_iterate_3level`.

    Attributes
    ----------
    populations : Tensor of shape (N_tau, L=3)
        Populations at the last Picard iterate (converged if iter_count < max_iter).
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


def _anderson_step(
    x_history: list[np.ndarray],
    f_history: list[np.ndarray],
    m: int,
) -> np.ndarray:
    """Compute the Anderson AA(m) update given history of iterates and residuals.

    Standard formulation (Walker & Ni 2011, Algorithm 2):
      Given x_k and f_k = T(x_k) - x_k (fixed-point residual), the AA(m) update
      minimises ||sum_i c_i f_{k-i}|| s.t. sum_i c_i = 1, then sets:

          x_{k+1} = sum_i c_i * T(x_{k-i})

    This numpy-only implementation uses the least-squares residual form:
      Build matrix F = [f_{k-m+1}, ..., f_k] of shape (n, m).
      Solve min_theta ||F theta||^2 s.t. sum(theta) = 1.
      Equivalent to: solve unconstrained problem on differences, apply constraints.

    Practical implementation (shifted formulation):
      Define delta_F[:, i] = f_{k-m+i+1} - f_{k-m+i}  (shape (n, m-1))
      delta_x[:, i] = x_{k-m+i+1} - x_{k-m+i}         (shape (n, m-1))
      Solve least-squares: delta_F @ gamma = -f_k  (gamma shape (m-1,))
      Then: beta = gamma;  mixing weights with sum-1 constraint.

    Parameters
    ----------
    x_history : list of np.ndarray
        The last min(k, m)+1 iterates x_{k-m}, ..., x_k (most recent last).
    f_history : list of np.ndarray
        The last min(k, m)+1 residuals f_i = T(x_i) - x_i.
    m : int
        History window size.

    Returns
    -------
    x_new : np.ndarray
        Anderson-accelerated next iterate.
    """
    # Use the last min(len(x_history), m+1) entries.
    m_actual = min(len(x_history), m + 1)
    xs = x_history[-m_actual:]    # list of (n,) arrays, length m_actual
    fs = f_history[-m_actual:]    # list of (n,) arrays, length m_actual

    if m_actual < 2:
        # Not enough history for AA; fall back to plain Picard step.
        return xs[-1] + fs[-1]

    # Build F and X matrices from the window.
    # F[:, j] = f_{j}  (the residuals in the window)
    # X[:, j] = x_{j}  (the iterates in the window)
    n = xs[0].shape[0]
    F = np.column_stack(fs)   # (n, m_actual)
    X = np.column_stack(xs)   # (n, m_actual)

    # Difference form: dF[:, i] = F[:, i+1] - F[:, i]
    dF = np.diff(F, axis=1)   # (n, m_actual - 1)
    dX = np.diff(X, axis=1)   # (n, m_actual - 1)

    # Last residual: f_k = F[:, -1]
    f_k = F[:, -1]             # (n,)

    # Least-squares solve: dF @ gamma = -f_k.
    # gamma has shape (m_actual - 1,).
    gamma, _, _, _ = np.linalg.lstsq(dF, -f_k, rcond=None)

    # Anderson update: x_new = X[:, -1] + F[:, -1] - (dX + dF) @ gamma
    # = x_k + f_k - (dX + dF) @ gamma
    x_new = X[:, -1] + f_k - (dX + dF) @ gamma

    return x_new


def picard_iterate_3level(
    atmosphere: Atmosphere,
    se_solver: Callable,
    *,
    initial_x: torch.Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 500,
    formal_solve_mode: str = "linear",
    accelerator: str = "none",
    anderson_history: int = 5,
    anderson_trigger_window: int = 200,
    anderson_trigger_ratio: float = 0.1,
) -> PicardResult3Level:
    """Picard iteration x_{k+1} = T_e(x_k) on populations.

    Convergence path used by the empirical-amplification test
    `I-resolvent-predicts-3level` (definition (ii) per §6.2.1).
    Picard exercises the rate-matrix coupling NONLINEARLY through
    the emulator's population output, which feeds back into the
    formal solver on the next iteration.  Direct mode short-circuits
    this and is therefore degenerate for definition (ii).

    The cycle step is T_e(x) = _cycle_3level(x, atmosphere, se_solver,
    formal_solve_mode).  Picard residual is |Δx| = ||x_{k+1} - x_k||_inf
    relative to ||x_k||_inf.

    Anderson acceleration (numpy-only)
    ----------------------------------
    Anderson AA(m) with m = anderson_history.  Triggered automatically
    when accelerator="anderson" AND plain-Picard residual reduction
    is < anderson_trigger_ratio (default 0.1×) over the last
    anderson_trigger_window iterations (default 200) — this catches
    deep-eps stalls where the cycle's spectral radius approaches 1.
    Anderson is implemented via numpy least-squares on the residual
    history; no scipy / no new dependencies (per NFR-001 and the
    M1b LHS sampler precedent).

    SPEC_threelevel §M3d TASK-024.

    Parameters
    ----------
    atmosphere : Atmosphere
        3-level atmosphere (must have atomic_model.n_levels == 3).
    se_solver : Callable
        SE3Solver: (J_bar_lines: Tensor[N_tau, K], T: Tensor[N_tau],
                    n_e: Tensor[N_tau]) -> Tensor[N_tau, 3].
        Typically the APNN adapter from make_cycle_adapter_3level.
    initial_x : Tensor of shape (N_tau, 3) or None
        Initial population guess.  If None, uses LTE populations from the
        reference SE solver at J_bar = B(T, nu_k) for each line.
    tol : float, default 1e-6
        Convergence tolerance on max relative change ||Δx|| / ||x||_inf.
    max_iter : int, default 500
        Maximum number of Picard iterations.
    formal_solve_mode : str, default 'linear'
        Passed to _cycle_3level (and hence to formal_solve).
    accelerator : str, default 'none'
        'none' for plain Picard; 'anderson' for Anderson AA(m) with
        automatic trigger.
    anderson_history : int, default 5
        History window m for Anderson AA(m).
    anderson_trigger_window : int, default 200
        Number of iterations over which to check for stall.
    anderson_trigger_ratio : float, default 0.1
        Residual reduction threshold: if residual_history[-1] >
        anderson_trigger_ratio * residual_history[-trigger_window], Anderson
        is triggered (meaning residual dropped by less than 10× in the window).

    Returns
    -------
    PicardResult3Level
        populations    : converged (or last) populations.
        iter_count     : total iterations performed.
        residual_history : per-iteration max relative change.
        anderson_engaged : whether Anderson acceleration was activated.
        converged       : True if tol was met before max_iter.
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
    # -----------------------------------------------------------------------
    if initial_x is None:
        # LTE: call reference SE solver at J_bar ≈ B(T, nu_k) per line.
        # Build J_bar_lines_lte of shape (N_tau, K) as Planck function values.
        K = atmosphere.atomic_model.K
        J_bar_lte_list = []
        for k in range(K):
            tr = atmosphere.atomic_model.transitions[k]
            B_k = _planck_nu(tr.nu_0, T)     # (N_tau,)
            J_bar_lte_list.append(B_k)
        J_bar_lte = torch.stack(J_bar_lte_list, dim=1)   # (N_tau, K)
        x = se_solver(J_bar_lte, T, n_e).detach().clone()  # (N_tau, 3)
    else:
        x = initial_x.to(torch.float64).detach().clone()

    residual_history: list[float] = []
    anderson_engaged = False
    converged = False

    # -----------------------------------------------------------------------
    # Anderson acceleration buffers (used only when anderson is triggered).
    # -----------------------------------------------------------------------
    # These store numpy flat vectors for the AA computation.
    aa_x_history: list[np.ndarray] = []    # iterates
    aa_f_history: list[np.ndarray] = []    # residuals f(x) = T(x) - x
    aa_active = False                       # whether AA has been triggered

    for iteration in range(max_iter):
        # One cycle step: T_e(x).
        x_new = _cycle_3level(
            x, atmosphere, se_solver,
            formal_solve_mode=formal_solve_mode,
        ).detach()  # (N_tau, 3)

        # Residual: f(x) = x_new - x (the fixed-point defect).
        f_x = x_new - x   # (N_tau, 3)

        # Convergence metric: max |Δx| / max(|x|, eps).
        safe_x = x.abs().clamp(min=1e-300)
        rel_change = (f_x.abs() / safe_x).max().item()
        residual_history.append(rel_change)

        # -----------------------------------------------------------------------
        # Anderson acceleration trigger check.
        # -----------------------------------------------------------------------
        if accelerator == "anderson" and not aa_active:
            # Check if we're stalling: residual reduction over the last
            # anderson_trigger_window iterations is < anderson_trigger_ratio.
            w = anderson_trigger_window
            if len(residual_history) >= w + 1:
                r_old = residual_history[-(w + 1)]
                r_now = residual_history[-1]
                if r_old > 0 and r_now / r_old > anderson_trigger_ratio:
                    # Stall detected: activate Anderson acceleration.
                    aa_active = True
                    anderson_engaged = True
                    # History starts empty; the current iterate will be
                    # appended in the aa_active block below on this iteration.

        # -----------------------------------------------------------------------
        # Anderson step or plain Picard update.
        # -----------------------------------------------------------------------
        if aa_active:
            x_np = x.numpy().ravel()
            f_np = f_x.numpy().ravel()
            # Append current iterate and residual to history.
            aa_x_history.append(x_np)
            aa_f_history.append(f_np)
            # Keep at most m+1 entries.
            if len(aa_x_history) > anderson_history + 1:
                aa_x_history = aa_x_history[-(anderson_history + 1):]
                aa_f_history = aa_f_history[-(anderson_history + 1):]

            x_aa_np = _anderson_step(aa_x_history, aa_f_history, anderson_history)
            x_next = torch.tensor(x_aa_np.reshape(N_tau, 3), dtype=torch.float64)
            # Clamp to [0, 1] since populations must be non-negative and <=1.
            x_next = x_next.clamp(min=0.0, max=1.0)
            # Renormalize to enforce sum=1 (Anderson may violate conservation slightly).
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

    return PicardResult3Level(
        populations=x,
        iter_count=len(residual_history),
        residual_history=residual_history,
        anderson_engaged=anderson_engaged,
        converged=converged,
    )
