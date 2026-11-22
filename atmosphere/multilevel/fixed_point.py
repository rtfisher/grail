"""Coupled N-level NLTE cycles with an explicit positive-opacity contract.

Tree atoms admit independent line-source ALI solves. Closed transition graphs
require the composed radiation-to-rate population equilibrium solved here in
independent log-population ratios. Picard acceleration preserves positivity
without projecting onto simplex faces; Newton uses the actual SE Jacobian.
Returned convergence flags test population and line-source defects against the
original formal cycle, including its affine thermal lower boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..atmosphere import Atmosphere
from ..formal_solver import formal_solve, line_profile_average
from ..atomic_model import AtomicModel
from .rate_equations import rate_equation_solve_nlevel
from ..twolevel.fixed_point import (
    _compute_lambda_full,
    _compute_lambda_star_diagonal,
    _planck_nu,
)

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant [J s]
_C = 2.99792458e8     # speed of light [m/s]
_KB = 1.380649e-23    # Boltzmann constant [J/K]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AliResultMultilevel:
    """Converged output from :func:`ali_loop_nlevel`.

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
        For a direct transition-tree solve, reported as 1. Closed graphs
        report the actual coupled Newton iteration count.
    residual_history : list[float]
        Max |ΔS/S| at each iteration.
    """

    populations: torch.Tensor
    S_lines: torch.Tensor
    J_bar_lines: torch.Tensor
    iter_count: int
    residual_history: list[float] = field(default_factory=list)
    converged: bool = False
    failure_reason: str | None = None


@dataclass
class PicardResultMultilevel:
    """Output from :func:`picard_iterate_nlevel`.

    Attributes
    ----------
    populations : Tensor of shape (N_tau, L)
        Populations at the last Picard iterate.
    iter_count : int
        Number of Picard iterations performed.
    residual_history : list[float]
        Max |Δx| / |x| at each iteration.
    anderson_engaged : bool
        True if Anderson acceleration was triggered.
    converged : bool
        True if the residual dropped below tol before max_iter.
    """

    populations: torch.Tensor
    iter_count: int
    residual_history: list[float] = field(default_factory=list)
    anderson_engaged: bool = False
    converged: bool = False
    source_residual_history: list[float] = field(default_factory=list)
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# Source function from populations (Einstein-relation form, multi-line)
# ---------------------------------------------------------------------------

def _source_from_populations_nlevel(
    populations: torch.Tensor,
    atomic_model: AtomicModel,
) -> torch.Tensor:
    """Compute K line source functions from level populations.

    For each transition k = (lower=l_k, upper=u_k):
        S_k = B_star_k / (n_{l_k} * g_{u_k} / (n_{u_k} * g_{l_k}) - 1)

    where B_star_k = 2 h nu_k^3 / c^2.

    Direct generalisation of _source_from_populations_3level with L/K
    read from atomic_model dynamically.

    Parameters
    ----------
    populations : Tensor of shape (N_tau, L)
        Level populations at each depth.
    atomic_model : AtomicModel
        Provides transitions, frequencies, and statistical weights.

    Returns
    -------
    S_lines : Tensor of shape (N_tau, K), float64
    """
    if not torch.isfinite(populations).all() or (populations <= 0).any():
        raise ValueError("line sources require finite, strictly positive populations")
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

        # This fixed-opacity transfer model excludes population inversions.
        # A denominator floor would silently turn a maser into a huge positive
        # source and create an artificial saturated neural fixed point.
        opacity_population = n_lower * (g_upper / g_lower) - n_upper
        if (opacity_population <= 0).any():
            bad = int((opacity_population <= 0).nonzero()[0, 0])
            raise ValueError(f"nonpositive line opacity at depth {bad}, line {k} ({tr.label})")
        S_lines[:, k] = B_star_k * n_upper / opacity_population

    return S_lines


# ---------------------------------------------------------------------------
# Per-line atmosphere view (nu_0 swap for formal solver)
# ---------------------------------------------------------------------------

def _atm_for_line(atm: Atmosphere, line_index: int) -> Atmosphere:
    """Build a per-line atmosphere view with nu_0 set to line k's frequency.

    Byte-identical to atmosphere/threelevel/fixed_point.py::_atm_for_line.
    The formal solver reads atm.nu_0 for the deep boundary condition
    (Planck function at tau_max).  For multi-line setups we create a shallow
    copy with only nu_0 (and epsilon) swapped to line k's values.

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
# N-level cycle step
# ---------------------------------------------------------------------------

def _cycle_nlevel(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    se_solver: Callable,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT^K o T_SE step for K transitions.

    Steps:
      1. Compute S_lines = _source_from_populations_nlevel(x, atomic_model).
      2. For each line k:
           a. atm_k = _atm_for_line(atmosphere, k)
           b. Broadcast S_lines[:, k] to all frequencies (CRD).
           c. J_nu = formal_solve(S_full, atm_k, mode=formal_solve_mode).
           d. J_bar_k = line_profile_average(J_nu, atm_k).
      3. Stack J_bar_lines of shape (N_tau, K).
      4. new_x = se_solver(J_bar_lines, T, n_e).

    Parameters
    ----------
    x : Tensor of shape (N_tau, L)
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
    S_lines = _source_from_populations_nlevel(x, atmosphere.atomic_model)  # (N_tau, K)

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
# Reference SE solver builder
# ---------------------------------------------------------------------------

def _make_reference_se_solver_nlevel(atmosphere: Atmosphere) -> Callable:
    """Build a 3-arg adapter (J_bar_lines, T, n_e) -> populations (N_tau, L).

    Captures atomic_model (and abundance_fractions, when present on the
    atmosphere) from the atmosphere.  The abundance_fractions field is
    optional (None for single-species setups); when set it is forwarded as
    species_fractions to rate_equation_solve_nlevel so that per-species
    population normalisation is applied correctly.  This is the multi-species
    fix for Phase S: without it the solver normalises the entire population
    vector to 1.0 regardless of species blocks, producing H-block sums of
    ~0.5 instead of ~0.9216.

    SPEC_phase_s §2.3 — species_fractions must be propagated from
    atmosphere.abundance_fractions to the SE solver.
    """
    model = atmosphere.atomic_model
    # Extract abundance_fractions from atmosphere; None for single-species.
    species_fractions = getattr(atmosphere, "abundance_fractions", None)

    def reference_solver(
        J_bar_lines: torch.Tensor,
        T: torch.Tensor,
        n_e: torch.Tensor,
    ) -> torch.Tensor:
        return rate_equation_solve_nlevel(
            J_bar_lines, T, n_e,
            atomic_model=model,
            species_fractions=species_fractions,
        )

    return reference_solver


# ---------------------------------------------------------------------------
# Full Λ matrix per line
# ---------------------------------------------------------------------------

_TRANSPORT_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _compute_lambda_full_nlevel(
    atmosphere: Atmosphere,
    formal_solve_mode: str = "linear",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build cached affine formal-transport operators per line.

    The geometry and prescribed opacity profile are common to all lines;
    only their Planck lower-boundary intensities differ. Cache Lambda and
    the unit-boundary response for up to eight geometries. Return a list
    of (Lambda, boundary) pairs with J_bar_k = Lambda @ S_k + boundary_k.
    """
    # Every line uses the same prescribed opacity profile and geometry. Only
    # its thermal lower-boundary intensity changes. Cache that affine operator;
    # the final acceptance check still calls the original formal cycle.
    key = (formal_solve_mode, str(atmosphere.T.device)) + tuple(
        tuple(t.detach().cpu().reshape(-1).tolist()) for t in (
            atmosphere.depth.tau, atmosphere.frequency.nu,
            atmosphere.frequency.profile, atmosphere.frequency.weights,
            atmosphere.angle.mu, atmosphere.angle.weights,
        )
    )
    if key not in _TRANSPORT_CACHE:
        atm0 = _atm_for_line(atmosphere, 0)
        lam, boundary = _compute_lambda_full(atm0, formal_solve_mode=formal_solve_mode)
        bottom = _planck_nu(atm0.nu_0, atm0.T[-1:])[0]
        if len(_TRANSPORT_CACHE) >= 8:
            _TRANSPORT_CACHE.pop(next(iter(_TRANSPORT_CACHE)))
        _TRANSPORT_CACHE[key] = (lam.detach(), (boundary / bottom).detach())
    lam, unit_boundary = _TRANSPORT_CACHE[key]
    return [(lam, unit_boundary * _planck_nu(tr.nu_0, atmosphere.T[-1:])[0])
            for tr in atmosphere.atomic_model.transitions]


def radiation_from_populations_nlevel(populations, atmosphere, formal_solve_mode="linear"):
    """Return actual population-derived sources and affine formal radiation."""
    sources = _source_from_populations_nlevel(populations, atmosphere.atomic_model)
    operators = _compute_lambda_full_nlevel(atmosphere, formal_solve_mode)
    radiation = torch.stack([lam @ sources[:, k] + boundary
                             for k, (lam, boundary) in enumerate(operators)], dim=1)
    return sources, radiation


# ---------------------------------------------------------------------------
# Public cycle functions
# ---------------------------------------------------------------------------

def cycle_truth_nlevel(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT^K o T_SE^ref cycle using the reference N-level SE solver.

    Parameters
    ----------
    x : Tensor of shape (N_tau, L)
        Current populations.
    atmosphere : Atmosphere
        Must have atomic_model with n_levels >= 2 and K >= 1 transitions.
    formal_solve_mode : str, default 'linear'

    Returns
    -------
    new_x : Tensor of shape (N_tau, L)
    """
    se_solver = _make_reference_se_solver_nlevel(atmosphere)
    return _cycle_nlevel(x, atmosphere, se_solver, formal_solve_mode=formal_solve_mode)


def cycle_emulator_nlevel(
    x: torch.Tensor,
    atmosphere: Atmosphere,
    emulator: Callable,
    formal_solve_mode: str = "linear",
) -> torch.Tensor:
    """One T_RT^K o T_SE^emulator cycle with the APNN emulator replacing T_SE.

    Parameters
    ----------
    x : Tensor of shape (N_tau, L)
        Current populations.
    atmosphere : Atmosphere
    emulator : callable
        Signature: (J_bar_lines: Tensor[N_tau, K], T: Tensor[N_tau],
                    n_e: Tensor[N_tau]) -> Tensor[N_tau, L].
    formal_solve_mode : str, default 'linear'

    Returns
    -------
    new_x : Tensor of shape (N_tau, L)
    """
    return _cycle_nlevel(x, atmosphere, emulator, formal_solve_mode=formal_solve_mode)


# ---------------------------------------------------------------------------
# ALI driver
# ---------------------------------------------------------------------------

def ali_loop_nlevel(
    atmosphere: Atmosphere,
    se_solver: Callable | None = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 200,
    initial_S: torch.Tensor | None = None,
    mode: str = "direct",
    formal_solve_mode: str = "linear",
) -> AliResultMultilevel:
    """Solve an N-level NLTE equilibrium.

    For a transition tree, ``direct`` solves the independent line equations
    including the affine lower-boundary term, while ``ali`` iterates them.
    For closed transition graphs, both modes solve the actual coupled
    population cycle by damped Newton. Independent master-epsilon line
    sources do not generally correspond to a common population vector on
    these graphs. ``S_lines`` and ``J_bar_lines`` describe the returned
    populations, and ``converged``/``failure_reason`` report actual success.
    """
    if se_solver is None:
        se_solver = _make_reference_se_solver_nlevel(atmosphere)

    if mode not in ("direct", "ali"):
        raise ValueError(f"mode must be 'direct' or 'ali', got mode={mode!r}")

    # Independent line source solves are exact only on a transition tree.
    # Closed rate loops carry nonzero line currents: their population-derived
    # sources need not equal independently prescribed master-epsilon sources.
    # Solve the actual composed cycle for these atoms, in both legacy modes.
    blocks = atmosphere.atomic_model.species_blocks or ((0, atmosphere.atomic_model.n_levels),)
    if atmosphere.atomic_model.K > atmosphere.atomic_model.n_levels - len(blocks):
        solved = solve_equilibrium_nlevel(atmosphere, se_solver, tol=tol,
                                         max_iter=max_iter, formal_solve_mode=formal_solve_mode)
        sources, radiation = radiation_from_populations_nlevel(solved.populations, atmosphere,
                                                               formal_solve_mode)
        return AliResultMultilevel(solved.populations, sources, radiation, solved.iter_count,
                                   solved.source_residual_history, solved.converged, solved.failure_reason)

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
        lambda_results = _compute_lambda_full_nlevel(
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

            # J_bar_k from converged S_k.
            J_bar_k = Lambda_k @ S_k + b_k
            S_list.append(S_k)
            J_bar_list.append(J_bar_k)

        S_lines = torch.stack(S_list, dim=1)           # (N_tau, K)
        J_bar_lines = torch.stack(J_bar_list, dim=1)   # (N_tau, K)

        # Final populations from SE solver.
        populations = se_solver(J_bar_lines, T, n_e)   # (N_tau, L)

        # Diagnostic residual: max |S - ((1-eps)J_bar + eps B)| / |S|.
        residuals = []
        for k in range(K):
            eps_k = eps_list[k]
            S_k = S_lines[:, k]
            J_bar_k = J_bar_lines[:, k]
            B_k = B_lines[k]
            S_expected = (1.0 - eps_k) * J_bar_k + eps_k * B_k
            res_k = (
                (S_expected - S_k).abs() / S_k.abs().clamp(min=1e-300)
            ).max().item()
            residuals.append(res_k)
        max_residual = max(residuals)

        return AliResultMultilevel(
            populations=populations,
            S_lines=S_lines,
            J_bar_lines=J_bar_lines,
            iter_count=1,
            residual_history=[max_residual],
            converged=max_residual < tol,
        )

    # ------------------------------------------------------------------
    # ALI iterative mode: per-line Λ* preconditioner.
    # ------------------------------------------------------------------

    lambda_star_list = []
    for k in range(K):
        atm_k = _atm_for_line(atmosphere, k)
        lstar_k = _compute_lambda_star_diagonal(
            atm_k, formal_solve_mode=formal_solve_mode
        )   # (N_tau,)
        lambda_star_list.append(lstar_k)

    # Initial guess: LTE (S_k = B_k) if not provided.
    if initial_S is not None:
        S_lines = initial_S.to(torch.float64).clone()
    else:
        S_lines = torch.stack(B_lines, dim=1).clone()  # (N_tau, K)

    # ALI denominator per line: denom_k = 1 - (1-eps_k)*lambda_star_k.
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
            S_full = S_k.unsqueeze(1).expand(-1, N_nu)
            J_nu = formal_solve(S_full, atm_k, mode=formal_solve_mode)
            J_bar_k = line_profile_average(J_nu, atm_k)
            J_bar_list.append(J_bar_k)

        J_bar_lines = torch.stack(J_bar_list, dim=1)   # (N_tau, K)

        # Per-line ALI update.
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

    return AliResultMultilevel(
        populations=populations,
        S_lines=S_lines,
        J_bar_lines=J_bar_lines_final,
        iter_count=len(residual_history),
        residual_history=residual_history,
        converged=bool(residual_history and residual_history[-1] < tol),
    )


# ---------------------------------------------------------------------------
# Anderson acceleration helper (numpy-only)
# ---------------------------------------------------------------------------

def _anderson_step_nlevel(
    x_history: list[np.ndarray],
    f_history: list[np.ndarray],
    m: int,
) -> np.ndarray:
    """Anderson AA(m) update for N-level populations.

    Identical to _anderson_step in atmosphere/threelevel/fixed_point.py.
    The flat blocked (L*N_tau,) layout is used throughout (same as GRAIL seam).

    Parameters
    ----------
    x_history : list of np.ndarray
        The last min(k, m)+1 iterates (flat blocked, most recent last).
    f_history : list of np.ndarray
        The last min(k, m)+1 residuals f_i = T(x_i) - x_i.
    m : int
        History window size.

    Returns
    -------
    x_new : np.ndarray
        Anderson-accelerated next iterate (flat blocked).
    """
    m_actual = min(len(x_history), m + 1)
    xs = x_history[-m_actual:]
    fs = f_history[-m_actual:]

    if m_actual < 2:
        return xs[-1] + fs[-1]

    n = xs[0].shape[0]
    F = np.column_stack(fs)   # (n, m_actual)
    X = np.column_stack(xs)   # (n, m_actual)

    dF = np.diff(F, axis=1)   # (n, m_actual - 1)
    dX = np.diff(X, axis=1)   # (n, m_actual - 1)

    f_k = F[:, -1]   # (n,)

    # Type-II Anderson: dF @ gamma = f_k.  The former minus sign combined
    # with the subtraction below extrapolated in the wrong direction.
    gamma, _, _, _ = np.linalg.lstsq(dF, f_k, rcond=None)

    # Anderson update: x_new = x_k + f_k - (dX + dF) @ gamma.
    x_new = X[:, -1] + f_k - (dX + dF) @ gamma

    return x_new


# ---------------------------------------------------------------------------
# Picard iteration on populations
# ---------------------------------------------------------------------------

def picard_iterate_nlevel(
    atmosphere: Atmosphere,
    se_solver: Callable,
    *,
    initial_x: torch.Tensor | None = None,
    tol: float = 1e-7,
    max_iter: int = 5000,
    formal_solve_mode: str = "linear",
    accelerator: str = "anderson",
    anderson_history: int = 5,
    anderson_trigger_window: int = 20,
    anderson_trigger_ratio: float = 1e-3,
) -> PicardResultMultilevel:
    """Iterate the actual population cycle in positive log coordinates.

    Convergence requires componentwise log population-ratio and line-source
    cycle defects below ``tol``. An additional call to the original formal
    solver verifies the returned state. Anderson uses its standard Type-II
    sign and backtracks extrapolations that leave the positive-opacity
    domain. No zero clipping or population/source floors are applied.
    Species-block population sums are preserved by normalized exponentials.
    """
    if accelerator not in ("none", "anderson"):
        raise ValueError("accelerator must be 'none' or 'anderson'")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    model = atmosphere.atomic_model
    blocks = model.species_blocks or ((0, model.n_levels),)
    T, ne = atmosphere.T.double(), atmosphere.n_e.double()
    ref = _make_reference_se_solver_nlevel(atmosphere)
    if initial_x is None:
        planck = torch.stack([_planck_nu(tr.nu_0, T) for tr in model.transitions], 1)
        x = ref(planck, T, ne).detach()
    else:
        x = initial_x.double().detach().clone()
    totals = [x[:, lo:hi].sum(-1, keepdim=True) for lo, hi in blocks]

    def encode(p):
        return torch.cat([torch.log(p[:, lo:hi]) - torch.log(p[:, lo:lo+1])
                          for lo, hi in blocks], 1)

    def decode(z):
        return torch.cat([torch.softmax(z[:, lo:hi], -1) * total
                          for (lo, hi), total in zip(blocks, totals)], 1)

    history, source_history = [], []
    aa_x, aa_f = [], []
    engaged = False
    failure = None
    _source_from_populations_nlevel(x, model)  # fail before any iteration
    for iteration in range(max_iter):
        sources, radiation = radiation_from_populations_nlevel(x, atmosphere, formal_solve_mode)
        y = se_solver(radiation, T, ne).detach()
        try:
            y_sources = _source_from_populations_nlevel(y, model)
        except ValueError as exc:
            failure = f"SE output outside positive-opacity domain: {exc}"
            break
        z, zy = encode(x), encode(y)
        f = zy - z
        # Log residuals resolve every positive component and every line,
        # independently of its absolute population or source magnitude.
        defect = float(f.abs().max())
        source_defect = float(torch.log(y_sources / sources).abs().max())
        history.append(defect)
        source_history.append(source_defect)
        if max(defect, source_defect) < tol:
            # Verify the returned state against un-cached formal transport.
            actual = _cycle_nlevel(x, atmosphere, se_solver, formal_solve_mode).detach()
            actual_sources = _source_from_populations_nlevel(actual, model)
            actual_defect = max(float((encode(actual) - z).abs().max()),
                                float(torch.log(actual_sources / sources).abs().max()))
            if actual_defect < tol:
                return PicardResultMultilevel(x, len(history), history, engaged, True,
                                              source_history, None)
            failure = f"cached/formal acceptance disagreement: {actual_defect}"
            break
        aa_x.append(z.numpy().reshape(-1).copy())
        aa_f.append(f.numpy().reshape(-1).copy())
        aa_x, aa_f = aa_x[-anderson_history-1:], aa_f[-anderson_history-1:]
        next_z = zy
        if accelerator == "anderson" and len(history) >= anderson_trigger_window:
            engaged = True
            candidate = torch.from_numpy(_anderson_step_nlevel(aa_x, aa_f, anderson_history)).reshape_as(z)
            # Backtrack in log populations, preserving strict positivity and
            # conservation. Never clamp an extrapolation onto simplex faces.
            for backtrack in range(20):
                trial_z = z + (candidate - z) * (0.5 ** backtrack)
                trial = decode(trial_z)
                try:
                    _source_from_populations_nlevel(trial, model)
                except ValueError:
                    continue
                next_z = trial_z
                break
        x = decode(next_z)
    return PicardResultMultilevel(x, len(history), history, engaged, False, source_history,
                                  failure or "maximum iterations reached")


def solve_equilibrium_nlevel(
    atmosphere: Atmosphere,
    se_solver: Callable | None = None,
    *,
    initial_x: torch.Tensor | None = None,
    tol: float = 1e-9,
    max_iter: int = 40,
    formal_solve_mode: str = "linear",
) -> PicardResultMultilevel:
    """Damped Newton solve of the *coupled* positive population cycle.

    Uses independent log(n_i/n_0) coordinates and an exact chain-rule
    Jacobian of the supplied SE callable. Legacy black-box callables use
    centered local relative finite differences when autograd is unavailable.
    Every accepted step reduces the actual cycle
    defect and stays in the strictly positive, positive-opacity domain.
    No population or source floors enter the equations.
    """
    model = atmosphere.atomic_model
    blocks = model.species_blocks or ((0, model.n_levels),)
    se_solver = se_solver or _make_reference_se_solver_nlevel(atmosphere)
    T, ne = atmosphere.T.double(), atmosphere.n_e.double()
    if initial_x is None:
        planck = torch.stack([_planck_nu(tr.nu_0, T) for tr in model.transitions], 1)
        x = _make_reference_se_solver_nlevel(atmosphere)(planck, T, ne).detach()
    else:
        x = initial_x.double().detach().clone()
    totals = [x[:, lo:hi].sum(1, keepdim=True) for lo, hi in blocks]
    N, L = x.shape
    D = L - len(blocks)
    operators = _compute_lambda_full_nlevel(atmosphere, formal_solve_mode)
    lam = operators[0][0]

    def encode(p):
        return torch.cat([torch.log(p[:, lo+1:hi] / p[:, lo:lo+1])
                          for lo, hi in blocks], 1)

    def decode(q):
        parts = []
        offset = 0
        for (lo, hi), total in zip(blocks, totals):
            qb = q[:, offset:offset+hi-lo-1]
            parts.append(torch.softmax(torch.cat([torch.zeros_like(q[:, :1]), qb], 1), 1)*total)
            offset += hi-lo-1
        return torch.cat(parts, 1)

    def evaluate(q):
        p = decode(q)
        sources, radiation = radiation_from_populations_nlevel(p, atmosphere, formal_solve_mode)
        y = se_solver(radiation, T, ne)
        next_sources = _source_from_populations_nlevel(y, model)
        residual = encode(y) - q
        source_defect = torch.log(next_sources / sources)
        return residual, source_defect, sources, radiation, y

    q = encode(x)
    history, source_history = [], []
    failure = "maximum Newton iterations reached"
    for iteration in range(max_iter):
        q = q.detach().requires_grad_(True)
        try:
            residual, source_defect, sources, radiation, y = evaluate(q)
        except ValueError as exc:
            failure = f"outside physical source domain: {exc}"
            break
        defect = float(residual.detach().abs().max())
        sdefect = float(source_defect.detach().abs().max())
        history.append(defect)
        source_history.append(sdefect)
        if max(defect, sdefect) < tol:
            x = decode(q.detach())
            actual = _cycle_nlevel(x, atmosphere, se_solver, formal_solve_mode).detach()
            actual_sources = _source_from_populations_nlevel(actual, model)
            actual_defect = max(float((encode(actual) - q.detach()).abs().max()),
                                float(torch.log(actual_sources / sources.detach()).abs().max()))
            if actual_defect < tol:
                return PicardResultMultilevel(x, len(history), history, False, True, source_history)
            failure = f"formal-cycle defect {actual_defect} exceeds tolerance"
            break
        # Local derivatives are block diagonal in depth; only transfer couples
        # different depths. Assemble that chain rule without dense AD sweeps.
        ds_dq = torch.stack([torch.autograd.grad(sources[:, k].sum(), q, retain_graph=True)[0]
                             for k in range(model.K)], 1)
        jlocal = radiation.detach().requires_grad_(True)
        qnew = encode(se_solver(jlocal, T, ne))
        if qnew.requires_grad:
            dq_dj = torch.stack([torch.autograd.grad(qnew[:, a].sum(), jlocal, retain_graph=True)[0]
                                 for a in range(D)], 1)
        else:
            if (jlocal <= 0).any():
                failure = "legacy numerical SE derivative requires positive radiation"
                break
            columns = []
            for k in range(model.K):
                displacement = torch.zeros_like(jlocal)
                step_k = jlocal[:, k].detach() * 1e-4
                displacement[:, k] = step_k
                plus = encode(se_solver(jlocal.detach()+displacement, T, ne))
                minus = encode(se_solver(jlocal.detach()-displacement, T, ne))
                columns.append((plus-minus)/(2*step_k[:, None]))
            dq_dj = torch.stack(columns, -1)
        K = torch.einsum("nak,nm,mkb->namb", dq_dj, lam, ds_dq).reshape(N*D, N*D)
        try:
            step = torch.linalg.solve(torch.eye(N*D, dtype=q.dtype, device=q.device) - K,
                                      residual.detach().reshape(-1)).reshape_as(q)
        except torch.linalg.LinAlgError as exc:
            failure = f"singular coupled Newton matrix: {exc}"
            break
        accepted = False
        for backtrack in range(32):
            trial_q = q.detach() + step.detach() * 0.5**backtrack
            try:
                trial_r, trial_s, *_ = evaluate(trial_q)
                trial_norm = max(float(trial_r.abs().max()), float(trial_s.abs().max()))
            except ValueError:
                continue
            if trial_norm < max(defect, sdefect):
                q = trial_q
                accepted = True
                break
        if not accepted:
            failure = "Newton line search failed to reduce population/source cycle defect"
            break
    return PicardResultMultilevel(decode(q.detach()), len(history), history, False,
                                  False, source_history, failure)
