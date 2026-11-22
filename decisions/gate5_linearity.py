"""Gate 5 — nonlinear validity remains deferred; homogeneity is diagnostic.

Solving (I-K) dx = alpha*b at alpha={1, 1/2, 1/4} only checks homogeneity
of a linear solve.  It cannot measure a quadratic nonlinear remainder or
establish a finite linearization radius.  We retain that useful solver
check, require the original 1e-8 true defect for every solve, and keep it separate from the nonlinear gate, which returns DEFERRED.  A nonlinear gate requires independent perturbed
fixed points and actual error amplitudes; those are separate experiments.
The 5% threshold below is the historical diagnostic ceiling, not a claimed
nonlinear acceptance result.  Population probes conserve total population.
"""

from __future__ import annotations

import torch

from .runner import GateResult
from ._numerical_contracts import (
    conserving_probe, diagnose_linear_solve, record_population_inputs,
)

_NU_0 = 2.47e15
_LINEARITY_TOL = 0.05  # 5% relative residual threshold for this verification contract

# Multilevel operating-point defaults (M.6 spec).
_ML_T0 = 8000.0
_ML_N_E0 = 1e12
_ML_EPSILON = 1e-4


def _run_one_config_2level(n_tau: int, epsilon: float) -> tuple[list[float], bool]:
    """Run linear-solve homogeneity diagnostic for one 2-level atmosphere config."""
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid
    from atmosphere.fixed_point import ali_loop, _source_from_populations
    from atmosphere.formal_solver import formal_solve, line_profile_average
    from atmosphere.rate_equations import rate_equation_solve
    from grail import resolvent_solve

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=21, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=5)
    atm = Atmosphere(
        depth=depth, frequency=frequency, angle=angle,
        T=torch.full((n_tau,), 6000.0, dtype=torch.float64),
        n_e=torch.full((n_tau,), 1e10, dtype=torch.float64),
        epsilon=epsilon, nu_0=_NU_0,
    )
    N_nu = atm.frequency.nu.shape[0]

    result_truth = ali_loop(atm, tol=1e-8, max_iter=10, mode="direct")
    x_truth = result_truth.populations.detach()
    x_truth_flat = x_truth.T.flatten()
    source_truth = _source_from_populations(x_truth, atm)
    J_bar_truth = line_profile_average(formal_solve(
        source_truth[:, None].expand(-1, N_nu), atm), atm).detach()
    basepoint_cycle = rate_equation_solve(
        J_bar_truth, atm.T, atm.n_e,
        epsilon=atm.epsilon, nu_0=atm.nu_0).detach()

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=torch.float64)
        v_pops = v.reshape(2, n_tau).T
        pops_base = x_truth.detach().clone()

        def _rt_step(pops: torch.Tensor) -> torch.Tensor:
            sl = _source_from_populations(pops, atm)
            s_full = sl.unsqueeze(1).expand(-1, N_nu)
            jnu = formal_solve(s_full, atm)
            return line_profile_average(jnu, atm)

        _, jbar_tangent = torch.func.jvp(_rt_step, (pops_base,), (v_pops,))
        result_v = torch.zeros(2 * n_tau, dtype=torch.float64)
        result_v[:n_tau] = jbar_tangent
        return result_v

    def se_jvp(w: torch.Tensor) -> torch.Tensor:
        w = w.to(dtype=torch.float64)
        jbar_tangent = w[:n_tau]

        def _se_step(jbar: torch.Tensor) -> torch.Tensor:
            return rate_equation_solve(
                jbar, atm.T, atm.n_e, epsilon=epsilon, nu_0=_NU_0
            )

        _, pops_tangent = torch.func.jvp(
            _se_step, (J_bar_truth.clone(),), (jbar_tangent,)
        )
        return pops_tangent.T.flatten()

    delta_T_base = conserving_probe(n_tau, 2, 42 + int(epsilon * 1e4))

    return _homogeneity_check(
        x_truth_flat, delta_T_base, se_jvp, host_jvp,
        atmosphere=atm, populations=x_truth, j_bar=J_bar_truth,
        cycle_populations=basepoint_cycle,
        solver_converged=getattr(result_truth, "converged", True),
        nonlinear_iterations=result_truth.iter_count)


def _run_one_config_3level(
    n_tau: int, eps_lya: float, eps_ha: float
) -> tuple[list[float], bool]:
    """Run linear-solve homogeneity diagnostic for one 3-level atmosphere config."""
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid
    from atmosphere.threelevel.fixed_point import (
        ali_loop_3level,
        _source_from_populations_3level,
        _atm_for_line,
    )
    from atmosphere.formal_solver import formal_solve, line_profile_average
    from atmosphere.threelevel.rate_equations import rate_equation_solve_3level
    from grail import resolvent_solve

    L = 3
    K = 2
    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=21, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=5)
    atm = Atmosphere.three_level_hydrogen_isothermal(
        T0=6000.0, n_e0=1e10,
        epsilon_lya=eps_lya, epsilon_ha=eps_ha,
        depth=depth, frequency=frequency, angle=angle,
    )
    N_nu = atm.frequency.nu.shape[0]
    atomic_model = atm.atomic_model

    result_truth = ali_loop_3level(atm, tol=1e-8, max_iter=10, mode="direct")
    x_truth = result_truth.populations.detach()
    x_truth_flat = x_truth.T.flatten()
    source_truth = _source_from_populations_3level(x_truth, atomic_model)
    J_bar_lines_truth = torch.stack([
        line_profile_average(formal_solve(
            source_truth[:, k:k+1].expand(-1, N_nu), _atm_for_line(atm, k)),
            _atm_for_line(atm, k)) for k in range(K)], dim=1).detach()
    basepoint_cycle = rate_equation_solve_3level(
        J_bar_lines_truth, atm.T, atm.n_e, atomic_model=atm.atomic_model).detach()

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=torch.float64)
        v_pops = v.reshape(L, n_tau).T

        def _rt_step(pops: torch.Tensor) -> torch.Tensor:
            S_lines = _source_from_populations_3level(pops, atomic_model)
            J_bar_list = []
            for k in range(K):
                atm_k = _atm_for_line(atm, k)
                S_k = S_lines[:, k]
                S_full = S_k.unsqueeze(1).expand(-1, N_nu)
                J_nu = formal_solve(S_full, atm_k)
                J_bar_k = line_profile_average(J_nu, atm_k)
                J_bar_list.append(J_bar_k)
            return torch.stack(J_bar_list, dim=1)

        _, jbar_tangent = torch.func.jvp(_rt_step, (x_truth.detach().clone(),), (v_pops,))
        result_v = torch.zeros(L * n_tau, dtype=torch.float64)
        for k in range(K):
            result_v[k * n_tau : (k + 1) * n_tau] = jbar_tangent[:, k]
        return result_v

    def se_jvp(w: torch.Tensor) -> torch.Tensor:
        w = w.to(dtype=torch.float64)
        jbar_tangent = torch.zeros(n_tau, K, dtype=torch.float64)
        for k in range(K):
            jbar_tangent[:, k] = w[k * n_tau : (k + 1) * n_tau]

        def _se_step(jbar_lines: torch.Tensor) -> torch.Tensor:
            return rate_equation_solve_3level(
                jbar_lines, atm.T.to(torch.float64), atm.n_e.to(torch.float64),
                atomic_model=atomic_model,
            )

        _, pops_tangent = torch.func.jvp(
            _se_step, (J_bar_lines_truth.clone(),), (jbar_tangent,)
        )
        return pops_tangent.T.flatten()

    seed = 42 + int(eps_lya * 1e4) + int(eps_ha * 1e6)
    delta_T_base = conserving_probe(n_tau, L, seed)

    return _homogeneity_check(
        x_truth_flat, delta_T_base, se_jvp, host_jvp,
        atmosphere=atm, populations=x_truth, j_bar=J_bar_lines_truth,
        cycle_populations=basepoint_cycle,
        solver_converged=getattr(result_truth, "converged", True),
        nonlinear_iterations=result_truth.iter_count)


def _run_one_config_multilevel(
    n_tau: int, epsilon: float
) -> tuple[list[float], bool]:
    """Run linear-solve homogeneity diagnostic for one 5-level multilevel atmosphere config.

    Uses truth SE JVPs (same pattern as 2-level/3-level), not the APNN FD-JVP.
    The resolvent equation is exactly linear in δT, so the 5% residual threshold
    applies regardless of whether truth or emulator JVPs are used.  The truth-JVP
    path avoids the APNN's numerical instability at near-zero J_bar values.
    """
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid
    from atmosphere.multilevel.fixed_point import (
        ali_loop_nlevel,
        _compute_lambda_full_nlevel,
        _source_from_populations_nlevel,
    )
    from atmosphere.multilevel.rate_equations import rate_equation_solve_nlevel

    L = 5
    K = 10
    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=11, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=3)
    atm = Atmosphere.balmer_isothermal(
        T0=_ML_T0, n_e0=_ML_N_E0, epsilon=epsilon,
        depth=depth, frequency=frequency, angle=angle,
    )
    atomic_model = atm.atomic_model

    result_truth = ali_loop_nlevel(atm, tol=1e-8, max_iter=200, mode="direct")
    x_truth = result_truth.populations.detach()
    x_truth_flat = x_truth.T.flatten()
    J_bar_lines_truth = result_truth.J_bar_lines.detach()

    # Precomputed-Λ host JVP.
    lambda_records = _compute_lambda_full_nlevel(atm)
    Lambda_list = [lr[0] for lr in lambda_records]
    source_truth = _source_from_populations_nlevel(x_truth, atomic_model)
    J_bar_lines_truth = torch.stack([
        lam @ source_truth[:, k] + boundary
        for k, (lam, boundary) in enumerate(lambda_records)], dim=1).detach()
    basepoint_cycle = rate_equation_solve_nlevel(
        J_bar_lines_truth, atm.T, atm.n_e, atomic_model=atm.atomic_model).detach()
    x_base = x_truth.detach().clone().to(torch.float64)

    def _S_from_pops(pops: torch.Tensor) -> torch.Tensor:
        return _source_from_populations_nlevel(pops, atomic_model)

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=torch.float64).reshape(-1)
        v_pops = v.reshape(L, n_tau).T.clone()
        _, dS_lines = torch.func.jvp(_S_from_pops, (x_base,), (v_pops,))
        dJ_bar_list = [Lambda_list[k] @ dS_lines[:, k] for k in range(K)]
        return torch.stack(dJ_bar_list, dim=1).T.reshape(-1)   # (K*N_tau,)

    # Truth SE JVP.
    def se_jvp(w: torch.Tensor) -> torch.Tensor:
        w = w.to(dtype=torch.float64).reshape(-1)
        jbar_tangent = w.reshape(K, n_tau).T.clone()

        def _se_step(jbar_lines: torch.Tensor) -> torch.Tensor:
            return rate_equation_solve_nlevel(
                jbar_lines,
                atm.T.to(torch.float64),
                atm.n_e.to(torch.float64),
                atomic_model=atomic_model,
            )

        _, pops_tangent = torch.func.jvp(
            _se_step, (J_bar_lines_truth.clone(),), (jbar_tangent,),
        )
        return pops_tangent.T.reshape(-1)   # (L*N_tau,)

    seed = 42 + int(epsilon * 1e4)
    delta_T_base = conserving_probe(n_tau, L, seed)

    return _homogeneity_check(
        x_truth_flat, delta_T_base, se_jvp, host_jvp,
        atmosphere=atm, populations=x_truth, j_bar=J_bar_lines_truth,
        cycle_populations=basepoint_cycle,
        solver_converged=getattr(result_truth, "converged", True),
        nonlinear_iterations=result_truth.iter_count)


def _homogeneity_diagnostics(
    x_truth_flat: torch.Tensor,
    delta_T_base: torch.Tensor,
    se_jvp,
    host_jvp,
    *, atmosphere=None, populations=None, j_bar=None,
    cycle_populations=None, solver_converged=True, nonlinear_iterations=None,
) -> tuple[list[float], bool, list]:
    """Return homogeneity values and true equation defects, never nonlinear order."""
    from grail import resolvent_solve

    alphas = [1.0, 0.5, 0.25]
    dx_stars: dict[float, torch.Tensor] = {}
    diagnostics = []
    for alpha in alphas:
        res = resolvent_solve(
            x_truth_flat,
            alpha * delta_T_base,
            se_jvp,
            host_jvp,
            gmres_tol=1e-8,
            gmres_maxiter=200,
        )
        dx_stars[alpha] = res.delta_x_star
        # n_tau/n_levels are irrelevant without a dense restriction here.
        diagnostics.append(diagnose_linear_solve(
            res, alpha * delta_T_base, se_jvp, host_jvp,
            tolerance=1e-8, iteration_cap=200, n_tau=1,
            n_levels=len(x_truth_flat),
        ))
        if atmosphere is not None:
            record_population_inputs(diagnostics[-1], atmosphere, populations, j_bar,
                                     cycle_populations=cycle_populations,
                                     solver_converged=solver_converged,
                                     nonlinear_iterations=nonlinear_iterations)

    dx_ref = dx_stars[1.0]

    rel_residuals = []
    for alpha in alphas:
        dx_a = dx_stars[alpha]
        diff = (dx_a - alpha * dx_ref).norm().item()
        denom = (alpha * dx_ref).norm().item()
        rel = diff / denom if denom > 0 else 0.0
        rel_residuals.append(rel)

    passes = (all(r < _LINEARITY_TOL for r in rel_residuals)
              and all(d.converged and not d.cap_hit and d.basepoint_valid
                      for d in diagnostics))
    return rel_residuals, passes, diagnostics


def _homogeneity_check(x_truth_flat, delta_T_base, se_jvp, host_jvp, **inputs):
    residuals, passes, _ = _homogeneity_diagnostics(
        x_truth_flat, delta_T_base, se_jvp, host_jvp, **inputs)
    return residuals, passes


def run(fast: bool = True, setup: str = "2level") -> GateResult:
    """Defer nonlinear validity until an independent nonlinear family is configured.

    Private homogeneity helpers remain available as solver diagnostics. They
    cannot establish a quadratic remainder, including when they return zero.
    """
    if setup not in {"2level", "3level", "multilevel"}:
        raise ValueError(f"Unknown setup {setup!r}")
    return GateResult(
        gate_id=5, name="Linearity validity", status="DEFERRED",
        measured=None, expected=None,
        message=(
            f"[{setup}] Nonlinear validity unassessed: no independent "
            "re-equilibrated nonlinear family is configured in this legacy gate. "
            "Scaled linear solves measure homogeneity and cannot establish a "
            "quadratic remainder. The separate local nonlinear studies are "
            "independent evidence, not a pass of this generic gate."
        ),
    )
