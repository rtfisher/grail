"""Gate 3 — GMRES scaling: iteration count vs optical depth.

Verification contract: "GMRES scaling: iteration count vs optical depth within 2× of
ALI brackets."

Measurement
-----------
Run resolvent_solve on isothermal atmospheres at τ_max ∈ {1e2, 1e3, 1e4}
with synthetic emulator residuals.  Record GMRES iteration count at each
τ_max.

Pass criterion
--------------
iter_count(τ_max=1e4) < 2 × iter_count(τ_max=1e2).
This is the ALI-like scaling proxy: the GMRES iteration count should not
grow faster than linearly (a 2× increase per 2 decades of τ_max).

Implementation notes
--------------------
We use the atmospheric JVPs (real D(T_RT) and D(T_SE)) computed at the
truth fixed point of each atmosphere.  This ensures the gate measures
the real resolvent behavior, not synthetic linear algebra.

The synthetic δT is a seed=42 Gaussian in an orthonormal population-
conservation tangent basis, with coordinate amplitude 1e-3.  Success also
requires true relative defects <=1e-6 and no 300-iteration cap hits.
The legacy 3-level relative threshold is 5; multilevel additionally
retains the original SPEC M.6 absolute high-depth limit of <76 iterations.

fast=True : n_tau=30 per atmosphere (cost: ~3 resolvent solves).
fast=False: n_tau=60 per atmosphere.
"""

from __future__ import annotations

import torch

from .runner import GateResult
from ._numerical_contracts import (
    LinearDiagnostics, conserving_probe, diagnose_linear_solve, record_population_inputs,
)

_NU_0 = 2.47e15
_EPSILON = 1e-4
_TAU_MAX_SWEEP = [1e2, 1e3, 1e4]


_ML_T0 = 8000.0
_ML_N_E0 = 1e12
_ML_EPSILON = 1e-4


def _measure_at_tau_max(n_tau: int, tau_max: float, setup: str = "2level") -> LinearDiagnostics:
    """Solve the resolvent equation and re-evaluate its true relative defect.

    setup='2level' uses the original 2-level atmosphere and JVPs.
    setup='3level' uses the 3-level chained atmosphere.
    setup='multilevel' uses the 5-level Balmer-H atmosphere with the M.5 JVPs.
    """
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid
    from atmosphere.formal_solver import formal_solve, line_profile_average
    from grail import resolvent_solve

    # Multilevel uses a coarser frequency/angle grid for tractability.
    if setup == "multilevel":
        n_nu = 11
        n_mu = 3
    else:
        n_nu = 21
        n_mu = 5

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=tau_max, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=n_nu, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=n_mu)

    if setup == "multilevel":
        # Use truth SE JVPs (same as 2-level/3-level pattern) to measure real
        # GMRES scaling of the physical resolvent (not APNN-coupled).
        from atmosphere.multilevel.fixed_point import (
            ali_loop_nlevel,
            _compute_lambda_full_nlevel,
            _source_from_populations_nlevel,
        )
        from atmosphere.multilevel.rate_equations import rate_equation_solve_nlevel

        L = 5
        K = 10

        atm = Atmosphere.balmer_isothermal(
            T0=_ML_T0, n_e0=_ML_N_E0, epsilon=_ML_EPSILON,
            depth=depth, frequency=frequency, angle=angle,
        )
        atomic_model = atm.atomic_model

        result_truth = ali_loop_nlevel(atm, tol=1e-8, max_iter=200, mode="direct")
        x_truth = result_truth.populations.detach()
        x_truth_flat = x_truth.T.flatten()
        J_bar_lines_truth = result_truth.J_bar_lines.detach()

        # Precomputed-Λ host JVP (fast per GMRES call).
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
                _se_step,
                (J_bar_lines_truth.clone(),),
                (jbar_tangent,),
            )
            return pops_tangent.T.reshape(-1)   # (L*N_tau,)

        delta_T = conserving_probe(n_tau, L, 42)

        # Build selective Λ*/Λ preconditioner (M.6 option 1b): full Λ_k for the
        # two stiffest resonance lines (Ly-α k=0, H-α k=4) and diagonal Λ*_k
        # for the remaining 8 transitions.  Assembles the full (L·N_τ × L·N_τ)
        # K* matrix, LU-factors (I − K*) once, and applies via lu_solve per GMRES
        # iteration.  The diagonal-only version is retained but no longer the
        # gate 3 default; multilevel only.
        from atmosphere.multilevel.precond_lambda_star import (
            make_selective_lambda_preconditioner_multilevel,
        )
        precond = make_selective_lambda_preconditioner_multilevel(
            atm, x_truth, J_bar_lines_truth=J_bar_lines_truth
        )

        res = resolvent_solve(
            x_truth_flat, delta_T, se_jvp, host_jvp,
            gmres_tol=1e-6, gmres_maxiter=300,
            preconditioner=precond,
        )
        diagnostic = diagnose_linear_solve(
            res, delta_T, se_jvp, host_jvp, tolerance=1e-6,
            iteration_cap=300, n_tau=n_tau, n_levels=L,
        )
        return record_population_inputs(diagnostic, atm, x_truth, J_bar_lines_truth,
            cycle_populations=basepoint_cycle,
            solver_converged=getattr(result_truth, "converged", True),
            nonlinear_iterations=result_truth.iter_count)

    elif setup == "3level":
        from atmosphere.threelevel.fixed_point import (
            ali_loop_3level,
            _source_from_populations_3level,
            _atm_for_line,
        )
        from atmosphere.threelevel.rate_equations import rate_equation_solve_3level

        atm = Atmosphere.three_level_hydrogen_isothermal(
            T0=6000.0, n_e0=1e10,
            epsilon_lya=_EPSILON, epsilon_ha=_EPSILON,
            depth=depth, frequency=frequency, angle=angle,
        )
        N_nu = atm.frequency.nu.shape[0]
        L = 3
        K = 2
        atomic_model = atm.atomic_model

        result_truth = ali_loop_3level(atm, tol=1e-8, max_iter=10, mode="direct")
        x_truth = result_truth.populations.detach()
        x_truth_flat = x_truth.T.flatten()
        source_truth = _source_from_populations_3level(x_truth, atm.atomic_model)
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
            result = torch.zeros(L * n_tau, dtype=torch.float64)
            for k in range(K):
                result[k * n_tau : (k + 1) * n_tau] = jbar_tangent[:, k]
            return result

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

        delta_T = conserving_probe(n_tau, L, 42)

    else:  # 2-level
        from atmosphere.fixed_point import ali_loop, _source_from_populations
        from atmosphere.rate_equations import rate_equation_solve

        atm = Atmosphere(
            depth=depth, frequency=frequency, angle=angle,
            T=torch.full((n_tau,), 6000.0, dtype=torch.float64),
            n_e=torch.full((n_tau,), 1e10, dtype=torch.float64),
            epsilon=_EPSILON, nu_0=_NU_0,
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
            result = torch.zeros(2 * n_tau, dtype=torch.float64)
            result[:n_tau] = jbar_tangent
            return result

        def se_jvp(w: torch.Tensor) -> torch.Tensor:
            w = w.to(dtype=torch.float64)
            jbar_tangent = w[:n_tau]

            def _se_step(jbar: torch.Tensor) -> torch.Tensor:
                return rate_equation_solve(
                    jbar, atm.T, atm.n_e, epsilon=_EPSILON, nu_0=_NU_0
                )

            _, pops_tangent = torch.func.jvp(
                _se_step, (J_bar_truth.clone(),), (jbar_tangent,)
            )
            return pops_tangent.T.flatten()

        L = 2
        delta_T = conserving_probe(n_tau, L, 42)

    res = resolvent_solve(
        x_truth_flat, delta_T, se_jvp, host_jvp,
        gmres_tol=1e-6, gmres_maxiter=300,
    )
    diagnostic = diagnose_linear_solve(
        res, delta_T, se_jvp, host_jvp, tolerance=1e-6,
        iteration_cap=300, n_tau=n_tau, n_levels=L,
    )
    j_bar = J_bar_lines_truth if setup == "3level" else J_bar_truth
    return record_population_inputs(diagnostic, atm, x_truth, j_bar,
        cycle_populations=basepoint_cycle,
        solver_converged=getattr(result_truth, "converged", True),
        nonlinear_iterations=result_truth.iter_count)


def _count_iters_at_tau_max(n_tau: int, tau_max: float, setup: str = "2level") -> int:
    """Legacy count-only helper; acceptance uses the complete diagnostics below."""
    return _measure_at_tau_max(n_tau, tau_max, setup).iterations


def _summarize_cases(setup: str, n_tau: int,
                     cases: dict[float, LinearDiagnostics]) -> GateResult:
    """A capped or unconverged solve never earns a scaling pass."""
    low = cases[_TAU_MAX_SWEEP[0]].iterations
    high = cases[_TAU_MAX_SWEEP[-1]].iterations
    multiplier = 2.0 if setup == "2level" else 5.0
    converged = all(case.converged for case in cases.values())
    uncapped = all(not case.cap_hit for case in cases.values())
    scales = high < multiplier * low
    # SPEC M.6's absolute limit was previously replaced by a nonequivalent
    # relative proxy.  Keep both requirements visible; do not tune either.
    absolute_limit = setup != "multilevel" or high < 76
    basepoints_valid = all(case.basepoint_valid for case in cases.values())
    passes = converged and uncapped and scales and absolute_limit and basepoints_valid
    details = "\n".join(f"  tau_max={tau:g}: {case.record()}"
                         for tau, case in cases.items())
    message = (
        f"[{setup}] N_tau={n_tau}, population conservation tangent; "
        f"ratio={high / max(low, 1):.6g}, original threshold <{multiplier:g}; "
        f"valid base equilibria: {basepoints_valid}; "
        f"all true relative defects <=1e-6: {converged}; "
        f"no 300-iteration cap hits: {uncapped}; "
        f"multilevel absolute <76 requirement: {absolute_limit}.\n{details}"
    )
    return GateResult(3, "GMRES scaling", "PASS" if passes else "FAIL",
                      float(high), float(min(multiplier * low, 76)
                                         if setup == "multilevel" else multiplier * low),
                      message)


def run(fast: bool = True, setup: str = "2level") -> GateResult:
    """Record original-tolerance true defects and counts for all three depths."""
    if setup not in {"2level", "3level", "multilevel"}:
        raise ValueError(f"Unknown setup {setup!r}")
    n_tau = 80 if setup == "multilevel" else 30 if fast else 60
    cases = {tau: _measure_at_tau_max(n_tau, tau, setup) for tau in _TAU_MAX_SWEEP}
    return _summarize_cases(setup, n_tau, cases)
