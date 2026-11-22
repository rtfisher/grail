"""Gate 2 — directional gain and independently checked tangent inverse norm.

The physical truth-cycle JVPs are evaluated at the direct truth equilibrium.
A seeded synthetic residual conserves total population at every depth.  Its
realized gain ||G b||/||b|| is reported separately from A_op = ||G||_2.
For tangent dimension <=256, a dense Q.T (I-K) Q and its smallest singular
value give A_op in orthonormal conservation-tangent coordinates.  Each
probe is checked against its own dense inverse, never another case's
condition number.  The Hessenberg condition estimate is diagnostic only.
Larger systems are DEFERRED until an independent inverse norm is available.
A true linear defect above the original GMRES tolerance is a FAIL, including
capped solves.  No nonlinear surrogate or finite-amplitude claim is made.
"""

from __future__ import annotations

import math

import torch

from .runner import GateResult
from ._numerical_contracts import (
    LinearDiagnostics, conserving_probe, diagnose_linear_solve, record_population_inputs,
)

_NU_0_2LEVEL = 2.47e15
_DENSE_TANGENT_LIMIT = 256
# Original minimum amplification threshold; cross-line coupling requires a separate test.
_A_OP_MIN_3LEVEL = 1.1
# Minimum A_op required in multilevel setup at deep ε.
# Spec §M.6 expects ~10²–10³; a non-trivial threshold of 10 confirms amplification.
_A_OP_MIN_MULTILEVEL = 10.0

# Multilevel operating-point defaults (M.6 spec).
_ML_T0 = 8000.0
_ML_N_E0 = 1e12
_ML_EPSILON = 1e-4
_ML_TAU_MAX = 1e4


def _build_atmosphere_2level(n_tau: int, epsilon: float):
    """Build a 2-level isothermal atmosphere."""
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=21, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=5)
    return Atmosphere(
        depth=depth, frequency=frequency, angle=angle,
        T=torch.full((n_tau,), 6000.0, dtype=torch.float64),
        n_e=torch.full((n_tau,), 1e10, dtype=torch.float64),
        epsilon=epsilon, nu_0=_NU_0_2LEVEL,
    )


def _build_atmosphere_3level(n_tau: int, eps_lya: float, eps_ha: float):
    """Build a 3-level isothermal atmosphere."""
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=21, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=5)
    return Atmosphere.three_level_hydrogen_isothermal(
        T0=6000.0, n_e0=1e10,
        epsilon_lya=eps_lya, epsilon_ha=eps_ha,
        depth=depth, frequency=frequency, angle=angle,
    )


def _measure_one_2level(n_tau: int, epsilon: float) -> LinearDiagnostics:
    """Measure the directional gain and independent tangent inverse norm for one 2-level config.

    Returns independently evaluated linear diagnostics and raw arrays.
    """
    from atmosphere.fixed_point import ali_loop, _source_from_populations
    from atmosphere.formal_solver import formal_solve, line_profile_average
    from atmosphere.rate_equations import rate_equation_solve
    from grail import resolvent_solve

    atm = _build_atmosphere_2level(n_tau, epsilon)
    N_nu = atm.frequency.nu.shape[0]

    # Truth fixed point.
    result_truth = ali_loop(atm, tol=1e-8, max_iter=10, mode="direct")
    x_truth = result_truth.populations.detach()
    x_truth_flat = x_truth.T.flatten()
    source_truth = _source_from_populations(x_truth, atm)
    J_bar_truth = line_profile_average(formal_solve(
        source_truth[:, None].expand(-1, N_nu), atm), atm).detach()
    basepoint_cycle = rate_equation_solve(
        J_bar_truth, atm.T, atm.n_e,
        epsilon=atm.epsilon, nu_0=atm.nu_0).detach()

    # Host JVP (D(T_RT)): tangent in (2*N_tau,), output in (2*N_tau,).
    # The RT step maps populations -> J_bar; the state-vector convention
    # is flat blocked: first N_tau entries are n_1, next N_tau are n_2.
    # The RT step only populates the J_bar component (first N_tau entries
    # of the output); the population component is set to zero.
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

    # SE JVP (D(T_SE)).
    def se_jvp(w: torch.Tensor) -> torch.Tensor:
        w = w.to(dtype=torch.float64)
        jbar_tangent = w[:n_tau]

        def _se_step(jbar: torch.Tensor) -> torch.Tensor:
            return rate_equation_solve(
                jbar, atm.T, atm.n_e, epsilon=epsilon, nu_0=_NU_0_2LEVEL
            )

        _, pops_tangent = torch.func.jvp(
            _se_step, (J_bar_truth.clone(),), (jbar_tangent,)
        )
        return pops_tangent.T.flatten()

    # Synthetic δT injection (~0.1% of population scale).
    delta_T_inj = conserving_probe(n_tau, 2, 42 + int(epsilon * 1e4))

    res = resolvent_solve(
        x_truth_flat, delta_T_inj, se_jvp, host_jvp,
        gmres_tol=1e-8, gmres_maxiter=200,
    )
    diagnostic = diagnose_linear_solve(
        res, delta_T_inj, se_jvp, host_jvp, tolerance=1e-8,
        iteration_cap=200, n_tau=n_tau, n_levels=2,
        dense_limit=_DENSE_TANGENT_LIMIT,
    )
    return record_population_inputs(diagnostic, atm, x_truth, J_bar_truth,
        cycle_populations=basepoint_cycle,
        solver_converged=getattr(result_truth, "converged", True),
        nonlinear_iterations=result_truth.iter_count)


def _measure_one_3level(
    n_tau: int, eps_lya: float, eps_ha: float
) -> LinearDiagnostics:
    """Measure the directional gain and independent tangent inverse norm for one 3-level config.

    The state vector is flat blocked with length L*N_tau = 3*N_tau:
      positions 0..N_tau-1    : n_1 (level populations)
      positions N_tau..2*N_tau-1 : n_2
      positions 2*N_tau..3*N_tau-1 : n_3
    Convention: populations (N_tau, 3) -> flat via populations.T.flatten().
    Inverse: flat.reshape(3, N_tau).T.

    The host_jvp maps the population tangent to a J_bar_lines tangent
    (one J_bar per line, K=2).  The output layout packs the two J_bar
    tangents into the first 2*N_tau entries; the third N_tau block is
    zeroed (the populations block is updated only by the SE step).

    The se_jvp uses the truth SE solver at J_bar_lines_truth.

    Returns independently evaluated linear diagnostics and raw arrays.
    """
    from atmosphere.threelevel.fixed_point import (
        ali_loop_3level,
        _source_from_populations_3level,
        _atm_for_line,
    )
    from atmosphere.formal_solver import formal_solve, line_profile_average
    from atmosphere.threelevel.rate_equations import rate_equation_solve_3level
    from grail import resolvent_solve

    L = 3   # number of levels
    K = 2   # number of transitions (Ly-α + H-α)

    atm = _build_atmosphere_3level(n_tau, eps_lya, eps_ha)
    N_nu = atm.frequency.nu.shape[0]

    # Truth fixed point.
    result_truth = ali_loop_3level(atm, tol=1e-8, max_iter=10, mode="direct")
    x_truth = result_truth.populations.detach()   # (N_tau, 3)
    x_truth_flat = x_truth.T.flatten()            # (3*N_tau,)
    source_truth = _source_from_populations_3level(x_truth, atm.atomic_model)
    J_bar_lines_truth = torch.stack([
        line_profile_average(formal_solve(
            source_truth[:, k:k+1].expand(-1, N_nu), _atm_for_line(atm, k)),
            _atm_for_line(atm, k)) for k in range(K)], dim=1).detach()
    basepoint_cycle = rate_equation_solve_3level(
        J_bar_lines_truth, atm.T, atm.n_e, atomic_model=atm.atomic_model).detach()

    # Host JVP (D(T_RT)): tangent (3*N_tau,) -> (3*N_tau,).
    # The RT step: populations -> S_lines (N_tau, K) -> J_nu (N_tau, N_nu) per line
    # -> J_bar_lines (N_tau, K).  State-vector packing:
    #   out[0:N_tau]         = ∂J_bar_lya / ∂pops · v_pops
    #   out[N_tau:2*N_tau]   = ∂J_bar_ha  / ∂pops · v_pops
    #   out[2*N_tau:3*N_tau] = 0   (populations block; set by SE step)
    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=torch.float64)
        v_pops = v.reshape(L, n_tau).T    # (N_tau, 3)
        pops_base = x_truth.detach().clone()

        def _rt_step(pops: torch.Tensor) -> torch.Tensor:
            # Returns J_bar_lines of shape (N_tau, K).
            S_lines = _source_from_populations_3level(pops, atm.atomic_model)
            J_bar_list = []
            for k in range(K):
                atm_k = _atm_for_line(atm, k)
                S_k = S_lines[:, k]
                S_full = S_k.unsqueeze(1).expand(-1, N_nu)
                J_nu = formal_solve(S_full, atm_k)
                J_bar_k = line_profile_average(J_nu, atm_k)
                J_bar_list.append(J_bar_k)
            return torch.stack(J_bar_list, dim=1)   # (N_tau, K)

        _, jbar_tangent = torch.func.jvp(_rt_step, (pops_base,), (v_pops,))
        # jbar_tangent: (N_tau, K)
        result = torch.zeros(L * n_tau, dtype=torch.float64)
        for k in range(K):
            result[k * n_tau : (k + 1) * n_tau] = jbar_tangent[:, k]
        return result

    # SE JVP (D(T_SE)): tangent (3*N_tau,) -> (3*N_tau,).
    # The SE step: J_bar_lines (N_tau, K) -> populations (N_tau, 3).
    # Input packing:
    #   w[0:N_tau]         = ∂J_bar_lya (the J_bar tangent for Ly-α)
    #   w[N_tau:2*N_tau]   = ∂J_bar_ha  (the J_bar tangent for H-α)
    #   w[2*N_tau:3*N_tau] = 0   (not used by the SE step)
    # Output: populations tangent packed as (3*N_tau,) flat blocked.
    atomic_model = atm.atomic_model

    def se_jvp(w: torch.Tensor) -> torch.Tensor:
        w = w.to(dtype=torch.float64)
        # Extract J_bar tangents for each line.
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
        return pops_tangent.T.flatten()   # (3*N_tau,)

    # Synthetic δT injection (~0.1% of population scale).
    seed = 42 + int(eps_lya * 1e4) + int(eps_ha * 1e6)
    delta_T_inj = conserving_probe(n_tau, L, seed)

    res = resolvent_solve(
        x_truth_flat, delta_T_inj, se_jvp, host_jvp,
        gmres_tol=1e-8, gmres_maxiter=300,
    )
    diagnostic = diagnose_linear_solve(
        res, delta_T_inj, se_jvp, host_jvp, tolerance=1e-8,
        iteration_cap=300, n_tau=n_tau, n_levels=3,
        dense_limit=_DENSE_TANGENT_LIMIT,
    )
    return record_population_inputs(diagnostic, atm, x_truth, J_bar_lines_truth,
        cycle_populations=basepoint_cycle,
        solver_converged=getattr(result_truth, "converged", True),
        nonlinear_iterations=result_truth.iter_count)


def _build_atmosphere_multilevel(n_tau: int, epsilon: float):
    """Build a 5-level isothermal Balmer-H atmosphere at the M.6 operating point."""
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=_ML_TAU_MAX, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=11, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=3)
    return Atmosphere.balmer_isothermal(
        T0=_ML_T0, n_e0=_ML_N_E0, epsilon=epsilon,
        depth=depth, frequency=frequency, angle=angle,
    )


def _measure_one_multilevel(
    n_tau: int, epsilon: float
) -> LinearDiagnostics:
    """Measure the directional gain and independent tangent inverse norm for one multilevel config.

    Mirrors the 2-level and 3-level measurement functions: uses the *truth*
    SE JVP (torch.func.jvp on rate_equation_solve_nlevel) and the precomputed-Λ
    host_jvp, NOT the APNN FD-JVP.  This is the correct approach for measuring
    the operator-norm amplification A_op = ||(I - DT)^{-1}|| at x*_truth, which
    reflects the spectral structure of the physical cycle, not the APNN numerics.

    (The APNN FD-JVP is used only for the Born-remainder integration tests where
    we compare GRAIL's prediction to the actual APNN-coupled Picard fixed point.
    For gate 2 we measure the resolvent's operator norm, which uses the truth
    cycle's Jacobian exactly as gate2 does for 2-level and 3-level.)

    State vector: (L*N_tau,) = (5*N_tau,) flat-blocked.
    host_jvp: (L*N_tau,) -> (K*N_tau,) = (10*N_tau,) via precomputed Λ matrices.
    se_jvp:   (K*N_tau,) -> (L*N_tau,) via truth rate_equation_solve_nlevel.

    Returns independently evaluated linear diagnostics and raw arrays.
    """
    from atmosphere.multilevel.fixed_point import (
        ali_loop_nlevel,
        _compute_lambda_full_nlevel,
        _source_from_populations_nlevel,
    )
    from atmosphere.multilevel.rate_equations import rate_equation_solve_nlevel
    from grail import resolvent_solve

    L = 5
    K = 10

    atm = _build_atmosphere_multilevel(n_tau, epsilon)
    atomic_model = atm.atomic_model

    # Truth fixed point via direct-mode ALI.
    result_truth = ali_loop_nlevel(atm, tol=1e-8, max_iter=200, mode="direct")
    x_truth = result_truth.populations.detach()    # (N_tau, L)
    x_truth_flat = x_truth.T.flatten()             # (L*N_tau,)
    J_bar_lines_truth = result_truth.J_bar_lines.detach()   # (N_tau, K)

    # Host JVP: precomputed-Λ (fast at each GMRES call).
    # Mirrors make_host_jvp_5level but inlined here for consistency with the
    # 2/3-level gate pattern.
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
        dJ_bar = torch.stack(dJ_bar_list, dim=1)
        return dJ_bar.T.reshape(-1)   # (K*N_tau,)

    # SE JVP: torch.func.jvp through the truth N-level SE solver.
    def se_jvp(w: torch.Tensor) -> torch.Tensor:
        w = w.to(dtype=torch.float64).reshape(-1)
        # Unpack (K*N_tau,) -> (N_tau, K)
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

    # Synthetic δT injection (~0.1% of population scale).
    seed = 42 + int(epsilon * 1e4)
    delta_T_inj = conserving_probe(n_tau, L, seed)

    res = resolvent_solve(
        x_truth_flat, delta_T_inj, se_jvp, host_jvp,
        gmres_tol=1e-8, gmres_maxiter=300,
    )
    diagnostic = diagnose_linear_solve(
        res, delta_T_inj, se_jvp, host_jvp, tolerance=1e-8,
        iteration_cap=300, n_tau=n_tau, n_levels=5,
        dense_limit=_DENSE_TANGENT_LIMIT,
    )
    return record_population_inputs(diagnostic, atm, x_truth, J_bar_lines_truth,
        cycle_populations=basepoint_cycle,
        solver_converged=getattr(result_truth, "converged", True),
        nonlinear_iterations=result_truth.iter_count)


def _summarize_cases(setup: str, cases: list[tuple[str, LinearDiagnostics]]) -> GateResult:
    """Aggregate only independently checked, same-configuration quantities."""
    threshold = {"2level": 0.0, "3level": _A_OP_MIN_3LEVEL,
                 "multilevel": _A_OP_MIN_MULTILEVEL}[setup]
    failures = []
    unresolved = []
    details = []
    norms = []
    for label, case in cases:
        norm = case.tangent_inverse_norm
        if not case.basepoint_valid:
            failures.append(f"{label}: base equilibrium failed convergence/1e-8 cycle-defect contract")
        if not case.converged:
            failures.append(f"{label}: true defect exceeds 1e-8 or is nonfinite")
        if norm is None:
            unresolved.append(f"{label}: independent inverse norm not measured")
        else:
            if math.isfinite(norm):
                norms.append(norm)
            # The dense assembly/solve is independent of the Krylov iteration.
            # A 1e-8 defect budget also bounds the probe's inverse-norm ratio.
            if not math.isfinite(norm) or norm <= 0:
                failures.append(f"{label}: nonfinite/singular dense inverse")
            if (case.dense_relative_defect is None
                    or case.dense_relative_defect > 1e-8
                    or case.tangent_leakage is None
                    or case.tangent_leakage > 1e-10):
                failures.append(f"{label}: dense equation/tangent contract failed")
            if case.directional_gain > norm * (1.0 + 1e-8):
                failures.append(f"{label}: directional gain exceeds its own inverse norm")
        details.append(f"  {label}: {case.record()}")
    if norms and max(norms) <= threshold:
        failures.append(f"maximum tangent A_op <= original threshold {threshold}")
    status = "FAIL" if failures else "DEFERRED" if unresolved else "PASS"
    message = (
        f"[{setup}] Euclidean fractional-population conservation tangent; "
        "directional gains are not operator norms; projected conditions are not bounds. "
        + "; ".join(failures + unresolved) + "\n" + "\n".join(details)
    )
    return GateResult(2, "Amplification factor A", status,
                      max(norms) if norms else None, threshold, message)


def run(fast: bool = True, setup: str = "2level") -> GateResult:
    """Check each physical probe against the same case's independent dense norm."""
    if setup == "2level":
        configs = [(40, 1e-4)] if fast else [(80, 1e-2), (80, 1e-4)]
        cases = [(f"N_tau={n}, epsilon={ep:g}", _measure_one_2level(n, ep))
                 for n, ep in configs]
    elif setup == "3level":
        configs = [(30, 1e-2, 1e-2)] if fast else [
            (60, 1e-2, 1e-2), (60, 1e-2, 1e-4)]
        cases = [(f"N_tau={n}, epsilon=({a:g},{b:g})", _measure_one_3level(n, a, b))
                 for n, a, b in configs]
    elif setup == "multilevel":
        configs = [(80, _ML_EPSILON)] if fast else [(80, 1e-2), (80, _ML_EPSILON)]
        cases = [(f"N_tau={n}, epsilon={ep:g}", _measure_one_multilevel(n, ep))
                 for n, ep in configs]
    else:
        raise ValueError(f"Unknown setup {setup!r}")
    return _summarize_cases(setup, cases)
