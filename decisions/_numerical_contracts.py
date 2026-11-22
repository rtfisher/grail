"""Population-tangent coordinates and independently evaluated gate diagnostics.

All norms here are Euclidean in flat blocked fractional-population coordinates.
The orthonormal basis restricts each depth to sum_l delta n_l = 0; a random
direction is a probe of the inverse, never an estimate of its operator norm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


def population_tangent_basis(n_tau: int, n_levels: int) -> torch.Tensor:
    """Orthonormal Helmert basis in the repository's flat blocked layout."""
    helmert = torch.zeros(n_levels, n_levels - 1, dtype=torch.float64)
    for j in range(n_levels - 1):
        scale = math.sqrt((j + 1) * (j + 2))
        helmert[:j + 1, j] = 1.0 / scale
        helmert[j + 1, j] = -(j + 1) / scale
    return torch.kron(helmert, torch.eye(n_tau, dtype=torch.float64))


def conserving_probe(n_tau: int, n_levels: int, seed: int) -> torch.Tensor:
    """Deterministic synthetic RHS; amplitude is not a nonlinear experiment."""
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.randn((n_levels - 1) * n_tau, generator=generator,
                              dtype=torch.float64)
    return 1e-3 * (population_tangent_basis(n_tau, n_levels) @ coordinates)


@dataclass
class LinearDiagnostics:
    directional_gain: float
    true_relative_defect: float
    iterations: int
    iteration_cap: int
    cap_hit: bool
    converged: bool
    projected_condition_diagnostic: float
    tangent_inverse_norm: float | None = None
    dense_relative_defect: float | None = None
    gmres_relative_solution_error: float | None = None
    tangent_leakage: float | None = None
    basepoint_valid: bool = True
    basepoint_relative_defect: float | None = None
    basepoint_solver_converged: bool | None = None
    basepoint_iterations: int | None = None
    raw: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def record(self) -> dict:
        """JSON-ready scalars; nonfinite diagnostics are explicit nulls."""
        return {key: (None if isinstance(value, float) and not math.isfinite(value)
                      else value)
                for key, value in self.__dict__.items() if key != "raw"}


def record_population_inputs(diagnostic, atmosphere, populations, j_bar, *,
                             cycle_populations=None, solver_converged=True,
                             nonlinear_iterations=None):
    """Retain physical inputs and actual derivative evaluation points."""
    diagnostic.raw.update({
        "population_evaluation_point": populations.detach(),
        "se_radiation_evaluation_point": j_bar.detach(),
        "tau": atmosphere.depth.tau.detach(),
        "frequency_nodes": atmosphere.frequency.nu.detach(),
        "frequency_weights": atmosphere.frequency.weights.detach(),
        "frequency_profile": atmosphere.frequency.profile.detach(),
        "angle_mu": atmosphere.angle.mu.detach(),
        "angle_weights": atmosphere.angle.weights.detach(),
        "temperature": atmosphere.T.detach(),
        "electron_density": atmosphere.n_e.detach(),
    })
    if cycle_populations is not None:
        defect = cycle_populations - populations
        relative = float((defect.abs() / populations.abs().clamp(min=1e-300)).max())
        diagnostic.raw["basepoint_cycle_defect"] = defect.detach()
        diagnostic.basepoint_relative_defect = relative
        diagnostic.basepoint_solver_converged = bool(solver_converged)
        diagnostic.basepoint_iterations = nonlinear_iterations
        diagnostic.basepoint_valid = bool(solver_converged) and math.isfinite(relative) and relative <= 1e-8
    return diagnostic


def diagnose_linear_solve(res, rhs, se_jvp, host_jvp, *, tolerance: float,
                          iteration_cap: int, n_tau: int, n_levels: int,
                          dense_limit: int = 0) -> LinearDiagnostics:
    """Re-evaluate the original equation, optionally build a dense restriction.

    The Hessenberg condition number is retained only as a diagnostic.  The
    inverse norm is 1/sigma_min(Q.T (I-K) Q), computed independently of GMRES.
    A dense solve of that matrix also checks the propagated probe.  Exceeding
    dense_limit leaves the norm unmeasured rather than claiming a bound.
    """
    solution = res.delta_x_star.to(torch.float64)
    residual = solution - se_jvp(host_jvp(solution)) - rhs
    relative = float(residual.norm() / rhs.norm())
    finite = bool(torch.isfinite(solution).all()) and math.isfinite(relative)
    diagnostics = LinearDiagnostics(
        directional_gain=float(solution.norm() / rhs.norm()),
        true_relative_defect=relative, iterations=res.iter_count,
        iteration_cap=iteration_cap, cap_hit=res.iter_count >= iteration_cap,
        converged=finite and relative <= tolerance,
        projected_condition_diagnostic=res.condition_estimate,
        raw={"rhs": rhs.detach(), "solution": solution.detach(),
             "true_residual": residual.detach()},
    )
    dimension = (n_levels - 1) * n_tau
    if dimension > dense_limit:
        return diagnostics
    q = population_tangent_basis(n_tau, n_levels)
    aq = torch.stack([v - se_jvp(host_jvp(v)) for v in q.T], dim=1)
    reduced = q.T @ aq
    singular = torch.linalg.svdvals(reduced)
    inverse_norm = float(1.0 / singular[-1])
    direct = torch.linalg.solve(reduced, q.T @ rhs)
    direct_full = q @ direct
    direct_defect = direct_full - se_jvp(host_jvp(direct_full)) - rhs
    diagnostics.tangent_inverse_norm = inverse_norm
    diagnostics.dense_relative_defect = float(direct_defect.norm() / rhs.norm())
    diagnostics.gmres_relative_solution_error = float(
        (solution - direct_full).norm() / direct_full.norm())
    diagnostics.tangent_leakage = float((aq - q @ reduced).norm() / aq.norm())
    diagnostics.raw.update({"tangent_basis": q, "reduced_I_minus_K": reduced,
                            "singular_values": singular,
                            "dense_solution": direct_full})
    return diagnostics
