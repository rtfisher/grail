"""Tier R regression tests for the 2-level NLTE prototype + GRAIL solver.

Three tests load committed HDF5 references from `tests/regression/` and
assert that the host pipeline + GRAIL resolvent solve reproduce them to
documented tolerances.  Activated 2026-05-21 (PI sign-off; see
`examples/review_artifacts/tier_r_review.md`).

Test IDs:
  R-truth-fixedpoint   — `ali_loop` direct-mode + parabolic-S formal solver
                          on three isothermal Eddington-2-stream configs at
                          ε ∈ {1e-2, 1e-4, 1e-6}.  Verifies populations
                          match reference to 1e-6 relative plus a legacy
                          cancellation allowance (2 float64 eps) and
                          population conservation to 1e-8.
  R-formal-solver      — `formal_solve` on linear-S and Planck-on-power-law-T
                          sources.  Verifies J_ν matches committed
                          reference to 1e-6 rel tol, the Eddington-Barbier
                          surface anchor (J(0,x=0) = a/2 + b/4) to 1%,
                          and the Planck deep-boundary anchor to 4 sig figs.
  R-resolvent-output   — `resolvent_solve` on a deterministic n=20 synthetic
                          (A_rt, A_se) system (seed=42).  Verifies δx*
                          matches committed reference to 1e-6 rel tol and
                          iter_count within ±5 of the reference.

These tests are the 2-level analog of `test_regression_3level.py`.

the technical reference §6 pause-point items.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from pathlib import Path

import h5py
import pytest
import torch

_REG_DIR = Path(__file__).parent

# Planck constants (SI units; same as examples/generate_regression_proposals.py)
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23


def _eddington_atmosphere(T_array, epsilon, nu_0, n_e0, depth):
    """Eddington 2-stream (μ=1/√3, w=1) + single-frequency monochromatic
    (n_nu=1, φ=1) atmosphere.  Matches the
    `eddington_2stream_monochromatic_parabolic_OK` config used in the
    Avrett--Hummer-accurate reference generator.
    """
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.grid import AngleQuadrature, FrequencyGrid

    n_tau = depth.tau.shape[0]
    angle = AngleQuadrature(
        mu=torch.tensor([1.0 / math.sqrt(3.0)], dtype=torch.float64),
        weights=torch.tensor([1.0], dtype=torch.float64),
    )
    frequency = FrequencyGrid(
        nu=torch.zeros(1, dtype=torch.float64),
        weights=torch.ones(1, dtype=torch.float64),
        profile=torch.ones(1, dtype=torch.float64),
    )
    n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)
    return Atmosphere(
        depth=depth, frequency=frequency, angle=angle,
        T=T_array.to(torch.float64), n_e=n_e,
        epsilon=epsilon, nu_0=nu_0,
    )


# ---------------------------------------------------------------------------
# R-truth-fixedpoint
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_R_truth_fixedpoint() -> None:
    """R-truth-fixedpoint: ali_loop output matches committed reference.

    Three isothermal Eddington-2-stream configs at ε ∈ {1e-2, 1e-4, 1e-6}.
    The old fixtures formed n2 = 1 - n1, so their tiny excited fractions
    carry O(eps_machine) absolute cancellation error.  Preserve those files
    and the 1e-6 relative threshold; allow 2*eps_machine absolute error for
    the old reciprocal/subtraction evaluation at n_tot=1.  Independently
    verify today's n2 against a 60-digit Decimal SE formula to 32 float64
    eps relative, so this compatibility allowance cannot certify a poorly
    evaluated small fraction.  Population conservation remains <1e-8.
    """
    from atmosphere.fixed_point import ali_loop
    from atmosphere.grid import DepthGrid

    ref_path = _REG_DIR / "truth_fixedpoint.h5"
    assert ref_path.exists(), f"Reference file not found: {ref_path}"

    isothermal_keys = [
        "isothermal_eps1e-02",
        "isothermal_eps1e-04",
        "isothermal_eps1e-06",
    ]

    with h5py.File(ref_path, "r") as f:
        for key in isothermal_keys:
            grp = f[key]
            T0 = float(grp.attrs["T0"])
            eps = float(grp.attrs["epsilon"])
            nu_0 = float(grp.attrs["nu_0"])
            n_tau = int(grp.attrs["n_tau"])
            tau_max = float(grp.attrs["tau_max"])
            pops_ref = torch.tensor(grp["populations"][:], dtype=torch.float64)

            depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=tau_max, n_tau=n_tau)
            T_iso = torch.full((n_tau,), T0, dtype=torch.float64)
            atm = _eddington_atmosphere(T_iso, eps, nu_0, 1e10, depth)
            result = ali_loop(atm, mode="direct", formal_solve_mode="parabolic")
            pops = result.populations.detach()

            safe_ref = pops_ref.abs().clamp(min=1e-300)
            rel_err = ((pops - pops_ref).abs() / safe_ref).max().item()
            absolute_roundoff = 2.0 * torch.finfo(torch.float64).eps
            allowed = 1e-6 * pops_ref.abs() + absolute_roundoff
            assert bool(((pops - pops_ref).abs() <= allowed).all()), (
                f"R-truth-fixedpoint [{key}]: old-reference compatibility "
                f"failed: max absolute error={(pops-pops_ref).abs().max():.3e}, "
                f"max relative error={rel_err:.3e}; "
                f"allowance=1e-6*abs(reference)+{absolute_roundoff:.3e}."
            )
            # Independent arithmetic and algebra: n2=S/(B_star+2*S),
            # evaluated at the actual radiation field with 60 decimal digits.
            with localcontext() as context:
                context.prec = 60
                D = Decimal.from_float
                h, c, kb = map(D, (_H, _C, _KB))
                frequency, temperature, destruction = map(D, (nu_0, T0, eps))
                bstar = 2 * h * frequency**3 / c**2
                planck = bstar / ((h * frequency / (kb * temperature)).exp() - 1)
                stable = []
                for value in result.J_bar.detach().tolist():
                    source = (1 - destruction) * D(value) + destruction * planck
                    stable.append(float(source / (bstar + 2 * source)))
            n2_independent = torch.tensor(stable, dtype=torch.float64)
            stable_error = ((pops[:, 1] - n2_independent).abs()
                            / n2_independent.abs()).max().item()
            assert stable_error <= 32 * torch.finfo(torch.float64).eps, (
                f"R-truth-fixedpoint [{key}]: excited fraction disagrees with "
                f"independent 60-digit SE by {stable_error:.3e} relative."
            )
            pop_sum_err = (pops.sum(dim=-1) - 1.0).abs().max().item()
            assert pop_sum_err < 1e-8, (
                f"R-truth-fixedpoint [{key}]: population conservation "
                f"max |sum-1| = {pop_sum_err:.3e} > 1e-8."
            )


# ---------------------------------------------------------------------------
# R-formal-solver
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_R_formal_solver() -> None:
    """R-formal-solver: J_ν matches committed reference for both sources.

    Linear-S source: J matches to 1e-6 rel tol + Eddington-Barbier surface
    anchor J(0, x=0) = a/2 + b/4 holds to 1% (5-point Gauss-Legendre angle
    quadrature discretisation error).

    Planck-S on power-law T source: J matches to 1e-6 rel tol + the
    deep-boundary anchor J(τ_max, x=0) = B(T_max) holds to 4 sig figs.
    """
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.formal_solver import formal_solve
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

    ref_path = _REG_DIR / "formal_solver.h5"
    assert ref_path.exists(), f"Reference file not found: {ref_path}"

    with h5py.File(ref_path, "r") as f:
        T0 = float(f.attrs["T0"])
        nu_0 = float(f.attrs["nu_0"])
        a = float(f.attrs["S_linear_a"])
        b = float(f.attrs["S_linear_b"])
        J_linear_ref = torch.tensor(f["J_linear"][:], dtype=torch.float64)
        J_planck_ref = torch.tensor(f["J_planck"][:], dtype=torch.float64)
        T_powerlaw = torch.tensor(f["T_powerlaw"][:], dtype=torch.float64)

    n_tau, n_nu = J_linear_ref.shape

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=n_nu, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=5)

    atm_iso = Atmosphere(
        depth=depth, frequency=frequency, angle=angle,
        T=torch.full((n_tau,), T0, dtype=torch.float64),
        n_e=torch.full((n_tau,), 1e10, dtype=torch.float64),
        epsilon=0.5, nu_0=nu_0,
    )

    # --- Linear-S source: J + Eddington-Barbier surface ---
    S_linear = a + b * depth.tau.unsqueeze(1).expand(-1, n_nu)
    J_linear = formal_solve(S_linear.to(torch.float64), atm_iso, mode="linear")
    safe = J_linear_ref.abs().clamp(min=1e-300)
    rel_err = ((J_linear - J_linear_ref).abs() / safe).max().item()
    assert rel_err < 1e-6, (
        f"R-formal-solver [linear-S]: J differs by {rel_err:.3e} > 1e-6"
    )

    # Eddington-Barbier surface anchor: J(0, x=0) = a/2 + b/4 (within
    # 5-point Gauss-Legendre quadrature accuracy).
    eb_analytic = 0.5 * a + 0.25 * b
    j_surface = J_linear[0, n_nu // 2].item()
    rel_eb = abs(j_surface - eb_analytic) / abs(eb_analytic)
    assert rel_eb < 0.01, (
        f"R-formal-solver [Eddington-Barbier surface]: "
        f"J(0,x=0)={j_surface:.6e} vs a/2+b/4={eb_analytic:.6e}, "
        f"rel err {rel_eb:.3e} > 1%."
    )

    # --- Planck-S on power-law T source: J + deep-boundary anchor ---
    atm_pl = Atmosphere.power_law(
        T0=T0, alpha=0.1, n_e0=1e10, epsilon=0.5, nu_0=nu_0,
        depth=depth, frequency=frequency, angle=angle,
    )
    B_at_tau = (2.0 * _H * nu_0**3 / _C**2) / (
        torch.exp(_H * nu_0 / (_KB * atm_pl.T)) - 1.0
    )
    S_planck = B_at_tau.unsqueeze(1).expand(-1, n_nu)
    J_planck = formal_solve(S_planck.to(torch.float64), atm_pl, mode="linear")
    safe = J_planck_ref.abs().clamp(min=1e-300)
    rel_err = ((J_planck - J_planck_ref).abs() / safe).max().item()
    assert rel_err < 1e-6, (
        f"R-formal-solver [Planck-S]: J differs by {rel_err:.3e} > 1e-6"
    )

    T_max = float(T_powerlaw[-1])
    B_max = (2.0 * _H * nu_0**3 / _C**2) / (
        math.exp(_H * nu_0 / (_KB * T_max)) - 1.0
    )
    j_deep = J_planck[-1, n_nu // 2].item()
    rel_planck = abs(j_deep - B_max) / abs(B_max)
    assert rel_planck < 1e-4, (
        f"R-formal-solver [Planck deep boundary]: "
        f"J(tau_max,x=0)={j_deep:.6e} vs B(T_max)={B_max:.6e}, "
        f"rel err {rel_planck:.3e} > 4 sig figs."
    )


# ---------------------------------------------------------------------------
# R-resolvent-output
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_R_resolvent_output() -> None:
    """R-resolvent-output: GRAIL's resolvent_solve reproduces committed δx*.

    Deterministic n=20 synthetic (A_rt, A_se) system, seed=42, x*=0.
    Verifies δx* matches committed reference to 1e-6 rel tol and iter_count
    within ±5 of the reference (GMRES iter count is robust to ±1 across
    BLAS/LAPACK builds; we allow ±5 to be safe).
    """
    from grail.api import resolvent_solve

    ref_path = _REG_DIR / "resolvent_output.h5"
    assert ref_path.exists(), f"Reference file not found: {ref_path}"

    with h5py.File(ref_path, "r") as f:
        n = int(f.attrs["n"])
        iter_count_ref = int(f.attrs["iter_count"])
        A_rt = torch.tensor(f["A_rt"][:], dtype=torch.float64)
        A_se = torch.tensor(f["A_se"][:], dtype=torch.float64)
        delta_T = torch.tensor(f["delta_T"][:], dtype=torch.float64)
        dx_star_ref = torch.tensor(f["delta_x_star_ref"][:], dtype=torch.float64)

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        return A_rt @ v.to(torch.float64)

    def emul_jvp(v: torch.Tensor) -> torch.Tensor:
        return A_se @ v.to(torch.float64)

    x_star = torch.zeros(n, dtype=torch.float64)
    res = resolvent_solve(
        x_star, delta_T, emul_jvp, host_jvp,
        gmres_tol=1e-10, gmres_maxiter=500,
    )

    safe_ref = dx_star_ref.abs().clamp(min=1e-300)
    rel_err = ((res.delta_x_star - dx_star_ref).abs() / safe_ref).max().item()
    assert rel_err < 1e-6, (
        f"R-resolvent-output: δx* rel err = {rel_err:.3e} > 1e-6 "
        f"(ref ||δx*||={dx_star_ref.norm().item():.4f}; "
        f"computed ||δx*||={res.delta_x_star.norm().item():.4f})."
    )
    assert abs(res.iter_count - iter_count_ref) <= 5, (
        f"R-resolvent-output: iter_count={res.iter_count} vs "
        f"ref={iter_count_ref} (allowed ±5)."
    )
