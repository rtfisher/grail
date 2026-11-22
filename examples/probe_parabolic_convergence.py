"""Verify parabolic-S OK87 convergence rate vs linear OK87.

Theory (the example documentation §C question; reference guide §6, parabolic-S note):
  Linear OK87:    error ~ O(N_τ^-2)  asymptotically (per-segment O(Δτ^2))
  Parabolic-S:    error ~ O(N_τ^-3)  asymptotically (per-segment O(Δτ^3))

Test: surface S(0)/B in the Eddington 2-stream coherent-monochromatic
configuration vs analytic √ε.  Run linear and parabolic on the same
N_τ ladder; fit log(rel_err) vs log(N_τ) and report the slopes.

Expected slopes:
  linear:    -2 in the asymptotic regime
  parabolic: -3 in the asymptotic regime
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from atmosphere.atmosphere import Atmosphere
from atmosphere.fixed_point import ali_loop
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid


def measure_surface_error(
    epsilon: float, n_tau: int, mode: str, tau_max: float = 1e4
) -> float:
    """Run ALI direct + given formal-solver mode; return |S(0)/B - sqrt(eps)|/sqrt(eps)."""
    T0 = 6000.0
    nu_0 = 2.47e15
    H, C, KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    B = (2.0 * H * nu_0**3 / C**2) / (math.exp(H * nu_0 / (KB * T0)) - 1.0)

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=tau_max, n_tau=n_tau)
    angle = AngleQuadrature(
        mu=torch.tensor([1.0 / math.sqrt(3.0)], dtype=torch.float64),
        weights=torch.ones(1, dtype=torch.float64),
    )
    frequency = FrequencyGrid(
        nu=torch.zeros(1, dtype=torch.float64),
        weights=torch.ones(1, dtype=torch.float64),
        profile=torch.ones(1, dtype=torch.float64),
    )
    atm = Atmosphere(
        depth=depth, frequency=frequency, angle=angle,
        T=torch.full((n_tau,), T0, dtype=torch.float64),
        n_e=torch.full((n_tau,), 1e10, dtype=torch.float64),
        epsilon=epsilon, nu_0=nu_0,
    )
    result = ali_loop(atm, mode="direct", formal_solve_mode=mode)
    surface = float(result.S_l[0]) / B
    sqrt_eps = math.sqrt(epsilon)
    return abs(surface - sqrt_eps) / sqrt_eps


def fit_slope(n_taus: list[int], errors: list[float]) -> tuple[float, float]:
    """Fit log(error) = a*log(N_tau) + b; return (slope a, intercept b)."""
    x = np.log10(np.array(n_taus, dtype=float))
    y = np.log10(np.array(errors, dtype=float))
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def main() -> None:
    out_dir = Path(__file__).parent / "convergence_data"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Parabolic-S vs linear OK87 convergence rate")
    print("Surface S(0)/B error vs sqrt(eps) in Eddington 2-stream coherent config")
    print("=" * 70)

    # epsilon = 1e-2: near-asymptotic regime even at small N_tau
    # epsilon = 1e-4: asymptotic regime at moderate N_tau
    for epsilon, tau_max in [(1e-2, 1e4), (1e-4, 1e4)]:
        n_taus = [40, 80, 160, 320, 640]

        print(f"\n--- epsilon = {epsilon:.0e}, tau_max = {tau_max:.0e} ---")
        print(f"{'N_tau':>6} {'linear err':>14} {'parabolic err':>16}")

        lin_errs: list[float] = []
        par_errs: list[float] = []
        for n_tau in n_taus:
            e_lin = measure_surface_error(epsilon, n_tau, "linear", tau_max)
            e_par = measure_surface_error(epsilon, n_tau, "parabolic", tau_max)
            lin_errs.append(e_lin)
            par_errs.append(e_par)
            print(f"{n_tau:>6} {e_lin:>14.4e} {e_par:>16.4e}")

        # Fit slopes (use last 4 points for asymptotic regime)
        n_lin, e_lin = n_taus[-4:], lin_errs[-4:]
        n_par, e_par = n_taus[-4:], par_errs[-4:]
        slope_lin, _ = fit_slope(n_lin, e_lin)
        slope_par, _ = fit_slope(n_par, e_par)
        print(f"  asymptotic slope: linear = {slope_lin:.2f} (expected -2)")
        print(f"  asymptotic slope: parabolic = {slope_par:.2f} (expected -3)")

        # Save raw data
        np.savez(
            out_dir / f"convergence_eps{epsilon:.0e}.npz",
            n_taus=n_taus,
            linear_errors=lin_errs,
            parabolic_errors=par_errs,
            slope_linear=slope_lin,
            slope_parabolic=slope_par,
        )
        print(f"  saved: {out_dir / f'convergence_eps{epsilon:.0e}.npz'}")


if __name__ == "__main__":
    main()
