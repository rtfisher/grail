"""Probe convergence of S(0)/B vs analytic √ε at varying N_τ for ε=1e-6.

Purpose: confirm or refute the orchestrator's hypothesis that the
A-thermalization-02 failure at ε=1e-6 is an O(1/N_τ) discretization-error
issue from linear-OK87 short characteristics.

If the relative error scales as 1/N_τ, the parabolic-S upgrade
(the technical reference §3.2 configurable upgrade) is the right fix.
If it scales differently (e.g., constant), there's a deeper bug.
"""

from __future__ import annotations

import math

import torch

from atmosphere.atmosphere import Atmosphere
from atmosphere.fixed_point import ali_loop
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid


def probe(epsilon: float, tau_max: float, n_tau_values: list[int]) -> None:
    """Run ALI on isothermal config at varying n_tau; report S(0)/B vs sqrt(eps)."""
    T0 = 6000.0
    n_e0 = 1e10
    nu_0 = 2.47e15  # Lyman-alpha
    h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    B = (2.0 * h * nu_0**3 / c**2) / (math.exp(h * nu_0 / (kB * T0)) - 1.0)
    sqrt_eps = math.sqrt(epsilon)

    print(f"\n=== epsilon = {epsilon:.0e}, tau_max = {tau_max:.0e} ===")
    print(f"Analytic surface S(0)/B = sqrt(eps) = {sqrt_eps:.6e}")
    print(f"{'N_tau':>8} {'iters':>8} {'S(0)/B':>14} {'rel_err':>14} {'tol':>10}")

    for n_tau in n_tau_values:
        depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=tau_max, n_tau=n_tau)
        # Eddington two-stream: μ=1/√3, w=1 (matches the AH analytic)
        mu_1 = torch.tensor([1.0 / math.sqrt(3.0)], dtype=torch.float64)
        w_1 = torch.tensor([1.0], dtype=torch.float64)
        angle = AngleQuadrature(mu=mu_1, weights=w_1)
        # Single frequency (coherent): φ=1, weight=1
        freq = FrequencyGrid(
            nu=torch.zeros(1, dtype=torch.float64),
            weights=torch.ones(1, dtype=torch.float64),
            profile=torch.ones(1, dtype=torch.float64),
        )
        T = torch.full((n_tau,), T0, dtype=torch.float64)
        n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)
        atm = Atmosphere(
            depth=depth,
            frequency=freq,
            angle=angle,
            T=T,
            n_e=n_e,
            epsilon=epsilon,
            nu_0=nu_0,
        )
        result = ali_loop(atm, max_iter=20000, tol=1e-9)
        S_surface = result.S_l[0].item()
        ratio = S_surface / B
        rel_err = abs(ratio - sqrt_eps) / sqrt_eps
        print(
            f"{n_tau:>8} {result.iter_count:>8} {ratio:>14.6e} "
            f"{rel_err:>14.4f} {'10%':>10}"
        )


if __name__ == "__main__":
    print("Linear-OK87 convergence test: surface S(0)/B vs analytic sqrt(epsilon)")
    print("HM14 §13.3 / Avrett & Hummer 1965")

    # ε = 1e-2 baseline (passes Tier A): should show small error at all N_τ
    probe(1e-2, 1e4, [50, 100, 200, 400, 800])

    # ε = 1e-4 (passes at N_tau=1000): scan n_tau
    probe(1e-4, 1e4, [100, 200, 400, 800, 1600])

    # ε = 1e-6 (xfail): scan large n_tau to characterize convergence
    probe(1e-6, 1e6, [200, 400, 800, 1600, 3200, 6400])
