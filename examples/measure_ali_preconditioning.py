"""ALI-preconditioning decision sweep — steady radiative-transfer verification utility.

Answers the question: at what (ε, τ_max, N_τ) does the iterative ALI path
with diagonal Λ* preconditioning beat the direct linear solve in wall-clock,
and when should GRAIL's matrix-free GMRES add a Λ*-style preconditioner?

The two-level CRD problem at fixed T is exactly linear in S:

    J_bar = Λ S + b_BC,
    S     = (1 - ε) J_bar + ε B(T)
    ⟹  (I - (1-ε) Λ) S = (1-ε) b_BC + ε B(T)

Direct mode  (mode='direct'): builds Λ via N_τ+1 formal solves, then calls
torch.linalg.solve once — O(N_τ³) dense linear algebra after O(N_τ²) setup.

ALI iterative (mode='ali'):  uses only the diagonal Λ* (one pass, N_τ formal
solves) then iterates until max |ΔS/S| < tol with optional Ng (1974)
acceleration — O(k · N_τ) per iteration where k → 1/√ε near thermalization
depth.

Sweep:
  ε ∈ {1e-2, 1e-3, 1e-4, 1e-5, 1e-6}
  τ_max ∈ {1e2, 1e4, 1e6}
  N_τ ∈ {80, 400, 2000}

For each of the 5×3×3 = 45 configs the script:
  1. Builds an isothermal 2-level atmosphere.
  2. Times mode='direct' (single linear solve).
  3. Times mode='ali' (iterative, max_iter=10_000, tol=1e-6).
  4. Verifies populations agree between modes to 1e-4.
  5. Records timing, iter_count, residuals, and population disagreement.

Output: examples/plot_data/ali_preconditioning.npz
  Arrays (shape [5, 3, 3], indexed [ε, τ_max, N_τ]):
    eps, tau_max, n_tau          — config axes
    wall_clock_direct            — seconds for direct solve (incl. Λ build)
    wall_clock_ali               — seconds for ALI iterative solve (incl. Λ* build)
    iter_count_direct            — always 1 (one linear solve)
    iter_count_ali               — ALI iterations to tol (or max_iter if stalled)
    converged_ali                — bool: did ALI reach tol?
    residual_direct              — final |ΔS/S| from direct (machine precision)
    residual_ali                 — final |ΔS/S| from ALI
    ratio_pop_disagreement       — max |n_direct - n_ali| across all depths/levels

Usage (from repo root):
    PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE \\
        python \\
        examples/measure_ali_preconditioning.py [--fast]

--fast flag: skips N_τ=2000 rows (estimated 5-10× speedup for a quick check).

References
----------
- Olson, Auer & Buchler (1986), JQSRT 35, 431  — diagonal Λ* preconditioner.
- Hubeny & Mihalas (2014), §13                 — ALI convergence, thermalization depth.
- the callback contract §"Two ALI modes in ali_loop"       — implementation notes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Suppress OpenMP double-init abort on this conda env (torch + h5py conflict).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

# Ensure project root is on sys.path when run as a script.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atmosphere.atmosphere import Atmosphere
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid
from atmosphere.twolevel.fixed_point import ali_loop

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_NU_LYA: float = 2.467381e15   # Hz (Lyman-alpha line center, same as other scripts)
_T0: float = 8000.0             # K  (isothermal temperature)
_N_E0: float = 1e12             # cm^-3

_EPS_VALUES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
_TAU_MAX_VALUES = [1e2, 1e4, 1e6]
_N_TAU_VALUES = [80, 400, 2000]

# Fixed quadrature (not swept — adequate for the timing comparison).
_N_NU: int = 11
_N_MU: int = 3

_ALI_MAX_ITER: int = 10_000
_ALI_TOL: float = 1e-6

# Sanity-check tolerance on population disagreement between modes.
_POP_AGREE_TOL: float = 1e-4

_SCRATCH_DIR = Path(__file__).parent
_PLOT_DATA_DIR = _SCRATCH_DIR / "plot_data"


# ---------------------------------------------------------------------------
# Per-config measurement
# ---------------------------------------------------------------------------

def _measure_one_config(
    eps: float,
    tau_max: float,
    n_tau: int,
    *,
    verbose: bool = True,
) -> dict:
    """Run direct + ALI on a single (ε, τ_max, N_τ) config.

    Returns a flat dict with all scalar results for this config.

    Notes
    -----
    We use ``formal_solve_mode='linear'`` throughout — the timing question
    is about the outer ALI/direct loop, not the formal-solver interpolation
    order.  Parabolic-S is only needed for analytic-limit tests at deep ε;
    for head-to-head timing comparison the same formal_solve_mode is used
    in both branches.
    """
    # Build atmosphere.
    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=tau_max, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=_N_NU, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=_N_MU)
    atm = Atmosphere.isothermal(
        T0=_T0, n_e0=_N_E0,
        epsilon=eps, nu_0=_NU_LYA,
        depth=depth, frequency=frequency, angle=angle,
    )

    # ------------------------------------------------------------------
    # Direct mode: build Λ (N_τ+1 formal solves) then torch.linalg.solve.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    r_direct = ali_loop(
        atm,
        tol=_ALI_TOL,
        max_iter=10,        # direct always finishes in 1 iteration
        mode="direct",
        formal_solve_mode="linear",
    )
    wall_direct = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # ALI iterative mode: build Λ* (N_τ+1 formal solves for the diagonal)
    # then iterate with diagonal preconditioner + Ng acceleration.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    r_ali = ali_loop(
        atm,
        tol=_ALI_TOL,
        max_iter=_ALI_MAX_ITER,
        mode="ali",
        formal_solve_mode="linear",
    )
    wall_ali = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Sanity check: populations must agree between modes to 1e-4.
    # ------------------------------------------------------------------
    pop_disagreement = (r_direct.populations - r_ali.populations).abs().max().item()

    converged_ali = (
        len(r_ali.residual_history) > 0
        and r_ali.residual_history[-1] < _ALI_TOL
    )

    if verbose:
        status = "CONVERGED" if converged_ali else "STALLED"
        agree = "OK" if pop_disagreement < _POP_AGREE_TOL else "FAIL"
        speedup = wall_direct / wall_ali if wall_ali > 0 else float("nan")
        print(
            f"  ε={eps:.0e}  τ_max={tau_max:.0e}  N_τ={n_tau:4d}  "
            f"direct={wall_direct:6.2f}s (1 iter)  "
            f"ali={wall_ali:6.2f}s ({r_ali.iter_count:5d} iter, {status})  "
            f"speedup={speedup:.2f}×  pop_agree={agree}({pop_disagreement:.1e})"
        )

    return {
        "eps": eps,
        "tau_max": tau_max,
        "n_tau": n_tau,
        "wall_clock_direct": wall_direct,
        "wall_clock_ali": wall_ali,
        "iter_count_direct": r_direct.iter_count,
        "iter_count_ali": r_ali.iter_count,
        "converged_ali": converged_ali,
        "residual_direct": r_direct.residual_history[-1] if r_direct.residual_history else float("nan"),
        "residual_ali": r_ali.residual_history[-1] if r_ali.residual_history else float("nan"),
        "ratio_pop_disagreement": pop_disagreement,
    }


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_sweep(fast: bool = False) -> list[dict]:
    """Run the full (or fast) sweep.  Returns list of per-config dicts."""
    n_tau_values = [80, 400] if fast else _N_TAU_VALUES
    label = "(fast — skipping N_τ=2000)" if fast else ""
    print(f"\n{'='*72}")
    print(f"ALI preconditioning decision sweep {label}")
    print(f"  ε        : {_EPS_VALUES}")
    print(f"  τ_max    : {_TAU_MAX_VALUES}")
    print(f"  N_τ      : {n_tau_values}")
    print(f"  N_ν={_N_NU}, N_μ={_N_MU}")
    print(f"  ALI tol={_ALI_TOL:.0e}, max_iter={_ALI_MAX_ITER}")
    total_configs = len(_EPS_VALUES) * len(_TAU_MAX_VALUES) * len(n_tau_values)
    print(f"  Total configs: {total_configs}")
    print(f"{'='*72}\n")

    results = []
    t_sweep_start = time.perf_counter()
    config_count = 0
    for eps in _EPS_VALUES:
        for tau_max in _TAU_MAX_VALUES:
            for n_tau in n_tau_values:
                config_count += 1
                r = _measure_one_config(eps, tau_max, n_tau, verbose=True)
                results.append(r)
    elapsed = time.perf_counter() - t_sweep_start
    print(f"\n  Sweep complete: {config_count} configs in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _save_results(results: list[dict], out_path: Path) -> None:
    """Save all per-config measurements to a .npz file.

    Arrays are flat (n_configs,) — not reshaped to (n_eps, n_tau_max, n_n_tau)
    because the fast mode may have a different n_tau_values list.  The sweep
    axes can be reconstructed from the eps/tau_max/n_tau arrays.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        eps=np.array([r["eps"] for r in results], dtype=np.float64),
        tau_max=np.array([r["tau_max"] for r in results], dtype=np.float64),
        n_tau=np.array([r["n_tau"] for r in results], dtype=np.int64),
        wall_clock_direct=np.array([r["wall_clock_direct"] for r in results], dtype=np.float64),
        wall_clock_ali=np.array([r["wall_clock_ali"] for r in results], dtype=np.float64),
        iter_count_direct=np.array([r["iter_count_direct"] for r in results], dtype=np.int64),
        iter_count_ali=np.array([r["iter_count_ali"] for r in results], dtype=np.int64),
        converged_ali=np.array([r["converged_ali"] for r in results], dtype=bool),
        residual_direct=np.array([r["residual_direct"] for r in results], dtype=np.float64),
        residual_ali=np.array([r["residual_ali"] for r in results], dtype=np.float64),
        ratio_pop_disagreement=np.array([r["ratio_pop_disagreement"] for r in results], dtype=np.float64),
    )
    print(f"  Saved to: {out_path}")
    print("  Fields: eps, tau_max, n_tau, wall_clock_direct, wall_clock_ali,")
    print("          iter_count_direct, iter_count_ali, converged_ali,")
    print("          residual_direct, residual_ali, ratio_pop_disagreement")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    """Print a per-config summary table grouped by ε."""
    print(f"\n{'='*72}")
    print("Summary table — wall-clock direct vs ALI, iter counts")
    print(f"{'ε':>8}  {'τ_max':>7}  {'N_τ':>5}  "
          f"{'t_direct':>9}  {'t_ali':>9}  {'speedup':>8}  "
          f"{'iters_ali':>9}  {'ali_cvg':>7}  {'pop_disagree':>12}")
    print("-" * 90)
    for r in results:
        wall_d = r["wall_clock_direct"]
        wall_a = r["wall_clock_ali"]
        speedup = wall_d / wall_a if wall_a > 0 else float("nan")
        # Positive speedup > 1 means direct is slower (ALI wins).
        winner = " [ALI]" if speedup > 1.0 else "      "
        print(
            f"{r['eps']:8.0e}  {r['tau_max']:7.0e}  {r['n_tau']:5d}  "
            f"{wall_d:8.3f}s  {wall_a:8.3f}s  {speedup:8.2f}x{winner}  "
            f"{r['iter_count_ali']:9d}  {str(r['converged_ali']):>7}  "
            f"{r['ratio_pop_disagreement']:12.1e}"
        )
    print(f"{'='*72}")

    # Identify the crossover point.
    crossing_found = False
    for r in results:
        wall_d = r["wall_clock_direct"]
        wall_a = r["wall_clock_ali"]
        if wall_a > 0 and wall_d / wall_a > 1.0:
            if not crossing_found:
                print("\n  ALI iterative beats direct at:")
                crossing_found = True
            print(
                f"    ε={r['eps']:.0e}  τ_max={r['tau_max']:.0e}  "
                f"N_τ={r['n_tau']}  "
                f"speedup={wall_d/wall_a:.2f}×  (direct={wall_d:.2f}s, ali={wall_a:.2f}s)"
            )
    if not crossing_found:
        print("\n  Direct is faster for all swept configs.")
        print("  Recommendation: direct mode is appropriate for the prototype.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALI preconditioning decision sweep (GRAIL §M5 Task 1)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip N_τ=2000 configs (faster sanity run, ~5-10× speedup).",
    )
    args = parser.parse_args()

    results = run_sweep(fast=args.fast)
    _print_summary(results)
    out_path = _PLOT_DATA_DIR / "ali_preconditioning.npz"
    _save_results(results, out_path)
    print(f"\nDone.  Reproduce: PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE "
          f"python "
          f"examples/measure_ali_preconditioning.py")


if __name__ == "__main__":
    main()
