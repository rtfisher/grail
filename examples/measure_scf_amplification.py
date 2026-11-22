"""Preparatory P2 / R2 / P5 measurement script for the SCF amplification figure.

SPEC_scf §6 P2 (preparatory analytic sweep) + §8.10 R2 (UHF sweep across
the Coulson–Fischer point) + §6 P5 (full emulator-coupled sweep including
GRAIL Born-remainder identity).

Three modes:

- ``--method rks`` (default): closed-shell analytic-only sweep across
  ``R ∈ {0.74, 1.0, 1.48, 2.0, 2.22} Å`` for RKS/B3LYP/STO-3G.  Output:
  ``examples/plot_data/scf_amplification_p2.npz``.
- ``--method uhf``: UHF analytic-only sweep across ``R ∈ [1.10, 1.25] Å``
  with ``ΔR = 0.025 Å`` (7 points: 1.100, 1.125, 1.150, 1.175, 1.200,
  1.225, 1.250) — focused window around the Coulson–Fischer point
  R_CF ≈ 1.15 Å.  Output: ``examples/plot_data/scf_uhf_amplification_p5.npz``.
- ``--method uhf --emulator-sweep``: **P5 full sweep**.  Runs the truth
  SCF, the NN-coupled emulator SCF, the analytic CPKS Jacobian, AND
  the head-on GRAIL Born-remainder identity at each R across the focused
  R3-trained window ``R = linspace(1.10, 1.25, 8)`` Å.  Output:
  ``examples/plot_data/scf_uhf_amplification_sweep.npz``.  This is the
  data source for the example diagnostics §12 SCF amplification figure
  (``fig:scf-amplification``) and the reference-guide §4.4 numerical
  content.

In the analytic-only modes each R records:
  ``A_op_analytic``, ``spectral_radius``, ``lambda_max_positive``,
  ``gap_eV``, ``s_squared``, ``iter_count_truth``, ``iter_count_emul``
  (NaN; filled by the emulator-sweep mode).

In the P5 emulator-sweep mode each R additionally records:
  ``A_emp``, ``rel_err``, ``born_ratio``, ``born_bound``, ``gmres_iter``,
  ``iter_count_emul``, ``s_squared_emul``, ``gap_alpha_eV``,
  ``gap_beta_eV``, plus 2-D ``delta_x_actual``, ``delta_x_predicted``,
  ``delta_T_flat`` arrays (shape ``(N_R, 2*n_unique)``) for the
  bar-plot figure.

The output npz files are committed to git so example diagnostics figures can be
regenerated reproducibly.

Usage::

    PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE \\
        python \\
        examples/measure_scf_amplification.py [--method rks|uhf] [--emulator-sweep]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# OpenMP triple-init guard (SPEC_scf §8.5).  Must be set before any PySCF
# import path is touched.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# Ensure project root is on sys.path so ``scf.*`` imports resolve when this
# script is invoked directly (e.g. ``python examples/measure_scf_amplification.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scf._make_state import make_h2_state                # noqa: E402
from scf.host import picard_iterate_scf                  # noqa: E402
from scf.jacobian import (                               # noqa: E402
    A_op_analytic,
    cpks_jacobian_analytic,
    spectral_radius,
)


# Hartree → eV conversion (CODATA 2018 value, sufficient for figure labels).
_HARTREE_TO_EV = 27.211386245988

# Born-remainder constants (SPEC_scf §7).  Mirrored from the P4 integration
# test ``tests/scf/integration/test_resolvent_predicts_scf.py``.
_C = 1.25
_TOL_FLOOR = 0.02


# Sweep configuration — keep RKS in sync with
# ``tests/scf/analytic/test_amplification_sweep.py``.
_R_SWEEP_RKS = (0.74, 1.0, 1.48, 2.0, 2.22)
# UHF focused window around R_CF ≈ 1.15 Å, ΔR = 0.025 Å (7 points).  This
# is the R2 task brief's "focused R ∈ [1.10, 1.25] Å with ΔR = 0.025 Å"
# specification.  Keeps the dominant amplification peak (~R=1.15 Å) and
# its symmetric drop-off in view for the example diagnostics figure.
_R_SWEEP_UHF = tuple(round(1.100 + 0.025 * k, 4) for k in range(7))
# P5 emulator-coupled sweep: 8 evenly-spaced points across the R3-trained
# range ``[1.10, 1.25]`` Å (ΔR ≈ 0.0214 Å, matches SPEC_scf §6 P5
# "ΔR = 0.02 Å → 8 points" intent).  Anchored on the endpoints so that the
# trained NN's full operating range is sampled and the peak at R ≈ 1.15 Å
# is bracketed by interior points (1.143 and 1.164).
_R_SWEEP_P5 = tuple(round(float(R), 4) for R in np.linspace(1.10, 1.25, 8))
_BASIS = "sto-3g"
_XC = "B3LYP"

_PLOT_DATA_DIR = _PROJECT_ROOT / "examples/plot_data"
_OUTPUT_NPZ_RKS = _PLOT_DATA_DIR / "scf_amplification_p2.npz"
_OUTPUT_NPZ_UHF = _PLOT_DATA_DIR / "scf_uhf_amplification_p5.npz"
# P5 emulator-coupled sweep output.  Distinct path from the existing R2
# analytic-only sweep above so the two artifacts coexist.
_OUTPUT_NPZ_P5 = _PLOT_DATA_DIR / "scf_uhf_amplification_sweep.npz"
# Trained NN artifact (R3 hand-off).  Only UHF/STO-3G is available;
# 6-31G is out of scope per the P5 task brief.
_XC_NET_PATH = _PROJECT_ROOT / "examples" / "scf_xc_emulator_uhf_sto3g.h5"


def _gap_eV_closed_shell(state, result) -> float:
    """HOMO–LUMO gap (eV) for closed-shell methods."""
    n_occ = state.n_electrons // 2
    return float(
        (result.mo_energy[n_occ] - result.mo_energy[n_occ - 1]) * _HARTREE_TO_EV
    )


def _gap_eV_uhf(state, result, mol_nelec: tuple[int, int]) -> float:
    """Minimum HOMO–LUMO gap (eV) over both spin channels for UHF/UKS.

    Uses the per-spin occupation pattern from PySCF's ``mol.nelec`` to
    identify HOMO/LUMO per spin.  Returns the *minimum* of the two spin
    gaps — the physically relevant amplification dial (the smaller-gap
    spin channel dominates the cycle's near-singular eigenvalue).
    """
    gap_a, gap_b = _gaps_eV_uhf_per_spin(state, result, mol_nelec)
    return min(gap_a, gap_b)


def _gaps_eV_uhf_per_spin(
    state, result, mol_nelec: tuple[int, int],
) -> tuple[float, float]:
    """Per-spin (alpha, beta) HOMO–LUMO gaps in eV for UHF/UKS.

    Returned tuple is ``(gap_alpha_eV, gap_beta_eV)``.  Either entry is
    ``NaN`` if the corresponding spin channel has no virtual (i.e. the
    basis is fully occupied for that spin).  Used by the P5 sweep where
    the figure needs to separately track both channels — the alpha and
    beta gaps split asymmetrically once the broken-symmetry branch takes
    over past R_CF.
    """
    n_a, n_b = mol_nelec
    eps_a, eps_b = result.mo_energy
    gap_a = float((eps_a[n_a] - eps_a[n_a - 1]) * _HARTREE_TO_EV) if n_a < eps_a.size else float("nan")
    gap_b = float((eps_b[n_b] - eps_b[n_b - 1]) * _HARTREE_TO_EV) if n_b < eps_b.size else float("nan")
    return gap_a, gap_b


def _measure_one_rks(R: float) -> dict:
    """Measure (A_op, ρ, λ_max+, gap_eV, iter_count_truth) at one R for RKS."""
    state = make_h2_state(R=R, basis=_BASIS, method="rks", xc=_XC)
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200, accelerator="diis")
    if not result.converged:
        print(
            f"  R={R:.4f} Å: truth SCF DID NOT CONVERGE "
            f"(iter={result.iter_count}, last residual="
            f"{result.residual_history[-1] if result.residual_history else None:.2e})"
        )
        return {
            "R": R,
            "A_op_analytic": float("nan"),
            "spectral_radius": float("nan"),
            "lambda_max_positive": float("nan"),
            "gap_eV": float("nan"),
            "s_squared": float("nan"),
            "iter_count_truth": result.iter_count,
            "iter_count_emul": float("nan"),
        }

    DT = cpks_jacobian_analytic(
        state, result.P_star, result.mo_energy, result.mo_coeff,
    )
    A_op = A_op_analytic(DT)
    rho = spectral_radius(DT)
    lam_pos = float(np.max(np.linalg.eigvals(DT).real))
    gap_eV = _gap_eV_closed_shell(state, result)

    return {
        "R": R,
        "A_op_analytic": A_op,
        "spectral_radius": rho,
        "lambda_max_positive": lam_pos,
        "gap_eV": gap_eV,
        "s_squared": float("nan"),
        "iter_count_truth": result.iter_count,
        "iter_count_emul": float("nan"),
    }


def _measure_one_uhf(R: float) -> dict:
    """Measure (A_op, ρ, λ_max+, gap_eV, ⟨S²⟩, iter_count_truth) for UHF."""
    state = make_h2_state(R=R, basis=_BASIS, method="uhf")
    result = picard_iterate_scf(state, tol=1e-9, max_iter=200, accelerator="diis")
    if not result.converged:
        print(
            f"  R={R:.4f} Å: UHF SCF DID NOT CONVERGE "
            f"(iter={result.iter_count}, last residual="
            f"{result.residual_history[-1] if result.residual_history else None:.2e})"
        )
        return {
            "R": R,
            "A_op_analytic": float("nan"),
            "spectral_radius": float("nan"),
            "lambda_max_positive": float("nan"),
            "gap_eV": float("nan"),
            "s_squared": float("nan"),
            "iter_count_truth": result.iter_count,
            "iter_count_emul": float("nan"),
        }

    DT = cpks_jacobian_analytic(
        state, result.P_star, result.mo_energy, result.mo_coeff,
    )
    A_op = A_op_analytic(DT)
    rho = spectral_radius(DT)
    lam_pos = float(np.max(np.linalg.eigvals(DT).real))
    # PySCF's per-spin nelec from a freshly-built Mole.  We do not cache
    # it on SCFResult so reconstruct here — cheap for STO-3G H₂.
    from scf.fock_build import _build_mol
    mol = _build_mol(state)
    gap_eV = _gap_eV_uhf(state, result, mol.nelec)
    s_squared = float(result.s_squared) if result.s_squared is not None else float("nan")

    return {
        "R": R,
        "A_op_analytic": A_op,
        "spectral_radius": rho,
        "lambda_max_positive": lam_pos,
        "gap_eV": gap_eV,
        "s_squared": s_squared,
        "iter_count_truth": result.iter_count,
        "iter_count_emul": float("nan"),
    }


# ---------------------------------------------------------------------------
# P5 — full emulator-coupled sweep: truth + emulator SCF + GRAIL identity.
# ---------------------------------------------------------------------------
#
# This is the SPEC §6 P5 deliverable: at each R sample point it (i) converges
# both the truth and emulator-coupled fixed points, (ii) computes the analytic
# CPKS Jacobian at the truth fixed point, (iii) forms the per-cycle emulator
# residual ``delta_T`` at the truth fixed point, (iv) feeds it through GRAIL's
# matrix-free GMRES resolvent, (v) compares the predicted shift to the actual
# emulator-coupled offset, and (vi) records the Born-remainder ratio against
# the SPEC §7 acceptance bound ``C = 1.25``.
#
# The pipeline mirrors the P4 integration test ``test_resolvent_predicts_scf``
# (steps 1–8) point-by-point so the example diagnostics sweep data is consistent
# with the standing acceptance test.  The only divergence from the integration
# test is that the measurement script keeps going when Picard fails to
# converge at one R (records NaN; does not skip), and that it computes a
# few extra figure-data fields (per-spin gaps; the full 2-D delta arrays).
# ---------------------------------------------------------------------------

def _diag_rebuild_uhf(
    F_alpha: np.ndarray, F_beta: np.ndarray, S: np.ndarray,
    n_alpha: int, n_beta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One per-spin diagonalize+rebuild step (no DIIS).

    Mirrors the helper of the same name in the P4 integration test
    ``tests/scf/integration/test_resolvent_predicts_scf.py``.  Used here
    to evaluate ``T(P) = 𝐃[𝐅(P)]`` once at ``P = P*_truth`` so we can form
    ``delta_T = T_emul(P*_truth) - T_truth(P*_truth)`` — the per-cycle
    emulator residual GRAIL is asked to propagate.
    """
    from pyscf.scf import hf as scf_hf
    F_alpha = 0.5 * (F_alpha + F_alpha.T)
    F_beta = 0.5 * (F_beta + F_beta.T)
    _ea, Ca = scf_hf.eig(F_alpha, S)
    _eb, Cb = scf_hf.eig(F_beta, S)
    Pa = Ca[:, :n_alpha] @ Ca[:, :n_alpha].T
    Pb = Cb[:, :n_beta] @ Cb[:, :n_beta].T
    return 0.5 * (Pa + Pa.T), 0.5 * (Pb + Pb.T)


def _measure_one_p5(R: float, xc_net) -> dict:
    """P5 full emulator-coupled measurement at one R.

    Steps 1–8 of SPEC_scf §6 P4 / P5.  Returns a row dict with every
    column the figure-data npz needs.  On any Picard non-convergence the
    row carries NaN observables (the npz preserves the attempt; the
    measurement script does NOT skip — that behavior is reserved for
    the integration test).
    """
    # Local imports keep PySCF/torch off the module-load OpenMP path.
    import torch  # noqa: F401  (used implicitly via pack_density_matrix)
    from grail import resolvent_solve
    from scf.fock_build import _build_mf, _build_mol, build_fock_truth
    from scf.jacobian import cpks_jacobian_analytic_uhf
    from scf.jvp import (
        make_cycle_jvps_scf,
        make_fock_builder_with_nn,
        make_host_jvp_scf,
    )
    from scf.state import pack_density_matrix

    # Step 1: build UHF state at this R.
    state = make_h2_state(R=R, basis=_BASIS, method="uhf")

    # Step 2: truth SCF.
    res_truth = picard_iterate_scf(
        state, tol=1e-9, max_iter=200, accelerator="diis",
    )

    # Build the NN-substituted Fock builder upfront so we surface load errors
    # early if the artifact is mis-shaped.
    fb_nn = make_fock_builder_with_nn(xc_net, state)

    # Step 3: emulator SCF.
    res_emul = picard_iterate_scf(
        state, fock_builder=fb_nn, tol=1e-9, max_iter=200, accelerator="diis",
    )

    # The PySCF nelec lookup is needed in both branches (Picard or not),
    # so reconstruct here.
    mol = _build_mol(state)
    n_alpha, n_beta = mol.nelec

    # NaN-row template for the non-convergence early-exit and final assembly.
    row_nan = {
        "R": R,
        "S2_truth": float("nan"),
        "S2_emul": float("nan"),
        "lambda_max_positive": float("nan"),
        "spectral_radius": float("nan"),
        "A_op_analytic": float("nan"),
        "A_emp": float("nan"),
        "rel_err": float("nan"),
        "born_ratio": float("nan"),
        "born_bound": float("nan"),
        "gmres_iter": -1,
        "truth_picard_iter": res_truth.iter_count,
        "emul_picard_iter": res_emul.iter_count,
        "gap_alpha_eV": float("nan"),
        "gap_beta_eV": float("nan"),
        "delta_x_actual": np.full(2 * 3, np.nan, dtype=np.float64),  # STO-3G default
        "delta_x_predicted": np.full(2 * 3, np.nan, dtype=np.float64),
        "delta_T_flat": np.full(2 * 3, np.nan, dtype=np.float64),
        "converged_truth": res_truth.converged,
        "converged_emul": res_emul.converged,
    }

    if not (res_truth.converged and res_emul.converged):
        print(
            f"  R={R:.4f} Å: Picard did NOT converge "
            f"(truth_conv={res_truth.converged}, "
            f"emul_conv={res_emul.converged}, "
            f"truth_iter={res_truth.iter_count}, "
            f"emul_iter={res_emul.iter_count})"
        )
        return row_nan

    # ⟨S²⟩ pre/post-symmetrization records.  Per the P4 advisory, these flag
    # the topological-excursion regime where the truth converges to a singlet
    # (⟨S²⟩ ≈ 0) but the emulator drifts to the broken-symmetry branch
    # (⟨S²⟩ > 0.05) — or vice-versa.
    s2_truth = (
        float(res_truth.s_squared) if res_truth.s_squared is not None else float("nan")
    )
    s2_emul = (
        float(res_emul.s_squared) if res_emul.s_squared is not None else float("nan")
    )

    # Step 4: actual shift at the GRAIL boundary (doubled UHF flat seam).
    P_truth_flat = pack_density_matrix(res_truth.P_star, method="uhf")
    P_emul_flat = pack_density_matrix(res_emul.P_star, method="uhf")
    delta_x_actual = P_emul_flat - P_truth_flat

    # Step 5: per-cycle emulator residual ``delta_T`` at the truth fixed point.
    F_truth = build_fock_truth(state, res_truth.P_star)
    F_nn = fb_nn(state, res_truth.P_star)
    mf = _build_mf(state, mol=mol)
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    P_T_truth = _diag_rebuild_uhf(F_truth[0], F_truth[1], S, n_alpha, n_beta)
    P_T_nn = _diag_rebuild_uhf(F_nn[0], F_nn[1], S, n_alpha, n_beta)
    import torch as _torch  # local rename to avoid name-clash with outer imports
    delta_T = pack_density_matrix(
        (_torch.from_numpy(P_T_nn[0] - P_T_truth[0]),
         _torch.from_numpy(P_T_nn[1] - P_T_truth[1])),
        method="uhf",
    )

    # Step 6: GRAIL predicted shift (slot-naming inversion — SPEC §8.1).
    # host_jvp ↔ Fock-build differential (NN lives here; FD-JVP).
    # emul_jvp ↔ density-update differential (no NN here; FD for convenience).
    host_jvp, emul_jvp = make_cycle_jvps_scf(state, res_truth.P_star, fb_nn)
    res_grail = resolvent_solve(
        P_truth_flat, delta_T, emul_jvp, host_jvp,
        gmres_tol=1e-10, gmres_maxiter=50,
    )
    delta_x_pred = res_grail.delta_x_star

    # Step 7: Born-remainder diagnostics.
    norm_actual = float(delta_x_actual.norm().item())
    norm_pred = float(delta_x_pred.norm().item())
    norm_delta_T = float(delta_T.norm().item())
    diff = float((delta_x_actual - delta_x_pred).norm().item())
    rel_err = diff / norm_actual if norm_actual > 1e-12 else float("inf")
    A_emp = norm_actual / norm_delta_T if norm_delta_T > 1e-12 else float("inf")
    born_floor = 1.0 - 1.0 / A_emp if A_emp > 1e-12 else 0.0
    born_ratio = rel_err / (born_floor + 1e-12)
    born_bound = _C + _TOL_FLOOR / (born_floor + 1e-12)

    # Analytic CPKS at the truth fixed point (SPEC §6 P5 step 5).  R2's
    # ``cpks_jacobian_analytic_uhf`` is the canonical UHF entry.
    DT_analytic = cpks_jacobian_analytic_uhf(
        state,
        P_star_alpha=res_truth.P_star[0], P_star_beta=res_truth.P_star[1],
        mo_energy_alpha=res_truth.mo_energy[0],
        mo_energy_beta=res_truth.mo_energy[1],
        mo_coeff_alpha=res_truth.mo_coeff[0],
        mo_coeff_beta=res_truth.mo_coeff[1],
    )
    A_op = A_op_analytic(DT_analytic)
    rho = spectral_radius(DT_analytic)
    lam_pos = float(np.max(np.linalg.eigvals(DT_analytic).real))

    gap_alpha, gap_beta = _gaps_eV_uhf_per_spin(state, res_truth, (n_alpha, n_beta))

    return {
        "R": R,
        "S2_truth": s2_truth,
        "S2_emul": s2_emul,
        "lambda_max_positive": lam_pos,
        "spectral_radius": rho,
        "A_op_analytic": A_op,
        "A_emp": A_emp,
        "rel_err": rel_err,
        "born_ratio": born_ratio,
        "born_bound": born_bound,
        "gmres_iter": int(res_grail.iter_count),
        "truth_picard_iter": int(res_truth.iter_count),
        "emul_picard_iter": int(res_emul.iter_count),
        "gap_alpha_eV": gap_alpha,
        "gap_beta_eV": gap_beta,
        "delta_x_actual": delta_x_actual.detach().numpy().astype(np.float64),
        "delta_x_predicted": delta_x_pred.detach().numpy().astype(np.float64),
        "delta_T_flat": delta_T.detach().numpy().astype(np.float64),
        "norm_actual": norm_actual,
        "norm_predicted": norm_pred,
        "norm_delta_T": norm_delta_T,
        "converged_truth": True,
        "converged_emul": True,
    }


def _persist_npz_p5(rows: list[dict], output_path: Path) -> None:
    """Pack the P5 sweep row dicts to a single npz.

    Distinct from :func:`_persist_npz` (which serves the analytic-only
    P2 / R2 paths) because the P5 row schema is richer: per-row 2-D
    delta arrays, separate alpha/beta gaps, and the Born-remainder
    fields needed by the example diagnostics figure.
    """
    _PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def col(name, dtype=np.float64):
        return np.array([r[name] for r in rows], dtype=dtype)

    # Stack the 2-D delta arrays.  Each row's delta is length
    # ``2 * n_unique`` — 6 for STO-3G H₂.
    delta_x_actual_2d = np.stack([r["delta_x_actual"] for r in rows], axis=0)
    delta_x_predicted_2d = np.stack([r["delta_x_predicted"] for r in rows], axis=0)
    delta_T_flat_2d = np.stack([r["delta_T_flat"] for r in rows], axis=0)

    np.savez(
        output_path,
        R=col("R"),
        S2_truth=col("S2_truth"),
        S2_emul=col("S2_emul"),
        lambda_max_positive=col("lambda_max_positive"),
        spectral_radius=col("spectral_radius"),
        A_op_analytic=col("A_op_analytic"),
        A_emp=col("A_emp"),
        rel_err=col("rel_err"),
        born_ratio=col("born_ratio"),
        born_bound=col("born_bound"),
        gmres_iter=col("gmres_iter", dtype=np.int64),
        truth_picard_iter=col("truth_picard_iter", dtype=np.int64),
        emul_picard_iter=col("emul_picard_iter", dtype=np.int64),
        gap_alpha_eV=col("gap_alpha_eV"),
        gap_beta_eV=col("gap_beta_eV"),
        delta_x_actual=delta_x_actual_2d,
        delta_x_predicted=delta_x_predicted_2d,
        delta_T_flat=delta_T_flat_2d,
        converged_truth=col("converged_truth", dtype=bool),
        converged_emul=col("converged_emul", dtype=bool),
    )


def _print_p5_table(rows: list[dict]) -> None:
    """Print the example diagnostics figure data as a clean ASCII table.

    Single source of truth for both the in-script summary and the
    completion-report ASCII block.  Format chosen for direct LaTeX-table-row
    copy-paste (whitespace-separated, fixed widths, no decorative borders
    aside from a header rule).
    """
    print()
    print("example diagnostics figure data (scf_uhf_amplification_sweep.npz):")
    header = (
        f"{'R (Å)':>7s}  "
        f"{'S²_t':>7s}  {'S²_e':>7s}  "
        f"{'λ+':>8s}  {'A_op':>10s}  {'A_emp':>8s}  "
        f"{'rel_err':>8s}  {'ratio':>7s}  {'bound':>7s}  "
        f"{'GMRES':>5s}  {'P_t':>4s}  {'P_e':>4s}  "
        f"{'gap_α':>7s}  {'gap_β':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        # Format with NaN-safe blanks so the alignment survives non-convergence.
        def f(v, fmt):
            return fmt.format(v) if np.isfinite(v) else "      —"
        print(
            f"{r['R']:7.4f}  "
            f"{f(r['S2_truth'], '{:7.4f}')}  "
            f"{f(r['S2_emul'], '{:7.4f}')}  "
            f"{f(r['lambda_max_positive'], '{:+8.4f}')}  "
            f"{f(r['A_op_analytic'], '{:10.4f}')}  "
            f"{f(r['A_emp'], '{:8.4f}')}  "
            f"{f(r['rel_err'], '{:8.4f}')}  "
            f"{f(r['born_ratio'], '{:7.4f}')}  "
            f"{f(r['born_bound'], '{:7.4f}')}  "
            f"{r['gmres_iter']:5d}  "
            f"{r['truth_picard_iter']:4d}  "
            f"{r['emul_picard_iter']:4d}  "
            f"{f(r['gap_alpha_eV'], '{:7.3f}')}  "
            f"{f(r['gap_beta_eV'], '{:7.3f}')}"
        )


def _persist_npz(rows: list[dict], output_path: Path) -> None:
    """Pack the row dicts to a single npz file."""
    _PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        R=np.array([r["R"] for r in rows], dtype=np.float64),
        A_op_analytic=np.array([r["A_op_analytic"] for r in rows], dtype=np.float64),
        spectral_radius=np.array([r["spectral_radius"] for r in rows], dtype=np.float64),
        lambda_max_positive=np.array(
            [r["lambda_max_positive"] for r in rows], dtype=np.float64,
        ),
        gap_eV=np.array([r["gap_eV"] for r in rows], dtype=np.float64),
        s_squared=np.array([r["s_squared"] for r in rows], dtype=np.float64),
        iter_count_truth=np.array(
            [r["iter_count_truth"] for r in rows], dtype=np.int64,
        ),
        iter_count_emul=np.array(
            [r["iter_count_emul"] for r in rows], dtype=np.float64,
        ),
    )


def _run_p5_emulator_sweep() -> None:
    """Drive the P5 full emulator-coupled sweep across R ∈ [1.10, 1.25] Å.

    Loads the trained NN-XC artifact, runs ``_measure_one_p5`` at every
    R sample point, prints a wall-clock-tagged per-R log line, persists
    the npz, and prints the example diagnostics ASCII table.

    Wall-clock notes:
      - Picard convergence at the peak R≈1.15 Å can take ~60 iterations
        with DIIS; at the easy R=1.25 Å, ~16 iterations.
      - The dominant per-R cost is the host_jvp FD-JVP build, which
        evaluates the NN 2 * 6 = 12 times per R.  Total sweep wall-clock
        budgets at a few seconds per R on the conda env → ~30–60 s for 8 R.
    """
    import time
    from scf.emulator import load_xc_emulator
    if not _XC_NET_PATH.exists():
        raise FileNotFoundError(
            f"Trained XCNet artifact not found at {_XC_NET_PATH}.  Run "
            "examples/train_xc_emulator_uhf_sto3g.py (R3 hand-off) before "
            "the P5 emulator sweep."
        )
    import torch
    xc_net = load_xc_emulator(_XC_NET_PATH, dtype=torch.float64)

    print("=" * 80)
    print("measure_scf_amplification.py — P5 emulator-coupled sweep")
    print(f"  basis = {_BASIS}, method = uhf (broken-symmetry init), XC = {_XC}")
    print(f"  Trained NN artifact: {_XC_NET_PATH.name}")
    print(f"  R sweep (Å, {len(_R_SWEEP_P5)} pts): {list(_R_SWEEP_P5)}")
    print(f"  Born-remainder bound: C = {_C}, tol_floor = {_TOL_FLOOR}")
    print("=" * 80)

    t0 = time.perf_counter()
    rows: list[dict] = []
    for R in _R_SWEEP_P5:
        t_R0 = time.perf_counter()
        row = _measure_one_p5(R, xc_net)
        t_R = time.perf_counter() - t_R0
        # One-line per-R progress; concise so the operator can watch the sweep.
        if row["converged_truth"] and row["converged_emul"]:
            print(
                f"  R={R:5.4f} Å  "
                f"iter_t={row['truth_picard_iter']:>3d}  "
                f"iter_e={row['emul_picard_iter']:>3d}  "
                f"S²_t={row['S2_truth']:>+8.5f}  "
                f"S²_e={row['S2_emul']:>+8.5f}  "
                f"A_op={row['A_op_analytic']:>10.4f}  "
                f"A_emp={row['A_emp']:>8.4f}  "
                f"rel_err={row['rel_err']:>7.4f}  "
                f"ratio={row['born_ratio']:>7.4f}  "
                f"bound={row['born_bound']:>7.4f}  "
                f"GMRES={row['gmres_iter']:>2d}  "
                f"[{t_R:5.2f}s]"
            )
        else:
            print(
                f"  R={R:5.4f} Å  Picard FAIL  "
                f"truth_conv={row['converged_truth']}  "
                f"emul_conv={row['converged_emul']}  "
                f"[{t_R:5.2f}s]"
            )
        rows.append(row)
    wall_total = time.perf_counter() - t0

    # Persist before the table print so the artifact is on disk even if the
    # table loop trips over an unexpected NaN-handling case.
    _persist_npz_p5(rows, _OUTPUT_NPZ_P5)
    print("-" * 80)
    print(f"Total wall-clock: {wall_total:6.2f} s ({wall_total/60:.2f} min)")
    print(f"Output written to: {_OUTPUT_NPZ_P5}")
    print(
        "Columns: R, S2_truth, S2_emul, lambda_max_positive, "
        "spectral_radius, A_op_analytic, A_emp, rel_err, born_ratio, "
        "born_bound, gmres_iter, truth_picard_iter, emul_picard_iter, "
        "gap_alpha_eV, gap_beta_eV, delta_x_actual, delta_x_predicted, "
        "delta_T_flat, converged_truth, converged_emul"
    )
    _print_p5_table(rows)

    # Peak Born-remainder ratio summary (the load-bearing acceptance number).
    finite_rows = [
        r for r in rows
        if r["converged_truth"] and r["converged_emul"]
           and np.isfinite(r["born_ratio"])
    ]
    if finite_rows:
        peak_row = max(finite_rows, key=lambda r: r["born_ratio"])
        print("-" * 80)
        print(
            f"Peak Born-remainder ratio across the sweep: "
            f"{peak_row['born_ratio']:.4f} at R={peak_row['R']:.4f} Å "
            f"(bound={peak_row['born_bound']:.4f}, headroom="
            f"{peak_row['born_bound'] - peak_row['born_ratio']:.4f})."
        )
        all_within = all(r["born_ratio"] <= _C for r in finite_rows)
        print(
            f"All Born ratios <= C = {_C}: {all_within} "
            f"({sum(r['born_ratio'] <= _C for r in finite_rows)} / "
            f"{len(finite_rows)} converged R)."
        )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the SCF cycle's CPKS amplification across R."
    )
    parser.add_argument(
        "--method",
        choices=("rks", "uhf"),
        default="rks",
        help="Method for the truth SCF and CPKS Jacobian.  RKS triggers "
             "the original P2 closed-shell sweep across {0.74, 1.0, 1.48, "
             "2.0, 2.22} Å; UHF triggers the R2 focused sweep across "
             "[1.10, 1.25] Å (ΔR = 0.025 Å, 7 points).",
    )
    parser.add_argument(
        "--emulator-sweep",
        action="store_true",
        help="Run the P5 full emulator-coupled sweep (truth + emulator "
             "SCF + GRAIL Born-remainder identity at each R).  Requires "
             "--method uhf.  Output: examples/plot_data/"
             "scf_uhf_amplification_sweep.npz.",
    )
    parser.add_argument(
        "--basis",
        default="sto-3g",
        choices=("sto-3g",),
        help="Basis set.  Currently only STO-3G is supported — 6-31G is "
             "out of scope per the P5 task brief (no trained 6-31G NN).",
    )
    args = parser.parse_args()

    if args.basis != "sto-3g":
        raise NotImplementedError(
            f"basis={args.basis!r} not yet supported in the P5 sweep; "
            "only sto-3g (the R3-trained UHF NN's basis) is available."
        )

    if args.emulator_sweep:
        if args.method != "uhf":
            raise ValueError(
                "--emulator-sweep requires --method uhf (the P5 task brief "
                "is UHF-only; the R3-trained NN is UHF/STO-3G)."
            )
        _run_p5_emulator_sweep()
        return

    if args.method == "uhf":
        R_sweep = _R_SWEEP_UHF
        measure = _measure_one_uhf
        output_npz = _OUTPUT_NPZ_UHF
        header_method = "uhf (broken-symmetry init)"
        header_xc = "(n/a — Hartree–Fock)"
    else:
        R_sweep = _R_SWEEP_RKS
        measure = _measure_one_rks
        output_npz = _OUTPUT_NPZ_RKS
        header_method = "rks"
        header_xc = _XC

    print("=" * 72)
    print("measure_scf_amplification.py — P2 / R2 measurement")
    print(f"  basis = {_BASIS}, method = {header_method}, XC = {header_xc}")
    print(f"  R sweep (Å): {list(R_sweep)}")
    print("=" * 72)

    rows: list[dict] = []
    for R in R_sweep:
        row = measure(R)
        if args.method == "uhf":
            print(
                f"  R={R:5.3f} Å  iter={row['iter_count_truth']:>4d}  "
                f"gap={row['gap_eV']:>7.3f} eV  "
                f"⟨S²⟩={row['s_squared']:>7.4f}  "
                f"λ_max+={row['lambda_max_positive']:>+8.4f}  "
                f"A_op={row['A_op_analytic']:>10.4f}  "
                f"ρ={row['spectral_radius']:>8.4f}"
            )
        else:
            print(
                f"  R={R:5.2f} Å  iter={row['iter_count_truth']:>4d}  "
                f"gap={row['gap_eV']:>8.3f} eV  "
                f"λ_max+={row['lambda_max_positive']:>+8.4f}  "
                f"A_op={row['A_op_analytic']:>9.4f}  "
                f"ρ={row['spectral_radius']:>9.4f}"
            )
        rows.append(row)

    print("-" * 72)
    print("Summary:")
    for r in rows:
        if args.method == "uhf":
            print(
                f"  R={r['R']:5.3f} Å  ⟨S²⟩={r['s_squared']:7.4f}  "
                f"λ_max+={r['lambda_max_positive']:+8.4f}  "
                f"A_op={r['A_op_analytic']:10.4f}  "
                f"ρ={r['spectral_radius']:8.4f}  gap={r['gap_eV']:6.3f} eV"
            )
        else:
            print(
                f"  R={r['R']:5.2f} Å  A_op={r['A_op_analytic']:8.4f}  "
                f"ρ={r['spectral_radius']:8.4f}  gap={r['gap_eV']:7.3f} eV"
            )
    print("=" * 72)

    _persist_npz(rows, output_npz)
    print(f"Output written to: {output_npz}")
    print(
        "Columns: R, A_op_analytic, spectral_radius, lambda_max_positive, "
        "gap_eV, s_squared, iter_count_truth, iter_count_emul (NaN — "
        "fill in P3/P4/R3)."
    )


if __name__ == "__main__":
    main()
