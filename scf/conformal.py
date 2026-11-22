"""Split-conformal coverage on δP* for the SCF testbed.  SPEC_scf §6 P6, §8.9.

Thin adapter around :mod:`grail.conformal` for the doubled UHF state-vector
seam.  P6 (this module) joins the NLTE 2-level (TASK-037) and 3-level
(M3f-remediation) coverage curves as the cross-domain demonstration that
GRAIL's conformal wrapper holds across structurally distinct cycle
substrates — photon RT vs. electron-structure SCF.

Pipeline mirrors ``tests/integration/twolevel/_residual_ensemble.py``:

  1. :func:`build_one_residual_scf` — for one configuration ``(R, seed)``
     converge truth UHF SCF + emulator UHF SCF, compute the per-call
     residual ``δT = T_emul(P*_truth) − T_truth(P*_truth)``, run GRAIL's
     resolvent at ``P*_truth`` to predict ``δP*_GRAIL``, and record the
     actual ``δP*_actual = P*_emul − P*_truth``.  Per the SPEC §6 P6
     fallback path (used because the trained NN's R range is narrow and
     retraining 4 independent NNs would blow the wall-clock budget), the
     "perturbation seed" is implemented as a per-config **smooth
     affine** perturbation of the trained XCNet's ``V_xc`` output (see
     :class:`_NoisyXCNetWrapper` docstring for why smooth+linear-in-P
     and not per-call Gaussian noise — the latter breaks GRAIL's
     FD-JVP).

  2. :func:`assemble_ensemble_h5_scf` — writes the 32-config ensemble to
     HDF5 in the same schema as ``examples/two_level_residual_ensemble.h5``
     (M3g TASK-037) and ``examples/three_level_residual_ensemble.h5``
     (M3f), so the existing tooling carries over.

  3. :func:`stratified_split_indices_scf` — within-bucket stratified
     50/50 split by ``R`` bucket (8 buckets in the fallback layout —
     the seed dimension is exchangeable within each R bucket so the
     bucket key omits it).

  4. :func:`loo_cv_pca_rank` — re-exported via direct copy from the
     2-level/3-level helpers (dimension-agnostic; verbatim from
     ``tests/integration/twolevel/_residual_ensemble.py``).

  5. :func:`fit_split_conformal_scf` /
     :func:`evaluate_coverage_scf` — thin wrappers around
     :func:`grail.conformal.split_conformal` that keep the SCF-specific
     packing convention inside ``scf/`` so ``grail/`` stays host-agnostic.

The HDF5 schema mirrors the 2-level ensemble closely (top-level datasets
``residuals``, ``delta_x_actual``, ``x_truth``, ``A_emp``, ``config_params``,
plus a ``metadata`` group).  Differences from the NLTE schema:

  - ``config_params`` carries ``R`` and ``seed`` instead of
    ``(T0, n_e0, eps, tau_max)``.
  - ``metadata`` group adds ``basis``, ``method``, ``c_value``,
    ``ensemble_mode`` ∈ {``"noisy_output"``, ``"retrained"``} so a
    downstream consumer can tell which fallback path was used.
  - State-vector dimension ``n_state = 2 * M*(M+1)/2`` = 6 for STO-3G H₂.

References
----------
- SPEC_scf §6 P6 (this milestone).
- SPEC_scf §8.9 (P6 scope decision).
- ``tests/integration/twolevel/_residual_ensemble.py`` (2-level helper).
- ``tests/integration/threelevel/_residual_ensemble.py`` (3-level helper).
- ``grail/conformal.py`` (host-agnostic split-conformal kernel).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch

from grail.conformal import ConformalBand, split_conformal


# ---------------------------------------------------------------------------
# LOO-CV PCA rank selection (dimension-agnostic; verbatim copy from the
# 2-level helper at tests/integration/twolevel/_residual_ensemble.py —
# kept local so this module is independently importable and does not
# depend on any tests/ helper module).
# ---------------------------------------------------------------------------

def loo_cv_pca_rank(
    residuals: np.ndarray,
    r_max: int = 30,
) -> int:
    """Choose PCA truncation rank by leave-one-out cross-validation.

    For each candidate rank ``r ∈ {1, …, r_max}``, computes the LOO-CV
    reconstruction MSE sum and returns the ``r`` minimizing it.  See
    docstring in ``tests/integration/twolevel/_residual_ensemble.py``;
    algorithm copied verbatim.

    Parameters
    ----------
    residuals : ndarray of shape (N, d), float64
        The residual matrix to decompose.
    r_max : int
        Maximum rank to try (capped at ``min(r_max, N-1, d)``).

    Returns
    -------
    best_r : int
        LOO-CV-selected rank.
    """
    N, d = residuals.shape
    r_max_actual = min(r_max, N - 1, d)

    best_r = 1
    best_loo_mse = float("inf")

    for r in range(1, r_max_actual + 1):
        total_loo_mse = 0.0

        for i in range(N):
            loo_residuals = np.delete(residuals, i, axis=0)
            mean_loo = loo_residuals.mean(axis=0)
            X_c = loo_residuals - mean_loo

            _, _, Vh = np.linalg.svd(X_c, full_matrices=False)
            V_r = Vh[:r].T

            delta_T_i_centered = residuals[i] - mean_loo
            proj = V_r @ (V_r.T @ delta_T_i_centered)
            recon = mean_loo + proj
            err = residuals[i] - recon
            total_loo_mse += float(np.dot(err, err))

        if total_loo_mse < best_loo_mse:
            best_loo_mse = total_loo_mse
            best_r = r

    return best_r


# ---------------------------------------------------------------------------
# Stratified split.  SCF buckets are R-bucket-only because the seed axis
# is exchangeable within each R bucket (Gaussian noise realizations) —
# stratification on (R, seed) would degenerate to 1 sample per bucket and
# defeat the purpose.  In the retrained-NN path (not the default fallback)
# the same bucket-by-R-only convention applies because each retrained NN
# is a realization conditional on the same training-data distribution.
# ---------------------------------------------------------------------------

def stratified_split_indices_scf(
    config_params: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 50/50 calibration/test split by ``R`` bucket.

    The structured array ``config_params`` is expected to have an ``R``
    field (float64).  Each unique ``R`` value defines a bucket; within
    each bucket the 50/50 split is shuffled with a per-bucket seed.

    Mirrors :func:`stratified_split_indices_2level` from
    ``tests/integration/twolevel/_residual_ensemble.py`` (TASK-033
    pattern) but with a 1-D bucket key (``R``) instead of the 2-D
    ``(eps, tau_max)`` key from the NLTE case.

    Parameters
    ----------
    config_params : structured ndarray
        Output of :func:`assemble_ensemble_h5_scf` — must carry an ``R``
        field.
    seed : int, default 0
        RNG seed for the within-bucket shuffle.

    Returns
    -------
    (cal_indices, test_indices) : tuple of 1-D int arrays of dtype intp
        Index sets into ``config_params``.  Odd-sized buckets give the
        extra element to calibration so ``|N_cal - N_test| ≤ #buckets``.
    """
    n_configs = len(config_params)
    rng = np.random.default_rng(seed)

    bucket_dict: dict[float, list[int]] = {}
    for idx in range(n_configs):
        key = float(config_params[idx]["R"])
        bucket_dict.setdefault(key, []).append(idx)

    cal_indices_list: list[int] = []
    test_indices_list: list[int] = []

    for key in sorted(bucket_dict.keys()):
        bucket = np.array(bucket_dict[key], dtype=np.intp)
        perm = rng.permutation(len(bucket))
        bucket_shuffled = bucket[perm]

        # Odd bucket: extra element goes to calibration.
        n_cal = (len(bucket) + 1) // 2
        cal_indices_list.extend(bucket_shuffled[:n_cal].tolist())
        test_indices_list.extend(bucket_shuffled[n_cal:].tolist())

    return (
        np.array(cal_indices_list, dtype=np.intp),
        np.array(test_indices_list, dtype=np.intp),
    )


# ---------------------------------------------------------------------------
# Noisy-output XCNet wrapper for the fallback ensemble path.
# ---------------------------------------------------------------------------

class _NoisyXCNetWrapper:
    """Wrap :class:`scf.emulator.XCNet` with a deterministic, smooth, linear-in-P
    perturbation on its ``V_xc`` output.

    The fallback ensemble path (SPEC §6 P6 wall-clock-budget fallback)
    realizes the "different NN emulator" axis as different smooth affine
    perturbations on the trained XCNet's V_xc output.  This is a realistic
    stand-in for stochastic NN emulator usage; the underlying network
    parameters are not perturbed.

    **Why smooth/linear-in-P, not per-call Gaussian noise.**  GRAIL's
    Born-remainder identity assumes the emulator cycle is smooth at the
    truth fixed point — its FD-JVP captures the *effective* Jacobian of
    the emulator cycle.  A per-call non-smooth noise overlay (e.g.
    Gaussian samples keyed by hash(P)) destroys this assumption: nearby
    inputs produce uncorrelated noise outputs, so the FD-JVP step
    amplifies noise by 1/h and dominates the true derivative.  The
    Born-remainder identity then fails by orders of magnitude even though
    the per-call δT is small.  A smooth affine perturbation
    ``δV_xc(P) = scale · (M_seed · (P − P_anchor) + b_seed)`` preserves
    differentiability: GRAIL's FD-JVP through ``fb_nn`` accurately
    measures the perturbed effective Jacobian, and the Born-remainder
    identity holds.

    Per-seed perturbation:
      - ``M_seed`` : fixed ``(n_unique, n_unique)`` matrix with entries
        drawn from a per-seed RNG, normalized so the largest row L2 norm
        is 1.  Captures direction-dependent V_xc bias.
      - ``b_seed`` : fixed ``(n_unique,)`` constant offset, drawn from the
        same per-seed RNG, scaled the same way.
      - ``scale`` : ``noise_std`` parameter (Hartree).  Default 0.005,
        well below the R3 rel_mae gate of 3% and inside the perturbation
        regime the trained NN was characterized over.

    The wrapper exposes the minimal :class:`XCNet` API surface that
    :func:`scf.jvp.make_fock_builder_with_nn` consumes: ``forward``
    (callable), ``eval``, ``n_unique``, and full attribute proxying.
    """

    def __init__(self, base_net, *, seed: int, noise_std: float = 0.005):
        from scf.emulator.network import XCNet
        if not isinstance(base_net, XCNet):
            raise ValueError(
                f"_NoisyXCNetWrapper requires an XCNet base; got "
                f"type {type(base_net).__name__}."
            )
        self._base_net = base_net
        self._seed = int(seed)
        self._noise_std = float(noise_std)

        # Mirror the XCNet attributes that the consumer code reads.
        self.n_unique = base_net.n_unique
        self.hidden = base_net.hidden
        self.depth = base_net.depth

        # Draw the per-seed affine perturbation parameters once at
        # construction.  These are fixed for the lifetime of the wrapper
        # so the noisy NN is a deterministic smooth function of P.
        rng = np.random.default_rng(seed)
        n = base_net.n_unique
        M = rng.standard_normal(size=(n, n))
        # Normalize so the operator 2-norm of M is 1; the noise_std then
        # cleanly governs the perturbation magnitude.
        svals = np.linalg.svd(M, compute_uv=False)
        M = M / max(svals[0], 1e-12)
        b = rng.standard_normal(size=(n,))
        b = b / max(np.linalg.norm(b), 1e-12)
        self._M = torch.from_numpy(M.astype(np.float64))
        self._b = torch.from_numpy(b.astype(np.float64))

    def eval(self):
        self._base_net.eval()
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with deterministic smooth affine perturbation.

        Output is ``base_net(x) + noise_std · (M_seed @ x + b_seed)``
        (broadcast over batch).  Smooth and fully differentiable.
        """
        with torch.no_grad():
            out = self._base_net.forward(x)            # (B, n_unique) or (n_unique,), f64

        x_f64 = x.to(dtype=torch.float64)
        # Compute the affine perturbation in float64.  For (B, n) input
        # the noise is (B, n) too.
        if x_f64.dim() == 1:
            noise = self._M @ x_f64 + self._b              # (n,)
        else:
            # x_f64: (B, n).  M @ x.T → (n, B) → transpose to (B, n).
            noise = x_f64 @ self._M.T + self._b            # (B, n)
        return out + self._noise_std * noise


# ---------------------------------------------------------------------------
# Per-config residual builder.
# ---------------------------------------------------------------------------

def build_one_residual_scf(
    R: float,
    seed: int,
    *,
    xc_net_path: Path | str,
    basis: str = "sto-3g",
    method: str = "uhf",
    picard_tol: float = 1e-9,
    picard_max_iter: int = 200,
    gmres_tol: float = 1e-10,
    gmres_maxiter: int = 50,
    noise_std: float = 0.005,
    ensemble_mode: str = "noisy_output",
    evaluation_dtype: torch.dtype = torch.float64,
) -> dict:
    """Run one config of the SCF residual ensemble sweep.

    For the configuration ``(R, seed)``:

      1. Build the UHF :class:`MoleculeState` and converge the truth UHF
         SCF.
      2. Construct the seed-specific noisy XCNet wrapper.  (For the
         "retrained" mode — not used by default — a different code path
         would load a per-seed trained network; left as a future
         extension.)
      3. Converge the emulator UHF SCF using the noisy NN as the
         Fock-builder.
      4. Compute ``δP*_actual = P*_emul − P*_truth`` in the doubled UHF
         seam (length 6 for STO-3G).
      5. Compute the per-call emulator residual at the truth fixed point
         ``δT = T_emul(P*_truth) − T_truth(P*_truth)`` via one full
         cycle (Fock-build + diag+rebuild) on both sides.
      6. Build the FD-JVPs and call GRAIL's resolvent to obtain
         ``δP*_GRAIL``.
      7. Record everything.

    Parameters
    ----------
    R : float
        H–H bond length in Angstrom.  Must lie inside the trained NN's
        R range (default ``[1.10, 1.25]`` Å for STO-3G UHF).
    seed : int
        Per-config NN-perturbation seed.  Two configs with the same R
        and different seeds produce statistically independent residuals.
    xc_net_path : Path or str
        Path to the trained XCNet HDF5 artifact (R3 hand-off file at
        ``examples/scf_xc_emulator_uhf_<basis>.h5``).
    basis : str, default ``"sto-3g"``
        PySCF basis label.
    method : str, default ``"uhf"``
        UHF or UKS.
    picard_tol, picard_max_iter : convergence params for both SCF calls.
    gmres_tol, gmres_maxiter : convergence params for the GRAIL solve.
    noise_std : float, default 0.005
        Scale of a fixed seed-specific affine perturbation on the NN's
        V_xc output. The historical 32-case driver explicitly uses
        0.001 Ha. No new random noise is drawn at each forward call.
    evaluation_dtype : torch.dtype, default torch.float64
        Compute precision for the frozen checkpoint. Float64 avoids
        float32 finite-difference and equilibrium roundoff floors.
    ensemble_mode : ``"noisy_output"`` or ``"retrained"``
        Recorded as a metadata tag.  Only ``"noisy_output"`` is
        implemented; ``"retrained"`` is reserved for a future path
        that loads per-seed retrained XCNet artifacts.

    Returns
    -------
    dict with the per-config diagnostics (see source for the exact keys).
    Includes a ``picard_converged`` boolean and a ``skip_reason`` field
    populated when either SCF leg stalls — the assembler skips
    non-converged configs.
    """
    # Lazy imports of PySCF-touching modules to keep import cost low.
    from grail.api import resolvent_solve
    from scf._make_state import make_h2_state
    from scf.emulator import load_xc_emulator
    from scf.fock_build import _build_mf, _build_mol, build_fock_truth
    from scf.host import picard_iterate_scf
    from scf.jvp import (
        make_cycle_jvps_scf,
        make_fock_builder_with_nn,
    )
    from scf.state import pack_density_matrix

    if ensemble_mode not in ("noisy_output", "retrained"):
        raise ValueError(
            f"ensemble_mode must be 'noisy_output' or 'retrained', got "
            f"{ensemble_mode!r}."
        )

    # Step 1: build state and converge truth SCF.
    state = make_h2_state(R=R, basis=basis, method=method)
    res_truth = picard_iterate_scf(
        state, tol=picard_tol, max_iter=picard_max_iter, accelerator="diis",
    )

    # Step 2: load trained NN and wrap it with deterministic per-seed noise.
    base_net = load_xc_emulator(str(xc_net_path), dtype=evaluation_dtype)
    if ensemble_mode == "noisy_output":
        xc_net = _NoisyXCNetWrapper(base_net, seed=seed, noise_std=noise_std)
    else:
        # Reserved branch (not exercised in the P6 default path).
        raise NotImplementedError(
            "ensemble_mode='retrained' requires per-seed retrained NN "
            "artifacts; not implemented in the P6 default fallback path."
        )
    fb_nn = make_fock_builder_with_nn(xc_net, state)

    # Step 3: emulator SCF.
    res_emul = picard_iterate_scf(
        state, fock_builder=fb_nn,
        tol=picard_tol, max_iter=picard_max_iter, accelerator="diis",
    )

    n_state = 2 * (state_M(state) * (state_M(state) + 1) // 2)

    # Early exit on non-convergence: assembler will skip this row.
    if not (res_truth.converged and res_emul.converged):
        return {
            "R": float(R),
            "seed": int(seed),
            "picard_converged": False,
            "picard_converged_truth": bool(res_truth.converged),
            "picard_converged_emul": bool(res_emul.converged),
            "truth_iter": int(res_truth.iter_count),
            "emul_iter": int(res_emul.iter_count),
            "skip_reason": "picard_non_convergence",
            "n_state": n_state,
        }

    # Step 4: actual shift at the GRAIL boundary.
    P_truth_flat = pack_density_matrix(res_truth.P_star, method=method)
    P_emul_flat = pack_density_matrix(res_emul.P_star, method=method)
    delta_x_actual_t = P_emul_flat - P_truth_flat

    # Step 5: per-cycle emulator residual via one full T-cycle on each side.
    F_truth = build_fock_truth(state, res_truth.P_star)
    F_nn = fb_nn(state, res_truth.P_star)
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    n_alpha, n_beta = mol.nelec
    P_T_truth = _diag_rebuild_uhf(F_truth[0], F_truth[1], S, n_alpha, n_beta)
    P_T_nn = _diag_rebuild_uhf(F_nn[0], F_nn[1], S, n_alpha, n_beta)
    delta_T_t = pack_density_matrix(
        (torch.from_numpy(P_T_nn[0] - P_T_truth[0]),
         torch.from_numpy(P_T_nn[1] - P_T_truth[1])),
        method=method,
    )

    # Step 6: GRAIL predicted shift.
    host_jvp, emul_jvp = make_cycle_jvps_scf(state, res_truth.P_star, fb_nn)
    res_grail = resolvent_solve(
        P_truth_flat, delta_T_t, emul_jvp, host_jvp,
        gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
    )
    delta_x_pred_t = res_grail.delta_x_star

    # Step 7: assemble diagnostics.
    delta_x_actual = delta_x_actual_t.detach().numpy()
    delta_x_pred = delta_x_pred_t.detach().numpy()
    delta_T = delta_T_t.detach().numpy()

    norm_actual = float(np.linalg.norm(delta_x_actual))
    norm_delta_T = float(np.linalg.norm(delta_T))
    A_emp = norm_actual / norm_delta_T if norm_delta_T > 1e-14 else float("nan")

    norm_diff = float(np.linalg.norm(delta_x_actual - delta_x_pred))
    rel_err = norm_diff / norm_actual if norm_actual > 1e-14 else float("nan")

    # Born-remainder bookkeeping (SPEC §7: C=1.25, tol_floor=0.02).
    born_floor = 1.0 - 1.0 / A_emp if A_emp > 1e-12 else 0.0
    born_ratio = rel_err / (born_floor + 1e-12) if born_floor > 0 else float("nan")

    return {
        "R": float(R),
        "seed": int(seed),
        "picard_converged": True,
        "picard_converged_truth": True,
        "picard_converged_emul": True,
        "truth_iter": int(res_truth.iter_count),
        "emul_iter": int(res_emul.iter_count),
        "s_squared_truth": float(res_truth.s_squared) if res_truth.s_squared is not None else float("nan"),
        "s_squared_emul": float(res_emul.s_squared) if res_emul.s_squared is not None else float("nan"),
        "delta_x_actual": delta_x_actual,
        "delta_x_predicted": delta_x_pred,
        "delta_T_flat": delta_T,
        "x_truth_flat": P_truth_flat.detach().numpy(),
        "norm_actual": norm_actual,
        "norm_delta_T": norm_delta_T,
        "rel_err": rel_err,
        "A_emp": A_emp,
        "born_ratio": born_ratio,
        "gmres_iter_count": int(res_grail.iter_count),
        "cycle_defect_truth_l2": res_truth.cycle_defect_l2,
        "cycle_defect_emul_l2": res_emul.cycle_defect_l2,
        "equation_converged_truth": res_truth.equation_converged,
        "equation_converged_emul": res_emul.equation_converged,
        "fock_nn_at_truth": np.concatenate([F[np.triu_indices(F.shape[0])] for F in F_nn]),
        "x_emul_flat": P_emul_flat.detach().numpy(),
        "evaluation_dtype": str(evaluation_dtype),
        "noise_std": float(noise_std),
        "n_state": n_state,
    }


def state_M(state) -> int:
    """Helper: read ``M = mol.nao_nr()`` from a :class:`MoleculeState` once.

    Encapsulates the PySCF call inside this module so callers do not need to
    import PySCF.  The result is the AO-basis size for the given basis
    (e.g. 2 for STO-3G H₂, 4 for 6-31G H₂).
    """
    from scf.fock_build import _build_mol
    mol = _build_mol(state)
    return int(mol.nao_nr())


def _diag_rebuild_uhf(
    F_alpha: np.ndarray, F_beta: np.ndarray, S: np.ndarray,
    n_alpha: int, n_beta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One per-spin diagonalize+rebuild step (no DIIS).

    Mirrors :func:`scf.host._build_density_from_orbitals_uhf` after
    solving the per-spin generalized eigenproblem.  Used to compute
    ``T(P) = 𝐃[𝐅(P)]`` at the truth fixed point for the per-call
    residual ``δT = T_emul(P*) − T_truth(P*)``.
    """
    from pyscf.scf import hf as scf_hf
    F_alpha = 0.5 * (F_alpha + F_alpha.T)
    F_beta = 0.5 * (F_beta + F_beta.T)
    _ea, Ca = scf_hf.eig(F_alpha, S)
    _eb, Cb = scf_hf.eig(F_beta, S)
    Pa = Ca[:, :n_alpha] @ Ca[:, :n_alpha].T
    Pb = Cb[:, :n_beta] @ Cb[:, :n_beta].T
    return 0.5 * (Pa + Pa.T), 0.5 * (Pb + Pb.T)


# ---------------------------------------------------------------------------
# HDF5 ensemble writer / reader.
# ---------------------------------------------------------------------------

def assemble_ensemble_h5_scf(
    out_path: Path | str,
    results: list[dict],
    *,
    n_state: int,
    R_grid: np.ndarray,
    seeds: np.ndarray,
    basis: str,
    method: str,
    c_value: float,
    pca_rank_loo_cv: int,
    pca_explained_variance: float,
    xc_net_path: str,
    ensemble_mode: str,
    noise_std: float,
    git_sha: str,
    excluded_configs: list[str],
    creation_date: str,
) -> None:
    """Write the SCF residual ensemble to HDF5.

    Schema (parallel to the 2-level and 3-level ensembles, with SCF-specific
    fields)::

      /metadata
        n_configs              : int
        n_state                : int  (doubled-seam length, 6 for STO-3G H₂)
        pca_rank_loo_cv        : int
        pca_explained_variance : float
        basis                  : str
        method                 : str  ("uhf" or "uks")
        c_value                : float (SPEC §7 Born-remainder C)
        ensemble_mode          : str  ("noisy_output" or "retrained")
        noise_std              : float
        xc_net_path            : str
        git_sha                : str
        creation_date          : str
        R_grid                 : str  (JSON-encoded list)
        seeds                  : str  (JSON-encoded list)
        excluded_configs       : str  (JSON-encoded list)
      /residuals               : (n_configs, n_state) float64  — δT
      /delta_x_actual          : (n_configs, n_state) float64
      /delta_x_predicted       : (n_configs, n_state) float64
      /x_truth                 : (n_configs, n_state) float64
      /A_emp                   : (n_configs,) float64
      /rel_err                 : (n_configs,) float64
      /born_ratio              : (n_configs,) float64
      /config_params           : structured array {R, seed}
    """
    out_path = Path(out_path)
    n_configs = len(results)
    if n_configs == 0:
        raise ValueError("assemble_ensemble_h5_scf: results list is empty.")

    residuals = np.stack([r["delta_T_flat"] for r in results], axis=0)
    delta_x_actual = np.stack([r["delta_x_actual"] for r in results], axis=0)
    delta_x_predicted = np.stack([r["delta_x_predicted"] for r in results], axis=0)
    x_truth = np.stack([r["x_truth_flat"] for r in results], axis=0)
    A_emp = np.array([r["A_emp"] for r in results], dtype=np.float64)
    rel_err = np.array([r["rel_err"] for r in results], dtype=np.float64)
    born_ratio = np.array([r["born_ratio"] for r in results], dtype=np.float64)

    dtype_config = np.dtype([("R", np.float64), ("seed", np.int64)])
    config_arr = np.zeros(n_configs, dtype=dtype_config)
    for idx, r in enumerate(results):
        config_arr[idx]["R"] = float(r["R"])
        config_arr[idx]["seed"] = int(r["seed"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["n_configs"] = int(n_configs)
        meta.attrs["n_state"] = int(n_state)
        meta.attrs["pca_rank_loo_cv"] = int(pca_rank_loo_cv)
        meta.attrs["pca_explained_variance"] = float(pca_explained_variance)
        meta.attrs["basis"] = str(basis)
        meta.attrs["method"] = str(method)
        meta.attrs["c_value"] = float(c_value)
        meta.attrs["ensemble_mode"] = str(ensemble_mode)
        meta.attrs["noise_std"] = float(noise_std)
        meta.attrs["xc_net_path"] = str(xc_net_path)
        meta.attrs["git_sha"] = str(git_sha)
        meta.attrs["creation_date"] = str(creation_date)
        meta.attrs["R_grid"] = json.dumps([float(r) for r in R_grid])
        meta.attrs["seeds"] = json.dumps([int(s) for s in seeds])
        meta.attrs["excluded_configs"] = json.dumps(excluded_configs)

        f.create_dataset("residuals", data=residuals, dtype=np.float64)
        f.create_dataset("delta_x_actual", data=delta_x_actual, dtype=np.float64)
        f.create_dataset("delta_x_predicted", data=delta_x_predicted, dtype=np.float64)
        f.create_dataset("x_truth", data=x_truth, dtype=np.float64)
        f.create_dataset("A_emp", data=A_emp, dtype=np.float64)
        f.create_dataset("rel_err", data=rel_err, dtype=np.float64)
        f.create_dataset("born_ratio", data=born_ratio, dtype=np.float64)
        f.create_dataset("config_params", data=config_arr)


def load_ensemble_h5_scf(h5_path: Path | str) -> dict:
    """Load the SCF residual ensemble from HDF5.

    Returns a dict matching the schema in :func:`assemble_ensemble_h5_scf`,
    with the ``metadata`` group flattened into a sub-dict and the
    JSON-encoded list-valued metadata fields parsed back into Python
    lists.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        meta_grp = f["metadata"]
        metadata: dict = {k: meta_grp.attrs[k] for k in meta_grp.attrs}
        # Parse JSON-encoded list fields.
        for key in ("R_grid", "seeds", "excluded_configs"):
            if key in metadata:
                metadata[key] = json.loads(str(metadata[key]))

        return {
            "residuals": f["residuals"][:],
            "delta_x_actual": f["delta_x_actual"][:],
            "delta_x_predicted": f["delta_x_predicted"][:],
            "x_truth": f["x_truth"][:],
            "A_emp": f["A_emp"][:],
            "rel_err": f["rel_err"][:],
            "born_ratio": f["born_ratio"][:],
            "config_params": f["config_params"][:],
            "metadata": metadata,
        }


# ---------------------------------------------------------------------------
# Public API surface: split-conformal adapters for the SCF state seam.
# ---------------------------------------------------------------------------

def build_scf_residual_ensemble(
    h5_path: Path | str,
    *,
    split_seed: int = 0,
) -> dict:
    """Load the SCF residual ensemble and produce a stratified cal/test split.

    Convenience entry point used by the P6 integration tests.  Returns a
    dict containing the ensemble arrays plus the cal/test index sets from
    :func:`stratified_split_indices_scf` and the calibration-side
    residuals ``cal_residuals = δP*_actual − δP*_GRAIL`` ready for
    :func:`fit_split_conformal_scf`.

    Parameters
    ----------
    h5_path : Path or str
        Path to the residual-ensemble HDF5 (default
        ``examples/scf_residual_ensemble.h5``).
    split_seed : int, default 0
        Seed for the within-bucket shuffle in
        :func:`stratified_split_indices_scf`.

    Returns
    -------
    dict with keys:
      - ``ensemble`` — full output of :func:`load_ensemble_h5_scf`.
      - ``cal_indices``, ``test_indices`` — int arrays.
      - ``cal_residuals``, ``test_residuals`` — float64 ndarrays of shape
        ``(N_*, n_state)``.  Already ``δP*_actual − δP*_GRAIL``.
      - ``test_predicted`` — ndarray ``(N_test, n_state)`` of GRAIL
        predictions on the test split.
    """
    ensemble = load_ensemble_h5_scf(h5_path)
    config_params = ensemble["config_params"]

    cal_indices, test_indices = stratified_split_indices_scf(
        config_params, seed=split_seed,
    )

    delta_x_actual = ensemble["delta_x_actual"]
    delta_x_predicted = ensemble["delta_x_predicted"]

    cal_residuals = delta_x_actual[cal_indices] - delta_x_predicted[cal_indices]
    test_residuals = delta_x_actual[test_indices] - delta_x_predicted[test_indices]
    test_predicted = delta_x_predicted[test_indices]

    return {
        "ensemble": ensemble,
        "cal_indices": cal_indices,
        "test_indices": test_indices,
        "cal_residuals": cal_residuals,
        "test_residuals": test_residuals,
        "test_predicted": test_predicted,
    }


def fit_split_conformal_scf(
    cal_residuals: np.ndarray,
    test_predictions: np.ndarray,
    *,
    alpha: float = 0.10,
    nonconformity: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ConformalBand:
    """Fit a split-conformal L∞ band on SCF state-vector residuals.

    Thin wrapper around :func:`grail.conformal.split_conformal`: handles
    the dtype conversion to torch.float64 inside ``scf/`` so callers can
    pass numpy arrays and the host-agnostic GRAIL kernel stays untouched.

    Parameters
    ----------
    cal_residuals : ndarray (N_cal, n_state), float64
        Per-config residuals ``δP*_actual − δP*_GRAIL`` on the
        calibration split.
    test_predictions : ndarray (n_state,) or (N_test, n_state), float64
        GRAIL predicted shift(s) on the test split; the conformal band is
        centered on these.
    alpha : float, default 0.10
        Nominal miscoverage; the band guarantees coverage ≥ 1 − α.
    nonconformity : callable, optional
        Custom nonconformity score.  Default L∞ (same as
        :func:`grail.conformal.split_conformal`).

    Returns
    -------
    ConformalBand with ``low`` and ``high`` matching the shape of
    ``test_predictions``.
    """
    cal_t = torch.from_numpy(np.ascontiguousarray(cal_residuals, dtype=np.float64))
    test_t = torch.from_numpy(np.ascontiguousarray(test_predictions, dtype=np.float64))
    return split_conformal(cal_t, test_t, alpha=alpha, nonconformity=nonconformity)


def evaluate_coverage_scf(
    band: ConformalBand,
    test_residuals: np.ndarray,
    *,
    tol: float = 1e-12,
) -> float:
    """Empirical L∞ coverage on a held-out test split.

    Counts the fraction of test residuals (``δP*_actual − δP*_GRAIL`` on
    the test split) whose L∞ norm lies within the band's half-width
    ``q_hat`` (read from ``band.high − band.low`` divided by 2).  The
    band's center is irrelevant for L∞ coverage since the nonconformity
    score in :mod:`grail.conformal` is centered on each test prediction
    by construction.

    Parameters
    ----------
    band : ConformalBand
        Output of :func:`fit_split_conformal_scf`.  Must have been built
        from a single test prediction (``test_predictions`` of shape
        ``(n_state,)``) so the band half-width is unambiguous.
    test_residuals : ndarray (N_test, n_state), float64
        ``δP*_actual − δP*_GRAIL`` on the test split.
    tol : float, default 1e-12
        Numerical slack on the comparison.

    Returns
    -------
    coverage : float in [0, 1]
        Fraction ``mean(L∞(r_i) ≤ q_hat + tol)``.
    """
    # Band half-width: q_hat = (high - low) / 2.
    if band.low.dim() == 1:
        q_hat = float(((band.high - band.low)[0] / 2.0).item())
    else:
        q_hat = float(((band.high - band.low)[0, 0] / 2.0).item())
    l_inf = np.abs(np.ascontiguousarray(test_residuals, dtype=np.float64)).max(axis=1)
    return float((l_inf <= q_hat + tol).mean())
