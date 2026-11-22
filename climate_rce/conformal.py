'Statewise covariance propagation and split-conformal calibration for the climate reference model. Calibration guarantees require exchangeable physical instances and apply to the chosen statewise prediction target.'

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

# Host-agnostic conformal kernel — SPEC §6 P3: do NOT reimplement.
from grail.conformal import ConformalBand, split_conformal


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def loo_cv_pca_rank(
    residuals: np.ndarray,
    r_max: int = 30,
) -> int:
    """Choose PCA truncation rank by leave-one-out cross-validation.

    For each candidate rank ``r ∈ {1, …, r_max}``, computes the LOO-CV
    reconstruction MSE sum and returns the ``r`` that minimizes it.  For the
    RCE testbed the residual matrix is ``(N_configs, n_z)`` with
    ``n_z ∈ [60, 200]``, so the PCA rank is bounded by
    ``min(r_max, N−1, n_z)``.

    Parameters
    ----------
    residuals : ndarray (N, d), float64
        Stacked ``δx*_actual − δx*_GRAIL`` over the calibration set.
    r_max : int, default 30
        Maximum rank to try; capped at ``min(r_max, N−1, d)``.

    Returns
    -------
    best_r : int in [1, min(r_max, N-1, d)]
        LOO-CV-selected rank.
    """
    N, d = residuals.shape
    r_max_actual = min(r_max, N - 1, d)

    best_r = 1
    best_loo_mse = float("inf")

    for r in range(1, r_max_actual + 1):
        total_loo_mse = 0.0
        for i in range(N):
            loo = np.delete(residuals, i, axis=0)
            mean_loo = loo.mean(axis=0)
            X_c = loo - mean_loo
            _, _, Vh = np.linalg.svd(X_c, full_matrices=False)
            V_r = Vh[:r].T
            centered_i = residuals[i] - mean_loo
            proj = V_r @ (V_r.T @ centered_i)
            err = residuals[i] - (mean_loo + proj)
            total_loo_mse += float(np.dot(err, err))

        if total_loo_mse < best_loo_mse:
            best_loo_mse = total_loo_mse
            best_r = r

    return best_r


# ---------------------------------------------------------------------------
# Stratified split by (feedback-dial bucket × CO₂ bucket)
# ---------------------------------------------------------------------------

def stratified_split_indices_rce(
    config_params: np.ndarray,
    *,
    seed: int = 42,
    n_f_bins: int = 4,
    n_co2_bins: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 50/50 calibration/test split by (feedback-dial × CO₂ bucket).

    The structured array ``config_params`` must carry ``f`` (feedback factor,
    float64) and ``co2_ppm`` (float64) fields.  Bins are computed from
    quantiles of the observed distribution, mirroring the M3f TASK-033
    ``stratified_split_indices`` helper (SPEC §6 P6).

    The near-critical-``f`` bucket (highest ``f`` bin) drives conformal heavy
    tails (``A_emp → ∞`` as ``ρ(K) → 1``), analogous to the 3-level cross-ε
    heavy-tail bucket (the callback contract M3f / SPEC §6 P6 "Expectation note").
    The stratified split ensures the heavy-tail bucket is represented in
    both the calibration and test splits, preventing the positional-split
    undercovers documented for the NLTE 3-level case.

    Parameters
    ----------
    config_params : structured ndarray or 2-D float ndarray
        If structured: must carry ``f`` and ``co2_ppm`` fields.
        If 2-D float array of shape (N, 2): columns are [f, co2_ppm].
    seed : int, default 42
    n_f_bins : int, default 4
        Number of feedback-factor quantile bins.
    n_co2_bins : int, default 2
        Number of CO₂ quantile bins.

    Returns
    -------
    (cal_indices, test_indices) : tuple of 1-D int arrays, dtype intp
        50/50 split; odd-sized buckets give the extra element to calibration.
    """
    n_configs = len(config_params)
    rng = np.random.default_rng(seed)

    # Extract f and co2_ppm vectors
    if config_params.dtype.names is not None:
        # Structured array
        f_vals = np.asarray([config_params[i]["f"] for i in range(n_configs)], dtype=np.float64)
        co2_vals = np.asarray([config_params[i]["co2_ppm"] for i in range(n_configs)], dtype=np.float64)
    else:
        # Plain 2-D float array: columns [f, co2_ppm]
        arr = np.asarray(config_params, dtype=np.float64)
        f_vals = arr[:, 0]
        co2_vals = arr[:, 1]

    # Quantile bin edges
    f_edges = np.percentile(f_vals, np.linspace(0, 100, n_f_bins + 1))
    co2_edges = np.percentile(co2_vals, np.linspace(0, 100, n_co2_bins + 1))

    # Assign each config to a (f bin, CO₂ bin) bucket
    f_bin = np.clip(np.digitize(f_vals, f_edges[1:-1]), 0, n_f_bins - 1)
    co2_bin = np.clip(np.digitize(co2_vals, co2_edges[1:-1]), 0, n_co2_bins - 1)
    bucket_keys = f_bin * n_co2_bins + co2_bin

    bucket_dict: dict[int, list[int]] = {}
    for idx in range(n_configs):
        key = int(bucket_keys[idx])
        bucket_dict.setdefault(key, []).append(idx)

    cal_list: list[int] = []
    test_list: list[int] = []

    for key in sorted(bucket_dict.keys()):
        bucket = np.array(bucket_dict[key], dtype=np.intp)
        perm = rng.permutation(len(bucket))
        bucket_shuffled = bucket[perm]
        n_cal = (len(bucket) + 1) // 2   # odd bucket → extra to cal
        cal_list.extend(bucket_shuffled[:n_cal].tolist())
        test_list.extend(bucket_shuffled[n_cal:].tolist())

    return (
        np.array(cal_list, dtype=np.intp),
        np.array(test_list, dtype=np.intp),
    )


# ---------------------------------------------------------------------------
# Thin conformal wrappers (P3-required public surface)
# ---------------------------------------------------------------------------

def fit_split_conformal_rce(
    cal_residuals: np.ndarray,
    test_predictions: np.ndarray,
    *,
    alpha: float = 0.10,
    nonconformity: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ConformalBand:
    """Fit a split-conformal L∞ band on RCE column T-profile residuals.

    **Thin pass-through to :func:`grail.conformal.split_conformal`.**
    This function does NOT reimplement the quantile.  The Tier U test
    ``test_conformal_adapter.py`` verifies this explicitly.

    Parameters
    ----------
    cal_residuals : ndarray (N_cal, n_z) or (N_cal,), float64
        Per-config conformal residuals ``δx*_actual − δx*_GRAIL`` on the
        calibration split.  Shape ``(N_cal, n_z)`` for the T-profile state.
    test_predictions : ndarray (n_z,) or (N_test, n_z), float64
        GRAIL-predicted RCE state shifts on the test split.
    alpha : float, default 0.10
        Nominal miscoverage; band achieves coverage ≥ 1 − alpha.
    nonconformity : callable, optional
        Custom nonconformity score (default L∞ from ``grail.conformal``).

    Returns
    -------
    ConformalBand
        Imported directly from :mod:`grail.conformal` — no reimplementation.
    """
    # dtype cast + contiguous layout for grail.conformal
    cal_t = torch.from_numpy(np.ascontiguousarray(cal_residuals, dtype=np.float64))
    test_t = torch.from_numpy(np.ascontiguousarray(test_predictions, dtype=np.float64))
    # Delegate to grail.conformal (no quantile reimplemented here — SPEC §6 P3).
    return split_conformal(cal_t, test_t, alpha=alpha, nonconformity=nonconformity)


def evaluate_coverage_rce(
    band: ConformalBand,
    test_residuals: np.ndarray,
    *,
    tol: float = 1e-12,
) -> float:
    """Empirical L∞ coverage on a held-out RCE test split.

    Counts the fraction of test T-profile residuals whose L∞ norm lies
    within the band's half-width ``q_hat``.

    Parameters
    ----------
    band : ConformalBand
        Output of :func:`fit_split_conformal_rce`.
    test_residuals : ndarray (N_test, n_z), float64
        ``δx*_actual − δx*_GRAIL`` on the test split.
    tol : float, default 1e-12
        Numerical slack on the L∞ comparison.

    Returns
    -------
    coverage : float in [0, 1]
        ``mean(L∞(r_i) ≤ q_hat + tol)`` over the test set.
    """
    # Extract q_hat from the band half-width (band is symmetric about the prediction).
    if band.low.dim() == 1:
        q_hat = float(((band.high - band.low)[0] / 2.0).item())
    else:
        q_hat = float(((band.high - band.low)[0, 0] / 2.0).item())

    l_inf = np.abs(np.ascontiguousarray(test_residuals, dtype=np.float64)).max(axis=1)
    return float((l_inf <= q_hat + tol).mean())
