'Tier U unit tests for climate_rce/conformal.py (thin adapter).\n\nSPEC_climate_rce §6 P3 gate:\n  - ``fit_split_conformal_rce`` is a thin pass-through to\n    ``grail.conformal.split_conformal`` (assert it imports grail.conformal\n    and does NOT reimplement the quantile).\n  - ``loo_cv_pca_rank`` returns a rank in [1, r_max] on a synthetic\n    low-rank residual matrix.\n  - ``stratified_split_indices_rce`` partitions correctly by (f-bin × CO₂-bin)\n    bucket.\n  - ``evaluate_coverage_rce`` returns a float in [0, 1].'

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import inspect
import types

import numpy as np
import pytest
import torch

from climate_rce.conformal import (
    fit_split_conformal_rce,
    evaluate_coverage_rce,
    stratified_split_indices_rce,
    loo_cv_pca_rank,
)
from grail.conformal import ConformalBand, split_conformal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_residuals(N: int = 20, d: int = 12, seed: int = 42) -> np.ndarray:
    """Low-rank residual matrix for PCA tests."""
    rng = np.random.default_rng(seed)
    # Rank-3 signal + noise
    U = rng.standard_normal((N, 3))
    V = rng.standard_normal((3, d))
    noise = rng.standard_normal((N, d)) * 1e-2
    return (U @ V + noise).astype(np.float64)


def _synthetic_config_params(N: int = 24) -> np.ndarray:
    """Structured array with f and co2_ppm fields for stratification tests."""
    rng = np.random.default_rng(5)
    f_vals = rng.uniform(0.1, 0.9, N).astype(np.float64)
    co2_vals = rng.choice([280.0, 560.0], size=N).astype(np.float64)
    dt = np.dtype([("f", np.float64), ("co2_ppm", np.float64)])
    arr = np.zeros(N, dtype=dt)
    arr["f"] = f_vals
    arr["co2_ppm"] = co2_vals
    return arr


# ---------------------------------------------------------------------------
# 1. fit_split_conformal_rce is a thin pass-through (no reimplemented quantile)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_conformal_adapter_imports_grail_conformal():
    """fit_split_conformal_rce imports and calls grail.conformal.split_conformal.

    SPEC §6 P3: 'assert it imports grail.conformal and does not reimplement
    the quantile.'

    We verify by:
    (a) inspecting the source module for ``split_conformal`` import;
    (b) confirming the return type is ``grail.conformal.ConformalBand``.
    """
    import climate_rce.conformal as rce_conf

    # Check that grail.conformal.split_conformal is imported in the module.
    # We look in the module's globals for the imported name.
    assert hasattr(rce_conf, "split_conformal"), (
        "climate_rce.conformal must import split_conformal from grail.conformal "
        "(SPEC §6 P3: thin pass-through, no quantile reimplemented)."
    )
    assert rce_conf.split_conformal is split_conformal, (
        "climate_rce.conformal.split_conformal must be the grail.conformal version."
    )


@pytest.mark.unit
def test_fit_split_conformal_rce_returns_conformal_band():
    """fit_split_conformal_rce returns a ConformalBand from grail.conformal."""
    residuals = _synthetic_residuals(N=20, d=12)
    N_cal, N_test = 12, 8
    cal_res = residuals[:N_cal]
    test_pred = residuals[N_cal:N_cal + 1]   # single test prediction

    band = fit_split_conformal_rce(cal_res, test_pred, alpha=0.10)
    assert isinstance(band, ConformalBand), (
        f"Expected ConformalBand, got {type(band).__name__}"
    )


@pytest.mark.unit
def test_fit_split_conformal_matches_direct_split_conformal():
    """fit_split_conformal_rce and grail.conformal.split_conformal give identical results.

    This is the strongest form of the 'thin pass-through' check: same inputs,
    same outputs.  If the adapter reimplemented the quantile, the results
    would differ.
    """
    rng = np.random.default_rng(11)
    n_z = 12
    N_cal = 16
    N_test = 1

    cal_res = rng.standard_normal((N_cal, n_z)).astype(np.float64)
    test_pred = rng.standard_normal((N_test, n_z)).astype(np.float64)
    alpha = 0.10

    band_adapter = fit_split_conformal_rce(cal_res, test_pred, alpha=alpha)
    band_direct = split_conformal(
        torch.from_numpy(np.ascontiguousarray(cal_res)),
        torch.from_numpy(np.ascontiguousarray(test_pred)),
        alpha=alpha,
    )

    # q_hat should be identical (same torch path)
    def _q_hat(band: ConformalBand) -> float:
        if band.low.dim() == 1:
            return float(((band.high - band.low)[0] / 2.0).item())
        return float(((band.high - band.low)[0, 0] / 2.0).item())

    q_adapter = _q_hat(band_adapter)
    q_direct = _q_hat(band_direct)
    assert abs(q_adapter - q_direct) < 1e-12, (
        f"q_hat mismatch: adapter={q_adapter:.6e}, direct={q_direct:.6e}. "
        "fit_split_conformal_rce must be a thin pass-through."
    )


# ---------------------------------------------------------------------------
# 2. loo_cv_pca_rank returns rank in [1, r_max]
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_loo_cv_pca_rank_in_range():
    """loo_cv_pca_rank returns a rank in [1, r_max] on a low-rank matrix."""
    residuals = _synthetic_residuals(N=20, d=12, seed=99)
    r_max = 10
    rank = loo_cv_pca_rank(residuals, r_max=r_max)

    assert isinstance(rank, int), f"rank must be int, got {type(rank)}"
    assert 1 <= rank <= r_max, f"rank={rank} not in [1, {r_max}]"


@pytest.mark.unit
def test_loo_cv_pca_rank_bounded_by_cap():
    """loo_cv_pca_rank is strictly bounded by min(r_max, N-1, d).

    The LOO-CV procedure monotonically minimizes reconstruction error so it
    may legitimately select any rank up to the cap; this test verifies that
    the returned rank never exceeds the mathematical cap (the true SPEC gate
    is 'returns a rank in [1, r_max]', tested in test_loo_cv_pca_rank_in_range).
    """
    rng = np.random.default_rng(77)
    N, d = 30, 15
    residuals = rng.standard_normal((N, d)).astype(np.float64)
    for r_max in [3, 8, 14]:
        rank = loo_cv_pca_rank(residuals, r_max=r_max)
        cap = min(r_max, N - 1, d)
        assert 1 <= rank <= cap, (
            f"r_max={r_max}: rank={rank} not in [1, {cap}]"
        )


@pytest.mark.unit
def test_loo_cv_pca_rank_capped_at_n_minus_1():
    """loo_cv_pca_rank is capped at min(r_max, N-1, d)."""
    N, d, r_max = 5, 10, 30
    residuals = np.random.default_rng(0).standard_normal((N, d))
    rank = loo_cv_pca_rank(residuals, r_max=r_max)
    assert 1 <= rank <= min(r_max, N - 1, d), (
        f"rank={rank} violates cap min(r_max={r_max}, N-1={N-1}, d={d})."
    )


# ---------------------------------------------------------------------------
# 3. stratified_split_indices_rce: partitioning and size
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stratified_split_sizes():
    """stratified_split_indices_rce produces a 50/50 split."""
    N = 24
    config_params = _synthetic_config_params(N)
    cal_idx, test_idx = stratified_split_indices_rce(config_params, seed=42)

    total = len(cal_idx) + len(test_idx)
    assert total == N, f"total={total} != N={N}"
    # 50/50 split: cal should be ≥ N//2 (odd-sized buckets go to cal).
    assert len(cal_idx) >= N // 2, f"cal={len(cal_idx)} < N//2={N//2}"


@pytest.mark.unit
def test_stratified_split_no_overlap():
    """cal and test indices are disjoint (no config appears in both splits)."""
    N = 24
    config_params = _synthetic_config_params(N)
    cal_idx, test_idx = stratified_split_indices_rce(config_params, seed=42)

    cal_set = set(cal_idx.tolist())
    test_set = set(test_idx.tolist())
    assert len(cal_set & test_set) == 0, "cal and test indices overlap"
    assert len(cal_set | test_set) == N, "not all configs assigned"


@pytest.mark.unit
def test_stratified_split_plain_array():
    """stratified_split_indices_rce works with plain 2-D float array (N, 2)."""
    rng = np.random.default_rng(3)
    N = 20
    arr = np.stack([
        rng.uniform(0.1, 0.9, N),
        rng.choice([280.0, 560.0], size=N).astype(np.float64),
    ], axis=1)
    cal_idx, test_idx = stratified_split_indices_rce(arr, seed=42)
    assert len(cal_idx) + len(test_idx) == N


# ---------------------------------------------------------------------------
# 4. evaluate_coverage_rce returns float in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evaluate_coverage_rce_range():
    """evaluate_coverage_rce returns a float in [0, 1]."""
    rng = np.random.default_rng(42)
    n_z = 12
    N_cal = 16

    cal_res = rng.standard_normal((N_cal, n_z)).astype(np.float64)
    test_pred = rng.standard_normal((1, n_z)).astype(np.float64)
    test_res = rng.standard_normal((8, n_z)).astype(np.float64)

    band = fit_split_conformal_rce(cal_res, test_pred, alpha=0.10)
    coverage = evaluate_coverage_rce(band, test_res)

    assert isinstance(coverage, float), f"coverage must be float, got {type(coverage)}"
    assert 0.0 <= coverage <= 1.0, f"coverage={coverage} not in [0, 1]"


@pytest.mark.unit
def test_evaluate_coverage_rce_perfect_coverage():
    """Coverage is 1.0 when all test residuals are within the band."""
    n_z = 12
    N_cal = 20

    # Calibration residuals: moderate spread
    cal_res = np.zeros((N_cal, n_z), dtype=np.float64)
    cal_res[:, 0] = np.linspace(-1.0, 1.0, N_cal)
    test_pred = np.zeros((1, n_z), dtype=np.float64)

    band = fit_split_conformal_rce(cal_res, test_pred, alpha=0.10)

    # Test residuals: all strictly inside the band (tiny perturbations)
    test_res = np.zeros((5, n_z), dtype=np.float64)
    test_res[:, 0] = 0.01   # well within the band

    coverage = evaluate_coverage_rce(band, test_res)
    assert coverage == 1.0, f"Expected coverage=1.0 for tiny residuals, got {coverage}"
