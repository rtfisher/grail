"""Split-conformal calibration of state or observable residual vectors.

The caller chooses and constructs the target residuals before calibration.
For an emergent-flux target, its state-to-flux projection belongs to the host;
this generic calibrator does not perform a projection or import a solver.

Theory (Angelopoulos & Bates 2023, "A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification")
------------------------------------------------------
Given n calibration pairs (y_i, ŷ_i) and a nonconformity score s_i = |y_i - ŷ_i|,
the split-conformal threshold q̂ is the k-th smallest calibration score,
where k = ceil((n+1)(1-α)). If k > n, q̂ = +infinity. Under joint
exchangeability of calibration and test scores, this gives marginal coverage:

    P(|Y_{n+1} - Ŷ_{n+1}| ≤ q̂) ≥ 1 - α

The implementation here uses the scalar quantile version applied to the
maximum absolute error across all observable dimensions, so the band is
simultaneously valid across the full spectrum.

Caller API
----------
  band = split_conformal(
      calibration_residuals,   # (N_cal,) or (N_cal, d): true - predicted
      test_predictions,        # (d,) or (N_test, d): what we predict for new points
      alpha=0.1,
  )
  # band.low, band.high are (d,)-shaped coverage intervals.

Float64 throughout.  No RNG used in this module.

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005), Algorithmic Learning in a Random
World, Springer.
Angelopoulos, A. & Bates, S. (2023), arXiv:2107.07511.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .backend import Backend, get_backend


def conformal_quantile(scores, *, alpha: float = 0.1) -> float:
    """Return the corrected calibration order statistic, including +infinity.

    Scores must be finite, nonnegative scalars. No interpolated quantile is
    used: the one-based index is ceil((n+1)(1-alpha)). This numerical rule
    matches the threshold in the paper's split-conformal theorem; whether
    scores are exchangeable remains an application assumption.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("scores must be finite and nonnegative")
    k = math.ceil((values.size + 1) * (1.0 - alpha))
    if k > values.size:
        return math.inf
    return float(np.partition(values, k - 1)[k - 1])


@dataclass
class ConformalBand:
    """Distribution-free coverage interval at miscoverage alpha.  the technical reference §5.5.

    Attributes
    ----------
    low : array of shape (d,)
        Lower endpoint of the prediction interval.  Native backend type.
    high : array of shape (d,)
        Upper endpoint of the prediction interval.  Native backend type.
    alpha : float
        Nominal miscoverage level; coverage is >= 1 - alpha with finite-sample
        guarantees.
    n_calibration : int
        Number of calibration samples used to compute the quantile.
    """

    low: object   # native backend array
    high: object  # native backend array
    alpha: float
    n_calibration: int


def split_conformal(
    calibration_residuals,
    test_predictions,
    *,
    alpha: float = 0.1,
    nonconformity: Optional[Callable] = None,
    backend: Optional[Backend] = None,
) -> ConformalBand:
    """Split-conformal calibration on held-out residuals.

    Returns a finite-sample distribution-free coverage band at miscoverage
    alpha.  the technical reference §5.5.

    Parameters
    ----------
    calibration_residuals : array of shape (N_cal,) or (N_cal, d), float64
        Calibration nonconformity values: true_observable - predicted_observable
        on the held-out split.  If 2-D, the nonconformity score for each
        calibration sample is reduced to a scalar via the default score or
        the user-supplied nonconformity function.  May be any backend type.
    test_predictions : array of shape (d,) or (N_test, d), float64
        Model predictions for the test point(s).  The coverage band is
        centered on each test prediction.  May be any backend type.
    alpha : float in (0, 1), default 0.1
        Miscoverage level; the band achieves coverage >= 1 - alpha.
    nonconformity : callable (d,) -> scalar, optional
        Maps a residual vector to a nonnegative scalar score.  Default:
        max absolute value across all observable dimensions (sup-norm), which
        gives a simultaneous band valid at every frequency.  The user may
        supply a score that dominates the sup-norm for every candidate
        residual, such as the 2-norm. That condition makes score acceptance
        imply membership of the returned coordinate box. Arbitrary scores
        require their own prediction-set inversion, rather than this box.
    backend : Backend or None
        Active backend for returned arrays.  Defaults to the module-level
        default (torch).

    Returns
    -------
    ConformalBand with low and high arrays matching the shape of
    test_predictions (native backend type), and alpha and n_calibration set
    accordingly.

    Notes
    -----
    The finite-sample order-statistic index is:

        k = ceil((N_cal + 1) * (1 - alpha))

    The threshold is the k-th smallest score, or +infinity when k>N_cal.
    The guarantee assumes joint exchangeability of calibration and test
    scores under a scoring rule fixed independently of calibration outcomes.
    Freezing a rule alone does not establish exchangeability.

    For vector residuals (d > 1), the default nonconformity score is the
    L∞ norm of the residual.  The resulting q̂ is then used as a symmetric
    ±q̂ band around test_predictions.  This is conservative for individual
    frequency channels but valid simultaneously over all channels.
    """
    bk = backend if backend is not None else get_backend()

    # Convert inputs to numpy for the calibration arithmetic.
    try:
        cal_np = bk.to_numpy(calibration_residuals).astype(np.float64)
    except Exception:
        cal_np = np.asarray(calibration_residuals, dtype=np.float64)

    try:
        pred_np = bk.to_numpy(test_predictions).astype(np.float64)
    except Exception:
        pred_np = np.asarray(test_predictions, dtype=np.float64)

    # Ensure calibration_residuals is 2-D: (N_cal, d).
    if cal_np.ndim == 1:
        cal_np = cal_np[:, np.newaxis]
    if cal_np.ndim != 2 or min(cal_np.shape) == 0:
        raise ValueError("calibration_residuals must have nonempty shape (N_cal, d)")
    if not np.isfinite(cal_np).all() or not np.isfinite(pred_np).all():
        raise ValueError("calibration residuals and test predictions must be finite")
    N_cal = cal_np.shape[0]

    # Compute nonconformity scores for each calibration sample.
    if nonconformity is not None:
        scores_list = []
        for i in range(N_cal):
            row = bk.from_numpy(cal_np[i])
            s = nonconformity(row)
            try:
                score_array = np.asarray(bk.to_numpy(s))
            except Exception:
                score_array = np.asarray(s)
            if score_array.size != 1:
                raise ValueError("nonconformity must return one scalar per residual")
            s_val = float(score_array.item())
            if s_val < float(np.abs(cal_np[i]).max()):
                raise ValueError("a custom score for this coordinate box must dominate the sup-norm")
            scores_list.append(s_val)
        scores_np = np.array(scores_list, dtype=np.float64)
    else:
        # Default: L∞ norm of the absolute residual.
        scores_np = np.abs(cal_np).max(axis=-1)  # (N_cal,)

    q_hat = conformal_quantile(scores_np, alpha=alpha)

    # Build the coverage band centered on test_predictions.
    low_np = pred_np - q_hat
    high_np = pred_np + q_hat

    return ConformalBand(
        low=bk.from_numpy(low_np),
        high=bk.from_numpy(high_np),
        alpha=float(alpha),
        n_calibration=int(N_cal),
    )
