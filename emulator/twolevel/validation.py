"""Held-out residuals against the reference rate-equation solver.  the technical reference §4.3, §5.4.

The residuals δT = T_SE^APNN - T_SE^ref on the validation set are GRAIL's
input to PCA basis construction and to the conformal calibrator.

All computations in float64 for consistency with the atmosphere stack.
"""

from __future__ import annotations

import torch

from ..apnn import APNN
from .training import TrainingData


def compute_residuals(model: APNN, val_data: TrainingData) -> torch.Tensor:
    """Return δT residuals on the held-out validation set.

    Residuals are defined as:
        δT = emulator_prediction - ground_truth_label

    These are the raw population residuals (Δn_1, Δn_2) per validation sample.

    Parameters
    ----------
    model : APNN
        Trained model (should be in eval mode).
    val_data : TrainingData
        Held-out validation split containing inputs and ground-truth targets.

    Returns
    -------
    residuals : Tensor of shape (N_val, 2), float64
        emulator_prediction - ground_truth_label, per validation sample.
        Column 0: Δn_1;  Column 1: Δn_2.
    """
    model.eval()
    with torch.no_grad():
        pred = model(val_data.inputs)        # (N_val, 2), float64
        residuals = pred - val_data.targets  # (N_val, 2), float64
    return residuals


def validation_mae(model: APNN, val_data: TrainingData) -> dict:
    """Compute validation MAE metrics for the M1b gate.

    The M1b acceptance criterion is: relative MAE < 1% on populations
    (docs/testing.md — Tier M1b validation MAE gate).

    Relative MAE definition:
      Since populations are normalised (n_1 + n_2 = n_tot = 1.0), the natural
      "relative" scale is n_tot = 1.  We therefore define:

          rel_mae = mean_over_samples(max(|Δn_1|, |Δn_2|))

      equivalently, the per-sample maximum absolute error divided by n_tot = 1.
      This is equivalent to the absolute MAE on the worst-case level per sample.

      This definition is robust when one level has near-zero population
      (e.g., n_2 ≈ 0 for high-excitation lines at low T), where the
      per-level relative error |Δn_i| / n_i would be pathologically large.

      The < 1% criterion means errors are small relative to the [0, 1]
      population range — a natural and physically meaningful scale.

    Parameters
    ----------
    model : APNN
        Trained model (should be in eval mode).
    val_data : TrainingData
        Held-out validation split.

    Returns
    -------
    metrics : dict with keys:
        'mae_n1'      : float — mean absolute error on n_1.
        'mae_n2'      : float — mean absolute error on n_2.
        'rel_mae'     : float — relative MAE = mean(max(|Δn_1|, |Δn_2|))
                        normalised by n_tot = 1.  This is the quantity gated
                        by the M1b < 1% threshold.
        'max_abs_err' : float — max absolute error across all samples and levels.
    """
    model.eval()
    with torch.no_grad():
        pred = model(val_data.inputs)      # (N_val, 2), float64
        targets = val_data.targets         # (N_val, 2), float64
        residuals = pred - targets         # (N_val, 2), float64

        mae_n1 = residuals[:, 0].abs().mean().item()
        mae_n2 = residuals[:, 1].abs().mean().item()

        # Relative MAE: mean per-sample max absolute error, normalised by n_tot=1.
        # This is the correct definition for normalised populations (see docstring).
        per_sample_max = residuals.abs().max(dim=-1).values  # (N_val,)
        rel_mae = per_sample_max.mean().item()

        max_abs_err = residuals.abs().max().item()

    return {
        "mae_n1":      mae_n1,
        "mae_n2":      mae_n2,
        "rel_mae":     rel_mae,
        "max_abs_err": max_abs_err,
    }
