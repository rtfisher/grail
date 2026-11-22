"""5-level APNN validation metrics. SPEC_nlte_scaleup §M.4.

Parallel to emulator/threelevel/validation.py but for (N, 5) population outputs.

Relative MAE definition (acceptance gate):
  rel_mae = mean_over_samples(max_over_levels(|Δn_l|))

The acceptance gate for the 5-level APNN is < 1.5% (looser than the 2-/3-level
1% gate because the 12-D LHS is sparser per dimension and the 5-level output
has more degrees of freedom).
"""

from __future__ import annotations

import torch

from ..apnn import APNN
from .training import TrainingDataMultilevel


def compute_residuals_multilevel(
    model: APNN,
    val_data: TrainingDataMultilevel,
) -> torch.Tensor:
    """Return δT residuals on the held-out multilevel validation set.

    Residuals are defined as:
        δT = emulator_prediction - ground_truth_label

    Parameters
    ----------
    model : APNN
        Trained model (should be in eval mode).
    val_data : TrainingDataMultilevel
        Held-out validation split.

    Returns
    -------
    residuals : Tensor of shape (N_val, L), float64
        emulator_prediction - ground_truth_label.
    """
    model.eval()
    with torch.no_grad():
        pred      = model(val_data.inputs)        # (N_val, L), float64
        residuals = pred - val_data.targets        # (N_val, L), float64
    return residuals


def validation_mae_multilevel(
    model: APNN,
    val_data: TrainingDataMultilevel,
) -> dict:
    """Compute validation MAE metrics for the M.4 acceptance gate.

    The M.4 acceptance criterion is: relative MAE < 1.5% (SPEC_nlte_scaleup §M.4).

    Relative MAE definition:
      rel_mae = mean_over_samples(max_over_levels(|Δn_l|))

    Parameters
    ----------
    model : APNN
        Trained model (should be in eval mode).
    val_data : TrainingDataMultilevel
        Held-out validation split.

    Returns
    -------
    metrics : dict with keys:
        'per_level_mae' : list[float] — mean absolute error per level.
        'rel_mae'       : float — mean(max_l |Δn_l|); this is the gated quantity.
        'max_abs_err'   : float — max absolute error across all samples and levels.
    """
    model.eval()
    with torch.no_grad():
        pred      = model(val_data.inputs)         # (N_val, L), float64
        targets   = val_data.targets               # (N_val, L), float64
        residuals = pred - targets                 # (N_val, L), float64

        per_level_mae = residuals.abs().mean(dim=0).tolist()

        # Relative MAE: mean per-sample max absolute error over all L levels.
        per_sample_max = residuals.abs().max(dim=-1).values   # (N_val,)
        rel_mae = per_sample_max.mean().item()

        max_abs_err = residuals.abs().max().item()

    return {
        "per_level_mae": per_level_mae,
        "rel_mae":       rel_mae,
        "max_abs_err":   max_abs_err,
    }
