"""5-level APNN training pipeline. SPEC_nlte_scaleup §M.4.

Extends the 3-level training pipeline (emulator/threelevel/training.py) to the
12-D input space:
    (log10(J_bar_l0), ..., log10(J_bar_l9), log10(T), log10(n_e)) -> (n_1..n_5)

Key differences from the 3-level path:
  - 12-D LHS instead of 4-D.
  - Default n_samples = 32,000 (SPEC §M.4; per-dim density 32000^(1/12) ≈ 2.4/dim).
  - J_bar bounds: flat common bounds [1e-5, 1e+1] for all 10 transitions (Q6
    resolution in SPEC §11 — escalation to per-line bounds only if rel_mae > 1.5%).
  - T range: [5000, 15000] K (per SPEC §M.4; slightly tighter than 3-level).
  - n_e range: [10^9, 10^14] cm^-3.
  - targets shape (N, 5) from rate_equation_solve_nlevel.
  - Training loss: MSE on the 5-component population vector (same as 3-level).
  - Acceptance gate: val_rel_mae < 1.5% (loosened from 3-level's 1%).

LHS sampling note (Q6 decision)
--------------------------------
J_bar_l ∈ [1e-5, 1e1] (log-space bounds [-5, +1]) for ALL 10 transitions.
This flat-common-bounds approach matches the 3-level precedent.  At 5-level the
10 Balmer transitions span Br-alpha (7.4e13 Hz) to Ly-delta (3.2e15 Hz), a 43×
frequency range; the flat LHS bounds accept some IR/UV imbalance as a known
limitation.  Per-line bounds are the R2 escalation path if rel_mae > 1.5%.

Determinism
-----------
All entry points seed torch, numpy, and Python random at top for reproducible
training (R-emulator-weights).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..apnn import APNN
from .._progress import progress_bar
from atmosphere.atomic_model import AtomicModel


# ---------------------------------------------------------------------------
# Flat common log-J bounds for all 10 Balmer transitions (Q6 resolution).
#
# Calibrated to bracket the J̄ regime produced by ali_loop_nlevel at the
# canonical M.5 operating points (T₀=8000 K, n_e=1e12, τ_max=1e4,
# ε ∈ {1e-2, 1e-4}):
#   Lyman lines     (k = 0..3):  log10(J̄) ∈ [-13.5, -9.9]
#   Balmer / Paschen / Brackett (k = 4..9):  log10(J̄) ∈ [-9.9, -6.8]
# The training range [-15, -5] covers all 10 lines with headroom on both
# ends.
#
# Historical note: the original M.4 value was (-5, +1), six orders of
# magnitude above where the cycle actually operates — every J̄ input at
# the test operating points fell outside the training distribution and the
# APNN extrapolated incoherently.  A first-pass fix used the 2-level /
# 3-level convention (-30, -10), but that drove the radiation field so
# weak that the rate-equation labels collapsed to LTE Boltzmann
# everywhere (level-1 = 1.000, levels 2-5 ~ 1e-16 to 1e-6), reducing the
# training data to a near-constant target distribution.  The (-15, -5)
# range is the regime where radiation actually matters AND covers the
# operating-point J̄ values exercised by the M.5 Born-remainder test.
# ---------------------------------------------------------------------------
_LOG_J_LO: float = -15.0   # log10(1e-15) W m^-2 Hz^-1 sr^-1
_LOG_J_HI: float =  -5.0   # log10(1e-5)


@dataclass
class TrainingDataMultilevel:
    """LHS-sampled training set for the 5-level APNN.

    Attributes
    ----------
    inputs : Tensor of shape (N, K+2), float32
        log10-scaled inputs: [log10(J_bar_l0), ..., log10(J_bar_l9),
                              log10(T), log10(n_e)].
        Float32 matches APNN internal compute.
    targets : Tensor of shape (N, L), float64
        Normalised level populations from rate_equation_solve_nlevel.
        Float64 for consistency with the atmosphere reference solver.
    n_lines : int
        K (number of transitions); read from inputs.shape[1] - 2.
    n_levels : int
        L (number of levels); read from targets.shape[1].
    """

    inputs: torch.Tensor    # (N, K+2), float32
    targets: torch.Tensor   # (N, L), float64

    @property
    def n_lines(self) -> int:
        return self.inputs.shape[1] - 2

    @property
    def n_levels(self) -> int:
        return self.targets.shape[1]


def _lhs_sample(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin hypercube sample on [0, 1]^n_dims.

    Each dimension has exactly one sample per stratum of width 1/n_samples.
    Row i is a single sample vector; columns are dimensions.

    Numpy-only (no scipy); mirrors emulator/threelevel/training.py::_lhs_sample.

    Parameters
    ----------
    n_samples : int
    n_dims : int
    rng : np.random.Generator

    Returns
    -------
    samples : ndarray of shape (n_samples, n_dims), in [0, 1].
    """
    samples = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples).astype(float)
        u = rng.uniform(size=n_samples)
        samples[:, d] = (perm + u) / n_samples
    return samples


def generate_training_data_multilevel(
    n_samples: int,
    *,
    atomic_model: AtomicModel,
    log_J_bounds: tuple[float, float] = (_LOG_J_LO, _LOG_J_HI),
    log_T_bounds: tuple[float, float] = (3.6990, 4.1761),   # log10(5000)..log10(15000) K
    log_ne_bounds: tuple[float, float] = (9.0, 14.0),
    seed: int = 0,
    species_fractions: dict | None = None,
) -> TrainingDataMultilevel:
    'Generate (K+2)-D LHS training data for the N-level APNN.\n\n    Samples (J_bar_l0..l_{K-1}, T, n_e) over physically plausible log-space\n    ranges using a Latin hypercube, then labels each sample via the reference\n    N-level rate-equation solver.  SPEC_nlte_scaleup §M.4, SPEC_phase_s §3.\n\n    All K J_bar lines use the same log10 bounds (flat common bounds, Q6\n    decision).  Lines are sampled independently so the LHS covers all\n    cross-frequency combinations.\n\n    Default bounds:\n      log10(J_bar_l) ∈ [-15, -5]:   J_bar ∈ [1e-15, 1e-5] W m^-2 Hz^-1 sr^-1\n      log10(T)       ∈ [3.699, 4.176]: T ∈ [5000, 15000] K\n      log10(n_e)     ∈ [9, 14]:       n_e ∈ [1e9, 1e14] cm^-3\n\n    Returns\n    -------\n    TrainingDataMultilevel\n        inputs  : (n_samples, K+2) float32 tensor, log10-scaled.\n        targets : (n_samples, L)   float64 tensor, populations from truth solver.'
    K = atomic_model.K
    L = atomic_model.n_levels
    n_dims = K + 2   # K lines + T + n_e

    # Seed everything for determinism.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # (K+2)-D Latin hypercube on [0,1], then rescale.
    rng = np.random.default_rng(seed)
    unit_cube = _lhs_sample(n_samples, n_dims=n_dims, rng=rng)  # (N, K+2)

    # Build bounds array: first K dims are J_bar lines, then T, then n_e.
    # log_J_bounds may be either a flat 2-tuple (applied to all K lines —
    # Phase M convention) or a per-line array of shape (K, 2) (Phase S
    # post-Q-S-3 escalation step (c): per-line Planck-anchored bounds
    # spanning the ~9 orders of magnitude of B_nu across UV He I lines
    # to IR Brackett / He 2.058 μm lines).
    bounds = np.empty((n_dims, 2), dtype=float)
    log_J_arr = np.asarray(log_J_bounds, dtype=float)
    if log_J_arr.ndim == 1 and log_J_arr.shape == (2,):
        for k in range(K):
            bounds[k, :] = log_J_arr
    elif log_J_arr.ndim == 2 and log_J_arr.shape == (K, 2):
        bounds[:K, :] = log_J_arr
    else:
        raise ValueError(
            f"log_J_bounds must have shape (2,) for flat bounds or "
            f"(K={K}, 2) for per-line bounds; got shape {log_J_arr.shape}."
        )
    bounds[K, :]     = log_T_bounds
    bounds[K + 1, :] = log_ne_bounds

    # Rescale: x_phys = lo + (hi - lo) * x_unit.
    log_inputs = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * unit_cube  # (N, K+2)

    # Convert to physical units.
    J_bar_vals_list = [
        torch.tensor(10.0 ** log_inputs[:, k], dtype=torch.float64)
        for k in range(K)
    ]
    T_vals   = torch.tensor(10.0 ** log_inputs[:, K],     dtype=torch.float64)
    n_e_vals = torch.tensor(10.0 ** log_inputs[:, K + 1], dtype=torch.float64)

    # J_bar_lines: shape (N, K), columns match atomic_model.transitions order.
    J_bar_lines = torch.stack(J_bar_vals_list, dim=1)   # (N, K), float64

    # Label via reference N-level SE solver.
    from atmosphere.multilevel.rate_equations import rate_equation_solve_nlevel

    with torch.no_grad():
        targets = rate_equation_solve_nlevel(
            J_bar_lines, T_vals, n_e_vals,
            atomic_model=atomic_model,
            species_fractions=species_fractions,
        )   # (N, L), float64

    # Store inputs as float32 (APNN internal precision).
    inputs_tensor = torch.tensor(log_inputs, dtype=torch.float32)

    return TrainingDataMultilevel(inputs=inputs_tensor, targets=targets)


def train_val_split_multilevel(
    data: TrainingDataMultilevel,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[TrainingDataMultilevel, TrainingDataMultilevel]:
    """Deterministic 80/20 split for multilevel training data.

    Identical logic to the 3-level train_val_split_3level but operates on
    TrainingDataMultilevel.

    Parameters
    ----------
    data : TrainingDataMultilevel
    val_fraction : float, default 0.2
    seed : int, default 0

    Returns
    -------
    (train_data, val_data) : (TrainingDataMultilevel, TrainingDataMultilevel)
    """
    n = data.inputs.shape[0]
    n_val = int(n * val_fraction)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    val_idx   = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    val_idx_t   = torch.from_numpy(val_idx).long()
    train_idx_t = torch.from_numpy(train_idx).long()

    train_data = TrainingDataMultilevel(
        inputs=data.inputs[train_idx_t],
        targets=data.targets[train_idx_t],
    )
    val_data = TrainingDataMultilevel(
        inputs=data.inputs[val_idx_t],
        targets=data.targets[val_idx_t],
    )
    return train_data, val_data


def train_multilevel(
    model: APNN,
    train_data: TrainingDataMultilevel,
    val_data: TrainingDataMultilevel,
    *,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    loss_type: str = "log_mse",
    log_eps: float = 1e-12,
    progress: bool = False,
) -> dict:
    """Train the 5-level APNN with AdamW.

    Parallel to emulator/threelevel/training.py::train_3level but for
    (N, K+2) inputs and (N, L) targets.  Seeds torch, numpy, and Python
    random at entry for deterministic training.

    Loss
    ----
    loss_type = "log_mse" (default for the 5-level case):
        MSE on log10(populations).  Necessary at 5-level because the
        population dynamic range spans ~10 orders of magnitude
        (ground-state level 1 ~ 1.0; upper-level n_5 ~ 1e-10 at the
        canonical operating points).  Raw-MSE gradients on the small
        components vanish and the optimizer stalls (the network drives
        all upper levels to ~0 instead of learning their J̄-response).
        log_mse equalises the per-level loss contribution.  Safe as
        default for the (-15, -5) log-J training range, where all
        target populations stay above 1e-10 (well above the log_eps
        floor).  Wider LHS bounds (e.g., (-30, -10), where some
        targets fall below 1e-12) can re-introduce the cold-corner
        degeneracy noted in the 3-level training docs.

    loss_type = "mse":
        MSE on the raw (n_1, ..., n_L) population outputs.  Equal absolute
        penalty per level — biased toward large-magnitude populations.
        Matches 2-/3-level convention but inadequate at 5-level for the
        dynamic-range reason above.  Available for diagnostic comparison
        and for narrower (e.g., (-10, -5)) ranges where the dynamic range
        of populations is compressed.

    Relative MAE definition (acceptance gate):
        rel_mae = mean_over_samples(max_over_levels(|Δn_l|))

    Parameters
    ----------
    model : APNN
        Model to train in-place (should have n_in == K+2, n_out == L).
    train_data : TrainingDataMultilevel
    val_data : TrainingDataMultilevel
    n_epochs : int, default 100
    lr : float, default 1e-3
    batch_size : int, default 256
    seed : int, default 0
    loss_type : str, default "mse"
    log_eps : float, default 1e-12
        Floor for log_mse (unused when loss_type="mse").
    progress : bool, default False
        When True, render a live tqdm epoch bar (loss + val_rel_mae in the
        postfix) instead of the every-10-epoch print.  Default False keeps
        the loop byte-identical to the un-instrumented path (silent test
        runs, unchanged stdout).

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'val_loss', 'val_rel_mae', 'val_log_rel_mae',
              'final_val_mae'.
        Each list has one entry per epoch.
    """
    if loss_type not in {"mse", "log_mse"}:
        raise ValueError(
            f"loss_type must be 'mse' or 'log_mse', got {loss_type!r}"
        )

    # Seed all RNGs at entry.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    L = train_data.n_levels

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss in float32 to match interior compute precision."""
        pred_f32   = pred.to(torch.float32)
        target_f32 = target.to(torch.float32)
        if loss_type == "mse":
            return torch.nn.functional.mse_loss(pred_f32, target_f32)
        log_pred   = torch.log10(torch.clamp(pred_f32,   min=log_eps))
        log_target = torch.log10(torch.clamp(target_f32, min=log_eps))
        return ((log_pred - log_target) ** 2).mean()

    n_train = train_data.inputs.shape[0]

    history: dict[str, Any] = {
        "train_loss":     [],
        "val_loss":       [],
        "val_rel_mae":    [],
        "val_log_rel_mae": [],
    }

    epoch_bar = progress_bar(
        range(n_epochs), desc="train (multilevel)", enable=progress, unit="ep"
    )
    for epoch in epoch_bar:
        model.train()

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx    = perm[start : start + batch_size]
            x_b    = train_data.inputs[idx]          # (B, K+2), float32
            y_b    = train_data.targets[idx]          # (B, L),   float64

            optimizer.zero_grad()
            pred   = model(x_b)                       # (B, L),   float64
            loss   = compute_loss(pred, y_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ------------------------------------------------------------------
        # Validation.
        # ------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_pred    = model(val_data.inputs)      # (N_val, L), float64
            val_targets = val_data.targets             # (N_val, L), float64

            val_loss    = compute_loss(val_pred, val_targets).item()

            # Linear relative MAE: mean per-sample max |Δn_l| over all levels.
            val_residuals = val_pred - val_targets    # (N_val, L)
            rel_mae = val_residuals.abs().max(dim=-1).values.mean().item()

            # Log-space MAE (diagnostic; reported regardless of loss_type).
            log_pred_d    = torch.log10(torch.clamp(val_pred,    min=log_eps))
            log_target_d  = torch.log10(torch.clamp(val_targets, min=log_eps))
            log_rel_mae   = (log_pred_d - log_target_d).abs().max(dim=-1).values.mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_rel_mae"].append(rel_mae)
        history["val_log_rel_mae"].append(log_rel_mae)

        if progress:
            epoch_bar.set_postfix(
                train_loss=f"{avg_train_loss:.3e}",
                val_rel_mae=f"{rel_mae:.3e}",
            )
        elif (epoch + 1) % 10 == 0:
            # Brief progress report every 10 epochs (visible in foreground run).
            print(
                f"  epoch {epoch + 1:3d}/{n_epochs}  "
                f"train_loss={avg_train_loss:.4e}  "
                f"val_rel_mae={rel_mae:.4e}"
            )

    # Final summary.
    history["final_val_mae"] = {
        "rel_mae":      history["val_rel_mae"][-1],
        "log_rel_mae":  history["val_log_rel_mae"][-1],
        "val_loss":     history["val_loss"][-1],
    }
    # Convenience: per-level MAE on last epoch.
    model.eval()
    with torch.no_grad():
        vp = model(val_data.inputs)
        vt = val_data.targets
        per_level_mae = (vp - vt).abs().mean(dim=0).tolist()
        history["final_val_mae"]["per_level_mae"] = per_level_mae

    return history
