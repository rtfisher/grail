"""Training data generation, training loop, validation split. the technical reference §4.3.

Training-data bounds (physically motivated):
  log₁₀(J_bar) : [-30, -10]
    The Planck function B_ν at line center (ν₀ = 2.47e15 Hz, Lyman-α) spans
    roughly 1e-30 to 1e-10 W/m²/Hz/sr over T ∈ [3000, 12000] K.  J_bar
    lives between 0 (empty space) and the Planck function at the hottest
    depth.  Sampling over 20 decades covers the sub-thermalization (J_bar ≪ B)
    and thermalized (J_bar ~ B) regimes.
  log₁₀(T)     : [log₁₀(3000), log₁₀(12000)]   ≈ [3.477, 4.079]
    Covers the isothermal and power-law canned atmospheres (T0 = 6000 K typical).
  log₁₀(n_e)   : [8, 12]
    Typical stellar-atmosphere electron densities in cm⁻³.

APNN inputs are log₁₀-scaled so that the network sees approximately uniform
dynamic range across all three inputs.  The rate-equation solver is called in
SI units for J_bar and Kelvin for T (as required by atmosphere/rate_equations.py).

LHS implementation:
  scipy is not available in this environment; we implement LHS directly using
  numpy.  The key property of a Latin hypercube is that each dimension has
  exactly one sample per stratum of width 1/n_samples.  We achieve this by:
    1. Permuting {0, ..., n-1} independently per dimension.
    2. Adding a uniform draw within each stratum: x_ij = (perm_ij + U_ij) / n.
  This is the standard stratified LHS construction and produces the same
  uniformity guarantee as scipy.stats.qmc.LatinHypercube.

Determinism (R-emulator-weights):
  All training entry points seed torch, numpy, and Python random at the top.
  CPU-only mode is recommended for the regression test to avoid CUDA
  non-determinism (see apnn.py module docstring for details).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..apnn import APNN


@dataclass
class TrainingData:
    """LHS-sampled training set for the APNN.  the technical reference §4.3.

    Attributes
    ----------
    inputs : Tensor of shape (N, 3), float32
        log₁₀-scaled inputs: [log₁₀(J_bar), log₁₀(T), log₁₀(n_e)].
        Float32 matches APNN internal compute (cast to float64 only at the
        forward-pass output boundary).
    targets : Tensor of shape (N, 2), float64
        Normalised level populations (n_1, n_2) from rate_equation_solve.
        Float64 for consistency with the atmosphere reference solver.
    """

    inputs: torch.Tensor    # (N, 3), float32 — log10-scaled
    targets: torch.Tensor   # (N, 2), float64 — populations from rate_equation_solve


def _lhs_sample(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin hypercube sample on [0, 1]^n_dims.

    Each dimension has exactly one sample per stratum of width 1/n_samples.
    Row i is a single sample vector; columns are dimensions.

    Replacement for scipy.stats.qmc.LatinHypercube that uses only numpy.

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    n_dims : int
        Number of dimensions.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    samples : ndarray of shape (n_samples, n_dims), in [0, 1].
    """
    # Build stratified samples: stratum k covers [k/n, (k+1)/n).
    strata = np.arange(n_samples, dtype=float)  # (n_samples,)
    samples = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        # Permute stratum indices independently per dimension.
        perm = rng.permutation(n_samples).astype(float)
        # Add a uniform draw within each stratum.
        u = rng.uniform(size=n_samples)
        samples[:, d] = (perm + u) / n_samples
    return samples


def generate_training_data(
    n_samples: int,
    *,
    log_J_bounds: tuple[float, float] = (-30.0, -10.0),
    log_T_bounds: tuple[float, float] = (3.477, 4.079),
    log_ne_bounds: tuple[float, float] = (8.0, 12.0),
    epsilon: float,
    nu_0: float,
    seed: int = 0,
) -> "TrainingData":
    """Generate LHS training data for the APNN.

    Samples (J_bar, T, n_e) over physically plausible log-space ranges using
    a Latin hypercube, then labels each sample via the reference rate-equation
    solver.  the technical reference §4.3.

    Default bounds rationale (see module docstring for full derivation):
      log₁₀(J_bar) ∈ [-30, -10]: spans sub-thermalization to thermalized regime
      log₁₀(T) ∈ [3.477, 4.079]: T ∈ [3000, 12000] K (canned atmosphere range)
      log₁₀(n_e) ∈ [8, 12]:      typical stellar n_e in cm⁻³

    Parameters
    ----------
    n_samples : int
        Number of LHS samples.
    log_J_bounds : (float, float)
        (min, max) for log₁₀(J_bar) in W/m²/Hz/sr.
    log_T_bounds : (float, float)
        (min, max) for log₁₀(T) in Kelvin.
    log_ne_bounds : (float, float)
        (min, max) for log₁₀(n_e) in cm⁻³.
    epsilon : float
        Photon destruction probability; forwarded to rate_equation_solve.
    nu_0 : float
        Line-center frequency in Hz; forwarded to rate_equation_solve.
    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    TrainingData
        inputs  : (n_samples, 3) float32 tensor, log₁₀-scaled.
        targets : (n_samples, 2) float64 tensor, populations from truth solver.
    """
    # -----------------------------------------------------------------------
    # Seed everything for determinism (R-emulator-weights requirement).
    # -----------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------------------------------------------------------
    # Latin hypercube on [0, 1]^3, then rescale to physical log-space bounds.
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    unit_cube = _lhs_sample(n_samples, n_dims=3, rng=rng)  # (N, 3) in [0,1]

    bounds = np.array([
        log_J_bounds,
        log_T_bounds,
        log_ne_bounds,
    ], dtype=float)  # (3, 2)

    # Rescale: x_phys = lo + (hi - lo) * x_unit.
    log_inputs = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * unit_cube  # (N, 3)

    # -----------------------------------------------------------------------
    # Convert to physical units and label via reference solver.
    # -----------------------------------------------------------------------
    J_bar_vals = torch.tensor(10.0 ** log_inputs[:, 0], dtype=torch.float64)
    T_vals     = torch.tensor(10.0 ** log_inputs[:, 1], dtype=torch.float64)
    n_e_vals   = torch.tensor(10.0 ** log_inputs[:, 2], dtype=torch.float64)

    # Import here to avoid circular imports (training.py is in emulator/).
    from atmosphere.rate_equations import rate_equation_solve

    with torch.no_grad():
        targets = rate_equation_solve(
            J_bar_vals, T_vals, n_e_vals,
            epsilon=epsilon, nu_0=nu_0,
        )  # (N, 2), float64

    # Store inputs as float32 (APNN internal precision).
    inputs_tensor = torch.tensor(log_inputs, dtype=torch.float32)

    return TrainingData(inputs=inputs_tensor, targets=targets)


def train_val_split(
    data: "TrainingData",
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple["TrainingData", "TrainingData"]:
    """Deterministic 80/20 split.  the technical reference §4.3.

    The split is reproducible given seed.  Index sets are disjoint and cover
    all n_samples samples (gated by U-validation-01).

    Parameters
    ----------
    data : TrainingData
        Full dataset (inputs + targets).
    val_fraction : float, default 0.2
        Fraction to assign to the validation set.
    seed : int, default 0
        Seed for the random permutation used to select validation indices.

    Returns
    -------
    (train_data, val_data) : (TrainingData, TrainingData)
    """
    n = data.inputs.shape[0]
    n_val = int(n * val_fraction)

    # Deterministic permutation using numpy (seeded).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    # Sort indices to preserve relative ordering within each split (makes
    # debugging easier; the shuffle already provides randomness).
    val_idx   = np.sort(val_idx)
    train_idx = np.sort(train_idx)

    val_idx_t   = torch.from_numpy(val_idx).long()
    train_idx_t = torch.from_numpy(train_idx).long()

    train_data = TrainingData(
        inputs=data.inputs[train_idx_t],
        targets=data.targets[train_idx_t],
    )
    val_data = TrainingData(
        inputs=data.inputs[val_idx_t],
        targets=data.targets[val_idx_t],
    )
    return train_data, val_data


def train(
    model: APNN,
    train_data: "TrainingData",
    val_data: "TrainingData",
    *,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    """Train the APNN with AdamW and MSE loss on populations.

    Seeds torch, numpy, and Python random at the top for deterministic
    training (required by R-emulator-weights).

    Loss:
      MSE on the raw (n_1, n_2) population outputs.  Since the APNN output
      is a softmax, populations are already in [0, 1] and sum to 1.  MSE on
      fractional populations is equivalent to minimising the mean squared
      error on the Boltzmann ratio.

    Validation MAE is computed each epoch to track convergence toward the
    M1b acceptance criterion (relative MAE < 1% on populations).

    CUDA non-determinism note:
      This function sets CPU seeds.  For GPU training, the caller must also
      set the device seed and optionally enable deterministic algorithms.
      The R-emulator-weights regression test should be run on CPU.

    Parameters
    ----------
    model : APNN
        Model to train in-place.
    train_data : TrainingData
        Training split.
    val_data : TrainingData
        Validation split.
    n_epochs : int, default 100
        Number of full passes over the training set.
    lr : float, default 1e-3
        AdamW learning rate.
    batch_size : int, default 256
        Mini-batch size.
    seed : int, default 0
        Seed for all RNGs.  Controls mini-batch shuffle order.

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'val_loss', 'val_mae_n1', 'val_mae_n2',
              'val_rel_mae', 'final_val_mae'.
        Each loss/mae list has one entry per epoch.
        'final_val_mae' is a dict summarising the last epoch's validation
        metrics (same keys as validation_mae()).
    """
    # -----------------------------------------------------------------------
    # Seed all RNGs at entry for determinism.
    # -----------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------------------------------------------------------
    # Training setup.
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    n_train = train_data.inputs.shape[0]

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss":   [],
        "val_mae_n1": [],
        "val_mae_n2": [],
        "val_rel_mae": [],
    }

    for epoch in range(n_epochs):
        model.train()

        # Shuffle training indices for this epoch.
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            x_batch = train_data.inputs[idx]          # (B, 3), float32
            y_batch = train_data.targets[idx]         # (B, 2), float64

            optimizer.zero_grad()
            pred = model(x_batch)                     # (B, 2), float64
            # Cast targets to float32 for loss (network operates in f32 space;
            # final f64 cast is at the boundary — see apnn.py).  Using f32
            # loss avoids mixed-precision issues.
            loss = loss_fn(
                pred.to(torch.float32),
                y_batch.to(torch.float32),
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # -----------------------------------------------------------------------
        # Validation.
        # -----------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_pred = model(val_data.inputs)          # (N_val, 2), float64
            val_targets = val_data.targets             # (N_val, 2), float64

            val_loss = loss_fn(
                val_pred.to(torch.float32),
                val_targets.to(torch.float32),
            ).item()

            mae_n1 = (val_pred[:, 0] - val_targets[:, 0]).abs().mean().item()
            mae_n2 = (val_pred[:, 1] - val_targets[:, 1]).abs().mean().item()
            # Relative MAE: mean per-sample max absolute error, normalised by
            # n_tot = 1.  This is robust for near-zero populations (see
            # validation.py docstring for full rationale).
            val_residuals = val_pred - val_targets  # (N_val, 2)
            rel_mae = val_residuals.abs().max(dim=-1).values.mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_n1"].append(mae_n1)
        history["val_mae_n2"].append(mae_n2)
        history["val_rel_mae"].append(rel_mae)

    # Final summary.
    history["final_val_mae"] = {
        "mae_n1":      history["val_mae_n1"][-1],
        "mae_n2":      history["val_mae_n2"][-1],
        "rel_mae":     history["val_rel_mae"][-1],
        "max_abs_err": float("nan"),  # computed by validation_mae() on demand
    }
    return history
