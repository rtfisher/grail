"""Unit tests for the 5-level APNN architecture. SPEC_nlte_scaleup §M.4.

Test IDs:
  U-apnn5-architecture       : shape contract + softmax conservation.
  U-apnn5-training-determinism: identical seed -> identical loss curves.
  A-apnn5-validation-rel-mae : val_rel_mae < 1.5% after full training.
                               Marked @pytest.mark.slow — skipped in default
                               CI; included in full test run.
  U-apnn5-cycle-adapter      : make_cycle_adapter_5level shape/dtype contract.

No __init__.py (rootdir-relative discovery, --import-mode=importlib).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# U-apnn5-architecture
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_apnn5_architecture() -> None:
    """U-apnn5-architecture: APNN(n_in=12, n_out=5) shape contract and softmax.

    Verifies:
    1. Instantiation succeeds with hidden_dims=(256,)*4.
    2. Forward on batch (3, 12) gives output (3, 5).
    3. Each output row sums to 1.0 to float64 precision (softmax gate).
    4. All output values are non-negative (populations are probabilities).
    5. Output dtype is float64 (boundary cast convention).
    """
    from emulator.apnn import APNN

    model = APNN(n_in=12, n_out=5, hidden_dims=(256, 256, 256, 256))
    model.eval()

    # Count parameters.
    n_params = sum(p.numel() for p in model.parameters())
    # Expected: 12*256+256 + 3*(256*256+256) + 256*5+5 = 3328+65792*3+1285 = 202k
    # Accept any value in [180_000, 220_000].
    assert 180_000 <= n_params <= 220_000, (
        f"Parameter count {n_params} outside expected range [180k, 220k]; "
        "check hidden_dims=(256,)*4 architecture."
    )

    # Shape contract.
    torch.manual_seed(42)
    batch = torch.randn(3, 12, dtype=torch.float32)
    out = model(batch)

    assert out.shape == (3, 5), (
        f"Expected output shape (3, 5), got {out.shape}"
    )
    assert out.dtype == torch.float64, (
        f"Expected float64 output (boundary cast), got {out.dtype}"
    )

    # Softmax conservation: each row sums to 1.
    row_sums = out.sum(dim=-1)
    max_dev = (row_sums - 1.0).abs().max().item()
    assert max_dev < 1e-6, (
        f"Softmax sum != 1: max deviation = {max_dev:.2e}"
    )

    # Non-negativity.
    assert (out >= 0.0).all().item(), "APNN output contains negative values"

    # Test with a larger batch (N_tau = 200) — the working scale.
    big_batch = torch.randn(200, 12, dtype=torch.float32)
    big_out = model(big_batch)
    assert big_out.shape == (200, 5), (
        f"Expected (200, 5), got {big_out.shape}"
    )
    big_sums = big_out.sum(dim=-1)
    assert (big_sums - 1.0).abs().max().item() < 1e-6, (
        "Softmax conservation fails at batch size 200"
    )


# ---------------------------------------------------------------------------
# U-apnn5-training-determinism
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_apnn5_training_determinism() -> None:
    """U-apnn5-training-determinism: same seed -> identical loss curves.

    Runs train_multilevel with a tiny dataset (200 samples, 2 epochs, batch 64)
    twice with seed=0, compares loss histories to 1e-10 tolerance.
    """
    from emulator.multilevel.apnn import make_apnn_5level
    from emulator.multilevel.training import (
        generate_training_data_multilevel,
        train_multilevel,
        train_val_split_multilevel,
    )
    from atmosphere.atomic_model import AtomicModel

    atomic_model = AtomicModel.balmer_5level(epsilon=1e-2)

    # Generate a small dataset for speed.
    data = generate_training_data_multilevel(
        n_samples=200,
        atomic_model=atomic_model,
        seed=7,
    )
    train_data, val_data = train_val_split_multilevel(data, val_fraction=0.2, seed=7)

    def _run(seed_val: int) -> list[float]:
        # Seed torch before model construction so initial weights are identical.
        torch.manual_seed(seed_val)
        model = make_apnn_5level()
        history = train_multilevel(
            model, train_data, val_data,
            n_epochs=2, lr=1e-3, batch_size=64, seed=seed_val,
        )
        return history["train_loss"]

    losses_a = _run(0)
    losses_b = _run(0)

    assert len(losses_a) == 2 and len(losses_b) == 2

    for e, (a, b) in enumerate(zip(losses_a, losses_b)):
        assert abs(a - b) < 1e-10, (
            f"Epoch {e}: train_loss differs between seeds: {a} vs {b} "
            f"(diff={abs(a-b):.2e})"
        )


# ---------------------------------------------------------------------------
# A-apnn5-validation-rel-mae  (slow: full training)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.analytic
def test_A_apnn5_validation_rel_mae() -> None:
    """A-apnn5-validation-rel-mae: val_rel_mae < 1.5% after training.

    Trains the 5-level APNN on 32,000 LHS samples (80/20 split), 100 epochs,
    batch 256, seed 0.  Asserts that the final validation rel_mae is below
    the 1.5% acceptance gate (SPEC_nlte_scaleup §M.4).

    This test is slow (~2–5 minutes on CPU) and is skipped in default CI.
    Run explicitly: pytest -m slow tests/emulator/multilevel/test_apnn.py
    """
    import os
    from emulator.multilevel.apnn import make_apnn_5level
    from emulator.multilevel.training import (
        generate_training_data_multilevel,
        train_multilevel,
        train_val_split_multilevel,
    )
    from emulator.multilevel.validation import validation_mae_multilevel
    from atmosphere.atomic_model import AtomicModel

    _N_SAMPLES   = 32_000
    _N_EPOCHS    = 100
    _BATCH_SIZE  = 256
    _LR          = 1e-3
    _SEED        = 0
    _REL_MAE_GATE = 0.015   # 1.5%

    atomic_model = AtomicModel.balmer_5level(epsilon=1e-4)

    print(f"\nGenerating {_N_SAMPLES} LHS samples ...")
    data = generate_training_data_multilevel(
        n_samples=_N_SAMPLES,
        atomic_model=atomic_model,
        seed=_SEED,
    )
    train_data, val_data = train_val_split_multilevel(data, val_fraction=0.2, seed=_SEED)
    print(f"  train={train_data.inputs.shape[0]}, val={val_data.inputs.shape[0]}")

    model = make_apnn_5level()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model n_params = {n_params:,}")

    print(f"Training {_N_EPOCHS} epochs ...")
    import time
    t0 = time.time()
    history = train_multilevel(
        model, train_data, val_data,
        n_epochs=_N_EPOCHS, lr=_LR, batch_size=_BATCH_SIZE, seed=_SEED,
    )
    elapsed = time.time() - t0
    print(f"  training wall-clock: {elapsed:.1f} s ({elapsed/60:.1f} min)")

    metrics = validation_mae_multilevel(model, val_data)
    rel_mae = metrics["rel_mae"]
    print(f"  val_rel_mae = {rel_mae:.4f}  (gate < {_REL_MAE_GATE:.3f})")
    for i, m in enumerate(metrics["per_level_mae"]):
        print(f"    level {i+1}: mae = {m:.4e}")

    # Save trained model for reuse by the integration test (M.5).
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "examples", "apnn_5level.pt"
    )
    checkpoint_path = os.path.normpath(checkpoint_path)
    torch.save({"model_state_dict": model.state_dict(), "history": history}, checkpoint_path)
    print(f"  checkpoint saved: {checkpoint_path}")

    assert rel_mae < _REL_MAE_GATE, (
        f"val_rel_mae = {rel_mae:.4f} exceeds 1.5% gate "
        f"(SPEC_nlte_scaleup §M.4 acceptance criterion)"
    )
