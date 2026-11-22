"""Train the STO-3G UHF XC-net artifact for the SCF testbed.

SPEC_scf §6 P3, §8.10 R3.  Produces
``examples/scf_xc_emulator_uhf_sto3g.h5`` — the trained UHF XCNet artifact
that P4 (GRAIL integration) loads to evaluate ``V_xc^NN`` inside the
broken-symmetry UHF Fock build.

Usage::

    PYTHONPATH=/path/to/grail KMP_DUPLICATE_LIB_OK=TRUE \\
      python \\
      examples/train_xc_emulator_uhf_sto3g.py

Prints a one-line summary every ~50 epochs and a final wall-clock /
rel_mae line.  The script aborts non-zero if rel_mae >= 0.03 (the
SPEC_scf §8.10 R3 gate) so a regression in training data or
hyperparameters is loud at the CI / manual-run boundary.

Configuration (SPEC_scf §8.10 R3):
  Basis            : STO-3G  →  M = 2, doubled seam length = 6
  Method           : UHF + broken-symmetry init guess (1.5×/0.5×)
  Truth XC         : B3LYP
  Hidden × depth    : 64 × 2  (~5 k parameters; SPEC §7 default)
  n_samples (LHS)  : 8000
  Epochs           : 300
  Batch size        : 64
  Optimizer         : Adam, lr=1e-3
  R range           : [1.10, 1.25] Å (R_CF ≈ 1.16 Å)
  Perturb max       : 0.03 per packed-flat entry (R3 default; P3 empirical sweet spot)
  R anchors         : 8
  Seed              : 42  (data generation + training)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# OpenMP triple-init guard (PyTorch + h5py + PySCF).  Must be set BEFORE
# any torch / h5py / PySCF import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scf._make_state import make_h2_state            # noqa: E402
from scf.emulator import (                            # noqa: E402
    XCNet,
    generate_training_data_uhf,
    save_xc_emulator,
    train_xc_emulator_uhf,
    validation_rel_mae_uhf,
)

_OUT_PATH = _REPO_ROOT / "examples" / "scf_xc_emulator_uhf_sto3g.h5"


def main() -> int:
    print("=" * 70)
    print("Train STO-3G UHF XC-net  (SPEC_scf §8.10 R3)")
    print("=" * 70)

    template = make_h2_state(R=1.18, basis="sto-3g", method="uhf", xc="B3LYP")
    # doubled-seam length for STO-3G UHF: 2 * 2*(2+1)/2 = 6.
    n_unique_total = 6

    # ----- Data generation ----------------------------------------------------
    print("\n[1/3] Generating UHF training data (8000 LHS × 8 R anchors)…")
    t0 = time.time()
    data = generate_training_data_uhf(
        template,
        n_samples=8000,
        seed=42,
        R_lo=1.10,
        R_hi=1.25,
        perturb_max=0.03,
        n_R_anchors=8,
    )
    t_data = time.time() - t0
    print(
        f"        done in {t_data:.1f}s.  P_flat={data['P_flat'].shape}, "
        f"Vxc_flat={data['Vxc_flat'].shape}, "
        f"R ∈ [{data['R'].min():.3f}, {data['R'].max():.3f}] Å."
    )
    print(
        f"        |Vxc| range: [{abs(data['Vxc_flat']).min():.3e}, "
        f"{abs(data['Vxc_flat']).max():.3e}] Ha."
    )

    # ----- Training -----------------------------------------------------------
    print("\n[2/3] Training XCNet (hidden=64, depth=2, ~5 k params)…")
    net = XCNet(n_unique=n_unique_total, hidden=64, depth=2)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"        parameter count: {n_params}.")

    t0 = time.time()
    history = train_xc_emulator_uhf(
        net,
        data,
        epochs=300,
        batch_size=64,
        lr=1e-3,
        val_fraction=0.2,
        seed=42,
    )
    t_train = time.time() - t0
    print(
        f"        done in {t_train:.1f}s.  "
        f"final train_loss={history['train_loss'][-1]:.3e}, "
        f"val_loss={history['val_loss'][-1]:.3e}, "
        f"val_rel_mae={history['final_val_rel_mae']:.4f}."
    )
    for milestone in (0, 49, 99, 149, 199, 249, 299):
        if milestone < len(history["val_rel_mae"]):
            print(
                f"          epoch {milestone+1:>3d}: "
                f"train={history['train_loss'][milestone]:.3e}  "
                f"val={history['val_loss'][milestone]:.3e}  "
                f"rel_mae={history['val_rel_mae'][milestone]:.4f}"
            )

    # ----- Save + holdout gate ------------------------------------------------
    print("\n[3/3] Saving artifact and verifying holdout gate…")
    save_xc_emulator(net, _OUT_PATH)
    file_kb = _OUT_PATH.stat().st_size / 1024.0
    print(f"        wrote {_OUT_PATH} ({file_kb:.1f} kB).")

    # Independent-seed holdout (seed=43 — same as the Tier A test, but a
    # larger N so the gate is more meaningful at the publishing boundary).
    holdout = generate_training_data_uhf(
        template,
        n_samples=512,
        seed=43,
        R_lo=1.10,
        R_hi=1.25,
        perturb_max=0.03,
        n_R_anchors=8,
    )
    holdout_rel_mae = validation_rel_mae_uhf(net, holdout)
    print(f"        holdout rel_mae (seed=43, N=512): {holdout_rel_mae:.4f}.")

    gate = 0.03
    if holdout_rel_mae >= gate:
        print(
            f"\nFAIL: holdout rel_mae={holdout_rel_mae:.4f} >= gate={gate:.4f}. "
            f"SPEC_scf §8.10 R3 publication gate violated."
        )
        return 1
    print(f"\nPASS: holdout rel_mae < {gate:.4f}.")
    print(f"Total wall-clock: {t_data + t_train:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
