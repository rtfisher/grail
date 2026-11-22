"""Tier U tests for ``scf/emulator/`` (UHF path).  SPEC_scf §6 P3, §8.10 R3.

Covers the parts of the R3 UHF XC-surrogate package that do NOT require a
trained model:

- :class:`XCNet` forward-pass shape on a doubled-seam packed-flat density
  (length 6 for STO-3G H₂, length 20 for 6-31G H₂).
- :class:`XCNet` is autograd-differentiable through its forward pass.
- ``save_xc_emulator → load_xc_emulator`` is byte-identical at the
  output level.
- :func:`generate_training_data_uhf` returns the expected number of
  samples with valid shapes.

These tests exercise the network on randomly initialized weights — they
do NOT load the trained artifact (that's the Tier A path in
``tests/scf/analytic/test_uhf_xc_emulator_accuracy.py``).

PySCF is only required by :func:`generate_training_data_uhf`.  Tests that
exercise the bare network skip PySCF; the data-generation test imports
PySCF via ``pytest.importorskip``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scf._make_state import make_h2_state
from scf.emulator import XCNet, load_xc_emulator, save_xc_emulator


# ---------------------------------------------------------------------------
# XCNet forward-pass shape (UHF: doubled seam)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_xcnet_forward_shape_uhf_sto3g() -> None:
    """A single STO-3G H₂ UHF density (length 6) maps to a 6-vector output."""
    net = XCNet(n_unique=6, hidden=64, depth=2)
    # 6-vector: [P_α[0,0], P_α[0,1], P_α[1,1], P_β[0,0], P_β[0,1], P_β[1,1]].
    P_flat = torch.tensor([0.6, 0.55, 0.6, 0.6, 0.55, 0.6], dtype=torch.float64)
    V_flat = net(P_flat.unsqueeze(0))     # (1, 6)
    assert V_flat.shape == (1, 6)
    assert V_flat.dtype == torch.float64


@pytest.mark.unit
def test_xcnet_forward_batch_shape_uhf() -> None:
    """A (B=5, n_unique=6) UHF batch maps to (5, 6) output."""
    net = XCNet(n_unique=6, hidden=64, depth=2)
    P_batch = torch.randn(5, 6, dtype=torch.float64)
    V_batch = net(P_batch)
    assert V_batch.shape == (5, 6)
    assert V_batch.dtype == torch.float64


@pytest.mark.unit
def test_xcnet_forward_shape_uhf_631g() -> None:
    """A 6-31G H₂ UHF density (length 20) maps to (B, 20) output."""
    net = XCNet(n_unique=20, hidden=64, depth=2)
    P_batch = torch.randn(4, 20, dtype=torch.float64)
    V_batch = net(P_batch)
    assert V_batch.shape == (4, 20)
    assert V_batch.dtype == torch.float64


@pytest.mark.unit
def test_xcnet_rejects_bad_config() -> None:
    """Constructor rejects non-positive n_unique / hidden / depth."""
    with pytest.raises(ValueError):
        XCNet(n_unique=0)
    with pytest.raises(ValueError):
        XCNet(n_unique=6, hidden=0)
    with pytest.raises(ValueError):
        XCNet(n_unique=6, depth=0)


@pytest.mark.unit
def test_xcnet_small_initial_output_uhf() -> None:
    """Untrained XCNet output magnitude is ≤ 1e-3 (small-init contract).

    Per SPEC_scf §8.10 R3 the final layer is rescaled by 1e-3 at construction
    time so the initial network output is a tiny perturbation around zero.
    Gives the supervised loss room to learn the V_xc landscape without
    being dominated by random initial output magnitude.
    """
    torch.manual_seed(0)
    net = XCNet(n_unique=6, hidden=64, depth=2)
    net.eval()
    # Sample a broad range of inputs; check that the network output remains
    # uniformly small.
    P_flat = torch.randn(64, 6, dtype=torch.float64)
    with torch.no_grad():
        V_flat = net(P_flat)
    # ≤ 1e-3 in V_xc units (Hartree).  We allow a modest factor of safety
    # because random-init weights times a bounded input distribution can
    # marginally exceed the nominal cap; the bound is per the R3 spec
    # ("≤ 1e-3 in V_xc units").
    max_abs = float(V_flat.abs().max().item())
    assert max_abs <= 1e-3, (
        f"Untrained XCNet output magnitude {max_abs:.4e} exceeds 1e-3 cap; "
        "the small-init rescaling has regressed."
    )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_xcnet_autograd_through_forward_uhf() -> None:
    """Autograd flows through the forward pass on a length-6 input.

    Required so the FD-JVP step-size sanity check (P4 / P5 diagnostic) can
    cross-validate against a coarse autograd JVP if needed.  The
    load-bearing FD-JVP path itself does NOT use autograd (TASK-035 /
    SPEC_scf §5), but the network must remain differentiable.
    """
    net = XCNet(n_unique=6, hidden=64, depth=2)
    P_flat = torch.tensor(
        [[0.6, 0.55, 0.6, 0.6, 0.55, 0.6]],
        dtype=torch.float32, requires_grad=True,
    )
    V_flat = net(P_flat)
    loss = V_flat.sum()
    loss.backward()
    assert P_flat.grad is not None
    assert P_flat.grad.shape == P_flat.shape
    # At least one input gradient is non-zero (the network is dense, but
    # small-init shrinks the magnitude — still must be non-zero).
    assert torch.any(P_flat.grad != 0.0)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_xcnet_save_load_roundtrip_uhf(tmp_path) -> None:
    """save_xc_emulator → load_xc_emulator preserves UHF-config outputs.

    Float32 internal storage is preserved bitwise across the HDF5
    round-trip; the forward pass output is therefore identical to within
    float32 reproducibility.
    """
    torch.manual_seed(7)
    net = XCNet(n_unique=6, hidden=64, depth=2)
    net.eval()

    P_flat = torch.tensor(
        [[0.5, 0.25, 0.6, 0.6, 0.25, 0.5], [0.6, 0.55, 0.6, 0.55, 0.5, 0.6]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        V_before = net(P_flat).numpy()

    out_path = tmp_path / "xcnet_uhf_roundtrip.h5"
    save_xc_emulator(net, out_path)

    net_loaded = load_xc_emulator(out_path)
    # Config matches.
    assert net_loaded.n_unique == 6
    assert net_loaded.hidden == 64
    assert net_loaded.depth == 2

    with torch.no_grad():
        V_after = net_loaded(P_flat).numpy()

    assert np.array_equal(V_before, V_after), (
        f"save/load round-trip changed output: max|Δ| = "
        f"{np.max(np.abs(V_before - V_after)):.3e}"
    )


@pytest.mark.unit
def test_xcnet_load_missing_file_raises(tmp_path) -> None:
    """load_xc_emulator raises FileNotFoundError on a missing path."""
    with pytest.raises(FileNotFoundError):
        load_xc_emulator(tmp_path / "does_not_exist.h5")


# ---------------------------------------------------------------------------
# generate_training_data_uhf (requires PySCF)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_generate_training_data_uhf_shape_sto3g() -> None:
    """generate_training_data_uhf returns the expected shapes and finite values.

    Uses a small n_samples + n_R_anchors so the test runs in a few
    seconds.  The dominant cost is the truth-UHF convergence per anchor
    plus the ~10–15 ms-per-sample spin-resolved V_xc evaluation.
    """
    pytest.importorskip("pyscf")
    from scf.emulator.training import generate_training_data_uhf

    template = make_h2_state(R=1.18, basis="sto-3g", method="uhf", xc="B3LYP")
    n_samples = 16
    data = generate_training_data_uhf(
        template,
        n_samples=n_samples,
        seed=0,
        R_lo=1.10,
        R_hi=1.25,
        perturb_max=0.03,
        n_R_anchors=4,
    )
    # Keys present.
    assert set(data.keys()) >= {
        "P_flat", "Vxc_flat", "R", "P_star_anchors", "R_anchors",
    }
    # Shapes — doubled seam length 6 for STO-3G H₂.
    assert data["P_flat"].shape == (n_samples, 6)
    assert data["Vxc_flat"].shape == (n_samples, 6)
    assert data["R"].shape == (n_samples,)
    # Anchor cache: (n_R_anchors, 2 spins, M, M) for M=2.
    assert data["P_star_anchors"].shape == (4, 2, 2, 2)
    assert data["R_anchors"].shape == (4,)
    # All finite.
    assert np.all(np.isfinite(data["P_flat"]))
    assert np.all(np.isfinite(data["Vxc_flat"]))
    # V_xc entries are O(1) Hartree for B3LYP — sanity-check magnitude.
    assert np.max(np.abs(data["Vxc_flat"])) < 5.0
