"""Tier A — UHF XC-emulator accuracy gate and symmetry property.

SPEC_scf §6 P3, §7, §8.10 R3.

This module loads the trained UHF STO-3G XC-net artifact committed at
``examples/scf_xc_emulator_uhf_sto3g.h5`` and verifies:

  A-scf-uhf-xcnet-trained  — relative MAE on a held-out validation set
                             (generated with seed=43, disjoint from the
                             training seed=42) is below the SPEC_scf §8.10
                             R3 **3 % gate**.

  A-scf-uhf-xcnet-symmetry — output preserves the symmetry of the input
                             ``(P_α, P_β)`` *within each spin block*: PySCF
                             evaluates the XC quadrature against the
                             symmetric density per spin and returns a
                             symmetric ``V_xc^σ`` matrix.  The packed-flat
                             representation discards lower-triangular
                             entries by construction, so this property
                             reduces to a sanity check that the network
                             does not introduce spurious antisymmetric
                             content within either spin block (a
                             per-spin-block analogue of the σ_h-swap
                             check used in the P3 closed-shell test).

The Tier A gate runs on PR-to-main, so we generate the validation data
on the fly (PySCF needed) but keep it cheap: 64 held-out samples are
enough to discriminate a 3 % gate against a ~0.6 % outcome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# All Tier A tests on PySCF physics require an importable PySCF.
pytest.importorskip("pyscf")
# h5py is required to load the trained artifact.
pytest.importorskip("h5py")

from scf._make_state import make_h2_state
from scf.emulator import load_xc_emulator
from scf.emulator.training import generate_training_data_uhf
from scf.emulator.validation import validation_rel_mae_uhf
from scf.state import pack_density_matrix, unpack_density_matrix

# Trained UHF XCNet artifact path.  ``parents[3]`` walks:
# analytic → scf → tests → repo_root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_XCNET_UHF_STO3G_PATH = _REPO_ROOT / "examples" / "scf_xc_emulator_uhf_sto3g.h5"


# ---------------------------------------------------------------------------
# A-scf-uhf-xcnet-trained: rel_mae < 3 % on disjoint-seed holdout
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_A_scf_uhf_xcnet_trained() -> None:
    """A-scf-uhf-xcnet-trained: rel_mae < 3 % on a seed=43 held-out set.

    The training artifact at ``examples/scf_xc_emulator_uhf_sto3g.h5`` is
    trained with seed=42 (see ``examples/train_xc_emulator_uhf_sto3g.py``).
    Here we generate a fresh ``(P, V_xc)`` set with seed=43, evaluate
    the loaded XCNet, and assert relative MAE below the SPEC_scf §8.10 R3
    gate of 0.03 (loosened from P3's 0.02 because UHF V_xc near the
    Coulson–Fischer point has higher local sensitivity).

    Generation parameters mirror the training set so that the holdout
    samples the same operating distribution (R range, perturbation
    amplitude).  N_holdout is small (64) to keep the test under ~10 s
    while still resolving the 3 % / 1 % difference with negligible
    sampling noise.
    """
    if not _XCNET_UHF_STO3G_PATH.exists():
        pytest.skip(
            f"Trained UHF STO-3G XCNet artifact not found at "
            f"{_XCNET_UHF_STO3G_PATH!s}; run "
            "examples/train_xc_emulator_uhf_sto3g.py to regenerate."
        )

    net = load_xc_emulator(_XCNET_UHF_STO3G_PATH)
    # Doubled-seam length for STO-3G UHF: 2 * 2*(2+1)/2 = 6.
    assert net.n_unique == 6
    template = make_h2_state(R=1.18, basis="sto-3g", method="uhf", xc="B3LYP")
    holdout = generate_training_data_uhf(
        template,
        n_samples=64,
        seed=43,                # disjoint from training seed=42
        R_lo=1.10,
        R_hi=1.25,
        perturb_max=0.03,
        n_R_anchors=4,          # smaller than training; sufficient for 64 samples
    )
    rel_mae = validation_rel_mae_uhf(net, holdout)
    assert rel_mae < 0.03, (
        f"A-scf-uhf-xcnet-trained failed: rel_mae={rel_mae:.4f}, "
        f"SPEC_scf §8.10 R3 gate is 0.03.  Retrain or escalate."
    )


# ---------------------------------------------------------------------------
# A-scf-uhf-xcnet-symmetry: per-spin-block invariance under index swap
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_A_scf_uhf_xcnet_symmetry() -> None:
    """A-scf-uhf-xcnet-symmetry: XCNet output preserves per-spin V_xc symmetry.

    The truth V_xc^σ is exactly symmetric per spin: PySCF's spin-resolved
    XC quadrature operates on the (symmetric per spin) density and
    returns a symmetric V_xc^σ matrix per spin.  The emulator therefore
    should approximately satisfy ``V_xc^σ_{NN}[i,j] = V_xc^σ_{NN}[j,i]``
    for σ ∈ {α, β}.

    Because the XCNet's input is the packed-flat (upper-triangular)
    representation per spin, this invariance is trivial under input-side
    repacking of the same symmetric matrix.  We therefore exercise a
    different *physical* symmetry: H₂ has a σ_h plane through the bond
    midpoint perpendicular to the bond axis, so swapping the two atomic
    orbitals (the M=2 STO-3G basis) leaves the chemistry invariant.
    Under this swap, ``P_σ[0,0]`` exchanges with ``P_σ[1,1]``, ``P_σ[0,1]``
    is invariant, and the truth ``V_xc^σ`` respects the analogous swap
    per spin.

    For UHF the broken-symmetry density localizes α-density on one atom
    and β-density on the other, so the σ_h swap also exchanges α ↔ β
    spin blocks.  The composite operation we check is therefore:
    swap basis indices (0 ↔ 1) within each spin block AND swap the two
    spin blocks.  The truth V_xc^(α,β) is invariant under this composite
    operation; the trained XCNet should learn the same.

    Construction:
      input  P_α = [a, c, b]; P_β = [d, e, f]
             flat = [a, c, b, d, e, f]
             NN(flat) = [V_α^{0,0}, V_α^{0,1}, V_α^{1,1},
                         V_β^{0,0}, V_β^{0,1}, V_β^{1,1}]
      swapped P_α' = [f, e, d] (was P_β with basis swap)
              P_β' = [b, c, a] (was P_α with basis swap)
              flat' = [f, e, d, b, c, a]
              expected NN(flat')[k] swaps in the same pattern
    """
    if not _XCNET_UHF_STO3G_PATH.exists():
        pytest.skip(
            f"Trained UHF STO-3G XCNet artifact not found at "
            f"{_XCNET_UHF_STO3G_PATH!s}; run "
            "examples/train_xc_emulator_uhf_sto3g.py to regenerate."
        )

    net = load_xc_emulator(_XCNET_UHF_STO3G_PATH)

    rng = np.random.default_rng(7)
    rel_errs: list[float] = []
    # Sample several plausible broken-symmetry densities; check the σ_h
    # spin-and-basis swap invariance on each.  Values are pulled from the
    # broken-symmetry neighborhood (α concentrated on bf 0, β on bf 1).
    for _ in range(20):
        # α-spin: high on bf 0, low on bf 1.
        a = float(rng.uniform(0.7, 1.1))
        b = float(rng.uniform(0.1, 0.3))
        c = float(rng.uniform(0.1, 0.3))
        # β-spin: mirror image of α (high on bf 1, low on bf 0).
        d = float(rng.uniform(0.1, 0.3))
        e = float(rng.uniform(0.1, 0.3))
        f = float(rng.uniform(0.7, 1.1))

        flat1 = torch.tensor([a, c, b, d, e, f], dtype=torch.float64)
        # Composite σ_h: swap basis (0 ↔ 1) within each spin, then swap
        # α ↔ β.  Per-block swap is [a, c, b] → [b, c, a]; full block
        # swap then yields [f, e, d, b, c, a].
        flat2 = torch.tensor([f, e, d, b, c, a], dtype=torch.float64)

        with torch.no_grad():
            V1 = net(flat1.unsqueeze(0)).squeeze(0).numpy()
            V2 = net(flat2.unsqueeze(0)).squeeze(0).numpy()
        # Expected: V2 = swap(V1) under the same composite operation.
        # V1 packing: [V_α^{0,0}, V_α^{0,1}, V_α^{1,1}, V_β^{0,0}, V_β^{0,1}, V_β^{1,1}].
        # Per-block index swap: [V_α^{1,1}, V_α^{0,1}, V_α^{0,0}, V_β^{1,1}, V_β^{0,1}, V_β^{0,0}].
        # α ↔ β swap on top: [V_β^{1,1}, V_β^{0,1}, V_β^{0,0}, V_α^{1,1}, V_α^{0,1}, V_α^{0,0}].
        ref = np.array(
            [V1[5], V1[4], V1[3], V1[2], V1[1], V1[0]],
            dtype=np.float64,
        )
        denom = np.abs(ref) + 1e-8
        rel_errs.append(float(np.mean(np.abs(V2 - ref) / denom)))

    mean_rel = float(np.mean(rel_errs))
    # The 3 % R3 gate is the publication threshold; the swap-invariance is
    # a symmetry the truth respects exactly, so the NN's deviation should
    # be at most a small multiple of the training precision.  Allow up to
    # 8 % as a soft gate that catches regressions (e.g. forgetting the
    # broken-symmetry training coverage) without flagging routine
    # training noise.
    assert mean_rel < 0.08, (
        f"A-scf-uhf-xcnet-symmetry failed: mean relative swap error = "
        f"{mean_rel:.4f}.  Either the network forgot the σ_h symmetry "
        f"or the artifact was retrained without symmetric coverage."
    )
