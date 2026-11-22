"""Validation gate for the UHF XC surrogate.  SPEC_scf §6 P3, §7, §8.10 R3.

The trained :class:`scf.emulator.XCNet` must achieve relative MAE on a
held-out set below the SPEC_scf §8.10 R3 gate of **3 %** — loosened from
the original 2 % gate of the closed-shell P3 path because the UHF V_xc
near the Coulson–Fischer point has higher local sensitivity.

The gate is looser than the NLTE 1 % gate (M1b) for the reasons documented in
SPEC_scf §6 P3 and tightened in §8.10:
  1. V_xc matrix entries span a wider relative dynamic range than line
     populations (entries near zero where the ``+1e-8`` regularization in
     the relative-error denominator matters).
  2. The ``(P, R)`` joint perturbation neighborhood is larger than the
     NLTE 1-D LHS coverage.
  3. Per-call NN error feeds into the Born-remainder bound ``C = 1.25``
     downstream; the 3 % allowance keeps headroom on that bound.

This module only computes the metric; it does not enforce the gate (the
test layer does that, in
``tests/scf/analytic/test_uhf_xc_emulator_accuracy.py``).
"""

from __future__ import annotations

import numpy as np
import torch

from .network import XCNet


def validation_rel_mae_uhf(
    xc_net: XCNet,
    holdout_data: dict[str, np.ndarray],
) -> float:
    """Relative MAE on a held-out UHF ``(P, V_xc)`` set.

    Metric (SPEC_scf §6 P3, generalized for the doubled UHF seam):

        rel_mae = mean( |V_xc^NN - V_xc^B3LYP| / (|V_xc^B3LYP| + 1e-8) )

    Both terms are float64 ndarrays of shape ``(N, 2*n_unique)``; the
    relative error is taken elementwise (per packed-flat entry per
    sample), then averaged over all entries and samples.  The
    ``+1e-8`` regularization protects against division by zero where the
    truth ``V_xc`` is small.

    Parameters
    ----------
    xc_net : XCNet
        Trained network (call site is expected to put it in eval mode;
        we re-enforce ``eval()`` defensively).
    holdout_data : dict
        Output of :func:`generate_training_data_uhf` (or any compatible
        dict).  Must contain ``"P_flat"`` and ``"Vxc_flat"`` keys, each
        an ndarray of shape ``(N, 2*n_unique)``.  The other keys (R,
        anchors) are ignored.

    Returns
    -------
    rel_mae : float
        The gate-quantity.  SPEC_scf §8.10 R3: production UHF XCNets must
        satisfy ``rel_mae < 0.03`` on a stratified holdout.

    Notes
    -----
    All compute happens in float64 (the input is cast on the way in via
    the ``XCNet.forward`` output cast).
    """
    P_flat = np.ascontiguousarray(holdout_data["P_flat"], dtype=np.float64)
    Vxc_truth = np.ascontiguousarray(holdout_data["Vxc_flat"], dtype=np.float64)
    if P_flat.shape != Vxc_truth.shape:
        raise ValueError(
            f"P_flat and Vxc_flat shape mismatch: "
            f"{P_flat.shape!r} vs {Vxc_truth.shape!r}."
        )
    if P_flat.shape[1] != xc_net.n_unique:
        raise ValueError(
            f"holdout n_unique={P_flat.shape[1]} does not match "
            f"xc_net.n_unique={xc_net.n_unique}."
        )

    xc_net.eval()
    with torch.no_grad():
        P_t = torch.from_numpy(P_flat)              # (N, 2*n_unique), float64
        V_pred = xc_net(P_t).numpy()                # (N, 2*n_unique), float64

    residuals = np.abs(V_pred - Vxc_truth)
    denom = np.abs(Vxc_truth) + 1e-8
    rel_mae = float(np.mean(residuals / denom))
    return rel_mae
