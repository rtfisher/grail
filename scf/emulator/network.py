"""Neural surrogate for the UHF/UKS exchange–correlation matrix.

SPEC_scf §3.2, §6 P3, §7, §8.10 (R3).

The :class:`XCNet` is a feed-forward network that maps the packed-flat unique
entries of an open-shell density pair ``(P_α, P_β)`` to the packed-flat unique
entries of the spin-resolved exchange–correlation matrix pair
``(V_xc^α[P_α, P_β], V_xc^β[P_α, P_β])``.  Both packed-flat vectors use the
*doubled* α | β upper-triangular row-major convention of
``scf.state.pack_density_matrix`` for ``method ∈ {"uhf", "uks"}``
(SPEC_scf §8.10, ``scf/INTERFACES.md`` §8.2): total length
``2 * M*(M+1)/2`` — 6 for STO-3G H₂, 20 for 6-31G H₂.

Architecture (SPEC_scf §7, unchanged from P3 closed-shell):
  Input dim  : 2 * n_unique  (6 for STO-3G, 20 for 6-31G)
  Hidden    : depth=2 layers of width=64, GELU activations
  Output dim : 2 * n_unique  (raw V_xc unique entries per spin, Hartree)
  Parameter count (STO-3G H₂): (6·64+64) + (64·64+64) + (64·6+6)
                              = 448 + 4160 + 390 = 4998  (~5k)
  Parameter count (6-31G H₂) : (20·64+64) + (64·64+64) + (64·20+20)
                              = 1344 + 4160 + 1300 = 6804  (~7k)
  Both well under the 100k upper bound suggested by the orchestrator.

Precision strategy (mirrors ``emulator/apnn.py`` and P3 closed-shell):
  - Network parameters stored in float32 (default PyTorch init).
  - Forward computation follows the parameter dtype, default float32;
    ``net.double()`` explicitly enables float64 evaluations of the same
    frozen weights.  Output is always float64 at the SCF boundary.

V_xc is taken directly (no log transform) because B3LYP matrix entries
typically lie in roughly ``[-1, 0]`` Hartree and are well-conditioned for
direct supervised regression.  Mirrors the SPEC_scf §6 P3 recommendation.

The network input/output dimensions both double versus the P3 closed-shell
network — the architecture (hidden=64, depth=2, GELU) is preserved per
SPEC_scf §8.10 R3.

Small-output initialization (SPEC_scf §8.10 R3):  V_xc is a small
perturbation around zero at the start of training (Hartree-scale entries
~ -1, but the network's untrained output should not dominate the
supervised target).  We rescale the final-layer weight and bias by
``1e-3`` so the initial output magnitude is ≤ 1e-3 in V_xc units.  Keeps
the early-epoch gradient signal-to-noise ratio sane on the UHF V_xc
landscape, which has higher local sensitivity near the Coulson–Fischer
point than the closed-shell B3LYP V_xc.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class XCNet(nn.Module):
    """Feed-forward NN for the UHF/UKS exchange–correlation matrix surrogate.

    Maps a packed-flat doubled density vector
    ``P_flat ∈ R^{2 n_unique}`` (concatenated α | β packings) to a
    packed-flat doubled V_xc vector
    ``V_xc_flat ∈ R^{2 n_unique}``.  The mapping is unconstrained at the
    network level — symmetry of each spin's ``V_xc`` matrix block follows
    from the symmetric packing convention, not from any architectural
    constraint.  See module docstring for the precision strategy and
    R3 contract.

    Parameters
    ----------
    n_unique : int
        *Total* length of the packed-flat vectors at the GRAIL boundary.
        For UHF/UKS this is ``2 * M*(M+1)/2`` — i.e. 6 for STO-3G H₂,
        20 for 6-31G H₂.  Named ``n_unique`` rather than ``2n_unique`` for
        save/load compatibility with the closed-shell P3 HDF5 schema
        (which used ``n_unique`` for the singlet packed length).
    hidden : int, default 64
        Hidden-layer width (SPEC_scf §7).
    depth : int, default 2
        Number of hidden layers (SPEC_scf §7).

    Notes
    -----
    Forward pass is batch-agnostic: input shape ``(B, n_unique)`` returns
    output ``(B, n_unique)``.  A bare ``(n_unique,)`` input is accepted via
    a leading-axis broadcast (handled by ``nn.Linear``).
    """

    def __init__(self, n_unique: int, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        if n_unique < 1:
            raise ValueError(f"n_unique must be >= 1, got n_unique={n_unique!r}.")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got hidden={hidden!r}.")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got depth={depth!r}.")
        # Stash the config for save/load round-trip (see scf/emulator/io.py).
        self.n_unique = n_unique
        self.hidden = hidden
        self.depth = depth

        # Build the layer stack: input -> [Linear, GELU] * depth -> output.
        layers: list[nn.Module] = []
        in_dim = n_unique
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            in_dim = hidden
        out_layer = nn.Linear(in_dim, n_unique)
        layers.append(out_layer)
        # No output activation: V_xc entries are real-valued, typically
        # in [-1, 0] Hartree but not constrained.  The closed-shell P3
        # convention is preserved here.
        self.net = nn.Sequential(*layers)

        # Small-output initialization (R3 §8.10): rescale the final linear
        # layer by 1e-3 so the untrained network outputs ≤ ~1e-3 in V_xc
        # units across the doubled state seam.  This is the standard "small
        # initial perturbation" trick — it keeps the supervised loss
        # dominated by the truth signal in the first few epochs rather
        # than by the random output magnitude (~O(hidden^{-1/2}) for
        # vanilla Kaiming init, i.e. ~0.1 Hartree on the 64-wide layer).
        with torch.no_grad():
            out_layer.weight.mul_(1e-3)
            out_layer.bias.mul_(1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``(B, n_unique) → (B, n_unique)``, output float64.

        Parameters
        ----------
        x : Tensor of shape ``(B, n_unique)`` or ``(n_unique,)``
            Packed-flat doubled density-pair entries (concatenated α | β
            upper-triangular row-major; see
            :func:`scf.state.pack_density_matrix` with ``method="uhf"``).
            Any input dtype is accepted; compute follows parameter dtype.

        Returns
        -------
        out : Tensor of shape matching ``x``, float64
            Packed-flat doubled ``V_xc`` entries in Hartree.

        Notes
        -----
        - Default float32 evaluations preserve legacy predictions.
        - ``net.double()`` uses float64 arithmetic without changing the
          checkpoint weights.  Central differences still evaluate the
          actual network; no truth derivative substitutes for its JVP.
        """
        # The default model remains float32.  Explicit net.double() makes
        # the entire evaluation smooth in float64, avoiding quantization
        # floors in finite differences and tight equilibrium solves.
        parameter = next(self.parameters())
        out = self.net(x.to(dtype=parameter.dtype, device=parameter.device))
        return out.to(dtype=torch.float64)
