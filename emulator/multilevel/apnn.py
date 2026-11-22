'Shared multilevel neural population models. The generic species-normalized head supplies the five-level model factory and conserves each species separately.'

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..apnn import APNN


class SpeciesPartitionedSoftmax(nn.Module):
    """Output head enforcing per-species population conservation.

    Applies softmax independently to each species block, then rescales
    each block's output by the corresponding species fraction so that:

        sum_{i ∈ block_s} n_i = fractions[s]   for each species s.

    This replaces the global softmax used in the single-species APNN and
    is C^∞ smooth (composition of softmax and scalar multiplication), so
    FD-JVP through the forward pass is well-conditioned.

    Parameters
    ----------
    species_blocks : sequence of (lo, hi)
        Half-open index slices of the output vector belonging to each
        species block.  Consistent with ``AtomicModel.species_blocks``.
    fractions : sequence of float
        Per-block species fraction.  Must sum to 1 (not enforced here;
        validated by the caller).  Length must equal len(species_blocks).

    SPEC_phase_s §4.1 (per-species softmax head), v1.1.
    """

    def __init__(
        self,
        species_blocks: Sequence[tuple[int, int]],
        fractions: Sequence[float],
    ) -> None:
        super().__init__()
        if len(species_blocks) != len(fractions):
            raise ValueError(
                f"len(species_blocks)={len(species_blocks)} must equal "
                f"len(fractions)={len(fractions)}."
            )
        # Store as plain Python data (not nn.Parameters — these are fixed).
        self._blocks: list[tuple[int, int]] = list(species_blocks)
        self._fractions: list[float] = [float(f) for f in fractions]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-block softmax and scale by species fraction.

        Parameters
        ----------
        x : Tensor of shape (B, n_out), any float dtype
            Raw logit outputs from the final linear layer.

        Returns
        -------
        out : Tensor of shape (B, n_out), same dtype as x
            Per-species-block softmax, each block scaled by its fraction.
        """
        parts: list[torch.Tensor] = []
        for (lo, hi), frac in zip(self._blocks, self._fractions):
            block_logits = x[:, lo:hi]
            # Softmax within this block then scale by the species fraction.
            block_pops = torch.softmax(block_logits, dim=-1) * frac
            parts.append(block_pops)
        return torch.cat(parts, dim=-1)


class SpeciesAPNN(nn.Module):
    """APNN with per-species softmax output head.

    Replaces the global-softmax ``APNN`` for multi-species setups where
    each species block must independently sum to a given fraction.

    Architecture: n_in → [hidden_dims] → n_out with GELU activations and
    a ``SpeciesPartitionedSoftmax`` output gate.

    Float precision: internal compute in float32; output cast to float64
    (same boundary convention as the single-species APNN).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dims: tuple[int, ...],
        species_blocks: Sequence[tuple[int, int]],
        fractions: Sequence[float],
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_dims = hidden_dims

        layers: list[nn.Module] = []
        in_dim = n_in
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_out))
        # Per-species softmax output gate.
        layers.append(SpeciesPartitionedSoftmax(species_blocks, fractions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, n_in) → (B, n_out), output in float64."""
        x_f32 = x.to(dtype=torch.float32)
        out_f32 = self.net(x_f32)
        return out_f32.to(dtype=torch.float64)


def make_apnn_for_atomic_model(
    atomic_model,   # AtomicModel — avoid circular import at top level
    hidden_dims: tuple[int, ...] = (256, 256, 256, 256),
    fractions: dict | None = None,
) -> nn.Module:
    'Parameterised APNN factory for any AtomicModel.\n\n    Reads n_in = K + 2 and n_out = n_levels from the atomic_model.\n    When atomic_model.species_blocks is not None (multi-species), returns a\n    ``SpeciesAPNN`` with a per-species softmax head.  When species_blocks\n    is None (single-species), falls through to the global-softmax ``APNN``.\n\n    Returns\n    -------\n    nn.Module\n        Untrained APNN (either ``APNN`` or ``SpeciesAPNN``).\n\n    SPEC_phase_s §4.1 (APNN factory), v1.1.'
    n_in = atomic_model.K + 2
    n_out = atomic_model.n_levels

    if atomic_model.species_blocks is None or len(atomic_model.species_blocks) <= 1:
        # Single-species path: global softmax (byte-identical to make_apnn_5level).
        return APNN(n_in=n_in, n_out=n_out, hidden_dims=hidden_dims)

    # Multi-species path: per-species softmax.
    blocks = atomic_model.species_blocks
    species = atomic_model.species

    if fractions is None:
        # Uniform distribution across blocks as a fallback.
        frac_list = [1.0 / len(blocks)] * len(blocks)
    else:
        frac_list = [float(fractions.get(species[lo], 1.0 / len(blocks)))
                     for lo, hi in blocks]

    return SpeciesAPNN(
        n_in=n_in,
        n_out=n_out,
        hidden_dims=hidden_dims,
        species_blocks=blocks,
        fractions=frac_list,
    )


def make_apnn_5level() -> APNN:
    """Construct the 5-level APNN with the M.4 architecture.

    Returns an APNN with:
        n_in=12, n_out=5, hidden_dims=(256, 256, 256, 256)

    The returned model is freshly initialised (random weights).  Call
    training.train_multilevel() to fit the weights, then adapter.make_cycle_adapter_5level()
    to wire it into the atmosphere cycle.

    This is a thin shim over ``make_apnn_for_atomic_model`` for
    backwards compatibility with existing code.

    Returns
    -------
    APNN
        Untrained 5-level APNN model.
    """
    return APNN(n_in=12, n_out=5, hidden_dims=(256, 256, 256, 256))


__all__ = [
    "APNN",
    "SpeciesAPNN",
    "SpeciesPartitionedSoftmax",
    "make_apnn_5level",
    "make_apnn_for_atomic_model",
]
