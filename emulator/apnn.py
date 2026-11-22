"""APNN model class. the technical reference §4.1, §4.4.

Architecture
------------
This conserving population MLP supplies a reference two-level closure.
The current physics-structured closure is implemented separately in
``emulator.twolevel.full_envelope.structured``.

Chosen architecture:
  Input dim  : 3   (log J_bar, log T, log n_e)
  Hidden dims: 3 × 128 units each
  Output dim : 2   (n_1, n_2)
  Activations: GELU between hidden layers; softmax over the output dim to
               enforce population conservation n_1 + n_2 = 1 for n_tot=1
               (satisfies U-rateeq-02 by construction).
  Parameter count: (3×128 + 128) + (128×128 + 128) + (128×128 + 128)
                   + (128×2 + 2)
                 = 512 + 16512 + 16512 + 258 = 33,794 params.

Float precision notes:
  - network internal compute: float32 (default PyTorch).
  - boundary cast: inputs are cast to float32 on entry; outputs are cast to
    float64 on exit so that population arithmetic downstream matches the
    atmosphere stack.
  - The cast at the boundary is documented here and in forward().

CUDA non-determinism note (relevant to R-emulator-weights):
  The regression test that gates checkpoint reproducibility runs in CPU-only
  mode.  On GPU, cuDNN convolutions and atomics introduce non-determinism
  even with a fixed seed; to disable this at training time, set
  torch.use_deterministic_algorithms(True) and export CUBLAS_WORKSPACE_CONFIG.
  All training entry points in training.py set CPU seeds; GPU callers must
  additionally pin the device seed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class APNN(nn.Module):
    """Atomic Physics Neural Network: (log J_bar, log T, log n_e) -> (n_1, n_2).

    Maps packed log-scaled inputs to normalised level populations.

    Architecture: 3 → [128] → [128] → [128] → 2 with GELU activations and a
    softmax output gate that enforces n_1 + n_2 = 1 (normalised populations).

    Parameter count: 33,794 (see module docstring for derivation).

    The forward pass is batch-agnostic (the technical reference §4.2): the model evaluates on a
    single depth point (B=1), a depth column (B=N_τ), or a multi-column
    training batch without code changes.

    Inputs are expected to be log-scaled (log₁₀ or natural — consistent with
    the training data generator in training.py).  See training.py for the exact
    scaling convention.

    Float precision: internal compute defaults to float32; explicit .double()
    promotes evaluation. Output is cast to float64 so
    that population arithmetic in the atmosphere stack maintains full precision.
    """

    def __init__(
        self,
        n_in: int = 3,
        n_out: int = 2,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
    ) -> None:
        """Initialise the APNN.

        Parameters
        ----------
        n_in : int, default 3
            Input features: (log J_bar, log T, log n_e).
        n_out : int, default 2
            Output features: (n_1, n_2) normalised populations.
        hidden_dims : tuple of int, default (128, 128, 128)
            Width of each hidden layer.  Three layers at 128 units gives a
            good balance between expressivity and parameter efficiency for the
            3→2 closed-form map.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_dims = hidden_dims

        # Build the layer sequence dynamically from hidden_dims.
        layers: list[nn.Module] = []
        in_dim = n_in
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_out))
        # Softmax over last dim enforces sum(populations) = 1 (population
        # conservation U-rateeq-02) by construction.
        layers.append(nn.Softmax(dim=-1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, n_in) → (B, n_out), output in float64.

        Parameters
        ----------
        x : Tensor of shape (B, n_in), any dtype
            Log-scaled inputs.  Typically float32 or float64.

        Returns
        -------
        out : Tensor of shape (B, n_out), float64
            Normalised level populations (n_1, n_2).
            Each row sums to 1 (enforced by the softmax output gate).

        Notes
        -----
        Precision strategy:
          - The network parameters are stored in float32 (default PyTorch init).
          - Inputs are cast to float32 before the network, and the output is
            cast to float64 before returning.  This keeps network compute in
            float32 for efficiency and satisfies the float64 convention of the
            atmosphere stack.
          - For torch.func.jvp or torch.autograd.gradcheck (which supply
            float64 inputs), the jvp() wrapper in jvp.py casts the model
            parameters to float64 using torch.func.functional_call, so that
            the functional transform sees a consistent dtype throughout.
            The forward() method itself does NOT need to handle that case —
            the wrapper owns the dtype promotion responsibility.
          - The output is always float64 so downstream atmosphere arithmetic
            maintains full precision (matching atmosphere/rate_equations.py).
        """
        # Match the evaluation dtype to the parameters.
        # Default weights remain float32. Explicit model.double() evaluates
        # the same stored weights as a smooth float64 map for derivative and
        # remainder studies, without a hidden cast back to float32.
        x_compute = x.to(dtype=self.net[0].weight.dtype)
        out_compute = self.net(x_compute)
        # Cast output to float64 for downstream atmosphere arithmetic.
        return out_compute.to(dtype=torch.float64)
