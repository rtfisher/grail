"""TorchBackend: PyTorch implementation of the GRAIL Backend protocol.

All operations are float64 (torch.float64).  This backend is the default
and preserves 100% of the pre-P1 numerical behaviour.

SPEC_jax.md §7 (P1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from grail.backend import Backend, register_backend


class TorchBackend(Backend):
    """Dispatches GRAIL numerical primitives to ``torch.*`` and
    ``torch.linalg.*``.  All arrays are ``torch.float64`` tensors."""

    # ------------------------------------------------------------------
    # Factory / conversion
    # ------------------------------------------------------------------

    def from_numpy(self, a: np.ndarray) -> torch.Tensor:
        """Convert a numpy array to a float64 torch.Tensor.

        Works for any numeric dtype; always upcasts to float64.
        """
        return torch.as_tensor(a, dtype=torch.float64)

    def to_numpy(self, a: torch.Tensor) -> np.ndarray:
        """Convert a torch.Tensor to a float64 numpy ndarray."""
        return a.detach().cpu().to(dtype=torch.float64).numpy()

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def zeros(self, *shape: int) -> torch.Tensor:
        """Return a float64 zero tensor of the given shape."""
        return torch.zeros(shape, dtype=torch.float64)

    def clone(self, a: torch.Tensor) -> torch.Tensor:
        """Return a detached copy of *a*.

        Uses ``.clone()`` to avoid aliasing; detach is implicit because
        GRAIL never needs gradients through the backend.
        """
        return a.clone()

    # ------------------------------------------------------------------
    # Reduction / combination
    # ------------------------------------------------------------------

    def dot(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Inner product of two 1-D float64 tensors; returns a scalar tensor."""
        return torch.dot(a, b)

    def norm(self, a: torch.Tensor) -> float:
        """Euclidean norm of a 1-D float64 tensor as a Python float."""
        return a.norm().item()

    def concat(self, arrays: list[Any], *, dim: int = 0) -> torch.Tensor:
        """Concatenate a list of tensors along *dim*."""
        return torch.cat(arrays, dim=dim)

    def where(self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise conditional select."""
        return torch.where(condition, x, y)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def solve_triangular(
        self, R: torch.Tensor, b: torch.Tensor, *, upper: bool = True
    ) -> torch.Tensor:
        """Solve R @ x = b for upper (or lower) triangular R.

        Both R and b must be 2-D; b is a column vector of shape (m, 1).
        Returns x of shape (m, 1).
        """
        return torch.linalg.solve_triangular(R, b, upper=upper)

    def svdvals(self, A: torch.Tensor) -> torch.Tensor:
        """Singular values of *A* in descending order."""
        return torch.linalg.svdvals(A.to(dtype=torch.float64))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def quantile(self, a: torch.Tensor, q: float) -> torch.Tensor:
        """Return the q-th quantile of a 1-D float64 tensor."""
        return torch.quantile(a, q)


# Register under the canonical name used by get_backend("torch").
register_backend("torch", TorchBackend)
