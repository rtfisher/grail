'JAX backend for the matrix-free linear response solver. Enable jax.config.update("jax_enable_x64", True) before creating JAX arrays for float64 work.'

from __future__ import annotations

from typing import Any

import numpy as np

from grail.backend import Backend, register_backend


# Enable float64 globally when the JAX backend is imported.
# This must be called before any JAX computation; importing this module
# satisfies the requirement.  SPEC_jax.md §2.2.
try:
    import jax as _jax_mod
    _jax_mod.config.update("jax_enable_x64", True)
except ImportError:
    pass  # ImportError surfaces when get_backend("jax") is called without JAX installed.


def _jax():
    """Lazy import of jax to avoid hard dependency at grail/ import time."""
    import jax
    return jax


def _jnp():
    """Lazy import of jax.numpy."""
    import jax.numpy as jnp
    return jnp


def _jsp_linalg():
    """Lazy import of jax.scipy.linalg."""
    import jax.scipy.linalg
    return jax.scipy.linalg


class JaxBackend(Backend):
    'Dispatches GRAIL numerical primitives to ``jax.numpy.*`` and\n    ``jax.scipy.linalg.*``.'

    # ------------------------------------------------------------------
    # Factory / conversion
    # ------------------------------------------------------------------

    def from_numpy(self, a: np.ndarray) -> Any:
        """Convert a numpy float64 array to a JAX float64 array."""
        jnp = _jnp()
        return jnp.array(np.asarray(a, dtype=np.float64), dtype=jnp.float64)

    def to_numpy(self, a: Any) -> np.ndarray:
        """Convert a JAX array to a writable float64 numpy ndarray.

        JAX arrays are immutable; ``np.asarray`` of a JAX array returns a
        read-only view.  We copy to return a mutable array consistent with
        TorchBackend.to_numpy (which shares writable memory via .numpy()).
        """
        return np.array(a, dtype=np.float64)  # np.array always copies → writable

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def zeros(self, *shape: int) -> Any:
        """Return a float64 zero JAX array of the given shape."""
        jnp = _jnp()
        return jnp.zeros(shape, dtype=jnp.float64)

    def clone(self, a: Any) -> Any:
        """Return a copy of *a*.

        JAX arrays are immutable, so this is a no-op (any modification
        creates a new array automatically).  We explicitly call jnp.array
        to ensure the returned value is a fresh JAX array.
        """
        jnp = _jnp()
        return jnp.array(a)

    # ------------------------------------------------------------------
    # Reduction / combination
    # ------------------------------------------------------------------

    def dot(self, a: Any, b: Any) -> Any:
        """Inner product of two 1-D float64 JAX arrays."""
        jnp = _jnp()
        return jnp.dot(a, b)

    def norm(self, a: Any) -> float:
        """Euclidean norm of a 1-D float64 JAX array as a Python float."""
        jnp = _jnp()
        return float(jnp.linalg.norm(a))

    def concat(self, arrays: list[Any], *, dim: int = 0) -> Any:
        """Concatenate a list of JAX arrays along *dim*."""
        jnp = _jnp()
        return jnp.concatenate(arrays, axis=dim)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Element-wise conditional select."""
        jnp = _jnp()
        return jnp.where(condition, x, y)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def solve_triangular(
        self, R: Any, b: Any, *, upper: bool = True
    ) -> Any:
        """Solve R @ x = b for upper (or lower) triangular R via
        jax.scipy.linalg.solve_triangular.

        Both R and b must be 2-D; b is a column vector of shape (m, 1).
        Returns x of shape (m, 1).
        """
        jsp = _jsp_linalg()
        return jsp.solve_triangular(R, b, lower=not upper)

    def svdvals(self, A: Any) -> Any:
        """Singular values of *A* in descending order.

        JAX 0.6 exposes jnp.linalg.svd; we use compute_uv=False to get
        singular values only, consistent with the torch.linalg.svdvals API.
        """
        jnp = _jnp()
        # jnp.linalg.svd returns s in descending order (same as numpy).
        return jnp.linalg.svd(A, full_matrices=False, compute_uv=False)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def quantile(self, a: Any, q: float) -> Any:
        """Return the q-th quantile of a 1-D float64 JAX array.

        JAX provides jnp.quantile with the same interpolation convention
        as numpy (linear by default).
        """
        jnp = _jnp()
        return jnp.quantile(a, q)


# Register under the canonical name used by get_backend("jax").
register_backend("jax", JaxBackend)
