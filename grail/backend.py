"""Backend protocol and registry for GRAIL's array-library abstraction.

GRAIL can dispatch its internal tensor operations through either PyTorch
(the default) or JAX.  The abstraction is a small explicit protocol — about
10 methods — that covers the GRAIL numerical surface:

    dot, norm, zeros, clone, concat, where, solve_triangular,
    svdvals, quantile, from_numpy, to_numpy

All internal `grail/` modules accept a ``backend: Backend`` parameter
(defaulting to the active TorchBackend) and dispatch through it.  Callers
that do not care about the backend need not touch this module.

The active default backend is controlled by:

    grail.set_default_backend("torch")   # default
    grail.set_default_backend("jax")

or overridden per-call via the ``backend`` kwarg on ``resolvent_solve``.

SPEC_jax.md §3 / §7 (P1).
"""

from __future__ import annotations

import abc
from typing import Any, Optional

import numpy as np


class Backend(abc.ABC):
    """Abstract base class defining GRAIL's internal array-operation surface.

    Each method maps closely to a single primitive from the underlying library
    so that the implementation is mechanical (torch.linalg.X → one method),
    not clever.  This minimises the risk of subtle semantic differences
    between backends.
    """

    # ------------------------------------------------------------------
    # Factory / conversion
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def from_numpy(self, a: np.ndarray) -> Any:
        """Convert a numpy float64 array to this backend's native type."""

    @abc.abstractmethod
    def to_numpy(self, a: Any) -> np.ndarray:
        """Convert a native array to a numpy float64 ndarray."""

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def zeros(self, *shape: int) -> Any:
        """Return a float64 zero-filled array of the given shape."""

    @abc.abstractmethod
    def clone(self, a: Any) -> Any:
        """Return a copy of *a* (detaches from computation graph if relevant)."""

    # ------------------------------------------------------------------
    # Reduction / combination
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        """Inner product of two 1-D float64 arrays; returns a 0-D scalar array."""

    @abc.abstractmethod
    def norm(self, a: Any) -> float:
        """Euclidean norm of a 1-D float64 array as a Python float."""

    @abc.abstractmethod
    def concat(self, arrays: list[Any], *, dim: int = 0) -> Any:
        """Concatenate a list of arrays along *dim*."""

    @abc.abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Element-wise conditional: return x where condition is True, else y."""

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def solve_triangular(self, R: Any, b: Any, *, upper: bool = True) -> Any:
        """Solve R @ x = b where R is upper (or lower) triangular.

        Parameters
        ----------
        R : 2-D array of shape (m, m), float64
        b : 2-D array of shape (m, 1), float64 — column vector.

        Returns
        -------
        x : 2-D array of shape (m, 1), float64.
        """

    @abc.abstractmethod
    def svdvals(self, A: Any) -> Any:
        """Singular values of *A* in descending order."""

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def quantile(self, a: Any, q: float) -> Any:
        """Return the q-th quantile (0 ≤ q ≤ 1) of 1-D array *a*."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Backend]] = {}
_DEFAULT_BACKEND_NAME: str = "torch"


def register_backend(name: str, cls: type[Backend]) -> None:
    """Register a Backend implementation under a name.  Called by each backend
    module at import time so callers do not need to import the concrete class."""
    _REGISTRY[name] = cls


def get_backend(name: Optional[str] = None) -> Backend:
    """Return a Backend instance for *name* (or the current default).

    Parameters
    ----------
    name : str or None
        One of "torch", "jax", or any registered name.  If None, returns the
        module-level default (initially "torch").

    Raises
    ------
    ValueError
        If *name* is not registered.
    ImportError
        If the required library for *name* is not installed.
    """
    resolved = name if name is not None else _DEFAULT_BACKEND_NAME
    if resolved not in _REGISTRY:
        # Attempt lazy import in case the module has not been imported yet.
        _try_lazy_import(resolved)
    if resolved not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {resolved!r}. "
            f"Registered backends: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[resolved]()


def set_default_backend(name: str) -> None:
    """Set the module-level default backend used by resolvent_solve when no
    ``backend`` kwarg is supplied.  Also triggers lazy import of the backend
    module to surface ImportError at set-time rather than call-time.

    Parameters
    ----------
    name : str
        One of "torch" or "jax".
    """
    global _DEFAULT_BACKEND_NAME
    # Trigger the import now so errors surface immediately.
    get_backend(name)
    _DEFAULT_BACKEND_NAME = name


def _try_lazy_import(name: str) -> None:
    """Attempt to import the named backend module, which self-registers."""
    if name == "torch":
        import grail.backends.torch_backend  # noqa: F401
    elif name == "jax":
        import grail.backends.jax_backend  # noqa: F401
    # Unknown names are silently ignored; the caller will raise ValueError.
