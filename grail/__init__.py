"""GRAIL: Green's-function Resolvent for AI-accelerated Iteration Libraries.

Public API is one function and one return type, plus backend control.
the technical reference §5.2; SPEC_jax.md §3.
"""

from .api import ResolventResult, resolvent_solve
from .backend import get_backend, set_default_backend

# Ensure TorchBackend is registered at import time.
from . import backends  # noqa: F401

__all__ = [
    "ResolventResult",
    "resolvent_solve",
    "get_backend",
    "set_default_backend",
]
