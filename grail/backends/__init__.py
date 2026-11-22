"""Concrete Backend implementations for GRAIL.

Import this package to register all built-in backends.  Individual
backend modules register themselves via ``grail.backend.register_backend``
at import time, so callers need only ``import grail.backends`` (or let
``get_backend`` do the lazy import automatically).
"""

from . import torch_backend  # noqa: F401 — registers TorchBackend

# JaxBackend is registered lazily (imported on demand) to avoid a hard
# JAX import when JAX is not installed.  It will be imported automatically
# by get_backend("jax") when needed.
