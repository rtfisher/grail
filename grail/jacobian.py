"""Chain-rule composition of emulator and host JVPs. the technical reference §5.1.

The composed cycle map is T = T_SE ∘ T_RT.  Its differential at a base point
is, by the chain rule:

    DT · v  =  D(T_SE) · (D(T_RT) · v)

where:
  - D(T_RT) · v  is the host JVP: populations tangent → J_bar tangent
  - D(T_SE) · w  is the emulator JVP: J_bar tangent → populations tangent

The resolvent GMRES inverts the operator (I − DT), so the matrix-vector
product it needs is:

    v  ↦  v − DT · v  =  (I − DT) · v

Both callbacks are opaque: GRAIL knows nothing about whether the host or
emulator uses PyTorch autograd, NumPy finite differences, or a Fortran
adjoint.  Host-agnosticism is gated by I-callback-isolation.

The ``backend`` parameter is accepted by the module-level functions so that
the composing closures inherit the active backend for callback-output
conversion.  The jacobian module itself does not allocate arrays; conversion
is done in the wrapper ``_to_backend``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .backend import Backend, get_backend


def _to_backend(v, bk: Backend):
    """Ensure *v* is the backend's native type, converting via numpy if needed."""
    try:
        # If already the backend's native type, from_numpy will still work
        # via the numpy detour — but many backends accept their own type directly.
        arr = bk.to_numpy(v)
    except Exception:
        arr = np.asarray(v, dtype=np.float64)
    return bk.from_numpy(np.asarray(arr, dtype=np.float64))


def composed_jvp(
    emulator_jvp: Callable,
    host_jvp: Callable,
    backend: Optional[Backend] = None,
) -> Callable:
    """Return v -> emulator_jvp(host_jvp(v))  ==  DT · v.

    Chain rule: DT = D(T_SE) ∘ D(T_RT).  The host JVP is applied first
    (D(T_RT) acts on the population tangent v to produce a J_bar tangent),
    then the emulator JVP maps the J_bar tangent back to a population tangent.

    Parameters
    ----------
    emulator_jvp : callable v -> D(T_SE) @ v
        Linearized emulator (SE step); maps population-space tangents → same.
    host_jvp : callable v -> D(T_RT) @ v
        Linearized host RT step; maps population-space tangents → same.
        Called first in the chain (T_RT acts before T_SE in the cycle).
    backend : Backend or None
        Active backend for output-type normalization.  Defaults to the
        module-level default (torch).

    Returns
    -------
    Callable that maps a tangent vector v to DT · v.

    Notes
    -----
    Callback outputs may be NumPy arrays, PyTorch tensors, or JAX arrays;
    the composed JVP normalizes via the active backend so that GMRES sees
    a consistent native array type throughout.  SPEC_jax.md §4.2.
    """
    bk = backend if backend is not None else get_backend()

    def _composed(v) -> object:
        # D(T_RT) · v  — host JVP (populations tangent → J_bar tangent)
        w = host_jvp(v)
        w = _to_backend(w, bk)
        # D(T_SE) · w  — emulator JVP (SE tangent → populations tangent)
        out = emulator_jvp(w)
        return _to_backend(out, bk)

    return _composed


def resolvent_matvec(
    emulator_jvp: Callable,
    host_jvp: Callable,
    backend: Optional[Backend] = None,
) -> Callable:
    """Return v -> v - DT · v  ==  (I − DT) · v.

    This is the matrix-vector product GMRES uses to solve the resolvent
    equation (I − DT) δx* = δT.

    Parameters
    ----------
    emulator_jvp : callable v -> D(T_SE) @ v
    host_jvp : callable v -> D(T_RT) @ v
    backend : Backend or None
        Active backend.  Defaults to the module-level default (torch).

    Returns
    -------
    Callable implementing the (I − DT) matrix-vector product.
    """
    bk = backend if backend is not None else get_backend()
    dt_v = composed_jvp(emulator_jvp, host_jvp, backend=bk)

    def _resolvent_mv(v) -> object:
        v_bk = _to_backend(v, bk)
        return v_bk - dt_v(v_bk)

    return _resolvent_mv
