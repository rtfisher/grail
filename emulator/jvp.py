"""Autograd JVP wrapper consumed by GRAIL via callback.  the technical reference §4.4.

Lives outside the model class to keep APNN clean of GRAIL-specific concerns.

Design notes
------------
torch.func.jvp vs torch.autograd.functional.jvp:
  Both compute (f(x), J(x) @ v).  torch.func.jvp uses the forward-mode
  functional transform API (vmap/functorch style) and is preferred for
  performance in batch JVP computations.  torch.autograd.functional.jvp is
  the legacy API.  U-jvp-01 requires our wrapper to match the legacy API to
  machine precision — which it does because both compute the same mathematical
  quantity via forward-mode AD.

Dtype consistency for torch.func.jvp:
  torch.func.jvp propagates dual-number tangents through every operation and
  requires all tensors (parameters and inputs) to share the same dtype.
  APNN.forward() casts inputs to float32 internally (for normal inference),
  but that cast breaks the dual-number graph when called from a float64
  functional transform.

  Solution: the jvp() wrapper here uses torch.func.functional_call with
  float64-promoted parameter copies so that the model runs purely in float64
  during the JVP computation.  This avoids mutating the model's stored
  parameters.  The APNN.forward() method is NOT called through this path;
  instead we build a pure functional version that:
    1. Casts all parameters to float64.
    2. Runs the network layers manually via functional_call.
    3. Applies softmax.
  This matches the model's architecture exactly.

make_cycle_adapter seam:
  cycle_emulator (atmosphere/fixed_point.py) calls the se_solver with three
  positional tensor arguments (J_bar, T, n_e) of shape (N_τ,).  The APNN
  forward pass expects a packed (B, 3) input.  make_cycle_adapter bridges
  this interface mismatch by stacking the three (N_τ,) tensors into (N_τ, 3)
  before calling the model, then returning the (N_τ, 2) output unchanged.

  Inputs to the adapter are log₁₀-scaled before passing to the network,
  matching the training convention in training.py.  The adapter handles the
  safe log conversion (clamping to a minimum positive value to avoid log(0)).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .apnn import APNN

# Minimum positive value for safe log10 conversion; below this we clamp.
# This should be smaller than any physically meaningful J_bar or n_e value.
_LOG_CLAMP_MIN = 1e-300


def _model_forward_f64(
    model: APNN,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run model.net as a pure function with float64 parameters.

    This is used by jvp() to ensure consistent dtype throughout the
    forward-mode AD computation.  We cast all module parameters to float64
    using torch.func.functional_call on model.net, which does NOT mutate the
    stored parameters.

    Parameters
    ----------
    model : APNN
    x : Tensor, float64

    Returns
    -------
    out : Tensor, float64 — softmax output
    """
    # Cast all parameters and buffers of model.net to float64.
    # model.named_parameters() returns names like "net.0.weight";
    # for functional_call on model.net, we strip the "net." prefix.
    net = model.net
    params = {
        name.removeprefix("net."): param.to(dtype=torch.float64)
        for name, param in model.named_parameters()
        if name.startswith("net.")
    }
    buffers = {
        name.removeprefix("net."): buf.to(dtype=torch.float64)
        for name, buf in model.named_buffers()
        if name.startswith("net.")
    }
    # functional_call runs net's forward() with the swapped float64 params,
    # bypassing APNN.forward()'s float32 cast.
    out = torch.func.functional_call(net, {**params, **buffers}, (x,))
    return out


def jvp(
    model: APNN,
    x: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (model(x), J(x) @ v) using torch.func.jvp.

    Signature is fixed by the technical reference §4.4.  Must match
    torch.autograd.functional.jvp to machine precision (gated by U-jvp-01).

    Parameters
    ----------
    model : APNN
        Trained (or untrained) model.  Must be in eval mode for consistent
        behaviour (no dropout, batchnorm, etc.).
    x : Tensor of shape (B, n_in), float64
        Input point at which to evaluate f and J.  Must be float64 so that
        the JVP computation is consistent (see module docstring).
    v : Tensor of shape (B, n_in), float64
        Tangent vector at x.

    Returns
    -------
    (f_x, jvp_val) : tuple[Tensor, Tensor], both float64
        f_x      — model(x), shape (B, n_out).
        jvp_val  — J(x) @ v (Jacobian-vector product), shape (B, n_out).

    Notes
    -----
    torch.func.jvp uses forward-mode AD.  The model parameters are treated as
    constants (not differentiated); only x is the differentiation variable.
    We use _model_forward_f64 to ensure consistent float64 dtype throughout.
    """
    # Ensure float64 for the JVP computation.
    x_f64 = x.to(dtype=torch.float64)
    v_f64 = v.to(dtype=torch.float64)

    def _f(x_: torch.Tensor) -> torch.Tensor:
        return _model_forward_f64(model, x_)

    f_x, jvp_val = torch.func.jvp(_f, (x_f64,), (v_f64,))  # type: ignore[misc]
    return f_x, jvp_val


# Back-compat re-export: make_cycle_adapter moved to emulator/twolevel/adapter.py
# as part of SPEC_threelevel §M3a.  All existing imports of the form
# ``from emulator.jvp import make_cycle_adapter`` continue to work.
from .twolevel.adapter import make_cycle_adapter  # noqa: E402, F401
