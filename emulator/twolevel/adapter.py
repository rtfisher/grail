"""Two-level cycle adapter: bridges APNN interface to the se_solver contract.

The canonical location for make_cycle_adapter as of SPEC_threelevel §M3a.
The original location (emulator/jvp.py) retains a re-export shim for
back-compat.

Design notes (see emulator/jvp.py for full context):
  cycle_emulator (atmosphere/twolevel/fixed_point.py) calls the se_solver with
  three positional tensor arguments (J_bar, T, n_e) of shape (N_τ,).  The APNN
  forward pass expects a packed (B, 3) input.  make_cycle_adapter bridges this
  interface mismatch by stacking the three (N_τ,) tensors into (N_τ, 3) before
  calling the model, then returning the (N_τ, 2) output unchanged.

  Inputs to the adapter are log₁₀-scaled before passing to the network,
  matching the training convention in training.py.  The adapter handles the
  safe log conversion (clamping to a minimum positive value to avoid log(0)).
"""

from __future__ import annotations

from typing import Callable

import torch

from ..apnn import APNN

# Minimum positive value for safe log10 conversion; below this we clamp.
# This should be smaller than any physically meaningful J_bar or n_e value.
_LOG_CLAMP_MIN = 1e-300


def make_cycle_adapter(model: APNN) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a callable that plugs the APNN into cycle_emulator's se_solver slot.

    The se_solver contract consumed by _cycle / ali_loop in
    atmosphere/twolevel/fixed_point.py is:

        se_solver(J_bar: Tensor[N_τ], T: Tensor[N_τ], n_e: Tensor[N_τ])
            -> Tensor[N_τ, 2]

    The APNN forward pass expects packed log-scaled inputs of shape (B, 3).
    This adapter:
        1. Stacks (J_bar, T, n_e) into shape (N_τ, 3).
        2. Applies log₁₀ scaling (matching the training convention in training.py).
        3. Runs model.forward() → (N_τ, 2).
        4. Returns the (N_τ, 2) float64 population tensor.

    Parameters
    ----------
    model : APNN
        Trained APNN model (should be in eval mode).

    Returns
    -------
    adapter : Callable[[Tensor, Tensor, Tensor], Tensor]
        The adapter callable satisfying the se_solver contract.

    Notes
    -----
    The adapter operates with no_grad for the forward pass.  If gradients
    through the adapter are required (e.g., for the GRAIL JVP), the caller
    should either:
      (a) call model.forward() directly with appropriate grad context, or
      (b) use emulator.jvp() on the packed-input form.
    """
    def adapter(
        J_bar: torch.Tensor,
        T: torch.Tensor,
        n_e: torch.Tensor,
    ) -> torch.Tensor:
        """Stack (J_bar, T, n_e) as log10-scaled batch, run APNN, return (N_τ, 2).

        Parameters
        ----------
        J_bar : Tensor of shape (N_τ,), float64
        T     : Tensor of shape (N_τ,), float64
        n_e   : Tensor of shape (N_τ,), float64

        Returns
        -------
        populations : Tensor of shape (N_τ, 2), float64
        """
        # Safe log₁₀ transform: clamp to avoid log(0).
        log_J = torch.log10(J_bar.clamp(min=_LOG_CLAMP_MIN))
        log_T = torch.log10(T.clamp(min=_LOG_CLAMP_MIN))
        log_ne = torch.log10(n_e.clamp(min=_LOG_CLAMP_MIN))

        # Pack into (N_τ, 3) float32 for the APNN.
        x = torch.stack([log_J, log_T, log_ne], dim=-1)  # (N_τ, 3)

        # Run the model.  Output is (N_τ, 2), float64 (cast in APNN.forward).
        return model(x)

    return adapter
