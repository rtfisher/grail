'Temperature-tendency network interface. The affine subclass supplies the controlled heating perturbation used in the climate demonstration. Jacobian-vector products differentiate the actual forward map.'

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from climate_rce.column import ColumnSpec


# ---------------------------------------------------------------------------
# SubgridNet
# ---------------------------------------------------------------------------

class SubgridNet(nn.Module):
    'Maps a non-dimensional T-only GRAIL state ``x`` (shape ``(n_state,)``)\n    to a non-dimensional subgrid tendency ``S^NN(x)`` (shape ``(n_state,)``).\n\n    Notes\n    -----\n    - Float64 input/output; float32 internal compute (cast at forward boundary).\n    - ``jvp(x, v)`` uses ``torch.func.jvp`` (Rule 2: autograd preferred).\n    - ``jvp_fd(x, v)`` uses central-FD at ``h=1e-4`` for cross-checks.\n    - Do NOT call ``jvp_fd`` from the main JVP path — it is a test fixture.'

    def __init__(
        self,
        spec: ColumnSpec,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.spec = spec
        n_in = spec.n_state
        n_out = spec.n_state

        # Build MLP: [n_in → h0 → h1 → ... → n_out], GELU activations.
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.network = nn.Sequential(*layers)

        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Input/output normalization arrays.  Filled by the loader; None means
        # identity (no normalization).  Shape (n_state,) float32.
        self.register_buffer("x_mean", torch.zeros(n_in, dtype=torch.float32))
        self.register_buffer("x_std",  torch.ones(n_in,  dtype=torch.float32))
        self.register_buffer("y_mean", torch.zeros(n_out, dtype=torch.float32))
        self.register_buffer("y_std",  torch.ones(n_out,  dtype=torch.float32))

        # Metadata set by the loader.
        self._weight_sha256: str | None = None
        self._is_smooth_fallback: bool = False

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the subgrid tendency.

        Parameters
        ----------
        x : Tensor
            ``(..., n_state)`` float64, non-dimensional T-only GRAIL state.

        Returns
        -------
        Tensor
            ``(..., n_state)`` float64, non-dimensional subgrid tendency.
        """
        # Preserve the checkpoint's default float32 behavior, while allowing
        # model.double() to define a smooth float64 evaluator consistently
        # for both prediction and differentiation.
        x32 = x.to(dtype=next(self.network.parameters()).dtype)
        # Normalize input.
        x_norm = (x32 - self.x_mean) / (self.x_std + 1e-8)
        # Forward through MLP.
        y_norm = self.network(x_norm)
        # De-normalize output.
        y32 = y_norm * (self.y_std + 1e-8) + self.y_mean
        # Cast output to float64 (repo convention: float64 at GRAIL boundary).
        return y32.to(dtype=torch.float64)

    # ------------------------------------------------------------------
    # predict (numpy, no grad)
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Numpy interface: compute subgrid tendency, no grad.

        Parameters
        ----------
        x : ndarray
            ``(n_state,)`` float64, non-dimensional T-only GRAIL state.

        Returns
        -------
        ndarray
            ``(n_state,)`` float64, non-dimensional subgrid tendency.
        """
        x_t = torch.tensor(x, dtype=torch.float64)
        with torch.no_grad():
            y_t = self(x_t)
        return y_t.numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # jvp (forward-mode autograd, Rule 2 default)
    # ------------------------------------------------------------------

    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute the JVP ``J·v`` via forward-mode autograd.

        **TASK-035 Rule 1 satisfied:** differentiates the *emulator's* own
        Jacobian ``∂S^NN/∂x``.  This is NOT the truth scheme's Jacobian.

        **Rule 2:** uses ``torch.func.jvp`` (preferred over central-FD when
        the NN is differentiable, per SPEC §4.3 and the callback contract).

        Parameters
        ----------
        x : Tensor
            ``(n_state,)`` float64, non-dimensional T-only state.
        v : Tensor
            ``(n_state,)`` float64, tangent vector.

        Returns
        -------
        Tensor
            ``(n_state,)`` float64, ``J·v``.
        """
        x_64 = x.to(dtype=torch.float64)
        v_64 = v.to(dtype=torch.float64)

        def _forward(x_in: torch.Tensor) -> torch.Tensor:
            return self(x_in)

        _, jvp_val = torch.func.jvp(_forward, (x_64,), (v_64,))
        return jvp_val.to(dtype=torch.float64)

    # ------------------------------------------------------------------
    # jvp_fd (central-FD cross-check; SPEC §4.3 / TASK-035 h=1e-4)
    # ------------------------------------------------------------------

    def jvp_fd(self, x: torch.Tensor, v: torch.Tensor, *, h: float = 1e-4) -> torch.Tensor:
        """Central-FD approximation to ``J·v``.

        Cross-check fixture for the ``test_jvp.py`` TASK-035 gate.  Do NOT
        use in the main JVP path.

        FD step ``h = 1e-4`` in non-dimensional units (SPEC §4.3 / TASK-035
        convention; same as NLTE and SCF testbeds).

        Parameters
        ----------
        x : Tensor
            ``(n_state,)`` float64, non-dimensional T-only state.
        v : Tensor
            ``(n_state,)`` float64, tangent direction (need not be unit).
        h : float
            Central-difference step (default 1e-4).

        Returns
        -------
        Tensor
            ``(n_state,)`` float64, central-FD approximation to ``J·v``.
        """
        x_64 = x.to(dtype=torch.float64)
        v_64 = v.to(dtype=torch.float64)
        with torch.no_grad():
            y_p = self(x_64 + h * v_64)   # (n_state,) float64
            y_m = self(x_64 - h * v_64)   # (n_state,) float64
        return (y_p - y_m) / (2.0 * h)
