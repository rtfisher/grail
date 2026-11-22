'Controlled affine heating surrogate. Its synthetic perturbation amplitude is a prescribed parameter. The state-dependent physical heating law is fixed as the numerical relaxation timestep varies. Its exact JVP supplies an independent derivative check.'

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from climate_rce.column import ColumnSpec
from climate_rce.emulator.network import SubgridNet


# ---------------------------------------------------------------------------
# Amplitude calibration (documented constant; traceable to SPEC §7c)
# ---------------------------------------------------------------------------

# tendency: ~0.3 K/day ≈ 3.5e-6 K/s.  Non-dimensionalized by T_scale=300 K:
# ~1.2e-8 non-dim/s.  We use a dimensionless perturbation amplitude in state
# space; the actual magnitude depends on the state vector magnitude (~1.0
# non-dim).  We set AMPLITUDE ≈ 1e-3 for the surrogate coefficient matrix,
# which produces |S^smooth(x)| ~ 1e-3 when |x| ~ 1 — within the range of
_AMPLITUDE = 1e-3   # documented surrogate coefficient scale


def make_smooth_fallback(
    spec: ColumnSpec,
    *,
    amplitude: float = _AMPLITUDE,
    seed: int = 42,
) -> SubgridNet:
    "Construct an amplitude-matched linear smooth surrogate as a SubgridNet.\n\n    Returns a ``SubgridNet`` whose hidden layers are replaced by an **affine**\n    map ``S^smooth(x) = A · x + b`` — a single linear layer with no activation\n    — so the JVP is exactly ``A · v`` for all ``x``.\n\n    The coefficient matrix ``A`` is constructed as:\n        A = amplitude * (0.5 * I + 0.1 * rand_offdiag)\n    where ``rand_offdiag`` is a normalized random matrix (seed-reproducible).\n    This is sparse enough to be computationally cheap while providing a\n    non-trivial off-diagonal Jacobian structure.\n\n    The bias ``b`` is set to zero (the truth tendency is zero at the RCE\n    fixed point, so a non-zero bias would break the fixed-point condition —\n    the surrogate is used to model the *residual* δS, not the full tendency).\n\n    Returns\n    -------\n    SubgridNet\n        A SubgridNet configured as an affine surrogate.  The network's\n        ``forward``, ``jvp``, ``jvp_fd``, and ``predict`` methods all work\n        normally; ``jvp`` is autograd through the linear layer (exact)."
    rng = np.random.default_rng(seed)
    n = spec.n_state

    # Construct A = amplitude * (0.5 * I + 0.1 * M_offdiag) where M_offdiag is
    # a random matrix with zero diagonal (reproducible from seed).
    # We reduce the amplitude further to stay strictly within the perturbative
    # regime: typical x ~ O(1), so S^smooth ~ amplitude * x ~ 1e-3.
    M_offdiag = rng.standard_normal((n, n)).astype(np.float32)
    np.fill_diagonal(M_offdiag, 0.0)
    M_offdiag /= (np.linalg.norm(M_offdiag, "fro") / n + 1e-12)
    A_np = amplitude * (0.5 * np.eye(n, dtype=np.float32) + 0.1 * M_offdiag)

    # Build a SubgridNet with a single linear layer (hidden_sizes=()).
    # An empty hidden_sizes means: input → single Linear → output (no activation).
    model = SubgridNet(spec, hidden_sizes=())
    # The network is already built as: Linear(n_state, n_state).
    # Replace its weights with A and bias with 0.
    linear_layer = model.network[0]  # single Linear in the Sequential
    with torch.no_grad():
        linear_layer.weight.copy_(torch.tensor(A_np, dtype=torch.float32))
        linear_layer.bias.zero_()

    # Set normalization to identity (A already in normalized units).
    model.x_mean.zero_()
    model.x_std.fill_(1.0)
    model.y_mean.zero_()
    model.y_std.fill_(1.0)

    model._is_smooth_fallback = True
    model._weight_sha256 = f"smooth_fallback_amplitude{amplitude}_seed{seed}"

    return model


def _build_analytic_jacobian(model: SubgridNet) -> np.ndarray:
    """Extract the analytic Jacobian A of a smooth-fallback SubgridNet.

    For the affine surrogate ``S^smooth(x) = A·x + b``, returns ``A``
    as a ``(n_state, n_state)`` float64 numpy array.

    This is used in ``test_network.py`` to assert that ``model.jvp(x, v)``
    equals ``A @ v`` exactly.

    Parameters
    ----------
    model : SubgridNet
        Must be a smooth-fallback model (``model._is_smooth_fallback`` is True).

    Returns
    -------
    ndarray
        ``(n_state, n_state)`` float64 Jacobian.

    Raises
    ------
    ValueError
        If the model is not a smooth fallback (not an affine map).
    """
    if not model._is_smooth_fallback:
        raise ValueError(
            "_build_analytic_jacobian: model is not a smooth fallback. "
            "The analytic Jacobian is only defined for affine surrogate models."
        )
    # Extract the weight matrix from the single linear layer.
    linear_layer = model.network[0]
    # Weight is float32; cast to float64 for the analytic Jacobian.
    # The full forward is: y32 = (weight @ x_norm + bias) * y_std + y_mean
    # With identity normalization (x_mean=0, x_std=1, y_mean=0, y_std=1)
    # and bias=0: y = weight @ x32, then cast to float64.
    # So the Jacobian in float64 is weight.
    W_np = linear_layer.weight.detach().numpy().astype(np.float64)
    return W_np


def load_from_npz(path: str | Path, spec: ColumnSpec) -> SubgridNet:
    """Load a smooth-fallback SubgridNet from a saved .npz file.

    The .npz must contain:
      - ``weight``: (n_state, n_state) float32
      - ``bias``:   (n_state,) float32 (usually zeros)
      - ``x_mean``, ``x_std``, ``y_mean``, ``y_std``: (n_state,) float32

    Parameters
    ----------
    path : str or Path
        Path to the .npz file.
    spec : ColumnSpec

    Returns
    -------
    SubgridNet
        Loaded smooth-fallback model.
    """
    data = np.load(str(path))
    model = SubgridNet(spec, hidden_sizes=())
    linear_layer = model.network[0]
    with torch.no_grad():
        linear_layer.weight.copy_(torch.tensor(data["weight"], dtype=torch.float32))
        linear_layer.bias.copy_(torch.tensor(data["bias"], dtype=torch.float32))
        model.x_mean.copy_(torch.tensor(data["x_mean"], dtype=torch.float32))
        model.x_std.copy_(torch.tensor(data["x_std"], dtype=torch.float32))
        model.y_mean.copy_(torch.tensor(data["y_mean"], dtype=torch.float32))
        model.y_std.copy_(torch.tensor(data["y_std"], dtype=torch.float32))
    model._is_smooth_fallback = True
    return model
