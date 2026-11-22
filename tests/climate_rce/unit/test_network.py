'Unit checks of the controlled smooth climate surrogate and its actual forward derivative.'

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from climate_rce.column import ColumnSpec
from climate_rce.emulator.network import SubgridNet
from climate_rce.emulator._smooth_fallback import make_smooth_fallback, _build_analytic_jacobian


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(n_z: int = 12) -> ColumnSpec:
    pressure = np.linspace(100.0, 1e5, n_z)
    return ColumnSpec(
        n_z=n_z,
        n_v=2,
        var_names=("T", "q"),
        pressure=pressure,
        var_scales=(300.0, 1e-2),
        n_state=n_z,
    )


def _make_x(spec: ColumnSpec, seed: int = 42) -> torch.Tensor:
    """Random in-distribution non-dim T-only state (values ≈ 0.8–0.95)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.8, 0.95, size=spec.n_state)
    return torch.tensor(x, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Smooth-fallback: basic properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_smooth_fallback_is_subgridnet():
    """make_smooth_fallback returns a SubgridNet instance."""
    spec = _make_spec()
    model = make_smooth_fallback(spec)
    assert isinstance(model, SubgridNet)
    assert model._is_smooth_fallback is True


@pytest.mark.unit
def test_smooth_fallback_forward_finite():
    """SubgridNet.forward returns a finite (n_state,) float64 tensor."""
    spec = _make_spec()
    model = make_smooth_fallback(spec)
    x = _make_x(spec)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (spec.n_state,), f"shape mismatch: {y.shape}"
    assert y.dtype == torch.float64, f"expected float64, got {y.dtype}"
    assert torch.isfinite(y).all(), "forward output contains non-finite values"


@pytest.mark.unit
def test_smooth_fallback_predict_numpy():
    """SubgridNet.predict returns float64 numpy array."""
    spec = _make_spec()
    model = make_smooth_fallback(spec)
    x = _make_x(spec).numpy()
    y = model.predict(x)
    assert isinstance(y, np.ndarray), "predict must return ndarray"
    assert y.shape == (spec.n_state,)
    assert y.dtype == np.float64


@pytest.mark.unit
def test_smooth_fallback_jvp_equals_analytic_jacobian():
    """Smooth-fallback jvp(x, v) equals the analytic affine Jacobian A @ v.

    The smooth fallback is a linear-in-state surrogate S^smooth(x) = A·x + b.
    Its JVP is therefore A·v exactly (independent of x).  This tests the
    load-bearing gate: "smooth-fallback: assert jvp equals the analytic
    affine Jacobian" (SPEC §6 P3 test_network).
    """
    spec = _make_spec(n_z=12)
    model = make_smooth_fallback(spec, seed=42)

    # Extract the analytic Jacobian A (n_state, n_state) float64
    A = _build_analytic_jacobian(model)   # (n_state, n_state) float64

    rng = np.random.default_rng(0)
    for trial in range(8):
        x = torch.tensor(rng.uniform(0.7, 1.0, spec.n_state), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(spec.n_state), dtype=torch.float64)

        jvp_autograd = model.jvp(x, v)                       # (n_state,) float64
        jvp_analytic = torch.tensor(A @ v.numpy(), dtype=torch.float64)

        # The smooth fallback uses float32 internal compute, so there is a
        # float32-cast rounding that introduces a small error.
        # Tolerance: float32 precision ≈ 1.2e-7 relative.
        rel_err = (torch.norm(jvp_autograd - jvp_analytic) /
                   (torch.norm(jvp_analytic) + 1e-15)).item()
        assert rel_err < 1e-5, (
            f"trial={trial}: jvp_autograd does not match analytic Jacobian: "
            f"rel_err={rel_err:.3e} (expected < 1e-5)"
        )


@pytest.mark.unit
def test_smooth_fallback_jvp_fd_agrees_with_autograd():
    "SubgridNet.jvp (autograd) agrees with jvp_fd (central-FD).\n\n    The TASK-035 cross-check (SPEC §6 P3 test_jvp): 'SubgridNet.jvp\n    (forward-mode autograd) matches central-FD (h=1e-4 non-dim) to 1e-5\n    relative' is exercised in test_jvp.py against the cycle-level JVPs."
    spec = _make_spec(n_z=12)
    model = make_smooth_fallback(spec, seed=42)

    rng = np.random.default_rng(1)
    for trial in range(4):
        x = torch.tensor(rng.uniform(0.7, 1.0, spec.n_state), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(spec.n_state), dtype=torch.float64)

        jvp_ag = model.jvp(x, v)
        jvp_fd = model.jvp_fd(x, v, h=1e-4)

        rel_err = (torch.norm(jvp_ag - jvp_fd) /
                   (torch.norm(jvp_fd) + 1e-15)).item()
        assert rel_err < 5e-3, (
            f"trial={trial}: autograd vs FD JVP mismatch: "
            f"rel_err={rel_err:.3e} (expected < 5e-3 — float32 FD floor limit)"
        )


@pytest.mark.unit
def test_smooth_fallback_spec_n_state():
    """SubgridNet.spec.n_state matches the ColumnSpec n_z (T-only seam)."""
    for n_z in (8, 12, 30):
        spec = _make_spec(n_z=n_z)
        model = make_smooth_fallback(spec)
        assert model.spec.n_state == n_z, (
            f"spec.n_state={model.spec.n_state} != n_z={n_z}"
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
