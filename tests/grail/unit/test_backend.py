"""Unit tests for grail.backend — TorchBackend method-by-method parity.

Each test verifies a single Backend method against a numpy reference
implementation.  This decouples backend correctness from GMRES / PCA /
conformal algorithm correctness: a failure here isolates to the dispatch
layer, not the algorithm.

SPEC_jax.md §7 (P1 acceptance).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from grail.backend import get_backend
from grail.backends.torch_backend import TorchBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bk() -> TorchBackend:
    """Return the registered TorchBackend instance."""
    b = get_backend("torch")
    assert isinstance(b, TorchBackend)
    return b


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Tier U: from_numpy / to_numpy round-trip
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_from_numpy_basic(bk: TorchBackend) -> None:
    """from_numpy converts a float64 ndarray to torch.float64 Tensor."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    t = bk.from_numpy(a)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float64
    np.testing.assert_allclose(t.numpy(), a)


@pytest.mark.unit
def test_U_backend_torch_from_numpy_upcasts_float32(bk: TorchBackend) -> None:
    """from_numpy always upcasts to float64, even for float32 input."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    t = bk.from_numpy(a)
    assert t.dtype == torch.float64


@pytest.mark.unit
def test_U_backend_torch_to_numpy_basic(bk: TorchBackend) -> None:
    """to_numpy converts a torch.float64 tensor to a float64 ndarray."""
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    a = bk.to_numpy(t)
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.float64
    np.testing.assert_allclose(a, np.array([1.0, 2.0, 3.0]))


@pytest.mark.unit
def test_U_backend_torch_roundtrip(bk: TorchBackend, rng: np.random.Generator) -> None:
    """from_numpy(to_numpy(t)) is elementwise-equal to t."""
    t_orig = torch.randn(50, dtype=torch.float64)
    a = bk.to_numpy(t_orig)
    t_back = bk.from_numpy(a)
    np.testing.assert_array_equal(t_orig.numpy(), bk.to_numpy(t_back))


# ---------------------------------------------------------------------------
# Tier U: zeros
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_zeros_1d(bk: TorchBackend) -> None:
    """zeros(n) returns a float64 tensor of shape (n,) filled with 0."""
    z = bk.zeros(10)
    assert isinstance(z, torch.Tensor)
    assert z.dtype == torch.float64
    assert z.shape == torch.Size([10])
    assert z.sum().item() == 0.0


@pytest.mark.unit
def test_U_backend_torch_zeros_2d(bk: TorchBackend) -> None:
    """zeros(m, n) returns a float64 tensor of shape (m, n)."""
    z = bk.zeros(4, 5)
    assert z.shape == torch.Size([4, 5])
    assert z.sum().item() == 0.0


# ---------------------------------------------------------------------------
# Tier U: clone
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_clone_no_alias(bk: TorchBackend) -> None:
    """clone returns a new tensor that does not alias the original."""
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    b = bk.clone(a)
    # Mutate b and confirm a is unchanged.
    b_np = bk.to_numpy(b)
    b_np[0] = 99.0
    assert a[0].item() == 1.0, "clone should not alias the original"


# ---------------------------------------------------------------------------
# Tier U: dot
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_dot_vs_numpy(bk: TorchBackend, rng: np.random.Generator) -> None:
    """dot(a, b) matches numpy inner product."""
    a_np = rng.standard_normal(30)
    b_np = rng.standard_normal(30)
    result = bk.dot(bk.from_numpy(a_np), bk.from_numpy(b_np))
    expected = np.dot(a_np, b_np)
    assert abs(float(bk.to_numpy(result).flat[0]) - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tier U: norm
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_norm_vs_numpy(bk: TorchBackend, rng: np.random.Generator) -> None:
    """norm(v) matches numpy.linalg.norm."""
    v_np = rng.standard_normal(50)
    result = bk.norm(bk.from_numpy(v_np))
    expected = np.linalg.norm(v_np)
    assert isinstance(result, float)
    assert abs(result - expected) < 1e-12


@pytest.mark.unit
def test_U_backend_torch_norm_zero(bk: TorchBackend) -> None:
    """norm of a zero vector is 0.0."""
    assert bk.norm(bk.zeros(20)) == 0.0


# ---------------------------------------------------------------------------
# Tier U: concat
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_concat_1d(bk: TorchBackend) -> None:
    """concat([a, b]) produces the concatenation in the expected order."""
    a = bk.from_numpy(np.array([1.0, 2.0]))
    b = bk.from_numpy(np.array([3.0, 4.0]))
    c = bk.concat([a, b])
    np.testing.assert_array_equal(bk.to_numpy(c), np.array([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.unit
def test_U_backend_torch_concat_axis1(bk: TorchBackend) -> None:
    """concat along dim=1 stacks columns."""
    a = bk.from_numpy(np.ones((3, 2)))
    b = bk.from_numpy(np.zeros((3, 2)))
    c = bk.concat([a, b], dim=1)
    assert c.shape == torch.Size([3, 4])


# ---------------------------------------------------------------------------
# Tier U: where
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_where_vs_numpy(bk: TorchBackend) -> None:
    """where(cond, x, y) matches numpy.where."""
    cond_np = np.array([True, False, True, False])
    x_np = np.array([1.0, 2.0, 3.0, 4.0])
    y_np = np.array([10.0, 20.0, 30.0, 40.0])
    result = bk.where(
        bk.from_numpy(cond_np.astype(np.float64)) > 0.5,
        bk.from_numpy(x_np),
        bk.from_numpy(y_np),
    )
    expected = np.where(cond_np, x_np, y_np)
    np.testing.assert_array_equal(bk.to_numpy(result), expected)


# ---------------------------------------------------------------------------
# Tier U: solve_triangular
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_solve_tri_vs_numpy(bk: TorchBackend, rng: np.random.Generator) -> None:
    """solve_triangular(R, b, upper=True) matches numpy.linalg.solve on upper triangular R."""
    m = 8
    A = rng.standard_normal((m, m))
    R_np = np.triu(A) + m * np.eye(m)  # upper triangular, well-conditioned
    b_np = rng.standard_normal((m, 1))
    result = bk.solve_triangular(
        bk.from_numpy(R_np),
        bk.from_numpy(b_np),
        upper=True,
    )
    expected = np.linalg.solve(R_np, b_np)
    np.testing.assert_allclose(bk.to_numpy(result), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_U_backend_torch_solve_tri_identity(bk: TorchBackend) -> None:
    """solve_triangular(I, b, upper=True) = b."""
    n = 5
    R = bk.from_numpy(np.eye(n))
    b_np = np.arange(1.0, n + 1.0).reshape(-1, 1)
    b = bk.from_numpy(b_np)
    x = bk.solve_triangular(R, b, upper=True)
    np.testing.assert_allclose(bk.to_numpy(x), b_np, atol=1e-15)


# ---------------------------------------------------------------------------
# Tier U: svdvals
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_svdvals_vs_numpy(bk: TorchBackend, rng: np.random.Generator) -> None:
    """svdvals(A) matches numpy.linalg.svd singular values (descending)."""
    A_np = rng.standard_normal((6, 4))
    sv_bk = bk.svdvals(bk.from_numpy(A_np))
    sv_ref = np.linalg.svd(A_np, compute_uv=False)  # numpy returns descending
    np.testing.assert_allclose(bk.to_numpy(sv_bk), sv_ref, rtol=1e-11, atol=1e-14)


@pytest.mark.unit
def test_U_backend_torch_svdvals_descending(bk: TorchBackend, rng: np.random.Generator) -> None:
    """svdvals output is in descending order."""
    A_np = rng.standard_normal((10, 7))
    sv = bk.to_numpy(bk.svdvals(bk.from_numpy(A_np)))
    assert all(sv[i] >= sv[i + 1] - 1e-14 for i in range(len(sv) - 1))


# ---------------------------------------------------------------------------
# Tier U: quantile
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_torch_quantile_vs_numpy(bk: TorchBackend, rng: np.random.Generator) -> None:
    """quantile(a, q) matches numpy.quantile for several q values."""
    a_np = rng.standard_normal(100)
    for q in [0.0, 0.1, 0.5, 0.9, 1.0]:
        result = float(bk.to_numpy(bk.quantile(bk.from_numpy(a_np), q)).flat[0])
        expected = float(np.quantile(a_np, q))
        assert abs(result - expected) < 1e-12, (
            f"quantile mismatch at q={q}: got {result}, expected {expected}"
        )


@pytest.mark.unit
def test_U_backend_torch_quantile_sorted(bk: TorchBackend) -> None:
    """quantile(0.0) = min, quantile(1.0) = max for sorted array."""
    a_np = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    q0 = float(bk.to_numpy(bk.quantile(bk.from_numpy(a_np), 0.0)).flat[0])
    q1 = float(bk.to_numpy(bk.quantile(bk.from_numpy(a_np), 1.0)).flat[0])
    assert q0 == pytest.approx(1.0)
    assert q1 == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Tier U: registry / factory
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_backend_registry_torch() -> None:
    """get_backend('torch') returns a TorchBackend instance."""
    b = get_backend("torch")
    assert isinstance(b, TorchBackend)


@pytest.mark.unit
def test_U_backend_registry_unknown() -> None:
    """get_backend with an unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent_backend_xyz")


@pytest.mark.unit
def test_U_backend_set_default(rng: np.random.Generator) -> None:
    """set_default_backend changes the module default and can be reset."""
    from grail.backend import set_default_backend

    # Confirm torch is the starting default.
    b0 = get_backend()
    assert isinstance(b0, TorchBackend)

    # Setting to "torch" explicitly is idempotent.
    set_default_backend("torch")
    b1 = get_backend()
    assert isinstance(b1, TorchBackend)
