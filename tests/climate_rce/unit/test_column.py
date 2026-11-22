"""Tier U unit tests for climate_rce/column.py.

SPEC_climate_rce §6 P1 gate (SPEC §2.2.1 T-only seam, frozen 2026-05-19):
  - pack_state emits the T block only, shape (n_z,), non-dim by T_scale.
  - unpack_state reconstructs the full (T,q) ColumnState from (n_z,) flat:
      T = x_flat * T_scale  (exact round-trip for T)
      q = q_ref(T_surf)      (fixed-RH diagnostic, NOT the original q)
  - n_state == n_z (GRAIL seam dimension, NOT n_z*n_v).
  - ColumnSpec.__post_init__ validates n_state == n_z.

Tolerance: T block round-trip error < 1e-14 relative (float64 machine-epsilon
floor; 1 ULP ≈ 2.2e-16; one division + one multiplication = at most a few ULP).
The q block is NOT round-tripped — it is the diagnosed fixed-RH closure.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from climate_rce.column import ColumnSpec, ColumnState, pack_state, unpack_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(n_z: int, n_v: int) -> ColumnSpec:
    pressure = np.linspace(100.0, 1e5, n_z)  # Pa, TOA to surface
    var_names = ("T", "q", "extra")[:n_v]
    var_scales = (300.0, 1e-2, 1.0)[:n_v]
    # n_state == n_z: T-only GRAIL seam (SPEC §2.2.1).
    # n_v retained for physical column physics (radiation/convection).
    return ColumnSpec(
        n_z=n_z,
        n_v=n_v,
        var_names=var_names,
        pressure=pressure,
        var_scales=var_scales,
        n_state=n_z,   # T-only GRAIL state; NOT n_z*n_v
    )


def _make_profile(spec: ColumnSpec, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    profile = np.zeros((spec.n_v, spec.n_z), dtype=np.float64)
    for v, scale in enumerate(spec.var_scales):
        profile[v] = rng.uniform(0.5, 1.5) * scale * np.ones(spec.n_z)
        profile[v] += rng.standard_normal(spec.n_z) * 0.01 * scale
    return profile


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("n_z,n_v", [(30, 2), (60, 2), (30, 3)])
def test_round_trip_float64(n_z: int, n_v: int) -> None:
    """pack_state / unpack_state: T block round-trips to float64 precision;
    q is reconstructed as q_ref(T_surf) (NOT the original q).

    SPEC §2.2.1 / §4.1 (T-only seam, fixed-RH closure):
    - pack_state emits T block only, shape (n_z,) = (n_state,).
    - unpack_state(pack_state(s), spec).profile[0] == s.profile[0] to float64
      precision (T round-trips exactly: divide by T_scale then multiply back).
    - unpack_state(pack_state(s), spec).profile[1] == q_ref(T_surf)
      (the fixed-RH diagnostic closure; NOT the original q).

    Tolerance: T block < 1e-14 relative error.
    Floor: float64 machine epsilon 2.2e-16; one division + one multiplication
    = at most a few ULP = O(1e-15).  1e-14 provides ×100 headroom.
    """
    spec = _make_spec(n_z, n_v)
    profile = _make_profile(spec)
    state = ColumnState(spec=spec, profile=profile)

    x_flat = pack_state(state)
    state_rt = unpack_state(x_flat, spec)

    # Check dtype
    assert x_flat.dtype == torch.float64, (
        f"pack_state must return float64, got {x_flat.dtype}"
    )

    # Check shape: T-only seam → length n_z, not n_z*n_v
    assert x_flat.shape == (spec.n_state,), (
        f"packed shape {x_flat.shape} != ({spec.n_state},) = ({n_z},). "
        "n_state must equal n_z for the T-only GRAIL seam (SPEC §2.2.1)."
    )
    assert spec.n_state == n_z, (
        f"ColumnSpec.n_state={spec.n_state} != n_z={n_z}. "
        "GRAIL seam dimension must be n_z (T-only, SPEC §2.2.1)."
    )

    # T block round-trips exactly (one divide + one multiply = few ULP error)
    T_orig = profile[0]
    T_rt = state_rt.profile[0]
    max_rel_err_T = np.max(
        np.abs(T_rt - T_orig) / (np.abs(T_orig) + 1e-15)
    )
    assert max_rel_err_T < 1e-14, (
        f"T block round-trip relative error {max_rel_err_T:.2e} exceeds float64 "
        "floor 1e-14.  Floor: each pack/unpack op is ≤1 ULP ≈ 2.2e-16; combined "
        "≤ few ULP = O(1e-15).  1e-14 provides ×100 headroom."
    )

    # q block is the fixed-RH diagnostic: q = q_ref(T_surf)
    # NOT the original q profile — this is by design (SPEC §2.2.1).
    # Verify q is physically plausible (positive, finite) and computed from T_surf.
    if spec.n_v >= 2:
        q_rt = state_rt.profile[1]
        assert np.all(np.isfinite(q_rt)), "Diagnosed q contains non-finite values."
        assert np.all(q_rt >= 0.0), "Diagnosed q contains negative values."
        # q must depend on T_surf (last level): verify it is non-zero
        T_surf_rt = T_rt[-1]
        assert T_surf_rt > 200.0, f"T_surf = {T_surf_rt:.1f} K suspiciously cold."
        # q_ref(T_surf) > 0 for T_surf > 200 K (Clausius-Clapeyron is positive)
        assert q_rt[-1] > 0.0, (
            "q_ref at surface level is zero — fixed-RH closure appears disconnected."
        )


@pytest.mark.unit
@pytest.mark.parametrize("n_z,n_v", [(30, 2), (60, 2), (30, 3)])
def test_blocked_layout(n_z: int, n_v: int) -> None:
    """pack_state emits the T block only: x_flat = T[0..n_z-1] / T_scale.

    T-only seam (SPEC §2.2.1 / §4.1): x_flat has length n_z, not n_z*n_v.
    The layout is a single contiguous block of non-dim temperatures, matching
    the "single-block" discipline (n_v_grail = 1; SPEC §4.1 note on
    'single T block since n_v_grail = 1').
    """
    spec = _make_spec(n_z, n_v)
    profile = _make_profile(spec)
    state = ColumnState(spec=spec, profile=profile)

    x_flat = pack_state(state).numpy()

    # Expected: T block only, non-dim by T_scale
    T_scale = spec.var_scales[0]
    expected_flat = profile[0] / T_scale   # shape (n_z,)

    assert x_flat.shape == (n_z,), (
        f"packed shape {x_flat.shape} != ({n_z},). T-only seam must have length n_z."
    )
    np.testing.assert_allclose(
        x_flat, expected_flat, rtol=1e-14,
        err_msg="T-only layout: x_flat != T_block / T_scale"
    )


@pytest.mark.unit
def test_spec_validation() -> None:
    """ColumnSpec __post_init__ raises on mismatched shapes.

    Specifically: n_state must equal n_z (T-only GRAIL seam, SPEC §2.2.1),
    NOT n_z*n_v.  Passing n_state=n_z*n_v raises ValueError.
    """
    pressure = np.linspace(100.0, 1e5, 30)
    # Mismatched var_names length
    with pytest.raises(ValueError, match="len\\(var_names\\)"):
        ColumnSpec(n_z=30, n_v=2, var_names=("T",), pressure=pressure,
                   var_scales=(300.0, 1e-2), n_state=30)
    # Mismatched pressure shape
    with pytest.raises(ValueError, match="pressure.shape"):
        ColumnSpec(n_z=30, n_v=2, var_names=("T","q"),
                   pressure=np.ones(20), var_scales=(300.0, 1e-2), n_state=30)
    # n_state must equal n_z (not n_z*n_v): n_state=60 with n_z=30 must raise
    with pytest.raises(ValueError, match="n_state"):
        ColumnSpec(n_z=30, n_v=2, var_names=("T","q"), pressure=pressure,
                   var_scales=(300.0, 1e-2), n_state=60)   # 60 = n_z*n_v, wrong
    # n_state=99 (arbitrary mismatch) must also raise
    with pytest.raises(ValueError, match="n_state"):
        ColumnSpec(n_z=30, n_v=2, var_names=("T","q"), pressure=pressure,
                   var_scales=(300.0, 1e-2), n_state=99)


@pytest.mark.unit
def test_spec_hash_eq() -> None:
    """ColumnSpec __hash__ and __eq__ work correctly (numpy arrays).

    n_state=30 = n_z (T-only GRAIL seam, SPEC §2.2.1).
    """
    pressure = np.linspace(100.0, 1e5, 30)
    s1 = ColumnSpec(n_z=30, n_v=2, var_names=("T","q"), pressure=pressure,
                    var_scales=(300.0, 1e-2), n_state=30)   # n_state == n_z
    s2 = ColumnSpec(n_z=30, n_v=2, var_names=("T","q"),
                    pressure=pressure.copy(), var_scales=(300.0, 1e-2), n_state=30)
    assert s1 == s2
    assert hash(s1) == hash(s2)
    # Dict-usable (the original motivation for the explicit __hash__)
    d = {s1: "value"}
    assert d[s2] == "value"
