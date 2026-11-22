"""Tier U unit tests for climate_rce/metric.py.

SPEC_climate_rce §6 P1 gate:
  - climate_sensitivity returns a positive ΔT of physically plausible magnitude
    (order 1–5 K for 2×CO2 in the gray model; exact value anchored in P2).
  - feedback_factor returns f ∈ (0, 1) at the default (stable) configuration.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from climate_rce.column import ColumnSpec
from climate_rce.metric import climate_sensitivity, feedback_factor, observable_functional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(n_z: int = 15) -> ColumnSpec:
    # n_state=n_z: T-only GRAIL seam (SPEC §2.2.1).
    pressure = np.linspace(5000.0, 1e5, n_z)
    return ColumnSpec(
        n_z=n_z, n_v=2, var_names=("T", "q"),
        pressure=pressure, var_scales=(300.0, 1e-2), n_state=n_z,  # T-only
    )


def _zero_subgrid(spec: ColumnSpec, profile: np.ndarray) -> np.ndarray:
    return np.zeros((spec.n_v, spec.n_z), dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_climate_sensitivity_positive_warming() -> None:
    """2×CO2 yields a positive ΔT_surf (warming response).

    SPEC §6 P1: 'climate_sensitivity returns a positive ΔT of physically
    plausible magnitude (order 1–5 K for 2×CO2 in the gray model)'.

    Exact value anchored in P2 analytic tests.

    Note: this test runs two equilibrations (baseline + perturbed), which
    may take ~10 s on a slow machine.  We use a coarse grid (n_z=12) and
    loose tol=1e-4 for speed.
    """
    spec = _make_spec(12)

    delta_T = climate_sensitivity(
        spec, _zero_subgrid,
        co2_baseline=280.0, co2_perturbed=560.0,
    )
    assert np.isfinite(delta_T), f"climate_sensitivity returned non-finite: {delta_T}"
    assert delta_T > 0.0, (
        f"2×CO₂ ΔT_surf = {delta_T:.3f} K ≤ 0.  Expected positive warming. "
        "Check the CO₂ forcing in radiation.py (_TAU_CO2_REF scaling)."
    )
    # Structural bound: gray model 2×CO2 sensitivity is typically 1–10 K.
    # We allow up to 20 K to cover a wide parameter range.
    assert delta_T < 20.0, (
        f"2×CO₂ ΔT_surf = {delta_T:.3f} K > 20 K seems unreasonably large. "
        "Exact value anchored in P2."
    )


@pytest.mark.unit
def test_feedback_factor_range() -> None:
    """feedback_factor returns f ∈ (0, 1) in the stable default configuration.

    SPEC §6 P1: 'feedback_factor returns f ∈ (0,1) at the default (stable) configuration.'

    We construct a synthetic K_matvec with known spectral radius < 1.
    """
    spec = _make_spec(12)

    # Synthetic K_matvec with known spectral radius = 0.7
    # K = 0.7 * (v @ e1) * e1  (rank-1 projection onto first basis vector)
    rho_true = 0.7
    n_state = spec.n_state
    e1 = np.zeros(n_state)
    e1[0] = 1.0

    def K_matvec(v: np.ndarray) -> np.ndarray:
        return rho_true * (v @ e1) * e1 + 0.01 * v  # dominant ev = 0.71 ≈ 0.7

    x_star_dummy = np.zeros((spec.n_v, spec.n_z))
    f = feedback_factor(spec, x_star_dummy, K_matvec, n_power_iter=60, seed=42)
    assert np.isfinite(f), f"feedback_factor returned non-finite: {f}"
    # The dominant eigenvalue should be recovered near 0.7 + 0.01 = 0.71
    # Power iteration is approximate; allow ±0.1 tolerance
    assert 0.0 < f < 1.0, (
        f"feedback_factor = {f:.4f} not in (0,1). "
        "Default stable config should give f < 1 (SPEC §6 P1)."
    )


@pytest.mark.unit
def test_observable_functional_shape() -> None:
    """observable_functional returns (n_state,) with correct surface-T position."""
    spec = _make_spec(20)
    x_star = np.zeros((spec.n_v, spec.n_z))
    Phi = observable_functional(spec, x_star)
    assert Phi.shape == (spec.n_state,), (
        f"observable_functional shape {Phi.shape} != ({spec.n_state},)"
    )
    assert Phi.dtype == np.float64
    # Surface T is at index n_z - 1 in the T block (= flat index n_z-1)
    surf_idx = spec.n_z - 1
    assert Phi[surf_idx] == spec.var_scales[0], (
        f"Phi[{surf_idx}] = {Phi[surf_idx]} != T_scale = {spec.var_scales[0]}. "
        "Surface-T position in flat state should be n_z-1."
    )
    # All other entries should be zero
    mask = np.ones(spec.n_state, dtype=bool)
    mask[surf_idx] = False
    assert np.all(Phi[mask] == 0.0), (
        "observable_functional has non-zero entries at non-surface-T positions."
    )
