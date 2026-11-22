"""Unit tests for the Atmosphere dataclass. docs/testing.md Tier U, atmosphere section.

All tests are marked ``@pytest.mark.unit`` per docs/testing.md conventions.

The grid class-method constructors (DepthGrid.log_spaced, etc.) are M0 stubs
that raise NotImplementedError, so we build grids directly from tensors.
"""

from __future__ import annotations

import pytest
import torch

from atmosphere.atmosphere import Atmosphere
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid


# ---------------------------------------------------------------------------
# Shared fixture: minimal but physically sensible grid objects
# ---------------------------------------------------------------------------

def _make_grids(n_tau: int = 20, n_nu: int = 11, n_mu: int = 3):
    """Return (depth, frequency, angle) grid objects for testing.

    Builds grids directly from tensors because the M0 class-method constructors
    (DepthGrid.log_spaced, FrequencyGrid.doppler, AngleQuadrature.gauss_legendre)
    are stubs that raise NotImplementedError.

    Depth grid: log-spaced from tau_min=1e-4 to tau_max=1e4 with n_tau points.
    Frequency grid: n_nu points centered at nu_0, uniform weights (placeholder).
    Angle quadrature: n_mu uniform mu values in (0, 1], uniform weights.
    """
    # Use the implemented grid constructors rather than constructing dataclass
    # instances by hand (the M0 stubs would have forced manual construction; on
    # main the .log_spaced/.doppler/.gauss_legendre class methods are real).
    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=n_tau)
    frequency = FrequencyGrid.doppler(n_nu=n_nu, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=n_mu)

    return depth, frequency, angle


# ---------------------------------------------------------------------------
# Test: isothermal constructor
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_isothermal_constructor() -> None:
    """Isothermal atmosphere has correct shapes, dtype, and constant T.

    Uses H-alpha defaults: T0=5800 K, n_e0=1e10 cm^-3, ε=1e-4, ν_0=4.57e14 Hz.
    """
    T0 = 5800.0       # K — approximate solar effective temperature
    n_e0 = 1.0e10     # cm^-3 — typical chromospheric electron density
    epsilon = 1.0e-4  # photon destruction probability
    nu_0 = 4.57e14    # Hz — hydrogen H-alpha line center

    depth, frequency, angle = _make_grids(n_tau=40)
    n_tau = depth.tau.shape[0]

    atm = Atmosphere.isothermal(
        T0, n_e0, epsilon, nu_0,
        depth=depth, frequency=frequency, angle=angle,
    )

    # Shape checks
    assert atm.T.shape == (n_tau,), (
        f"T should have shape ({n_tau},), got {atm.T.shape}"
    )
    assert atm.n_e.shape == (n_tau,), (
        f"n_e should have shape ({n_tau},), got {atm.n_e.shape}"
    )

    # Dtype check: float64 throughout
    assert atm.T.dtype == torch.float64, f"T dtype should be float64, got {atm.T.dtype}"
    assert atm.n_e.dtype == torch.float64, (
        f"n_e dtype should be float64, got {atm.n_e.dtype}"
    )

    # Scalar fields stored correctly
    assert atm.epsilon == epsilon
    assert atm.nu_0 == nu_0

    # Temperature is constant T0 at every depth point
    assert torch.all(atm.T == T0), (
        "T should equal T0 at every depth point for isothermal atmosphere"
    )

    # Electron density is constant n_e0 at every depth point
    assert torch.all(atm.n_e == n_e0), (
        "n_e should equal n_e0 at every depth point for isothermal atmosphere"
    )

    # Grid objects are stored by reference
    assert atm.depth is depth
    assert atm.frequency is frequency
    assert atm.angle is angle

    # SPEC_threelevel §M3a TASK-002: atomic_model is built by the constructor.
    assert atm.atomic_model.label == "two_level_master_epsilon"


# ---------------------------------------------------------------------------
# Test: power-law constructor
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_power_law_constructor() -> None:
    """Power-law atmosphere: T(τ) = T0·(τ/τ_min)^α at every grid point.

    Also verifies monotonicity for α > 0 (temperature rises with depth).
    """
    T0 = 5800.0
    alpha = 0.25     # positive exponent → T increases with τ
    n_e0 = 1.0e10
    epsilon = 1.0e-4
    nu_0 = 4.57e14

    depth, frequency, angle = _make_grids(n_tau=40)

    atm = Atmosphere.power_law(
        T0, alpha, n_e0, epsilon, nu_0,
        depth=depth, frequency=frequency, angle=angle,
    )

    tau = depth.tau.to(dtype=torch.float64)
    tau_min = tau[0]
    T_expected = T0 * (tau / tau_min) ** alpha

    # Shape and dtype
    n_tau = tau.shape[0]
    assert atm.T.shape == (n_tau,)
    assert atm.T.dtype == torch.float64
    assert atm.n_e.shape == (n_tau,)
    assert atm.n_e.dtype == torch.float64

    # Values match power-law formula at every grid point
    assert torch.allclose(atm.T, T_expected, rtol=1e-12, atol=0.0), (
        "T(τ) should match T0·(τ/τ_min)^α at every depth point"
    )

    # n_e is constant
    assert torch.all(atm.n_e == n_e0), (
        "n_e should equal n_e0 at every depth point for power-law atmosphere"
    )

    # For α > 0, T is strictly monotonically increasing (deeper = larger τ = hotter)
    diffs = atm.T[1:] - atm.T[:-1]
    assert torch.all(diffs > 0), (
        "T should be strictly increasing with τ when alpha > 0"
    )

    # Boundary condition: at τ = τ_min (first grid point), T = T0
    assert abs(atm.T[0].item() - T0) < 1e-10, (
        f"T at τ_min should equal T0={T0}, got {atm.T[0].item()}"
    )


# ---------------------------------------------------------------------------
# Test: input validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_input_validation() -> None:
    """Invalid scalar inputs raise ValueError with descriptive messages.

    Checks:
    - ε ≤ 0 raises ValueError
    - ε > 1 raises ValueError
    - T0 ≤ 0 raises ValueError
    - n_e0 ≤ 0 raises ValueError
    - nu_0 ≤ 0 raises ValueError
    """
    depth, frequency, angle = _make_grids()
    kwargs = dict(depth=depth, frequency=frequency, angle=angle)

    # Valid baseline values
    good_T0, good_n_e0, good_eps, good_nu0 = 5800.0, 1e10, 1e-4, 4.57e14

    # --- epsilon out of range ---
    with pytest.raises(ValueError, match="epsilon"):
        Atmosphere.isothermal(good_T0, good_n_e0, epsilon=0.0, nu_0=good_nu0, **kwargs)

    with pytest.raises(ValueError, match="epsilon"):
        Atmosphere.isothermal(good_T0, good_n_e0, epsilon=-0.5, nu_0=good_nu0, **kwargs)

    with pytest.raises(ValueError, match="epsilon"):
        Atmosphere.isothermal(good_T0, good_n_e0, epsilon=1.5, nu_0=good_nu0, **kwargs)

    # --- epsilon = 1 is valid (pure LTE) ---
    atm = Atmosphere.isothermal(good_T0, good_n_e0, epsilon=1.0, nu_0=good_nu0, **kwargs)
    assert atm.epsilon == 1.0

    # --- T0 out of range ---
    with pytest.raises(ValueError, match="T0"):
        Atmosphere.isothermal(0.0, good_n_e0, epsilon=good_eps, nu_0=good_nu0, **kwargs)

    with pytest.raises(ValueError, match="T0"):
        Atmosphere.isothermal(-100.0, good_n_e0, epsilon=good_eps, nu_0=good_nu0, **kwargs)

    # --- n_e0 out of range ---
    with pytest.raises(ValueError, match="n_e0"):
        Atmosphere.isothermal(good_T0, 0.0, epsilon=good_eps, nu_0=good_nu0, **kwargs)

    with pytest.raises(ValueError, match="n_e0"):
        Atmosphere.isothermal(good_T0, -1.0, epsilon=good_eps, nu_0=good_nu0, **kwargs)

    # --- nu_0 out of range ---
    with pytest.raises(ValueError, match="nu_0"):
        Atmosphere.isothermal(good_T0, good_n_e0, epsilon=good_eps, nu_0=0.0, **kwargs)

    with pytest.raises(ValueError, match="nu_0"):
        Atmosphere.isothermal(good_T0, good_n_e0, epsilon=good_eps, nu_0=-1e14, **kwargs)

    # --- Same validation for power_law ---
    with pytest.raises(ValueError, match="epsilon"):
        Atmosphere.power_law(
            good_T0, 0.25, good_n_e0, epsilon=0.0, nu_0=good_nu0, **kwargs
        )

    with pytest.raises(ValueError, match="T0"):
        Atmosphere.power_law(
            -1.0, 0.25, good_n_e0, epsilon=good_eps, nu_0=good_nu0, **kwargs
        )


# ---------------------------------------------------------------------------
# Test: field shapes and dtypes
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_field_shapes_and_dtypes() -> None:
    """T and n_e are torch.float64 with shape (n_τ,) matching depth.tau.shape[0].

    Exercises multiple grid sizes to confirm the shape contract is general.
    """
    for n_tau in (10, 40, 80, 200):
        depth, frequency, angle = _make_grids(n_tau=n_tau)

        # Isothermal
        atm_iso = Atmosphere.isothermal(
            5800.0, 1e10, 1e-4, 4.57e14,
            depth=depth, frequency=frequency, angle=angle,
        )
        assert atm_iso.T.shape == (n_tau,), (
            f"isothermal T shape mismatch at n_tau={n_tau}: {atm_iso.T.shape}"
        )
        assert atm_iso.n_e.shape == (n_tau,), (
            f"isothermal n_e shape mismatch at n_tau={n_tau}: {atm_iso.n_e.shape}"
        )
        assert atm_iso.T.dtype == torch.float64
        assert atm_iso.n_e.dtype == torch.float64

        # Power-law
        atm_pl = Atmosphere.power_law(
            5800.0, 0.25, 1e10, 1e-4, 4.57e14,
            depth=depth, frequency=frequency, angle=angle,
        )
        assert atm_pl.T.shape == (n_tau,), (
            f"power_law T shape mismatch at n_tau={n_tau}: {atm_pl.T.shape}"
        )
        assert atm_pl.n_e.shape == (n_tau,), (
            f"power_law n_e shape mismatch at n_tau={n_tau}: {atm_pl.n_e.shape}"
        )
        assert atm_pl.T.dtype == torch.float64
        assert atm_pl.n_e.dtype == torch.float64


# ---------------------------------------------------------------------------
# Test: 3-level hydrogen constructors (U-atmosphere3-01)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_atmosphere3_01_three_level_constructors() -> None:
    """U-atmosphere3-01: 3-level hydrogen constructors return Atmosphere with correct label.

    Checks:
    - atomic_model.label == "three_level_hydrogen_chained"
    - n_levels == 3, K == 2
    - T and n_e shapes match depth grid
    - isothermal: T is constant
    - power_law: T is strictly increasing for alpha > 0
    - epsilon and nu_0 come from the first transition (Ly-alpha)
    - ValueError raised for invalid T0, n_e0, or epsilon values

    SPEC_threelevel §M3b TASK-013.
    """
    depth, frequency, angle = _make_grids(n_tau=40)
    n_tau = depth.tau.shape[0]
    kwargs = dict(depth=depth, frequency=frequency, angle=angle)

    # ---  isothermal constructor ---
    atm_iso = Atmosphere.three_level_hydrogen_isothermal(
        T0=6000.0, n_e0=1e10,
        epsilon_lya=1e-2, epsilon_ha=1e-4,
        **kwargs,
    )

    assert atm_iso.atomic_model.label == "three_level_hydrogen_chained", (
        f"Expected label 'three_level_hydrogen_chained', got {atm_iso.atomic_model.label!r}"
    )
    assert atm_iso.atomic_model.n_levels == 3
    assert atm_iso.atomic_model.K == 2
    assert atm_iso.T.shape == (n_tau,)
    assert atm_iso.n_e.shape == (n_tau,)
    assert atm_iso.T.dtype == torch.float64
    assert atm_iso.n_e.dtype == torch.float64
    # Isothermal: T constant at T0.
    assert torch.all(atm_iso.T == 6000.0), "T should be constant at T0=6000 for isothermal"
    # Electron density constant.
    assert torch.all(atm_iso.n_e == 1e10)
    # Legacy fields from Ly-alpha (first transition).
    assert atm_iso.epsilon == 1e-2
    assert atm_iso.nu_0 == 2.467381e15

    # --- power-law constructor ---
    atm_pl = Atmosphere.three_level_hydrogen_power_law(
        T0=5000.0, alpha=0.25, n_e0=1e10,
        epsilon_lya=1e-2, epsilon_ha=1e-4,
        **kwargs,
    )

    assert atm_pl.atomic_model.label == "three_level_hydrogen_chained"
    assert atm_pl.atomic_model.n_levels == 3
    assert atm_pl.T.shape == (n_tau,)
    assert atm_pl.T.dtype == torch.float64
    # Power-law: T strictly increasing for alpha > 0.
    diffs = atm_pl.T[1:] - atm_pl.T[:-1]
    assert torch.all(diffs > 0), "T should be strictly increasing for alpha > 0"
    # T[0] should equal T0.
    assert abs(atm_pl.T[0].item() - 5000.0) < 1e-10

    # --- validation: invalid T0 ---
    with pytest.raises(ValueError, match="T0"):
        Atmosphere.three_level_hydrogen_isothermal(
            T0=-1.0, n_e0=1e10, epsilon_lya=1e-2, epsilon_ha=1e-4, **kwargs
        )

    # --- validation: invalid n_e0 ---
    with pytest.raises(ValueError, match="n_e0"):
        Atmosphere.three_level_hydrogen_isothermal(
            T0=6000.0, n_e0=0.0, epsilon_lya=1e-2, epsilon_ha=1e-4, **kwargs
        )

    # --- validation: invalid epsilon (delegated to AtomicModel.__post_init__) ---
    with pytest.raises(ValueError):
        Atmosphere.three_level_hydrogen_isothermal(
            T0=6000.0, n_e0=1e10, epsilon_lya=0.0, epsilon_ha=1e-4, **kwargs
        )
