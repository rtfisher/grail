"""Atmosphere configuration: T(tau), n_e(tau), line physics. the technical reference §3.1."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .atomic_model import AtomicModel
from .grid import AngleQuadrature, DepthGrid, FrequencyGrid


@dataclass
class Atmosphere:
    """Static 1-D plane-parallel atmosphere with prescribed thermodynamics.

    All tensors are float64 (torch.float64).  The depth axis has length N_τ.

    Fields
    ------
    depth : DepthGrid
        Logarithmic optical-depth grid.  ``depth.tau`` has shape ``(N_τ,)``.
    frequency : FrequencyGrid
        Frequency grid across the line profile.
    angle : AngleQuadrature
        Gauss-Legendre angle quadrature on μ ∈ [0, 1].
    T : torch.Tensor
        Temperature profile, shape ``(N_τ,)``, in Kelvin.
    n_e : torch.Tensor
        Electron density profile, shape ``(N_τ,)``, in cm⁻³.
    epsilon : float
        Photon destruction probability ε ∈ (0, 1].  the technical reference §3.1 and
        reference guide assumption A10: ε is the master thermalization parameter.
        Defined as collisional de-excitation rate / total de-excitation rate.
        Populated from ``atomic_model.transitions[0].epsilon`` by the
        class-method constructors.
    nu_0 : float
        Line-center frequency in Hz.  Populated from
        ``atomic_model.transitions[0].nu_0`` by the class-method constructors.
    atomic_model : AtomicModel
        Atomic data for the NLTE setup.  Built by the class-method constructors
        (e.g., ``two_level_master_epsilon``).  Provides a structured way to
        access all transitions and statistical weights.
    abundance_fractions : dict[str, float] or None, default None
        Per-species fractional abundance by number (sum = 1).  Used by the
        multi-species rate equation solver to normalise each species block
        independently.  Example: ``{"H": 0.9216, "He": 0.0784}`` (solar
        photospheric, Asplund et al. 2009, rounded to He/H ≈ 0.085).
        When None, single-species is assumed and all populations normalise
        to 1 (back-compat with pre-Phase-S code).
        SPEC_phase_s §2.3 (per-species conservation).

    the technical reference §3.1 — atmosphere structure.
    SPEC_threelevel §M3a TASK-002 — AtomicModel field added.
    SPEC_phase_s §2.3 — abundance_fractions field added.
    """

    depth: DepthGrid
    frequency: FrequencyGrid
    angle: AngleQuadrature
    T: torch.Tensor        # shape (N_τ,), float64, Kelvin
    n_e: torch.Tensor      # shape (N_τ,), float64, cm^-3
    epsilon: float         # photon destruction probability, in (0, 1]
    nu_0: float            # line-center frequency in Hz
    atomic_model: AtomicModel = field(default=None)  # type: ignore[assignment]
    abundance_fractions: dict | None = field(default=None)

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(epsilon: float, T0: float, n_e0: float, nu_0: float) -> None:
        """Raise ValueError for any out-of-range scalar parameter.

        Parameters
        ----------
        epsilon : float
            Photon destruction probability.  Must satisfy 0 < ε ≤ 1.
        T0 : float
            Reference temperature in Kelvin.  Must be positive.
        n_e0 : float
            Reference electron density in cm⁻³.  Must be positive.
        nu_0 : float
            Line-center frequency in Hz.  Must be positive.
        """
        if not (0.0 < epsilon <= 1.0):
            raise ValueError(
                f"epsilon must satisfy 0 < epsilon <= 1, got epsilon={epsilon!r}."
            )
        if T0 <= 0.0:
            raise ValueError(
                f"T0 must be positive (temperature in Kelvin), got T0={T0!r}."
            )
        if n_e0 <= 0.0:
            raise ValueError(
                f"n_e0 must be positive (electron density in cm^-3), got n_e0={n_e0!r}."
            )
        if nu_0 <= 0.0:
            raise ValueError(
                f"nu_0 must be positive (line-center frequency in Hz), got nu_0={nu_0!r}."
            )

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def isothermal(
        cls,
        T0: float,
        n_e0: float,
        epsilon: float,
        nu_0: float,
        *,
        depth: DepthGrid,
        frequency: FrequencyGrid,
        angle: AngleQuadrature,
    ) -> "Atmosphere":
        """Construct a uniform-temperature atmosphere.

        T(τ) = T0 for all τ.
        n_e(τ) = n_e0 for all τ.

        Parameters
        ----------
        T0 : float
            Temperature in Kelvin (constant throughout the atmosphere).
        n_e0 : float
            Electron density in cm⁻³ (constant throughout the atmosphere).
        epsilon : float
            Photon destruction probability ε ∈ (0, 1].
        nu_0 : float
            Line-center frequency in Hz.
        depth : DepthGrid
            Pre-constructed depth grid (keyword-only).
        frequency : FrequencyGrid
            Pre-constructed frequency grid (keyword-only).
        angle : AngleQuadrature
            Pre-constructed angle quadrature (keyword-only).

        Returns
        -------
        Atmosphere
        """
        cls._validate(epsilon, T0, n_e0, nu_0)
        n_tau = depth.tau.shape[0]
        T = torch.full((n_tau,), T0, dtype=torch.float64)
        n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)
        atomic_model = AtomicModel.two_level_master_epsilon(nu_0, epsilon)
        return cls(
            depth=depth,
            frequency=frequency,
            angle=angle,
            T=T,
            n_e=n_e,
            epsilon=epsilon,
            nu_0=nu_0,
            atomic_model=atomic_model,
        )

    @classmethod
    def power_law(
        cls,
        T0: float,
        alpha: float,
        n_e0: float,
        epsilon: float,
        nu_0: float,
        *,
        depth: DepthGrid,
        frequency: FrequencyGrid,
        angle: AngleQuadrature,
    ) -> "Atmosphere":
        """Construct a power-law temperature atmosphere.

        T(τ) = T0 · (τ / τ_min)^α  where τ_min = depth.tau[0].
        n_e(τ) = n_e0 for all τ.

        the technical reference §3.1: "simple power-law T(τ) as canned option".
        The depth grid is logarithmic starting at τ_min, so τ_min is the
        shallowest (surface-side) grid point: ``depth.tau[0]``.

        For α > 0 the temperature increases monotonically with depth (τ
        increases inward), as expected for a typical stellar atmosphere.

        Parameters
        ----------
        T0 : float
            Temperature at τ_min (surface-side boundary) in Kelvin.
        alpha : float
            Power-law exponent.  α > 0 → temperature rises with depth.
        n_e0 : float
            Electron density in cm⁻³ (constant throughout the atmosphere).
        epsilon : float
            Photon destruction probability ε ∈ (0, 1].
        nu_0 : float
            Line-center frequency in Hz.
        depth : DepthGrid
            Pre-constructed depth grid (keyword-only).
        frequency : FrequencyGrid
            Pre-constructed frequency grid (keyword-only).
        angle : AngleQuadrature
            Pre-constructed angle quadrature (keyword-only).

        Returns
        -------
        Atmosphere
        """
        cls._validate(epsilon, T0, n_e0, nu_0)
        tau = depth.tau.to(dtype=torch.float64)
        tau_min = tau[0]
        # Power-law profile: T(τ) = T0 · (τ / τ_min)^α
        T = T0 * (tau / tau_min) ** alpha
        n_tau = tau.shape[0]
        n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)
        atomic_model = AtomicModel.two_level_master_epsilon(nu_0, epsilon)
        return cls(
            depth=depth,
            frequency=frequency,
            angle=angle,
            T=T,
            n_e=n_e,
            epsilon=epsilon,
            nu_0=nu_0,
            atomic_model=atomic_model,
        )

    @classmethod
    def balmer_isothermal(
        cls,
        T0: float,
        n_e0: float,
        epsilon: float | tuple = 1e-4,
        *,
        depth: DepthGrid,
        frequency: FrequencyGrid,
        angle: AngleQuadrature,
    ) -> "Atmosphere":
        """Construct an isothermal 5-level hydrogen Balmer atmosphere.

        Builds the 5-level hydrogen Balmer model (n=1..5, 10 bound-bound
        transitions) from ``AtomicModel.balmer_5level``.

        T(τ) = T0 for all τ.  n_e(τ) = n_e0 for all τ.

        The legacy ``epsilon`` and ``nu_0`` fields are populated from the
        first transition (Ly_alpha) for back-compatibility with any code
        that reads those scalar fields directly.

        Parameters
        ----------
        T0 : float
            Temperature in Kelvin (constant throughout the atmosphere).
        n_e0 : float
            Electron density in cm⁻³ (constant throughout the atmosphere).
        epsilon : float or tuple of length 10
            Per-line photon destruction probability.  If scalar, applied to
            all 10 transitions.  If tuple of length 10, applied per transition.
        depth : DepthGrid
            Pre-constructed depth grid (keyword-only).
        frequency : FrequencyGrid
            Pre-constructed frequency grid (keyword-only).
        angle : AngleQuadrature
            Pre-constructed angle quadrature (keyword-only).

        Returns
        -------
        Atmosphere
            With ``atomic_model.label == "balmer_5level"``.

        SPEC_nlte_scaleup §M.3.
        """
        atomic_model = AtomicModel.balmer_5level(epsilon=epsilon)
        if T0 <= 0.0:
            raise ValueError(
                f"T0 must be positive (temperature in Kelvin), got T0={T0!r}."
            )
        if n_e0 <= 0.0:
            raise ValueError(
                f"n_e0 must be positive (electron density in cm^-3), got n_e0={n_e0!r}."
            )

        n_tau = depth.tau.shape[0]
        T = torch.full((n_tau,), T0, dtype=torch.float64)
        n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)

        # Legacy scalar fields from the first transition (Ly_alpha).
        tr0 = atomic_model.transitions[0]
        return cls(
            depth=depth,
            frequency=frequency,
            angle=angle,
            T=T,
            n_e=n_e,
            epsilon=tr0.epsilon,
            nu_0=tr0.nu_0,
            atomic_model=atomic_model,
        )

    @classmethod
    def balmer_power_law(
        cls,
        T0: float,
        alpha: float,
        n_e0: float,
        epsilon: float | tuple = 1e-4,
        *,
        depth: DepthGrid,
        frequency: FrequencyGrid,
        angle: AngleQuadrature,
    ) -> "Atmosphere":
        """Construct a power-law temperature 5-level hydrogen Balmer atmosphere.

        T(τ) = T0 · (τ / τ_min)^α.  n_e(τ) = n_e0 for all τ.
        Uses the 5-level hydrogen Balmer model (n=1..5, 10 transitions).

        Parameters
        ----------
        T0 : float
            Temperature at τ_min (surface-side boundary) in Kelvin.
        alpha : float
            Power-law exponent.  α > 0 → temperature rises with depth.
        n_e0 : float
            Electron density in cm⁻³ (constant throughout the atmosphere).
        epsilon : float or tuple of length 10
            Per-line photon destruction probability.
        depth : DepthGrid
            Pre-constructed depth grid (keyword-only).
        frequency : FrequencyGrid
            Pre-constructed frequency grid (keyword-only).
        angle : AngleQuadrature
            Pre-constructed angle quadrature (keyword-only).

        Returns
        -------
        Atmosphere
            With ``atomic_model.label == "balmer_5level"``.

        SPEC_nlte_scaleup §M.3.
        """
        atomic_model = AtomicModel.balmer_5level(epsilon=epsilon)
        if T0 <= 0.0:
            raise ValueError(
                f"T0 must be positive (temperature in Kelvin), got T0={T0!r}."
            )
        if n_e0 <= 0.0:
            raise ValueError(
                f"n_e0 must be positive (electron density in cm^-3), got n_e0={n_e0!r}."
            )

        tau = depth.tau.to(dtype=torch.float64)
        tau_min = tau[0]
        T = T0 * (tau / tau_min) ** alpha
        n_tau = tau.shape[0]
        n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)

        # Legacy scalar fields from the first transition (Ly_alpha).
        tr0 = atomic_model.transitions[0]
        return cls(
            depth=depth,
            frequency=frequency,
            angle=angle,
            T=T,
            n_e=n_e,
            epsilon=tr0.epsilon,
            nu_0=tr0.nu_0,
            atomic_model=atomic_model,
        )

    @classmethod
    def three_level_hydrogen_isothermal(
        cls,
        T0: float,
        n_e0: float,
        epsilon_lya: float,
        epsilon_ha: float,
        *,
        depth: DepthGrid,
        frequency: FrequencyGrid,
        angle: AngleQuadrature,
    ) -> "Atmosphere":
        """Construct an isothermal 3-level hydrogen atmosphere (Ly-α + H-α).

        Builds the toy hydrogen 3-level model (levels n=1, n=2, n=3 with
        Lyman-alpha 1↔2 and H-alpha 2↔3) from ``AtomicModel.three_level_hydrogen_chained``.

        T(τ) = T0 for all τ.  n_e(τ) = n_e0 for all τ.

        The legacy ``epsilon`` and ``nu_0`` fields are populated from the
        first transition (Ly-alpha) for back-compatibility with any code
        that reads those scalar fields directly.

        Parameters
        ----------
        T0 : float
            Temperature in Kelvin (constant throughout the atmosphere).
        n_e0 : float
            Electron density in cm⁻³ (constant throughout the atmosphere).
        epsilon_lya : float
            Photon destruction probability for Lyman-alpha ∈ (0, 1].
        epsilon_ha : float
            Photon destruction probability for H-alpha ∈ (0, 1].
        depth : DepthGrid
            Pre-constructed depth grid (keyword-only).
        frequency : FrequencyGrid
            Pre-constructed frequency grid (keyword-only).
        angle : AngleQuadrature
            Pre-constructed angle quadrature (keyword-only).

        Returns
        -------
        Atmosphere
            With ``atomic_model.label == "three_level_hydrogen_chained"``.

        SPEC_threelevel §M3b TASK-013.
        """
        # Build the atomic model first (validates epsilon values via __post_init__).
        atomic_model = AtomicModel.three_level_hydrogen_chained(
            epsilon_lya=epsilon_lya,
            epsilon_ha=epsilon_ha,
        )
        # Validate remaining scalar parameters.
        if T0 <= 0.0:
            raise ValueError(
                f"T0 must be positive (temperature in Kelvin), got T0={T0!r}."
            )
        if n_e0 <= 0.0:
            raise ValueError(
                f"n_e0 must be positive (electron density in cm^-3), got n_e0={n_e0!r}."
            )

        n_tau = depth.tau.shape[0]
        T = torch.full((n_tau,), T0, dtype=torch.float64)
        n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)

        # Legacy scalar fields from the first transition (Ly-alpha).
        tr0 = atomic_model.transitions[0]
        return cls(
            depth=depth,
            frequency=frequency,
            angle=angle,
            T=T,
            n_e=n_e,
            epsilon=tr0.epsilon,
            nu_0=tr0.nu_0,
            atomic_model=atomic_model,
        )

    @classmethod
    def three_level_hydrogen_power_law(
        cls,
        T0: float,
        alpha: float,
        n_e0: float,
        epsilon_lya: float,
        epsilon_ha: float,
        *,
        depth: DepthGrid,
        frequency: FrequencyGrid,
        angle: AngleQuadrature,
    ) -> "Atmosphere":
        """Construct a power-law temperature 3-level hydrogen atmosphere.

        T(τ) = T0 · (τ / τ_min)^α.  n_e(τ) = n_e0 for all τ.
        Uses the toy hydrogen 3-level model (Ly-α + H-α).

        Parameters
        ----------
        T0 : float
            Temperature at τ_min (surface-side boundary) in Kelvin.
        alpha : float
            Power-law exponent.  α > 0 → temperature rises with depth.
        n_e0 : float
            Electron density in cm⁻³ (constant throughout the atmosphere).
        epsilon_lya : float
            Photon destruction probability for Lyman-alpha ∈ (0, 1].
        epsilon_ha : float
            Photon destruction probability for H-alpha ∈ (0, 1].
        depth : DepthGrid
            Pre-constructed depth grid (keyword-only).
        frequency : FrequencyGrid
            Pre-constructed frequency grid (keyword-only).
        angle : AngleQuadrature
            Pre-constructed angle quadrature (keyword-only).

        Returns
        -------
        Atmosphere
            With ``atomic_model.label == "three_level_hydrogen_chained"``.

        SPEC_threelevel §M3b TASK-013.
        """
        atomic_model = AtomicModel.three_level_hydrogen_chained(
            epsilon_lya=epsilon_lya,
            epsilon_ha=epsilon_ha,
        )
        if T0 <= 0.0:
            raise ValueError(
                f"T0 must be positive (temperature in Kelvin), got T0={T0!r}."
            )
        if n_e0 <= 0.0:
            raise ValueError(
                f"n_e0 must be positive (electron density in cm^-3), got n_e0={n_e0!r}."
            )

        tau = depth.tau.to(dtype=torch.float64)
        tau_min = tau[0]
        T = T0 * (tau / tau_min) ** alpha
        n_tau = tau.shape[0]
        n_e = torch.full((n_tau,), n_e0, dtype=torch.float64)

        # Legacy scalar fields from the first transition (Ly-alpha).
        tr0 = atomic_model.transitions[0]
        return cls(
            depth=depth,
            frequency=frequency,
            angle=angle,
            T=T,
            n_e=n_e,
            epsilon=tr0.epsilon,
            nu_0=tr0.nu_0,
            atomic_model=atomic_model,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
