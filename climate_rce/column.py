"""Column state-vector seam: ColumnSpec, ColumnState, pack_state, unpack_state.

SPEC_climate_rce §4.1, §2.2.1 (fixed-RH closure decision, 2026-05-19).

**GRAIL state is temperature-only (n_state = n_z).**
Under the fixed-relative-humidity closure (Manabe & Wetherald 1967), specific
humidity is a *diagnostic* variable: q = q_ref(T_surf) — the saturation
specific humidity on the moist adiabat anchored at the surface temperature.
The column carries q physically (radiation needs it for the LW optical depth),
but q is NOT an independent GRAIL state coordinate and is never packed.

State layout (frozen; T-only, level-minor):
    x_flat = [ T(z_0..z_{n_z-1}) ]   # length n_z  (T block only)

``pack_state`` emits the T block only, non-dimensionalized by ``var_scales[0]``
(T_scale = 300 K by default).  ``unpack_state`` takes a ``(n_z,)`` flat vector
and reconstructs the full physical ``(T, q)`` ColumnState by diagnosing
q = q_ref(T_surf) via ``convection._build_moist_adiabat`` /
``convection._build_q_ref``.  The host's radiation and convection still consume
the physical (T, q) profile.

The ``ColumnSpec`` retains physical ``n_v``, ``var_names``, and ``var_scales``
for the column physics (radiation/convection), but the GRAIL seam dimension is
``n_state == n_z`` (not ``n_z * n_v``).  This is the SPEC §2.2.1 decision.

Non-dimensionalization: ``pack_state`` divides T[v=0] by ``var_scales[0]``.
``δx*`` returned by GRAIL's ``resolvent_solve`` is in the same non-dimensional
units; the caller re-dimensionalizes (analogous to SCF trace-drift discipline,
SPEC_scf §8.4 / SPEC_climate_rce §4.1).

The literal (T,q) Jacobian K_full is singular by construction (eigenvalue-1
humidity modes at stable upper levels; SPEC §2.2.1).  The GRAIL-facing operator
is the reduced n_z × n_z ``K_eff`` (the WV-feedback-inclusive T-cycle Jacobian);
K_full is never formed at the seam.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# ColumnSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Specification of a single-column vertical grid and state convention.

    Parameters
    ----------
    n_z : int
        Number of vertical levels.
    n_v : int
        Number of physical prognostic variables (2 = T, q for the core
        column).  Retained for the column physics (radiation/convection);
        NOT the GRAIL state dimension.
    var_names : tuple[str, ...]
        Length ``n_v``; e.g. ``("T", "q")``.
    pressure : np.ndarray
        Shape ``(n_z,)`` Pa, mid-layer pressure; the vertical grid.
    var_scales : tuple[float, ...]
        Length ``n_v``; per-variable characteristic scales.  ``var_scales[0]``
        is T_scale (default 300 K) used by ``pack_state``/``unpack_state``
        for the T block.  ``var_scales[1]`` is q_scale (default 1e-2 kg/kg)
        retained for internal physics use.
    n_state : int
        **GRAIL seam dimension: must equal ``n_z`` (T-only, SPEC §2.2.1).**
        The column carries ``n_v`` physical variables (T, q) for
        radiation/convection, but only T is packed into the GRAIL state.
        q is the diagnostic fixed-RH closure q = q_ref(T_surf).
    """

    n_z: int
    n_v: int
    var_names: tuple
    pressure: np.ndarray
    var_scales: tuple
    n_state: int

    def __post_init__(self) -> None:
        if len(self.var_names) != self.n_v:
            raise ValueError(
                f"len(var_names)={len(self.var_names)} != n_v={self.n_v}"
            )
        if self.pressure.shape != (self.n_z,):
            raise ValueError(
                f"pressure.shape={self.pressure.shape} != ({self.n_z},)"
            )
        # GRAIL seam is T-only (SPEC §2.2.1 / §4.1 fixed-RH closure decision).
        # n_state == n_z, NOT n_z*n_v.  The column carries n_v physical variables
        # internally (radiation/convection consume them), but only T is packed into
        # the GRAIL state.  K_full (n_z*n_v × n_z*n_v) is singular (eigenvalue-1
        # q-neutral modes); the GRAIL-facing K_eff is n_z × n_z.
        if self.n_state != self.n_z:
            raise ValueError(
                f"n_state={self.n_state} != n_z={self.n_z}. "
                "GRAIL seam is temperature-only (SPEC §2.2.1): n_state must equal n_z, "
                "not n_z*n_v.  The column carries n_v physical variables for physics "
                "but only T is packed into the GRAIL state vector."
            )
        if len(self.var_scales) != self.n_v:
            raise ValueError(
                f"len(var_scales)={len(self.var_scales)} != n_v={self.n_v}"
            )

    # Explicit __hash__ and __eq__ because numpy arrays are unhashable
    # in SPEC_climate_rce §4.4).
    def __hash__(self) -> int:
        return hash((self.n_z, self.n_v, self.var_names, self.n_state, self.var_scales))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColumnSpec):
            return NotImplemented
        return (
            self.n_z == other.n_z
            and self.n_v == other.n_v
            and self.var_names == other.var_names
            and self.n_state == other.n_state
            and self.var_scales == other.var_scales
            and np.array_equal(self.pressure, other.pressure)
        )


# ---------------------------------------------------------------------------
# ColumnState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnState:
    """A vertical column profile in physical units.

    Parameters
    ----------
    spec : ColumnSpec
        Associated grid and variable specification.
    profile : np.ndarray
        Shape ``(n_v, n_z)``.  Row ``v`` is the vertical profile of variable
        ``v`` in physical units (e.g. T in K, q in kg/kg).
    """

    spec: ColumnSpec
    profile: np.ndarray


# ---------------------------------------------------------------------------
# Seam functions
# ---------------------------------------------------------------------------


def pack_state(s: ColumnState) -> torch.Tensor:
    """Pack a ColumnState into a flat non-dimensional GRAIL state vector.

    **T-only seam (SPEC §2.2.1 / §4.1):**
    Returns a ``(n_state,)`` = ``(n_z,)`` float64 tensor containing only the
    temperature block.  Specific humidity is NOT packed — it is the diagnostic
    fixed-RH closure q = q_ref(T_surf) and is reconstructed by ``unpack_state``.

    Non-dimensionalization: ``x_flat[k] = profile[0, k] / var_scales[0]``
    (i.e. T[k] / T_scale, where T_scale = var_scales[0] = 300 K by default).

    The GRAIL resolvent operates on this ``(n_z,)`` reduced state via K_eff
    (the WV-feedback-inclusive T-cycle Jacobian, n_z × n_z).  K_full
    (n_z*n_v × n_z*n_v) is singular and never formed at the seam.

    Parameters
    ----------
    s : ColumnState
        Column state in physical units; shape ``(n_v, n_z)``.

    Returns
    -------
    torch.Tensor
        ``(n_z,)`` float64, non-dimensional T block only.
    """
    spec = s.spec
    T_scale = spec.var_scales[0]   # characteristic temperature, e.g. 300 K
    # Pack T block only (profile[0]); divide by T_scale for non-dimensionalization.
    T_nondim = s.profile[0] / T_scale   # (n_z,)
    return torch.tensor(T_nondim, dtype=torch.float64)


def unpack_state(x_flat: torch.Tensor | np.ndarray, spec: ColumnSpec) -> ColumnState:
    """Unpack a flat non-dimensional GRAIL state vector into a ColumnState.

    **T-only seam (SPEC §2.2.1 / §4.1):**
    Takes a ``(n_state,)`` = ``(n_z,)`` non-dimensional T vector and reconstructs
    the full physical ``(T, q)`` ColumnState.  q is diagnosed via the fixed-RH
    closure: q = q_ref(T_surf) — the saturation specific humidity on the moist
    adiabat anchored at the surface temperature T[-1].

    This uses ``convection._build_moist_adiabat`` and ``convection._build_q_ref``
    (lazy import to avoid circular dependency; convection.py imports column.py).

    **NOT a pure round-trip inverse of ``pack_state``:**
    ``pack_state(unpack_state(x_flat, spec))`` recovers x_flat exactly (T block).
    ``unpack_state(pack_state(s), spec).profile[0]`` recovers ``s.profile[0]``
    (T) to float64 precision.
    ``unpack_state(pack_state(s), spec).profile[1]`` is ``q_ref(T_surf)`` —
    NOT necessarily equal to the original ``s.profile[1]``.  This is by design:
    q is the diagnostic closure, not an independent state coordinate.

    Parameters
    ----------
    x_flat : tensor or ndarray
        ``(n_z,)`` float64, non-dimensional T block (T / T_scale).
    spec : ColumnSpec
        Grid/variable spec.

    Returns
    -------
    ColumnState
        Profile in physical units, shape ``(n_v, n_z)``.  Row 0 = T [K],
        row 1 = q [kg/kg] (diagnosed via q_ref(T_surf)), further rows = zeros
        for n_v > 2.
    """
    # Deferred import to avoid circular dependency:
    # convection.py imports column.py; column.py must not import convection.py at
    # module load time.
    from .convection import _build_moist_adiabat, _build_q_ref  # noqa: PLC0415

    if isinstance(x_flat, torch.Tensor):
        arr = x_flat.detach().numpy().astype(np.float64)
    else:
        arr = np.asarray(x_flat, dtype=np.float64)

    if arr.shape != (spec.n_z,):
        raise ValueError(
            f"unpack_state: x_flat.shape={arr.shape} != ({spec.n_z},). "
            "Expected T-only GRAIL state of length n_z (SPEC §2.2.1)."
        )

    T_scale = spec.var_scales[0]   # e.g. 300 K
    T_vec = arr * T_scale          # re-dimensionalize T, shape (n_z,)

    # Diagnose q via fixed-RH closure: q = q_ref(T_surf)
    # where T_surf = T_vec[-1] (bottom level = highest pressure = surface).
    T_surf = float(T_vec[-1])
    T_ref = _build_moist_adiabat(spec, T_surf)   # (n_z,) moist-adiabat reference
    q_vec = _build_q_ref(spec, T_ref)            # (n_z,) saturation q [kg/kg]

    # Build (n_v, n_z) profile: T block + diagnosed q + zeros for any extra vars
    profile = np.zeros((spec.n_v, spec.n_z), dtype=np.float64)
    profile[0] = T_vec
    if spec.n_v >= 2:
        profile[1] = q_vec
    # Any n_v > 2 variables remain zero (not in scope for the fixed-RH column)

    return ColumnState(spec=spec, profile=profile)


def reference_profile(spec: ColumnSpec) -> ColumnState:
    """Construct a simple isothermal + constant-RH reference profile.

    Used as the default ``profile_init`` in ``equilibrate_to_rce`` when no
    explicit initial profile is provided.

    Temperature: 250 K isothermal.
    Specific humidity: fixed relative-humidity (RH=0.5) with respect to a
    simple Clausius–Clapeyron approximation.

    Parameters
    ----------
    spec : ColumnSpec
        Column specification (pressure grid used for q profile).

    Returns
    -------
    ColumnState
        Simple reference profile in physical units.
    """
    T_iso = 250.0  # K, isothermal reference temperature
    T_profile = np.full(spec.n_z, T_iso, dtype=np.float64)

    # Clausius–Clapeyron approximation for saturation specific humidity
    # q_sat ≈ 0.622 * e_sat / p,  e_sat = 611.2 * exp(17.67*(T-273.15)/(T-29.65))
    # Use fixed RH = 0.5 as a simple starting profile.
    p = spec.pressure  # Pa
    T_K = T_profile
    e_sat = 611.2 * np.exp(17.67 * (T_K - 273.15) / (T_K - 29.65))
    q_sat = 0.622 * e_sat / np.maximum(p, 1.0)
    RH = 0.5
    q_profile = RH * q_sat

    if spec.n_v == 2:
        profile = np.stack([T_profile, q_profile], axis=0)   # (2, n_z)
    else:
        # For n_v > 2: fill additional vars with zeros
        profile = np.zeros((spec.n_v, spec.n_z), dtype=np.float64)
        profile[0] = T_profile
        profile[1] = q_profile

    return ColumnState(spec=spec, profile=profile)
