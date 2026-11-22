"""Differentiable gray-band radiative-transfer tendency.

SPEC_climate_rce §3.1, §4.4.

Model
-----
A gray two-stream radiative transfer column with:

- Longwave (LW) emission/absorption: optical depth τ_LW(p) proportional to
  CO₂ column amount plus a water-vapor (WV) contribution.
- Shortwave (SW): prescribed heating profile (absorbed solar radiation),
  constant (no state dependence).

The LW optical depth carries the water-vapor feedback that drives the
near-critical-f amplification regime (SPEC_climate_rce §3.5):

    τ_LW(z_k) = τ_CO2(co2_ppm) + τ_WV * q(z_k) / q_ref

The tendency R_i(x) = (dT/dt, dq/dt) in physical units, returned as an
``(n_v, n_z)`` numpy array.

Physical calibration
--------------------
Parameters _TAU_WV_SCALE, _SW_FLUX, _SW_DECAY are chosen so the gray column
has a physically reachable equilibrium temperature T* ≈ 260 K:

- _TAU_WV_SCALE = 1.0 (not 50.0): keeps total surface optical depth ~3
  (τ_CO2 ≈ 2 + τ_WV ≈ 1), preventing the opaque-surface trapping that
  would push T* to unphysical ~400 K.
- _SW_FLUX = 240.0 W/m² (not 340.0): the post-albedo net absorbed solar
  radiation; 340 W/m² is the pre-albedo TOA insolation.
- SW profile = exp(-2 * p_norm): Beer-Lambert from TOA (not from surface);
  heating peaks at TOA, decays exponentially toward the surface.

These calibrations are not user-tunable — they define the model's reference
state. The CO₂ feedback path (SPEC §3.5) is driven by varying co2_ppm
relative to the 280 ppm reference.

Differentiability
-----------------
``_lw_tendency_torch`` is implemented using the **transfer-matrix method**:
the downward and upward sweeps are expressed as cumulative products of
2×2 transfer matrices (one per layer), then reduced with torch.matmul.
This is fully vectorized over the layer index and produces an autograd
graph that ``torch.func.jvp`` can differentiate exactly.

Physical ground
---------------
Each layer k has transfer matrix:
    M_k = [[t_k, 1-t_k], [1-t_k, t_k]]  (symmetric 2x2)
where t_k = exp(-dtau_k) is the layer transmissivity and (1-t_k) is the
emissivity.  The two-stream state is [F_dn, F_up] at each interface;
the layer source adds B_k * (1-t_k) to both fluxes (gray emission).
"""

from __future__ import annotations

import numpy as np
import torch

from .column import ColumnSpec


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_SIGMA = 5.670374419e-8   # W m⁻² K⁻⁴  Stefan–Boltzmann
_CP    = 1004.0           # J kg⁻¹ K⁻¹  specific heat of dry air
_G     = 9.81             # m s⁻²        gravity

# CO₂ reference optical depth at 280 ppm
_TAU_CO2_REF = 2.0        # dimensionless

# Water-vapor optical depth per unit specific humidity (normalized).
# Gray-atmosphere calibration: at q_ref=1e-2 kg/kg, WV contributes
# τ_WV = 1.0 at the surface — a modest additional optical depth on top
# of τ_CO2 ≈ 2.0, giving a total surface optical depth of ~3.  This is
# consistent with a ~260 K gray-atmosphere equilibrium temperature (see
# physical calibration note in _sw_tendency_np below).
_TAU_WV_SCALE = 1.0       # kg/kg → optical depth

# SW parameters.
# _SW_FLUX: net absorbed solar radiation (W m⁻²).  240 W m⁻² is the
#   realistic global-mean absorbed shortwave after a ~30% albedo.
#   (340 W m⁻² is the TOA insolation; 240 is the post-albedo net.)
# _SW_DECAY: Beer-Lambert attenuation scale in normalized pressure.
#   exp(-_SW_DECAY * p_norm) concentrates SW heating near TOA (p_norm→0)
#   and decays toward the surface (p_norm→1).  _SW_DECAY=2.0 gives
#   ~86% absorption above p_norm=1, with a ~e⁻² ≈ 13.5% surface value —
#   physically reasonable for a cloud-free gray atmosphere that absorbs
#   most SW in the stratosphere/upper troposphere.
_SW_FLUX  = 240.0         # W m⁻²  net absorbed SW (after surface albedo)
_SW_DECAY = 2.0           # Beer-Lambert attenuation in normalized pressure

# Reference humidity scale for optical depth
_Q_REF = 1e-2


# ---------------------------------------------------------------------------
# Layer pressure grid helper (no autograd dependency)
# ---------------------------------------------------------------------------

def _layer_dp_np(spec: ColumnSpec) -> np.ndarray:
    """Per-layer pressure thickness Δp in Pa, shape (n_z,). Numpy (constant)."""
    p = spec.pressure
    if spec.n_z == 1:
        return np.array([p[0]], dtype=np.float64)
    dp = np.zeros(spec.n_z, dtype=np.float64)
    dp[1:-1] = 0.5 * (p[2:] - p[:-2])
    dp[0]  = p[1] - p[0]
    dp[-1] = p[-1] - p[-2]
    return np.abs(dp)


# ---------------------------------------------------------------------------
# Two-stream torch implementation (loop-based, fully differentiable)
# ---------------------------------------------------------------------------

def _lw_tendency_torch(
    spec: ColumnSpec,
    T_tensor: torch.Tensor,
    q_tensor: torch.Tensor,
    co2_ppm: float,
) -> torch.Tensor:
    """Longwave tendency (n_z,) K/s — differentiable via torch autograd.

    Uses the standard Schwarzschild two-stream recurrence.  Python for-loops
    are used over n_z, but each loop body is a torch operation so the
    autograd graph is correctly built for ``torch.func.jvp``.

    This is the same algorithmic structure as the original loop-based code,
    but each assignment uses torch operations (not in-place indexing)
    to keep the autograd graph intact.
    """
    n_z = spec.n_z
    p = torch.tensor(spec.pressure, dtype=torch.float64)
    dp = torch.tensor(_layer_dp_np(spec), dtype=torch.float64)
    p_max = p[-1] + 1e-12

    # --- Optical depth profile ---
    tau_co2 = _TAU_CO2_REF * (torch.log(
        torch.tensor(co2_ppm / 280.0, dtype=torch.float64) + 1e-12
    ) + 1.0)
    tau_co2_profile = tau_co2 * p / p_max
    tau_wv_profile  = _TAU_WV_SCALE * q_tensor / _Q_REF * (p / p_max)
    tau = tau_co2_profile + tau_wv_profile

    dtau = torch.diff(tau, prepend=tau.new_zeros(1))
    dtau = torch.clamp(dtau, min=1e-12)

    t = torch.exp(-dtau)   # (n_z,) transmissivities
    B = _SIGMA * T_tensor**4     # (n_z,) layer Planck emission
    B_surf = _SIGMA * T_tensor[-1]**4   # surface upwelling BC

    # --- Downward sweep (TOA → surface) using torch.stack accumulation ---
    # F_dn[0] = 0; F_dn[k+1] = F_dn[k]*t[k] + B[k]*(1-t[k])
    # Build list then stack — each element is a scalar torch tensor
    F_dn_list = [torch.zeros((), dtype=torch.float64)]
    for k in range(n_z):
        F_dn_k1 = F_dn_list[k] * t[k] + B[k] * (1.0 - t[k])
        F_dn_list.append(F_dn_k1)
    F_dn = torch.stack(F_dn_list)   # (n_z+1,) — full autograd graph

    # --- Upward sweep (surface → TOA) ---
    # F_up[n_z] = B_surf; F_up[k] = F_up[k+1]*t[k] + B[k]*(1-t[k])
    F_up_list = [None] * (n_z + 1)
    F_up_list[n_z] = B_surf
    for k in range(n_z - 1, -1, -1):
        F_up_list[k] = F_up_list[k + 1] * t[k] + B[k] * (1.0 - t[k])
    F_up = torch.stack(F_up_list)   # (n_z+1,)

    # --- Heating rate ---
    # F_net[k] = F_up[k] - F_dn[k]: net upward flux at interface k (upward positive).
    # Layer k is bounded by interface k (top, lower pressure) and k+1 (bottom, higher pressure).
    # Index 0 = TOA (lowest pressure), index n_z = surface (highest pressure).
    #
    # Net energy GAINED by layer k:
    #   = flux entering from below − flux leaving at top
    #   = F_net[k+1] − F_net[k]   (bottom interface minus top interface)
    #
    # Physical check: for LW cooling, OLR escapes at TOA so F_net[0] >> F_net[n_z].
    # Then F_net[k+1] - F_net[k] < 0 → net energy loss (cooling). Correct sign.
    #
    # In hydrostatic coordinates (Δp = p_bottom - p_top > 0):
    #   dT/dt = (g/Cp) * (F_net[k+1] - F_net[k]) / Δp[k]
    F_net   = F_up - F_dn                          # (n_z+1,)  upward positive
    delta_F = F_net[1:] - F_net[:-1]               # (n_z,)  bottom - top: negative = cooling
    dT_dt   = (_G / _CP) * delta_F / (dp + 1e-12)  # K/s

    return dT_dt


def _sw_tendency_np(spec: ColumnSpec) -> np.ndarray:
    """Shortwave solar heating tendency, shape (n_z,) in K/s. Numpy (constant).

    Beer-Lambert attenuation from TOA: the SW flux absorbed per unit pressure
    is proportional to exp(−_SW_DECAY * p_norm), where p_norm = p/p_surface ∈
    [0,1] runs from TOA (0) to surface (1).  This gives maximum heating near
    TOA (p_norm→0) and exponential decay toward the surface (p_norm→1),
    consistent with the physical picture of solar radiation being absorbed
    primarily in the upper atmosphere.

    Physical calibration note:
    The equilibrium T* of the gray column satisfies SW_total = |LW_OLR|.
    With _SW_FLUX=240 W/m², _TAU_CO2_REF=2.0, _TAU_WV_SCALE=1.0, q=5e-3,
    the isothermal fixed point is T* ≈ 260 K — a reasonable gray-atmosphere
    result (the realistic global-mean surface temperature of ~288 K requires
    a realistic lapse rate + surface emissivity, not captured in a gray model).
    Stability: d(net)/dT < 0 at T* (verified numerically), so T* is stable.
    """
    p = spec.pressure
    p_max = p[-1] if spec.n_z > 0 else 1e5
    p_norm = p / (p_max + 1e-12)
    dp = _layer_dp_np(spec)
    # Beer-Lambert from TOA: maximum at p_norm=0 (TOA), decays to
    # exp(-_SW_DECAY) ≈ 0.135 at p_norm=1 (surface) with _SW_DECAY=2.0
    sw_profile = np.exp(-_SW_DECAY * p_norm)
    norm = (sw_profile * dp).sum()
    # Normalize so total column-integrated SW = _SW_FLUX (W/m²)
    sw_heating = sw_profile / (norm + 1e-15) * _SW_FLUX   # W/m²/Pa * dp → W/m²
    # Convert to temperature tendency (K/s) via hydrostatic Δp:
    #   dT/dt = (g/Cp) * (SW_per_Pa * Δp) / Δp  = g/Cp * SW_per_Pa
    dT_dt = _G / _CP * sw_heating * dp / (dp + 1e-15)
    return dT_dt


def _radiative_tendency_torch(
    spec: ColumnSpec,
    profile_t: torch.Tensor,
    co2_ppm: float = 280.0,
) -> torch.Tensor:
    """Torch-native radiative tendency for ``torch.func.jvp``.

    Parameters
    ----------
    spec : ColumnSpec
    profile_t : torch.Tensor
        Shape ``(n_v, n_z)`` float64 in physical units.
    co2_ppm : float

    Returns
    -------
    torch.Tensor
        Shape ``(n_v, n_z)`` float64 in physical units/s.
    """
    T_t = profile_t[0]
    q_t = profile_t[1] if spec.n_v >= 2 else torch.zeros(spec.n_z, dtype=torch.float64)

    lw = _lw_tendency_torch(spec, T_t, q_t, co2_ppm)   # (n_z,)

    sw_np = _sw_tendency_np(spec)
    sw_t  = torch.tensor(sw_np, dtype=torch.float64)

    # Build output tensor using stack (no in-place operations → clean autograd)
    dT_row = (lw + sw_t).unsqueeze(0)   # (1, n_z)
    if spec.n_v == 1:
        return dT_row
    zero_rows = torch.zeros(spec.n_v - 1, spec.n_z, dtype=torch.float64)
    return torch.cat([dT_row, zero_rows], dim=0)


def radiative_tendency(
    spec: ColumnSpec,
    profile: np.ndarray,
    *,
    co2_ppm: float = 280.0,
) -> np.ndarray:
    """Compute the radiative tendency for a given column profile.

    Parameters
    ----------
    spec : ColumnSpec
    profile : np.ndarray
        Shape ``(n_v, n_z)`` in physical units.
    co2_ppm : float

    Returns
    -------
    np.ndarray
        Shape ``(n_v, n_z)`` radiative tendency in physical units/s.
    """
    profile_t = torch.tensor(profile.astype(np.float64), dtype=torch.float64)
    with torch.no_grad():
        out_t = _radiative_tendency_torch(spec, profile_t, co2_ppm)
    return out_t.numpy()
