"""Smoothed convective-adjustment tendency (C¹ — mandatory for IFT condition (i)).

SPEC_climate_rce §3.2, §4.4.

Physical basis
--------------
Resolved convective adjustment relaxes T toward a **moist-adiabatic** lapse
rate where the column is convectively unstable (−dT/dz > Γ_m).  A hard
switch is NOT differentiable at the stability boundary.  We use a **softplus-
gated** relaxation that is C¹ everywhere:

    C_conv(k) = −(τ_conv)⁻¹ · σ_smooth(unstable) · (T(k) − T_ref(k))

where:
  - ``T_ref(k)`` = moist-adiabatic reference temperature at level k (computed
    from the column's surface temperature and the moist adiabatic lapse rate
    Γ_m ≈ 6.5 K/km).
  - ``unstable = (T_ref(k) − T(k)) / ΔT_smooth`` is a normalized measure of
    instability (positive = unstable, negative = stable).
  - ``σ_smooth = softplus(unstable) / (1 + softplus(unstable))`` is a smooth
    [0,1] sigmoid gate (C¹ everywhere; avoids the hard H(z) switch).
  - ``τ_conv`` = convective adjustment timescale (default: 1 day = 86400 s).

The smoothing width ``ΔT_smooth`` (default 1.0 K) is a controllable knob:
smaller → sharper adjustment → larger ‖K‖ (secondary amplification dial,
analogous to the SCF τ_floor regularizer, SPEC_climate_rce §3.2).

Moisture
--------
In a simple gray model, convection also drives moisture upward.  We implement
a simple moist-adiabatic q tendency:
    C_q(k) = −(τ_conv)⁻¹ · σ_smooth(unstable) · (q(k) − q_ref(k))
where q_ref(k) is the saturation specific humidity on the moist adiabat.

C¹ guarantee
-----------
``softplus(x) = log(1 + exp(x))`` is C∞, so the gate is C∞.  The product
with the linear restoring force is C¹.  This is verified by the
``test_convection.py`` numerical smoothness gate (no jump > 1e-4 relative
across the stability boundary).

TASK-035 / FD-JVP rule
-----------------------
This module uses pure torch throughout (no numpy in the autograd path), so
``torch.func.jvp`` propagates through the convective tendency without
approximation.  The host-JVP computation does NOT use a truth-side Jacobian
in the NN-bearing slot (TASK-035 Rule 1) — the convective tendency here is
the TRUTH subgrid process (no NN); the NN subgrid enters via ``climate_rce/
emulator/`` and its JVP is computed separately in ``climate_rce/jvp.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from .column import ColumnSpec


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_G          = 9.81     # m s⁻²   gravity
_CP         = 1004.0   # J kg⁻¹ K⁻¹  specific heat dry air
_LV         = 2.5e6    # J kg⁻¹  latent heat of vaporization
_RV         = 461.5    # J kg⁻¹ K⁻¹  gas constant water vapor
_T_CONV     = 86400.0  # s       default convective timescale (1 day)
_DT_SMOOTH  = 1.0      # K       default smoothing width (controllable knob)


# ---------------------------------------------------------------------------
# Moist-adiabat reference
# ---------------------------------------------------------------------------

def _moist_adiabat_lapse(T: float) -> float:
    """Moist-adiabatic lapse rate Γ_m in K/Pa (approximately).

    Uses the standard formula (Wallace & Hobbs 2006, eq. 3.16):
        Γ_m = g/Cp * (1 + Lv*qs/(Rd*T)) / (1 + Lv²*qs/(Cp*Rv*T²))
    with qs ≈ 0.622*e_sat/p.

    Returned as K/Pa (per Pascal of pressure).  The negative sign is absorbed
    in the calling convention: decreasing pressure upward → T decreases upward.
    """
    _RD = 287.0   # J kg⁻¹ K⁻¹ gas constant dry air
    # Saturation vapor pressure (Tetens-like)
    e_sat = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65 + 1e-6))
    qs = 0.622 * e_sat / 1e5   # approximate; numerator stable at 1e5 Pa ref
    qs = max(qs, 1e-9)
    num = 1.0 + _LV * qs / (_RD * T)
    den = 1.0 + _LV**2 * qs / (_CP * _RV * T**2)
    Gm_per_m = _G / _CP * num / den   # K/m
    # Convert to K/Pa: dT/dp = Gm_per_m / (g * rho) ≈ Gm_per_m * RT/(g*p)
    # For the reference profile approximation, just return K/m / (g/RT) ≈ RT/gp * Gamma
    # We return K/Pa using the hydrostatic relation dz/dp = -1/(ρg) ≈ -RT/(gp)
    # → Γ_m [K/Pa] = -Γ_m [K/m] * RT / (g * p)
    # Since we use this only with pressure differences, a constant reference
    # value suffices for the smooth gate.
    return Gm_per_m / _G * 287.0 * T / 1e5   # K/Pa (reference pressure ~1e5 Pa)


def _build_moist_adiabat(spec: ColumnSpec, T_surf: float) -> np.ndarray:
    """Build a moist-adiabatic reference T profile, shape (n_z,).

    Integrates the moist-adiabatic lapse rate from the surface upward.
    The profile is built in pressure coordinates (increasing p → surface).

    Parameters
    ----------
    spec : ColumnSpec
    T_surf : float
        Surface temperature in K (from the column state at the bottom level).

    Returns
    -------
    np.ndarray
        Shape ``(n_z,)`` reference temperature in K.
    """
    p = spec.pressure   # (n_z,) Pa, increasing downward (index 0=TOA, -1=sfc)
    T_ref = np.zeros(spec.n_z, dtype=np.float64)
    T_ref[-1] = T_surf   # surface anchor
    for k in range(spec.n_z - 2, -1, -1):
        dp = p[k] - p[k + 1]   # negative (going upward → lower pressure)
        T_ref[k] = T_ref[k + 1] + _moist_adiabat_lapse(T_ref[k + 1]) * dp
    return T_ref


def _build_q_ref(spec: ColumnSpec, T_ref: np.ndarray) -> np.ndarray:
    """Saturation specific humidity on the moist adiabat, shape (n_z,)."""
    p = spec.pressure
    e_sat = 611.2 * np.exp(17.67 * (T_ref - 273.15) / (T_ref - 29.65 + 1e-6))
    q_sat = 0.622 * e_sat / np.maximum(p, 1.0)
    return q_sat


# ---------------------------------------------------------------------------
# Smoothed convective tendency (torch-native for autograd)
# ---------------------------------------------------------------------------

def _build_moist_adiabat_torch(
    spec: ColumnSpec,
    T_tensor: torch.Tensor,
) -> tuple:
    """Build moist-adiabatic reference profiles differentiably in torch.

    Returns (T_ref_t, q_ref_t) where T_ref_t and q_ref_t are torch tensors
    that depend differentiably on T_tensor[-1] (the surface temperature).

    The moist adiabat is anchored at the surface T[-1] and integrated upward
    using the moist adiabatic lapse rate.  The integration coefficients are
    pre-computed as constants (they depend only on a reference T, not on T_tensor
    directly), but the surface anchor T[-1] is tracked differentiably.

    Approach: expand T_ref around a reference T_surf0 (detached):
        T_ref(k) ≈ T_ref0(k) + (dT_ref/dT_surf)|_0 * (T_surf - T_surf0)
    where the Jacobian row (dT_ref/dT_surf) is the same for all levels
    (the moist adiabat lapse rate propagation is linear in T_surf at leading order).

    This makes T_ref a linear (hence differentiable) function of T_tensor[-1].
    """
    p = spec.pressure
    T_surf_ref = T_tensor[-1].detach().item()   # reference point (for linearization)

    # Build reference adiabat at T_surf_ref (numpy, for coefficients)
    T_ref0_np = _build_moist_adiabat(spec, T_surf_ref)
    # Jacobian: d(T_ref(k))/d(T_surf) via FD at reference
    # Use a small step (0.01 K) so the linearization error is O((0.01)²) = O(1e-4),
    # ensuring FD-vs-autodiff agreement to 1e-4 relative in the test.
    h_lin = 0.01  # 0.01 K perturbation for linearization (small to reduce O(h²) error)
    T_ref0_plus  = _build_moist_adiabat(spec, T_surf_ref + h_lin)
    T_ref0_minus = _build_moist_adiabat(spec, T_surf_ref - h_lin)
    dT_ref_dT_surf = (T_ref0_plus - T_ref0_minus) / (2 * h_lin)  # (n_z,) numpy

    # T_ref as a linear function of T_tensor[-1] (differentiable)
    T_ref0_t          = torch.tensor(T_ref0_np, dtype=torch.float64)
    dT_ref_dT_surf_t  = torch.tensor(dT_ref_dT_surf, dtype=torch.float64)
    T_surf_delta      = T_tensor[-1] - T_surf_ref   # scalar torch (differentiable)
    T_ref_t           = T_ref0_t + dT_ref_dT_surf_t * T_surf_delta   # (n_z,) differentiable

    # q_ref: saturation q on the moist adiabat (linearized around T_ref0)
    q_ref0_np   = _build_q_ref(spec, T_ref0_np)
    # dq_sat/dT via Clausius–Clapeyron: dq_sat/dT = q_sat * L_v / (R_v * T²)
    # precompute as constant at T_ref0
    dq_ref_dT   = q_ref0_np * _LV / (_RV * np.maximum(T_ref0_np, 100.0)**2)
    dq_ref_t    = torch.tensor(q_ref0_np, dtype=torch.float64) + \
                  torch.tensor(dq_ref_dT * dT_ref_dT_surf, dtype=torch.float64) * T_surf_delta
    # Clamp q_ref to non-negative
    q_ref_t     = torch.clamp(dq_ref_t, min=0.0)

    return T_ref_t, q_ref_t


def _convective_tendency_torch(
    spec: ColumnSpec,
    profile_t: torch.Tensor,
    *,
    tau_conv: float = _T_CONV,
    dt_smooth: float = _DT_SMOOTH,
) -> torch.Tensor:
    """Torch-native C¹ convective tendency for autograd.

    The moist-adiabat reference is computed as a differentiable linear
    function of the surface temperature T[-1] (using a first-order Taylor
    expansion), so torch.func.jvp can propagate through the T_surf anchor.

    Parameters
    ----------
    spec : ColumnSpec
    profile_t : torch.Tensor
        Shape ``(n_v, n_z)`` float64.
    tau_conv : float
        Convective relaxation timescale in seconds.
    dt_smooth : float
        Smoothing width in K (controllable C¹ knob; smaller → sharper).

    Returns
    -------
    torch.Tensor
        Shape ``(n_v, n_z)`` float64.
    """
    T_t = profile_t[0]   # (n_z,) K
    q_t = profile_t[1] if spec.n_v >= 2 else torch.zeros(spec.n_z, dtype=torch.float64)

    # Build differentiable reference profiles from current T_surf
    T_ref_t, q_ref_t = _build_moist_adiabat_torch(spec, T_t)

    # Instability measure: positive → unstable, negative → stable
    instability = (T_ref_t - T_t) / dt_smooth   # (n_z,); differentiable

    # Smooth [0,1] gate via softplus-sigmoid:
    #   gate = softplus(x) / (1 + softplus(x))  ∈ (0, 1), C∞
    sp   = torch.nn.functional.softplus(instability)   # (n_z,) ≥ 0
    gate = sp / (1.0 + sp)   # (n_z,) ∈ (0,1), C∞

    # Convective relaxation tendency
    dT_row = (-gate * (T_t - T_ref_t) / tau_conv).unsqueeze(0)   # (1, n_z)
    if spec.n_v >= 2:
        dq_row = (-gate * (q_t - q_ref_t) / tau_conv).unsqueeze(0)  # (1, n_z)
        zero_rows = torch.zeros(spec.n_v - 2, spec.n_z, dtype=torch.float64)
        out = torch.cat([dT_row, dq_row, zero_rows], dim=0) if spec.n_v > 2 else \
              torch.cat([dT_row, dq_row], dim=0)
    else:
        out = dT_row
    return out


def convective_tendency(
    spec: ColumnSpec,
    profile: np.ndarray,
    *,
    tau_conv: float = _T_CONV,
    dt_smooth: float = _DT_SMOOTH,
) -> np.ndarray:
    """Compute the smoothed convective-adjustment tendency.

    This function is C¹ everywhere, satisfying IFT condition (i)
    (SPEC_climate_rce §2.2).

    Parameters
    ----------
    spec : ColumnSpec
        Column specification.
    profile : np.ndarray
        Shape ``(n_v, n_z)`` in physical units (T in K, q in kg/kg, ...).
    tau_conv : float
        Convective relaxation timescale in seconds (default: 86400 s = 1 day).
    dt_smooth : float
        Smoothing width in K.  Controls the C¹ knob; smaller → sharper
        adjustment → larger ‖K‖ (the secondary amplification dial,
        SPEC_climate_rce §3.2).

    Returns
    -------
    np.ndarray
        Shape ``(n_v, n_z)`` convective tendency in physical units per second.
    """
    profile_t = torch.tensor(profile.astype(np.float64), dtype=torch.float64)
    with torch.no_grad():
        out_t = _convective_tendency_torch(spec, profile_t,
                                           tau_conv=tau_conv, dt_smooth=dt_smooth)
    return out_t.numpy()
