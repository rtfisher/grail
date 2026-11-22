"""GRAIL ↔ RCE integration: JVP wrappers and cycle-path adapter.

SPEC_climate_rce §4.2, §4.3, §6 P3.

This module wires the RCE column physics and the trained ``SubgridNet`` into
GRAIL's ``resolvent_solve`` for the climate RCE testbed.  Public entry points:

- :func:`make_subgrid_with_nn` — returns a ``subgrid_tendency`` closure
  (the ``equilibrate_to_rce`` contract) that wraps a SubgridNet.
- :func:`make_host_jvp_rce` — the GRAIL ``host_jvp`` slot.
- :func:`make_emulator_jvp_rce` — the GRAIL ``emulator_jvp`` slot.
- :func:`make_identity_emulator_jvp` — identity callback (Path B additive
  workaround partner).
- :func:`make_back_residual` — computes ``δ𝒯 = Δt·(S^NN(x*) − S^truth(x*))``
  (the GRAIL RHS).
- :func:`make_host_jvp_fd` — central-FD cross-check / fallback.

================================================================================
PATH A — COMPOSITIONAL (DEFAULT).  SPEC_climate_rce §4.2 / INTERFACES.md §4.2.
================================================================================

The cycle map is (SPEC §2.1 eq. 1):

    𝒯(x) = x + Δt · [ R(x) + C(x) + S^NN(x) ]

The equilibration driver implements this as a **split-implicit composition**:
    1. 𝒯_rad:  update T via radiation R (NN-free).
    2. 𝒯_phys: apply convective + subgrid adjustments C + S^NN.

This maps natively onto GRAIL's ``composed_jvp(emulator_jvp, host_jvp)``:
    - GRAIL ``host_jvp``     ←→  D𝒯_rad  (first-applied; NN-FREE)
    - GRAIL ``emulator_jvp`` ←→  D𝒯_phys (second-applied; CONTAINS the NN)

**NO slot-naming inversion in Path A** — this is the ONLY testbed where the
slot names align with their literal meanings (host = no NN, emulator = has NN).
This confirms the deferred ``first_jvp``/``second_jvp`` rename is the right
long-term fix (the callback contract "Slot-naming wart").

================================================================================
SLOT-NAMING WART — documentation for Path A (SPEC_climate_rce §4.2 / the callback contract)
================================================================================

GRAIL's :func:`grail.jacobian.composed_jvp` applies the slots in this order::

    composed_jvp(emulator_jvp, host_jvp)(v) = emulator_jvp(host_jvp(v))

i.e. ``host_jvp`` is applied FIRST.  For Path A:

    GRAIL slot           ↔   RCE step            ↔   NN lives here?
    -----------------------------------------------------------------
    ``host_jvp``         ↔   D𝒯_rad              ↔   NO  (radiation only)
    ``emulator_jvp``     ↔   D𝒯_phys             ↔   YES (subgrid NN)

Check (Path A):
    composed_jvp(emul, host)(v) = emul(host(v))
                                = D𝒯_phys(D𝒯_rad(v))
                                = D(𝒯_phys ∘ 𝒯_rad)(v)
                                = D𝒯(v)  ✓

    resolvent_matvec(emul, host)(v)
      = v − emul(host(v))
      = v − D𝒯(v)
      = (I − D𝒯)(v)  ✓

So GRAIL solves ``(I − D𝒯)·δx* = δ𝒯`` — exactly the resolvent identity.

For Path B (additive workaround, available but non-default):
    - ``host_jvp(v)``     ≔ ``v − K·v``  (full cycle; NN-bearing)
    - ``emulator_jvp(u)`` ≔ ``u``         (identity)
    Then ``resolvent_matvec(emul, host)(v) = v − (v − K·v) = K·v``,
    and GRAIL solves ``(I−K)·δx* = δ𝒯``.

================================================================================
TASK-035 FD-JVP RULE applied here.  SPEC_climate_rce §4.3.
================================================================================

**Rule 1 (architectural, hard, NEVER VIOLATE):**
The ``emulator_jvp`` (Path A) or NN-bearing ``host_jvp`` (Path B) MUST
differentiate the **emulator's** Jacobian ``∂S^NN/∂x``, NOT the truth subgrid
scheme ``∂S^truth/∂x``.  This is the NLTE TASK-035 hazard (wiring truth-SE
autograd → Born ratio ~478 vs bound ~1.42).  **In Path A:**
``make_emulator_jvp_rce`` differentiates the ``SubgridNet`` + convection
combined tendency ``C + S^NN``, NOT the truth convective tendency alone.

**Rule 2 (method-of-computation, soft):**
Default = ``torch.func.jvp`` through the NN-replaced physics step
(``C + S^NN`` differentiable in torch).  Central-FD (``h=1e-4`` non-dim)
is the cross-check fixture and opaque-NN fallback (``make_host_jvp_fd``).

================================================================================
Fixed-RH / T-only seam.  SPEC §2.2.1, INTERFACES.md §11.
================================================================================

All JVP callables operate on the **T-only GRAIL state** (``n_state = n_z``).
Physical profiles ``(n_v, n_z)`` are reconstructed from T via the fixed-RH
closure q = q_ref(T_surf).  The cycle Jacobian is ``K_eff`` (the WV-feedback-
inclusive T-cycle Jacobian, n_z × n_z); the literal ``(T,q)`` Jacobian
``K_full`` is singular and never formed here.

================================================================================
DIFFERENTIABILITY NOTE — moist adiabat in torch.
================================================================================

The WV-feedback coupling goes through q = q_ref(T_surf), where q_ref is the
saturation specific humidity on the moist adiabat anchored at T_surf = T[-1].

The frozen ``convection.py::_build_moist_adiabat_torch`` uses ``detach()`` to
break the autograd graph for T_surf, making ``torch.func.jvp`` give an
incomplete Jacobian (it misses the ∂R/∂q * ∂q_ref/∂T_surf WV-feedback term).
This module implements ``_build_moist_adiabat_exact_torch`` (a pure-torch
autograd-compatible version, no detach) so that both autograd and FD
differentiate the SAME function to float64 precision.

This is the load-bearing differentiability fix that enables the SPEC §6 P3
gate: ``composed_jvp(emul, host)(v) == K_eff·v to 1e-7``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.func

from climate_rce.column import ColumnSpec, ColumnState, pack_state, unpack_state
from climate_rce.radiation import _lw_tendency_torch, _sw_tendency_np
from climate_rce.emulator.network import SubgridNet


# ---------------------------------------------------------------------------
# FD step (TASK-035 convention: h=1e-4 non-dim)
# ---------------------------------------------------------------------------

# SPEC §4.3 / the callback contract TASK-035: central-FD step in non-dimensional units.
# The non-dimensional state is T / T_scale (T_scale ≈ 300 K), so h=1e-4
# corresponds to a physical perturbation of 1e-4 × 300 K = 0.03 K.
# This is safely perturbative (0.01% of T_scale) and well above float64
# round-off (ε_64 ≈ 2e-16 → FD noise ~ ε_64/h ~ 2e-12 << 1e-5 tolerance).
_FD_STEP = 1e-4

# Physical constants (matching convection.py for consistency)
_LV = 2.5e6    # J kg⁻¹ latent heat of vaporization
_RV = 461.5    # J kg⁻¹ K⁻¹ gas constant water vapor
_G  = 9.81     # m s⁻²  gravity
_CP = 1004.0   # J kg⁻¹ K⁻¹ specific heat dry air


# ---------------------------------------------------------------------------
# Exact differentiable moist adiabat (no detach — load-bearing for autograd)
# ---------------------------------------------------------------------------

def _moist_adiabat_lapse_torch(T_k: torch.Tensor) -> torch.Tensor:
    """Moist-adiabatic lapse rate Γ_m [K/Pa] — differentiable via torch.

    Identical formula to ``convection.py::_moist_adiabat_lapse`` but expressed
    in torch ops so ``torch.func.jvp`` can propagate tangents through it.

    Parameters
    ----------
    T_k : Tensor  scalar float64, temperature in K at level k

    Returns
    -------
    Tensor  scalar float64, lapse rate in K/Pa
    """
    _RD = 287.0
    # Saturation vapor pressure (Tetens-like, all torch ops)
    e_sat = 611.2 * torch.exp(17.67 * (T_k - 273.15) / (T_k - 29.65 + 1e-6))
    qs = torch.clamp(0.622 * e_sat / 1e5, min=1e-9)
    num = 1.0 + _LV * qs / (_RD * T_k)
    den = 1.0 + _LV**2 * qs / (_CP * _RV * T_k**2)
    Gm_per_m = _G / _CP * num / den
    return Gm_per_m / _G * 287.0 * T_k / 1e5  # K/Pa


def _build_moist_adiabat_exact_torch(
    spec: ColumnSpec,
    T_surf_t: torch.Tensor,
) -> torch.Tensor:
    """Moist-adiabatic reference T profile — differentiable in T_surf.

    Integrates the moist adiabatic lapse rate from the surface upward in
    pure torch (no ``detach()``), so ``torch.func.jvp`` can propagate
    tangents through T_surf to all levels.

    Parameters
    ----------
    spec : ColumnSpec
    T_surf_t : Tensor  scalar float64, surface temperature in K

    Returns
    -------
    Tensor  shape (n_z,) float64, reference temperature profile in K
    """
    p = spec.pressure  # (n_z,) Pa, index 0=TOA, -1=surface
    n_z = spec.n_z

    # Integrate from surface (index -1) upward to TOA (index 0).
    # Accumulate levels in a list then stack — no in-place writes → clean autograd.
    T_levels = [None] * n_z
    T_levels[n_z - 1] = T_surf_t  # surface anchor (differentiable)
    for k in range(n_z - 2, -1, -1):
        dp = p[k] - p[k + 1]  # negative (going upward)
        lapse = _moist_adiabat_lapse_torch(T_levels[k + 1])
        T_levels[k] = T_levels[k + 1] + lapse * dp
    return torch.stack(T_levels)  # (n_z,)


def _build_q_ref_exact_torch(
    spec: ColumnSpec,
    T_ref_t: torch.Tensor,
) -> torch.Tensor:
    """Saturation specific humidity on the moist adiabat — differentiable.

    Parameters
    ----------
    spec : ColumnSpec
    T_ref_t : Tensor  shape (n_z,) float64, reference temperature in K

    Returns
    -------
    Tensor  shape (n_z,) float64, saturation q in kg/kg
    """
    p_t = torch.tensor(spec.pressure, dtype=torch.float64)
    e_sat = 611.2 * torch.exp(
        17.67 * (T_ref_t - 273.15) / (T_ref_t - 29.65 + 1e-6)
    )
    q_sat = 0.622 * e_sat / torch.clamp(p_t, min=1.0)
    return q_sat


def _T_only_to_profile_exact(
    x_T: torch.Tensor,
    spec: ColumnSpec,
) -> torch.Tensor:
    """Convert a T-only non-dim flat tensor to a full (n_v, n_z) torch profile.

    Uses the exact differentiable moist adiabat (no detach) so that
    ``torch.func.jvp`` can propagate tangents through the WV-feedback coupling.

    This replaces the use of ``climate_rce.convection._build_moist_adiabat_torch``
    (which uses ``detach()`` and breaks autograd-vs-FD consistency).

    Parameters
    ----------
    x_T : Tensor  (n_z,) float64 non-dim T-only state
    spec : ColumnSpec

    Returns
    -------
    Tensor  (n_v, n_z) float64 physical units (T in K, q from exact moist adiabat)
    """
    T_scale = spec.var_scales[0]
    T_t = x_T * T_scale  # (n_z,) K, differentiable

    if spec.n_v == 1:
        return T_t.unsqueeze(0)

    # Build q via exact (fully differentiable) moist adiabat
    T_surf_t = T_t[-1]  # surface temperature (differentiable)
    T_ref_t = _build_moist_adiabat_exact_torch(spec, T_surf_t)
    q_ref_t = _build_q_ref_exact_torch(spec, T_ref_t)

    rows = [T_t, q_ref_t]
    for _ in range(spec.n_v - 2):
        rows.append(torch.zeros(spec.n_z, dtype=torch.float64))
    return torch.stack(rows, dim=0)  # (n_v, n_z)


def _radiative_tendency_exact_torch(
    spec: ColumnSpec,
    x_T: torch.Tensor,
    co2_ppm: float,
) -> torch.Tensor:
    """Radiative tendency via exact differentiable profile, shape (n_z,) K/s.

    Computes the radiation tendency using a fully differentiable moist adiabat
    for q (no detach), enabling correct autograd-vs-FD agreement.

    Parameters
    ----------
    spec : ColumnSpec
    x_T : Tensor  (n_z,) float64 non-dim T-only state
    co2_ppm : float

    Returns
    -------
    Tensor  (n_z,) float64 T-block radiative tendency in K/s
    """
    profile_t = _T_only_to_profile_exact(x_T, spec)
    T_t = profile_t[0]
    q_t = profile_t[1] if spec.n_v >= 2 else torch.zeros(spec.n_z, dtype=torch.float64)

    lw = _lw_tendency_torch(spec, T_t, q_t, co2_ppm)  # (n_z,)
    sw_np = _sw_tendency_np(spec)
    sw_t = torch.tensor(sw_np, dtype=torch.float64)
    return lw + sw_t  # (n_z,) K/s


# ---------------------------------------------------------------------------
# Internal torch-native cycle steps (Path A) — exact autograd
# ---------------------------------------------------------------------------

def _rad_step_exact(
    spec: ColumnSpec,
    x_T: torch.Tensor,
    *,
    dt: float,
    co2_ppm: float,
) -> torch.Tensor:
    """Radiative cycle substep 𝒯_rad: x → x + Δt·R(x).  Exact autograd.

    This is the **first-applied** (host) substep in Path A.  NN-FREE.
    Uses ``_T_only_to_profile_exact`` (no detach) so ``torch.func.jvp``
    gives the correct Jacobian including the WV-feedback term
    ∂R/∂q * ∂q_ref/∂T_surf.

    Parameters
    ----------
    spec : ColumnSpec
    x_T : Tensor (n_z,) float64 non-dim T-only state
    dt : float  pseudo-timestep in seconds
    co2_ppm : float  CO₂ forcing

    Returns
    -------
    Tensor (n_z,) float64 non-dim updated T state
    """
    T_scale = spec.var_scales[0]
    R_T_t = _radiative_tendency_exact_torch(spec, x_T, co2_ppm)  # (n_z,) K/s
    return x_T + (dt / T_scale) * R_T_t


def _convective_tendency_exact_torch(
    spec: ColumnSpec,
    x_T: torch.Tensor,
) -> torch.Tensor:
    """Convective tendency using exact differentiable moist adiabat, (n_z,) K/s.

    Uses ``_T_only_to_profile_exact`` (no detach) so autograd and FD agree
    to float64 precision.

    Parameters
    ----------
    spec : ColumnSpec
    x_T : Tensor (n_z,) float64 non-dim T-only state

    Returns
    -------
    Tensor (n_z,) float64 T-block convective tendency in K/s
    """
    from climate_rce.convection import _T_CONV, _DT_SMOOTH

    profile_t = _T_only_to_profile_exact(x_T, spec)
    T_t = profile_t[0]   # (n_z,) K
    q_t = profile_t[1] if spec.n_v >= 2 else torch.zeros(spec.n_z, dtype=torch.float64)

    # Build reference profiles from the SAME exact adiabat
    T_surf_t = T_t[-1]
    T_ref_t = _build_moist_adiabat_exact_torch(spec, T_surf_t)
    q_ref_t = _build_q_ref_exact_torch(spec, T_ref_t)

    tau_conv = _T_CONV
    dt_smooth = _DT_SMOOTH

    # Instability measure (positive = unstable)
    instability = (T_ref_t - T_t) / dt_smooth

    # Smooth gate (softplus-sigmoid, C∞)
    sp = torch.nn.functional.softplus(instability)
    gate = sp / (1.0 + sp)

    # T-block convective tendency (K/s)
    return -gate * (T_t - T_ref_t) / tau_conv  # (n_z,)


def _phys_step_nn_exact(
    spec: ColumnSpec,
    x_T: torch.Tensor,
    model: SubgridNet,
    *,
    dt: float,
    T_scale: float,
    output_timescale: float | None = None,
) -> torch.Tensor:
    """Physics cycle substep 𝒯_phys: x → x + Δt·(C(x) + S^NN(x)).  Exact autograd.

    This is the **second-applied** (emulator) substep in Path A.  CONTAINS
    the NN emulator.  Uses the exact differentiable convection.

    TASK-035 Rule 1: JVP differentiates S^NN, NOT S^truth.

    Parameters
    ----------
    spec : ColumnSpec
    x_T : Tensor (n_z,) float64 non-dim T-only state
    model : SubgridNet
    dt : float  pseudo-timestep
    T_scale : float  temperature scale

    Returns
    -------
    Tensor (n_z,) float64 non-dim updated T state
    """
    # Convective tendency (truth; exact differentiable)
    C_T_t = _convective_tendency_exact_torch(spec, x_T)  # (n_z,) K/s

    # SubgridNet tendency (NN-bearing; TASK-035 Rule 1: JVP from NN).
    # Model outputs are increments over their fixed physical output timescale.
    s_nn_nondim = model(x_T)  # (n_z,) float64 non-dim
    timescale = dt if output_timescale is None else output_timescale

    # Update T-only state (non-dim)
    return x_T + (dt / T_scale) * C_T_t + (dt / timescale) * s_nn_nondim


# ---------------------------------------------------------------------------
# make_subgrid_with_nn
# ---------------------------------------------------------------------------

def make_subgrid_with_nn(
    model: SubgridNet,
    *,
    T_scale: float = 300.0,
    dt: float = 1.0,
    output_timescale: float | None = None,
) -> Callable[[ColumnSpec, np.ndarray], np.ndarray]:
    """Return a ``subgrid_tendency`` closure wrapping a SubgridNet.

    The returned callable matches the ``equilibrate_to_rce`` contract:
        ``(ColumnSpec, np.ndarray (n_v, n_z) physical) → np.ndarray (n_v, n_z) physical/s``

    The SubgridNet operates on the **T-only non-dim GRAIL state** (SPEC §2.2.1):
      1. Pack T block from the physical profile: x = T / T_scale.
      2. Evaluate ``SubgridNet.predict(x)`` → non-dim tendency.
      3. Convert to physical T-tendency: S_T [K/s] = NN_out * T_scale / output_timescale.

    Parameters
    ----------
    model : SubgridNet
    T_scale : float  Temperature scale (default 300 K).
    dt : float
        Legacy output timescale, used only when output_timescale is omitted.
    output_timescale : float or None
        Fixed physical timescale in seconds represented by a model output.
        Hold this fixed when varying the numerical cycle timestep. The returned
        callable records it so the autograd callback uses the same heating law.
        None preserves the legacy conversion using dt.

    Returns
    -------
    Callable[[ColumnSpec, np.ndarray], np.ndarray]
        Subgrid tendency in physical units (K/s for T-block, 0 for q-block).
    """
    timescale = float(dt if output_timescale is None else output_timescale)
    if not np.isfinite(timescale) or timescale <= 0:
        raise ValueError("output_timescale must be finite and positive")

    def _subgrid_nn(spec: ColumnSpec, profile: np.ndarray) -> np.ndarray:
        T_vec = profile[0]          # (n_z,) K
        x_nondim = T_vec / T_scale  # (n_z,) non-dim
        s_nondim = model.predict(x_nondim)   # (n_z,) float64
        s_T_phys = s_nondim * T_scale / timescale   # (n_z,) K/s
        out = np.zeros((spec.n_v, spec.n_z), dtype=np.float64)
        out[0] = s_T_phys
        return out

    _subgrid_nn.output_timescale = timescale
    return _subgrid_nn


# ---------------------------------------------------------------------------
# make_host_jvp_rce (Path A: D𝒯_rad; Path B: v ↦ v − K·v)
# ---------------------------------------------------------------------------

def make_host_jvp_rce(
    spec: ColumnSpec,
    x_star: np.ndarray,
    subgrid_tendency: Callable,
    *,
    path: str = "compositional",
    dt: float = 1e4,
    co2_ppm: float = 280.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the GRAIL ``host_jvp`` slot for the RCE testbed.

    **Path A (default, "compositional"):**
    Returns ``v ↦ D𝒯_rad · v`` — the radiative substep Jacobian (NN-FREE).
    Computed via ``torch.func.jvp`` through ``_rad_step_exact``, which uses a
    fully differentiable moist adiabat (no ``detach()``).  This ensures
    autograd and FD agree to float64 precision (SPEC §6 P3 gate).

    **Path B ("additive"):**
    Returns ``v ↦ v − K·v`` — the full cycle Jacobian, NN-bearing.
    Used with ``make_identity_emulator_jvp``.  See the additive-cycle
    workaround in SPEC §4.2 and the callback contract.

    TASK-035 Rule 1 in Path A: the host_jvp slot differentiates 𝒯_rad
    (which is NN-FREE), so there is NO risk of wiring truth-side Jacobian.
    The emulator_jvp slot (``make_emulator_jvp_rce``) is the NN-bearing slot.

    Parameters
    ----------
    spec : ColumnSpec
    x_star : ndarray (n_v, n_z) or (n_z,) converged RCE state
    subgrid_tendency : Callable
        Subgrid tendency (S^NN). For Path B only.
    path : str  ``"compositional"`` (default) or ``"additive"``.
    dt : float  Pseudo-timestep in seconds.
    co2_ppm : float  CO₂ forcing in ppm.

    Returns
    -------
    Callable[[Tensor], Tensor]
        JVP callable mapping ``(n_state,)`` float64 → ``(n_state,)`` float64.
    """
    T_scale = float(spec.var_scales[0])
    n_state = spec.n_state

    # Extract T-only fixed point (non-dim)
    if x_star.ndim == 2:
        T_star_nondim = torch.tensor(x_star[0] / T_scale, dtype=torch.float64)
    else:
        T_star_nondim = torch.tensor(x_star / T_scale, dtype=torch.float64)

    if path == "compositional":
        # Path A: host_jvp = D𝒯_rad (NN-FREE, first-applied).
        # Use _rad_step_exact (exact differentiable moist adiabat; no detach).
        # This ensures torch.func.jvp agrees with FD to float64 precision.
        def _rad_step(x_T: torch.Tensor) -> torch.Tensor:
            return _rad_step_exact(spec, x_T, dt=dt, co2_ppm=co2_ppm)

        def _host_jvp_a(v: torch.Tensor) -> torch.Tensor:
            v_64 = v.to(dtype=torch.float64)
            if v_64.shape != (n_state,):
                raise ValueError(
                    f"make_host_jvp_rce (Path A): input shape must be ({n_state},); "
                    f"got {tuple(v_64.shape)!r}."
                )
            _, jvp_val = torch.func.jvp(_rad_step, (T_star_nondim,), (v_64,))
            return jvp_val.to(dtype=torch.float64)

        return _host_jvp_a

    elif path == "additive":
        # Path B: host_jvp = v − K·v (full cycle; NN-bearing).
        # Materialize K via central-FD on the NN-replaced cycle (numpy).
        def _full_cycle_T(T_vec: np.ndarray) -> np.ndarray:
            from climate_rce.host import _F_T_substituted
            return T_vec + dt * _F_T_substituted(
                spec, T_vec, subgrid_tendency, co2_ppm=co2_ppm
            )

        T_star_phys = T_star_nondim.numpy() * T_scale
        h_fd = _FD_STEP * T_scale
        n_z = spec.n_z
        K_mat = np.zeros((n_z, n_z), dtype=np.float64)
        for j in range(n_z):
            e_j = np.zeros(n_z)
            e_j[j] = h_fd
            Kv_p = _full_cycle_T(T_star_phys + e_j)
            Kv_m = _full_cycle_T(T_star_phys - e_j)
            K_mat[:, j] = (Kv_p - Kv_m) / (2.0 * h_fd)

        K_t = torch.tensor(K_mat, dtype=torch.float64)

        def _host_jvp_b(v: torch.Tensor) -> torch.Tensor:
            v_64 = v.to(dtype=torch.float64)
            if v_64.shape != (n_state,):
                raise ValueError(
                    f"make_host_jvp_rce (Path B): input shape must be ({n_state},); "
                    f"got {tuple(v_64.shape)!r}."
                )
            return v_64 - K_t @ v_64

        return _host_jvp_b

    else:
        raise ValueError(
            f"make_host_jvp_rce: unknown path={path!r}. "
            "Use 'compositional' (Path A, default) or 'additive' (Path B)."
        )


# ---------------------------------------------------------------------------
# make_emulator_jvp_rce (Path A: D𝒯_phys; Path B: identity)
# ---------------------------------------------------------------------------

def make_emulator_jvp_rce(
    spec: ColumnSpec,
    x_star: np.ndarray,
    subgrid_tendency: Callable,
    *,
    path: str = "compositional",
    dt: float = 1e4,
    co2_ppm: float = 280.0,
    model: SubgridNet | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the GRAIL ``emulator_jvp`` slot for the RCE testbed.

    **Path A (default, "compositional"):**
    Returns ``v ↦ D𝒯_phys · v`` — the physics substep Jacobian (NN-BEARING).
    Computed via ``torch.func.jvp`` through ``_phys_step_nn_exact``, evaluated
    at the **post-radiation intermediate point** x_rad = 𝒯_rad(x*) to satisfy
    the exact chain rule D𝒯 = D𝒯_phys(x_rad) · D𝒯_rad(x*).

    TASK-035 Rule 1: the emulator_jvp differentiates the NN-replaced physics
    step ``C + S^NN``, NOT the truth-convection-only cycle.

    **Path B ("additive"):**
    Returns the identity ``u ↦ u`` (the additive-cycle workaround partner).

    Parameters
    ----------
    spec : ColumnSpec
    x_star : ndarray (n_v, n_z) or (n_z,) converged RCE state
    subgrid_tendency : Callable  Subgrid tendency (S^NN).
        An adapter from make_subgrid_with_nn supplies its physical output
        timescale; its cycle increment is scaled by dt/output_timescale.
    path : str  ``"compositional"`` (Path A, default) or ``"additive"`` (Path B).
    dt : float  Pseudo-timestep (seconds).
    co2_ppm : float  CO₂ forcing, matching the host radiative substep.
    model : SubgridNet or None
        If provided, used directly for autograd (preferred).  If None,
        wraps ``subgrid_tendency`` via central-FD (fallback).

    Returns
    -------
    Callable[[Tensor], Tensor]
        JVP callable mapping ``(n_state,)`` float64 → ``(n_state,)`` float64.
    """
    T_scale = float(spec.var_scales[0])
    n_state = spec.n_state

    if path == "additive":
        return make_identity_emulator_jvp()

    # Path A: emulator_jvp = D𝒯_phys (NN-bearing).
    # Extract T-only fixed point (non-dim)
    if x_star.ndim == 2:
        T_star_nondim = torch.tensor(x_star[0] / T_scale, dtype=torch.float64)
    else:
        T_star_nondim = torch.tensor(x_star / T_scale, dtype=torch.float64)

    # Compute the intermediate point x_rad = 𝒯_rad(x*) for the exact chain rule.
    # D(𝒯_phys ∘ 𝒯_rad)(x*) = D𝒯_phys(𝒯_rad(x*)) · D𝒯_rad(x*)
    # The emulator_jvp must be evaluated AT x_rad, not at x*.
    with torch.no_grad():
        x_rad_nondim = _rad_step_exact(spec, T_star_nondim, dt=dt, co2_ppm=co2_ppm)

    if model is not None:
        # Preferred path: torch.func.jvp through the NN-replaced physics step.
        # Evaluated at x_rad (post-radiation intermediate point).
        # TASK-035 Rule 1: differentiating model (S^NN), not truth.
        # TASK-035 Rule 2: forward-mode autograd (preferred).
        _nn = model
        _dt = dt
        _T_scale = T_scale
        _spec = spec
        _output_timescale = getattr(subgrid_tendency, "output_timescale", dt)

        def _phys_step(x_T: torch.Tensor) -> torch.Tensor:
            return _phys_step_nn_exact(_spec, x_T, _nn, dt=_dt,
                                       T_scale=_T_scale, output_timescale=_output_timescale)

        def _emul_jvp_a(v: torch.Tensor) -> torch.Tensor:
            v_64 = v.to(dtype=torch.float64)
            if v_64.shape != (n_state,):
                raise ValueError(
                    f"make_emulator_jvp_rce (Path A): input shape must be ({n_state},); "
                    f"got {tuple(v_64.shape)!r}."
                )
            # JVP at the intermediate point x_rad (exact chain rule)
            _, jvp_val = torch.func.jvp(_phys_step, (x_rad_nondim,), (v_64,))
            return jvp_val.to(dtype=torch.float64)

        return _emul_jvp_a

    else:
        # Fallback: central-FD through the NN-replaced physics step.
        # TASK-035 Rule 1: subgrid_tendency is make_subgrid_with_nn(model).

        T_rad_phys = x_rad_nondim.numpy() * T_scale  # (n_z,) K at intermediate point
        h_phys = _FD_STEP * T_scale

        def _phys_step_np(T_vec: np.ndarray) -> np.ndarray:
            from climate_rce.convection import convective_tendency
            q_ref = _get_q_ref(spec, T_vec)
            prof = _build_profile(spec, T_vec, q_ref)
            C = convective_tendency(spec, prof)
            S = subgrid_tendency(spec, prof)
            return T_vec + dt * (C[0] + S[0])

        def _emul_jvp_fd(v: torch.Tensor) -> torch.Tensor:
            v_64 = v.to(dtype=torch.float64)
            v_np = v_64.numpy()
            v_phys = v_np * T_scale
            norm_v = np.linalg.norm(v_phys) + 1e-15
            v_unit = v_phys / norm_v
            Kv_p = _phys_step_np(T_rad_phys + h_phys * v_unit)
            Kv_m = _phys_step_np(T_rad_phys - h_phys * v_unit)
            Kv = (Kv_p - Kv_m) / (2.0 * h_phys)
            Kv_nondim = Kv / T_scale * norm_v
            return torch.tensor(Kv_nondim, dtype=torch.float64)

        return _emul_jvp_fd


# ---------------------------------------------------------------------------
# Helper: build physical profile from T_vec
# ---------------------------------------------------------------------------

def _get_q_ref(spec: ColumnSpec, T_vec: np.ndarray) -> np.ndarray:
    """Diagnose q = q_ref(T_surf) for a given T profile (numpy)."""
    from climate_rce.convection import _build_moist_adiabat, _build_q_ref
    T_surf = float(T_vec[-1])
    T_ref = _build_moist_adiabat(spec, T_surf)
    return _build_q_ref(spec, T_ref)


def _build_profile(spec: ColumnSpec, T_vec: np.ndarray, q_vec: np.ndarray) -> np.ndarray:
    """Build a (n_v, n_z) physical profile from T and q vectors."""
    prof = np.zeros((spec.n_v, spec.n_z), dtype=np.float64)
    prof[0] = T_vec
    if spec.n_v >= 2:
        prof[1] = q_vec
    return prof


# ---------------------------------------------------------------------------
# make_identity_emulator_jvp
# ---------------------------------------------------------------------------

def make_identity_emulator_jvp() -> Callable:
    """Return the identity callback ``u ↦ u``.

    This is the GRAIL ``emulator_jvp`` slot for the **additive-cycle
    workaround (Path B)** (SPEC §4.2; the callback contract "additive-cycle identity-
    partner workaround").  It contains no NN.

    With Path B:
        host_jvp(v) = v − K·v   (full cycle; NN-bearing)
        emulator_jvp(u) = u     (identity — no NN)
    Then:
        resolvent_matvec(emul_id, host)(v)
          = v − emul_id(host(v))
          = v − (v − K·v)
          = K·v  ✓

    GRAIL solves ``(I−K)·δx* = δ𝒯``.

    Returns
    -------
    Callable
        ``identity_jvp(u) → u`` unchanged.  Accepts torch.Tensor or numpy.
    """
    def _identity_jvp(u):
        if not isinstance(u, torch.Tensor):
            return torch.as_tensor(u, dtype=torch.float64)
        return u

    return _identity_jvp


# ---------------------------------------------------------------------------
# make_back_residual
# ---------------------------------------------------------------------------

def make_back_residual(
    spec: ColumnSpec,
    x_star: np.ndarray,
    subgrid_nn: Callable,
    subgrid_truth: Callable,
    *,
    dt: float = 1e4,
    co2_ppm: float = 280.0,
    path: str = "compositional",
) -> torch.Tensor:
    """Compute the difference of the substituted and reference cycle maps.

    For the split cycle, the common radiation step first produces ``T_mid``.
    Evaluate both subgrid terms there, with humidity rediagnosed at that
    intermediate temperature. The shared convective term cancels exactly.
    For the additive cycle the terms are evaluated at the input state.

    Parameters
    ----------
    spec : ColumnSpec
    x_star : ndarray (n_v, n_z) converged truth RCE profile (physical)
    subgrid_nn : Callable  NN subgrid tendency (S^NN)
    subgrid_truth : Callable  truth subgrid tendency (S^truth)
    dt : float  pseudo-timestep in seconds (default 1e4 s)
    co2_ppm : float  CO₂ forcing, matching the actual cycle.
    path : str  ``"compositional"`` (default) or ``"additive"``.

    Returns
    -------
    Tensor (n_state,) float64, non-dimensional δ𝒯
    """
    T_scale = float(spec.var_scales[0])

    from climate_rce.host import _q_ref_for_T

    if x_star.ndim == 2:
        T_vec = np.asarray(x_star[0], dtype=np.float64)
    elif x_star.shape == (spec.n_z,):
        T_vec = np.asarray(x_star, dtype=np.float64)
    else:
        raise ValueError("x_star must be a physical temperature vector or profile")
    if path == "compositional":
        from climate_rce.radiation import radiative_tendency
        profile_in = _build_profile(spec, T_vec, _q_ref_for_T(spec, T_vec))
        T_vec = T_vec + dt * radiative_tendency(spec, profile_in, co2_ppm=co2_ppm)[0]
    elif path != "additive":
        raise ValueError("path must be 'compositional' or 'additive'")
    profile = _build_profile(spec, T_vec, _q_ref_for_T(spec, T_vec))

    S_nn    = subgrid_nn(spec, profile)     # (n_v, n_z) physical/s
    S_truth = subgrid_truth(spec, profile)  # (n_v, n_z) physical/s

    delta_S_T = S_nn[0] - S_truth[0]   # (n_z,) K/s
    delta_T_nondim = dt * delta_S_T / T_scale   # (n_z,) non-dim

    return torch.tensor(delta_T_nondim, dtype=torch.float64)


# ---------------------------------------------------------------------------
# make_host_jvp_fd (central-FD cross-check; SPEC §4.3)
# ---------------------------------------------------------------------------

def make_host_jvp_fd(
    spec: ColumnSpec,
    x_star: np.ndarray,
    subgrid_tendency: Callable,
    *,
    path: str = "compositional",
    h: float = _FD_STEP,
    dt: float = 1e4,
    co2_ppm: float = 280.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Central-FD alternative to :func:`make_host_jvp_rce`.

    For Path A: differentiates the SAME ``_rad_step_exact`` function used by
    ``make_host_jvp_rce`` (autograd), so both methods give the same JVP to
    within FD truncation error O(h²) ≈ O(1e-8) at h=1e-4.

    For Path B: central-FD through the full NN-replaced cycle.

    Parameters
    ----------
    spec : ColumnSpec
    x_star : ndarray converged truth RCE state (physical)
    subgrid_tendency : Callable  NN-replaced subgrid tendency
    path : str  ``"compositional"`` or ``"additive"``
    h : float  Central-FD step in non-dimensional units (default 1e-4)
    dt : float  Pseudo-timestep (seconds)
    co2_ppm : float  CO₂ forcing (ppm)

    Returns
    -------
    Callable[[Tensor], Tensor]
        ``host_jvp_fd(v) → D𝒯_rad · v`` (Path A) or ``v − K·v`` (Path B).
    """
    T_scale = float(spec.var_scales[0])
    n_state = spec.n_state

    if x_star.ndim == 2:
        T_star_nondim = torch.tensor(x_star[0] / T_scale, dtype=torch.float64)
    else:
        T_star_nondim = torch.tensor(x_star / T_scale, dtype=torch.float64)

    if path == "compositional":
        # FD through _rad_step_exact (SAME function as autograd in make_host_jvp_rce).
        # Both autograd and FD differentiate the same function → agreement to O(h²).

        def _host_jvp_fd_a(v: torch.Tensor) -> torch.Tensor:
            v_64 = v.to(dtype=torch.float64)
            with torch.no_grad():
                Rp = _rad_step_exact(spec, T_star_nondim + h * v_64, dt=dt, co2_ppm=co2_ppm)
                Rm = _rad_step_exact(spec, T_star_nondim - h * v_64, dt=dt, co2_ppm=co2_ppm)
            return ((Rp - Rm) / (2.0 * h)).to(dtype=torch.float64)

        return _host_jvp_fd_a

    elif path == "additive":
        # FD through the full NN-replaced cycle (same as Path B, v − K·v)
        from climate_rce.host import _F_T_substituted

        T_star_phys = T_star_nondim.numpy() * T_scale

        def _full_cycle_np(T_vec: np.ndarray) -> np.ndarray:
            return T_vec + dt * _F_T_substituted(
                spec, T_vec, subgrid_tendency, co2_ppm=co2_ppm
            )

        h_phys = h * T_scale

        def _host_jvp_fd_b(v: torch.Tensor) -> torch.Tensor:
            v_64 = v.to(dtype=torch.float64).numpy()
            v_phys = v_64 * T_scale
            norm_v = np.linalg.norm(v_phys) + 1e-15
            v_unit = v_phys / norm_v
            Kv_p = _full_cycle_np(T_star_phys + h_phys * v_unit)
            Kv_m = _full_cycle_np(T_star_phys - h_phys * v_unit)
            Kv = (Kv_p - Kv_m) / (2.0 * h_phys)
            Kv_nondim = Kv / T_scale * norm_v
            Kv_nondim_t = torch.tensor(Kv_nondim, dtype=torch.float64)
            return v.to(dtype=torch.float64) - Kv_nondim_t

        return _host_jvp_fd_b

    else:
        raise ValueError(
            f"make_host_jvp_fd: unknown path={path!r}. "
            "Use 'compositional' or 'additive'."
        )
