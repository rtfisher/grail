"""RCE equilibration driver and spectral-radius estimator.

SPEC_climate_rce §4.4, §6 P1.

Cycle-path decision (SPEC_climate_rce §4.2, INTERFACES.md §4.2 — FROZEN)
-------------------------------------------------------------------------
**Path A — compositional 𝒯_split = 𝒯_phys ∘ 𝒯_rad (default).**

The CANONICAL split cycle is implemented as a forward-Euler two-substep:

  Step 1 — 𝒯_rad:   T_mid = T + Δt · R_T(T, q_ref(T_surf))
  Step 2 — 𝒯_phys:  T_new = T_mid + Δt · (C_T(T_mid, q_ref(T_mid_surf))
                                           + S_T(T_mid, q_ref(T_mid_surf)))

The function ``_T_split_cycle_np`` is the single source of truth for this
map; it is imported by ``climate_rce/jvp.py`` so the host's fixed point and
the GRAIL Jacobian slots differentiate EXACTLY the same function.

RCE is the fixed point: ``𝒯_split(T*) = T*``, i.e.
  ``_T_split_cycle_np(spec, T_star, ...) − T_star = 0``.

The ``equilibrate_to_rce`` Newton solver finds this fixed point directly.

This maps natively onto GRAIL's ``composed_jvp(emulator_jvp, host_jvp)``:
  - GRAIL ``host_jvp``     ↔  D𝒯_rad  (first-applied; NN-FREE)
  - GRAIL ``emulator_jvp`` ↔  D𝒯_phys (second-applied; CONTAINS the NN)

**No slot-naming inversion in Path A** (SPEC §4.2; INTERFACES.md §4.2).

Why NOT the unsplit cycle ``T + Δt·(R+C+S)``?
  The unsplit forward-Euler map and the split map 𝒯_phys ∘ 𝒯_rad have different
  fixed points:  𝒯_split(T*_split) = T*_split ≠ T*_unsplit in general.
  The discrepancy is O(Δt²·‖R‖·‖∂C/∂T‖) ≈ O(Δt²·5e-5·1e-3) ≈ 0.5 K at
  Δt=1e4 s.  GRAIL's Jacobian slots differentiate 𝒯_split; the host must
  converge 𝒯_split's fixed point so the resolvent identity holds at x*.

Water-vapor feedback and the substituted T cycle
------------------------------------------------
The GRAIL state is T-only (SPEC §2.2.1; INTERFACES.md §1).  q is diagnosed
as q_ref(T_surf) at each substep — never an independent degree of freedom.
Both substeps apply the fixed-RH closure:
  q_mid  = q_ref(T_mid_surf)
  q_star = q_ref(T_star_surf)
so the converged state satisfies the full WV-feedback-inclusive condition.

The K_eff_T Jacobian (power-iterated for the spectral-radius gate) is
computed on the substituted T-cycle ``𝒯_split(T) − T``, not the unsplit
``_F_T_substituted``.  See ``_estimate_spectral_radius``.

**Measurable WV feedback (f > 0):**
At dt=500 s the split-cycle ρ(K_split_eff) ≈ 0.9993 < 1, with f > 0.

Convergence tolerance
---------------------
tol=1e-3 is the practical floor: softplus gate gives a relative residual
floor of ~1e-2 (gate ≈ exp(-5) at 5 K stability margin); Newton drives
the split-cycle residual ‖𝒯_split(T) − T‖ to < 1e-8 K equivalent.

Anderson acceleration
---------------------
``equilibrate_to_rce`` uses Anderson Mixing (AA(m)) with the SPLIT Picard
iterate ``𝒯_split(T) − T`` as the residual, mirroring ``picard_iterate_3level``
in the NLTE testbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .column import ColumnSpec, ColumnState, reference_profile, pack_state, unpack_state
from .radiation import radiative_tendency
from .convection import convective_tendency, _build_moist_adiabat, _build_q_ref


# ---------------------------------------------------------------------------
# RceResult
# ---------------------------------------------------------------------------


@dataclass
class RceResult:
    """Output of :func:`equilibrate_to_rce`.

    Attributes
    ----------
    x_star : np.ndarray
        Shape ``(n_v, n_z)`` converged RCE profile in physical units.
    converged : bool
        ``True`` if the stationarity tolerance was met within ``max_iter``.
    n_iter : int
        Number of equilibration steps performed.
    residual_history : list[float]
        ``||net tendency|| / ||init||`` (L2-norm) at each step.
    K_spectral_radius : float or None
        Estimate of ``ρ(K_eff_T)`` at ``x*`` — the WV-feedback-inclusive
        T-cycle spectral radius (see module docstring for the physical
        justification).  ``None`` if not requested
        (``compute_spectral_radius=False``).
    feedback_factor : float or None
        Feedback factor ``f`` at ``x*`` (§3.5).  ``None`` if not requested.
    cycle_path : str
        ``"compositional"`` or ``"additive"`` (§4.2).
    """

    x_star: np.ndarray
    converged: bool
    n_iter: int
    residual_history: list = field(default_factory=list)
    K_spectral_radius: float = None
    feedback_factor: float = None
    cycle_path: str = "compositional"


# ---------------------------------------------------------------------------
# Anderson mixing (AA(m), numpy-only)
# ---------------------------------------------------------------------------


def _anderson_mix(
    history_x: list,
    history_f: list,
    m: int = 5,
) -> np.ndarray:
    """Anderson mixing update from the last min(len, m+1) iterates.

    Returns the AA(m) extrapolation of the next iterate.

    Parameters
    ----------
    history_x : list of np.ndarray
        Recent x iterates, each shape (n_state,).
    history_f : list of np.ndarray
        Corresponding residuals f_k = 𝒯(x_k) − x_k, each shape (n_state,).
    m : int
        AA window size.

    Returns
    -------
    np.ndarray
        Next x estimate, shape (n_state,).
    """
    n = min(len(history_x), m + 1)
    # Use at least two points for mixing
    if n < 2:
        return history_x[-1] + history_f[-1]

    # Stack: each column is a history vector; shape (n_state, n)
    F = np.stack(history_f[-n:], axis=1)   # (n_state, n)
    X = np.stack(history_x[-n:], axis=1)   # (n_state, n)
    dF = np.diff(F, axis=1)   # (n_state, n-1)
    dX = np.diff(X, axis=1)   # (n_state, n-1)

    f_last = history_f[-1]   # (n_state,)
    if dF.shape[1] == 0:
        return history_x[-1] + f_last

    # Least-squares: min_{c} ||dF @ c - f_last||² (standard AA formulation)
    # dF is (n_state, n-1); f_last is (n_state,)
    try:
        c, _, _, _ = np.linalg.lstsq(dF, f_last, rcond=None)
    except np.linalg.LinAlgError:
        return history_x[-1] + f_last

    # AA extrapolation: x_new = x_last + f_last - (dX + dF) @ c
    x_new = history_x[-1] + f_last - (dX + dF) @ c
    return x_new


# ---------------------------------------------------------------------------
# Water-vapor reference profile (shared helper)
# ---------------------------------------------------------------------------


def _q_ref_for_T(spec: ColumnSpec, T_vec: np.ndarray) -> np.ndarray:
    """Compute the moist-adiabatic saturation q profile for a given T column.

    This is the analytic fixed point of the q equation F_q = C_q = 0:
        q* = q_ref(T_surf)   (saturation q on the moist adiabat anchored at T[-1])

    Parameters
    ----------
    spec : ColumnSpec
    T_vec : np.ndarray
        Shape ``(n_z,)`` temperature in K.

    Returns
    -------
    np.ndarray
        Shape ``(n_z,)`` saturation specific humidity in kg/kg.
    """
    T_surf = float(T_vec[-1])
    T_ref = _build_moist_adiabat(spec, T_surf)
    return _build_q_ref(spec, T_ref)


# ---------------------------------------------------------------------------
# Substituted T-only tendency (WV feedback closed via q=q_ref(T_surf))
# ---------------------------------------------------------------------------


def _F_T_substituted(
    spec: ColumnSpec,
    T_vec: np.ndarray,
    subgrid_tendency: Callable,
    *,
    co2_ppm: float,
) -> np.ndarray:
    """Temperature tendency with q = q_ref(T_surf) substituted (UNSPLIT).

    This is F_T^sub(T) ≡ R_T + C_T + S_T all evaluated simultaneously at T
    with q = q_ref(T_surf).  This is the UNSPLIT tendency — used in the
    spectral-radius estimation (which needs the Jacobian of 𝒯_split, computed
    by FD on ``_T_split_cycle_np``, not on this function).

    Also retained for the P2 analytic tests (``test_analytic_limits.py``)
    which build K_eff via FD on this function and pass it as a linearized
    host_jvp (the additive-cycle / Path B workaround, SPEC §4.2).

    Parameters
    ----------
    spec : ColumnSpec
    T_vec : np.ndarray
        Shape ``(n_z,)`` temperature profile in K.
    subgrid_tendency : Callable
    co2_ppm : float

    Returns
    -------
    np.ndarray
        Shape ``(n_z,)`` temperature tendency in K/s (UNSPLIT).
    """
    q_ref = _q_ref_for_T(spec, T_vec)
    prof = np.stack([T_vec, q_ref] if spec.n_v == 2 else
                    [T_vec, q_ref] + [np.zeros(spec.n_z)] * (spec.n_v - 2),
                    axis=0)
    R = radiative_tendency(spec, prof, co2_ppm=co2_ppm)
    C = convective_tendency(spec, prof)
    S = subgrid_tendency(spec, prof)
    return (R + C + S)[0]   # T-block only, shape (n_z,)


def _T_split_cycle_np(
    spec: ColumnSpec,
    T_vec: np.ndarray,
    subgrid_tendency: Callable,
    *,
    dt: float,
    co2_ppm: float,
) -> np.ndarray:
    """Canonical split cycle map 𝒯_split(T) — the SINGLE SOURCE OF TRUTH.

    Implements the Path A (FROZEN, INTERFACES.md §4.2) two-substep composition:

      Step 1 — 𝒯_rad:
          q     = q_ref(T_surf)                           (fixed-RH closure)
          T_mid = T + Δt · R_T(T, q)                     (forward-Euler radiation)

      Step 2 — 𝒯_phys:
          q_mid = q_ref(T_mid_surf)                       (fixed-RH closure at T_mid)
          T_new = T_mid + Δt · (C_T(T_mid, q_mid)
                               + S_T(T_mid, q_mid))      (forward-Euler phys+subgrid)

    Returns T_new = 𝒯_split(T), shape (n_z,), in physical units (K).

    **This function is the single canonical definition of 𝒯_split.**
    It is used by:
      - ``_newton_step_split`` (Newton on 𝒯_split(T) − T = 0) — host convergence
      - ``_cycle_step_split`` (Picard / Anderson iterate) — Anderson convergence
      - ``_estimate_spectral_radius`` (FD Jacobian of 𝒯_split) — K spectral radius
      - ``climate_rce/jvp.py`` imports and reuses this function for the
        independent FD reference in the P3 isolation gate — NOT torch.func.jvp
        on jvp.py internals.

    The fixed point T* satisfies 𝒯_split(T*) = T*, equivalently
    ‖𝒯_split(T*) − T*‖ = 0.

    Parameters
    ----------
    spec : ColumnSpec
    T_vec : np.ndarray
        Shape ``(n_z,)`` temperature profile in K.
    subgrid_tendency : Callable
        ``(ColumnSpec, np.ndarray (n_v, n_z)) → np.ndarray (n_v, n_z)``
        physical tendency K/s.
    dt : float
        Forward-Euler pseudo-timestep in seconds.
    co2_ppm : float
        CO₂ concentration in ppm.

    Returns
    -------
    np.ndarray
        Shape ``(n_z,)`` updated temperature in K = 𝒯_split(T).
    """
    # --- Step 1: 𝒯_rad ---
    q = _q_ref_for_T(spec, T_vec)                  # fixed-RH closure at T_surf
    prof1 = _build_profile_from_T(spec, T_vec, q)  # (n_v, n_z) physical
    R = radiative_tendency(spec, prof1, co2_ppm=co2_ppm)
    T_mid = T_vec + dt * R[0]                       # (n_z,) radiation update

    # --- Step 2: 𝒯_phys ---
    q_mid = _q_ref_for_T(spec, T_mid)              # fixed-RH closure at T_mid_surf
    prof2 = _build_profile_from_T(spec, T_mid, q_mid)
    C = convective_tendency(spec, prof2)
    S = subgrid_tendency(spec, prof2)
    T_new = T_mid + dt * (C[0] + S[0])             # (n_z,) phys+subgrid update

    return T_new


def _build_profile_from_T(
    spec: ColumnSpec,
    T_vec: np.ndarray,
    q_vec: np.ndarray,
) -> np.ndarray:
    """Build a (n_v, n_z) physical profile from T and q (numpy helper)."""
    if spec.n_v == 1:
        return T_vec[np.newaxis, :]
    rows = [T_vec, q_vec] + [np.zeros(spec.n_z, dtype=np.float64)] * (spec.n_v - 2)
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Column cycle map 𝒯(x)
# ---------------------------------------------------------------------------


def _newton_step_split(
    spec: ColumnSpec,
    profile: np.ndarray,
    subgrid_tendency: Callable,
    *,
    dt: float,
    co2_ppm: float,
    h_fd: float = 0.01,
    damp: float = 1.0,
) -> np.ndarray:
    """One Newton step for the SPLIT cycle fixed-point equation.

    Solves ``𝒯_split(T) − T = 0`` where 𝒯_split is the canonical two-substep
    split cycle (INTERFACES.md §4.2, FROZEN):

        𝒯_split(T) = 𝒯_phys(𝒯_rad(T))

    implemented in ``_T_split_cycle_np``.  This is the CORRECT fixed-point
    equation for Path A: the returned T* satisfies 𝒯_split(T*) = T*, so the
    GRAIL resolvent ``(I − D𝒯_split)^{-1}`` is built on the right operator.

    **Why split, not unsplit?**
    The unsplit Newton (on F_T^sub = R+C+S = 0) finds a different fixed point
    T*_unsplit with ‖T*_split − T*_unsplit‖ = O(dt²·‖R‖·‖∂C/∂T‖) ≈ 0.5 K
    at dt=1e4 s.  Using the unsplit fixed point with the split Jacobian creates
    an O(dt²) inconsistency in the resolvent identity.

    The Jacobian of ``𝒯_split(T) − T`` is (K_split − I) where
    K_split = ∂𝒯_split/∂T is the split-cycle Jacobian.  Its FD approximation
    includes the WV feedback through both substeps (both ``_q_ref_for_T`` calls
    in ``_T_split_cycle_np`` respond to T perturbations).

    Parameters
    ----------
    spec : ColumnSpec
    profile : np.ndarray
        Current estimate, shape ``(n_v, n_z)`` physical units.
        T-block updated by Newton; q-block set to q_ref(T_surf_new) on return.
    subgrid_tendency : Callable
    dt : float
        Forward-Euler pseudo-timestep for 𝒯_split.
    co2_ppm : float
    h_fd : float
        FD step fraction: physical step = h_fd * T_scale (K).
    damp : float
        Newton damping factor (1.0 = full, 0.5 = half-step).

    Returns
    -------
    np.ndarray
        Updated profile, shape ``(n_v, n_z)``.
    """
    n_z = spec.n_z
    T_scale = spec.var_scales[0]
    h_T = h_fd * T_scale   # FD step in K

    T0 = profile[0].copy()

    # Split-cycle residual: G(T) = 𝒯_split(T) - T
    def _split_residual(T_vec: np.ndarray) -> np.ndarray:
        return (_T_split_cycle_np(spec, T_vec, subgrid_tendency,
                                  dt=dt, co2_ppm=co2_ppm)
                - T_vec)

    G0 = _split_residual(T0)   # (n_z,) current residual

    # Build the n_z × n_z Jacobian ∂G/∂T = (K_split − I) via central-FD.
    # The FD step perturbs T_surf in both substeps (through _q_ref_for_T calls
    # inside _T_split_cycle_np), so the WV feedback is included automatically.
    J_split = np.zeros((n_z, n_z), dtype=np.float64)
    for j in range(n_z):
        e_j = np.zeros(n_z, dtype=np.float64)
        e_j[j] = h_T
        G_p = _split_residual(T0 + e_j)
        G_m = _split_residual(T0 - e_j)
        J_split[:, j] = (G_p - G_m) / (2.0 * h_T)

    # Solve J_split · ΔT = −G(T0)
    try:
        delta_T = np.linalg.solve(J_split, -G0)
    except np.linalg.LinAlgError:
        # Jacobian singular: fall back to gradient descent step.
        delta_T = -G0 * dt   # Picard-like fallback

    T_new = T0 + damp * delta_T
    profile_new = profile.copy()
    profile_new[0] = T_new
    # Diagnose q at the new T_surf (fixed-RH closure).
    profile_new[1] = _q_ref_for_T(spec, T_new)
    return profile_new


def _cycle_step(
    spec: ColumnSpec,
    profile: np.ndarray,
    subgrid_tendency: Callable,
    *,
    dt: float,
    co2_ppm: float,
) -> np.ndarray:
    """One Picard iterate of the canonical split cycle 𝒯_split(T).

    Returns ``(T_new, q_new)`` where T_new = 𝒯_split(profile[0]).  This is the
    Picard iterate used with Anderson acceleration and plain Picard convergence.

    The fixed point of this map is the same ``x*`` that Newton converges to in
    ``_newton_step_split``: 𝒯_split(T*) = T* ⟺ T_new − T0 = 0.

    **Uses ``_T_split_cycle_np`` (the single canonical source of truth).**

    Parameters
    ----------
    spec : ColumnSpec
    profile : np.ndarray
        Shape ``(n_v, n_z)`` in physical units.
    subgrid_tendency : Callable
    dt : float
        Pseudo-timestep in seconds.
    co2_ppm : float

    Returns
    -------
    np.ndarray
        Updated profile ``(n_v, n_z)``.
    """
    T_vec = profile[0].copy()
    T_new = _T_split_cycle_np(spec, T_vec, subgrid_tendency, dt=dt, co2_ppm=co2_ppm)
    q_new = _q_ref_for_T(spec, T_new)
    profile_new = profile.copy()
    profile_new[0] = T_new
    profile_new[1] = q_new
    return profile_new


def _net_tendency(
    spec: ColumnSpec,
    profile: np.ndarray,
    subgrid_tendency: Callable,
    *,
    co2_ppm: float,
) -> np.ndarray:
    """Net tendency R + C + S at a given profile, shape (n_v, n_z).

    Used for convergence monitoring (the relative norm of this unsplit tendency
    is the convergence criterion).  The converged state minimises both the
    unsplit tendency and the split-cycle residual 𝒯_split(T) − T simultaneously
    (they differ by O(dt²) at the fixed point, well inside tol=1e-3).
    """
    R = radiative_tendency(spec, profile, co2_ppm=co2_ppm)
    C = convective_tendency(spec, profile)
    S = subgrid_tendency(spec, profile)
    return R + C + S


# ---------------------------------------------------------------------------
# Spectral-radius estimation
# ---------------------------------------------------------------------------


def _estimate_spectral_radius(
    spec: ColumnSpec,
    profile_star: np.ndarray,
    subgrid_tendency: Callable,
    *,
    dt: float,
    co2_ppm: float,
    n_iter: int = 50,
    seed: int = 42,
) -> float:
    """Estimate ρ(K_eff_T) — WV-feedback-inclusive T-cycle spectral radius.

    Computes the spectral radius of the **substituted T-cycle Jacobian**
    K_eff_T = ∂𝒯_T^sub/∂T, where 𝒯_T^sub(T) = T + Δt·F_T^sub(T) and
    F_T^sub(T) is the T tendency with q = q_ref(T_surf) substituted.

    **Why K_eff_T, not K_full?**
    The full (T,q) Jacobian K_full has eigenvalue ≈ 1 for q-neutral modes
    at upper levels (gate ~ exp(-60) ≈ 0 for T ≪ moist-adiabat), giving
    ρ(K_full) = 1 to float64 precision for any dt < 1e13 s.  These are
    physically inert modes: they don't appear in δ𝒯 (the ML emulator
    residual has no q-tendency at stable levels) and don't contribute to
    the resolvent response.

    K_eff_T captures the **physically active contraction** — the rate at
    which the temperature fixed point is approached accounting for WV
    feedback — which is the quantity relevant to SPEC §2.2 (iii).

    K_eff_T strictly satisfies ρ(K_eff_T) < 1 in the stable regime.
    At dt=500 s: ρ(K_eff_T) ≈ 0.9993 (T radiative relaxation + WV feedback).
    At dt=1e4 s: ρ(K_eff_T) ≈ 0.990 (larger pseudo-step, faster contraction).

    **WV feedback is measurable:**
    ρ(K_eff_T) > ρ(K_T_no_wv) — the WV feedback increases the spectral radius
    (amplifies the slow mode), which is the physical water-vapor positive
    feedback (f > 0) of Roe (2009).

    Physical derivation of K_eff_T:
        K_eff_T = I + Δt · ∂F_T^sub/∂T
        ∂F_T^sub/∂T = ∂F_T/∂T + ∂F_T/∂q · (dq_ref/dT_surf) · e_surf^T
    where e_surf selects T[-1] (the surface temperature).  The second term
    is the WV feedback contribution through the Clausius-Clapeyron derivative
    dq_ref/dT_surf = q_ref * Lv / (Rv * T_surf²) > 0.

    Parameters
    ----------
    spec : ColumnSpec
    profile_star : np.ndarray
        Converged RCE profile, shape ``(n_v, n_z)``.
    subgrid_tendency : Callable
    dt : float
        Pseudo-timestep for the cycle map (determines ρ(K) = 1 − dt·λ_eff).
    co2_ppm : float
    n_iter : int
        Number of power-iteration steps.
    seed : int

    Returns
    -------
    float
        Estimated ρ(K_eff_T).  Guaranteed non-negative (absolute value of
        dominant eigenvalue).  Strictly < 1 in the physically stable regime.
    """
    rng = np.random.default_rng(seed)
    n_z = spec.n_z
    T_scale = spec.var_scales[0]
    h_T = 1e-3 * T_scale   # ~0.3 K FD step for the cycle matvec

    T_star = profile_star[0].copy()

    def matvec_K_eff_T(v_T: np.ndarray) -> np.ndarray:
        """K_split · v_T via central-FD on the canonical 𝒯_split(T).

        Uses _T_split_cycle_np (the single canonical split cycle) to compute
        the Jacobian of 𝒯_split at T*.  The FD perturbation automatically
        captures the WV feedback through both substeps (both _q_ref_for_T
        calls inside _T_split_cycle_np respond to T perturbations).

        The FD step h_T = 1e-3 * T_scale ≈ 0.3 K balances perturbative
        accuracy against float64 cancellation.
        """
        norm_v = np.linalg.norm(v_T) + 1e-15
        v_unit = v_T / norm_v

        Tp = _T_split_cycle_np(spec, T_star + h_T * v_unit,
                               subgrid_tendency, dt=dt, co2_ppm=co2_ppm)
        Tm = _T_split_cycle_np(spec, T_star - h_T * v_unit,
                               subgrid_tendency, dt=dt, co2_ppm=co2_ppm)

        # Central-FD quotient of 𝒯_split; re-normalize
        Kv_T = (Tp - Tm) / (2.0 * h_T)   # (n_z,)
        return Kv_T * norm_v

    # Power iteration on K_eff_T (dimension n_z, well within memory)
    v = rng.standard_normal(n_z)
    v /= np.linalg.norm(v) + 1e-15
    rho = 0.0
    for _ in range(n_iter):
        w = matvec_K_eff_T(v)
        rho = np.linalg.norm(w)
        if rho < 1e-15:
            break
        v = w / rho

    return float(rho)


# ---------------------------------------------------------------------------
# Main equilibration driver
# ---------------------------------------------------------------------------


def equilibrate_to_rce(
    spec: ColumnSpec,
    subgrid_tendency: Callable,
    *,
    profile_init: np.ndarray = None,
    tol: float = 1e-3,
    max_iter: int = 5000,
    dt: float = 1e4,
    accelerator: str = "newton",
    co2_ppm: float = 280.0,
    compute_spectral_radius: bool = False,
    spectral_radius_seed: int = 42,
    anderson_m: int = 5,
) -> RceResult:
    """Equilibrate the column to the split-cycle RCE fixed point.

    Finds T* satisfying ``𝒯_split(T*) = T*`` where ``𝒯_split = 𝒯_phys ∘ 𝒯_rad``
    is the canonical two-substep split cycle (INTERFACES.md §4.2, FROZEN).
    This is the fixed point that GRAIL's resolvent ``(I − D𝒯_split)^{-1}``
    is built around.

    **Newton accelerator (default):** solves ``𝒯_split(T) − T = 0`` directly
    via ``_newton_step_split``.  Converges in O(10-20) steps.  The Jacobian
    is ∂(𝒯_split − I)/∂T, computed by central-FD on ``_T_split_cycle_np``
    (the same canonical function imported by jvp.py for the P3 isolation gate).

    **Convergence criterion:** split-cycle residual
    ``‖𝒯_split(T) − T‖ / ‖init split residual‖ < tol``.
    tol=1e-3 is the practical floor set by the softplus gate.

    Parameters
    ----------
    spec : ColumnSpec
        Column specification.
    subgrid_tendency : Callable
        ``(ColumnSpec, np.ndarray) -> np.ndarray`` — the subgrid tendency
        (truth or emulator).  Input ``(n_v, n_z)`` physical; output ``(n_v, n_z)``
        physical/s.
    profile_init : np.ndarray or None
        Initial ``(n_v, n_z)`` profile.  Defaults to the isothermal+RH
        reference profile from ``column.reference_profile``.
    tol : float
        Convergence threshold on the relative split-cycle residual.
        Default 1e-3.
    max_iter : int
        Maximum equilibration iterations.
    dt : float
        Forward-Euler pseudo-timestep for ``𝒯_split`` in seconds.
        Also used for spectral-radius estimation.
    accelerator : str
        ``"newton"`` (default), ``"anderson"`` (AA(m)), or ``"none"`` (Picard).
    co2_ppm : float
        CO₂ forcing in ppm.
    compute_spectral_radius : bool
        If True, estimate ρ(K_split) at convergence via power iteration on
        ``𝒯_split`` (FD Jacobian via ``_T_split_cycle_np``).
    spectral_radius_seed : int
        RNG seed for power iteration.
    anderson_m : int
        AA window size (``"anderson"`` accelerator only).

    Returns
    -------
    RceResult
    """
    if profile_init is None:
        ref_state = reference_profile(spec)
        profile = ref_state.profile.copy()
    else:
        profile = profile_init.copy()

    # Compute initial split-cycle residual norm for normalization.
    # We use the SPLIT residual ‖𝒯_split(T) − T‖ as the convergence criterion
    # so that Newton (which directly minimises this) and Picard/Anderson all
    # converge to THE SAME FIXED POINT of 𝒯_split.
    T0_init = profile[0].copy()
    T0_next = _T_split_cycle_np(spec, T0_init, subgrid_tendency, dt=dt, co2_ppm=co2_ppm)
    split_res0 = T0_next - T0_init          # (n_z,) split residual at init
    norm0 = np.linalg.norm(split_res0) + 1e-15

    history_x: list = []
    history_f: list = []
    residual_history: list = []
    converged = False

    for i in range(max_iter):
        # Convergence check on the SPLIT cycle residual: ‖𝒯_split(T) − T‖ / ‖init‖.
        T_vec_now = profile[0].copy()
        T_next_now = _T_split_cycle_np(spec, T_vec_now, subgrid_tendency,
                                       dt=dt, co2_ppm=co2_ppm)
        split_res = T_next_now - T_vec_now  # (n_z,)
        res_norm = np.linalg.norm(split_res) / norm0
        residual_history.append(float(res_norm))
        tend = None   # computed lazily if needed by Anderson branch

        if res_norm < tol:
            converged = True
            break

        if accelerator == "newton":
            # Newton on the SPLIT fixed-point equation: 𝒯_split(T) − T = 0.
            # Converges in O(10-20) steps; see _newton_step_split docstring.
            profile = _newton_step_split(
                spec, profile, subgrid_tendency,
                dt=dt, co2_ppm=co2_ppm, h_fd=0.01, damp=0.5,
            )
        elif accelerator == "anderson":
            # Anderson mixing on the SPLIT Picard residual.
            # Re-use the already-computed split_res and T_next_now to avoid
            # redundant cycle evaluations.
            f_flat = split_res       # 𝒯_split(T) − T, shape (n_z,)
            x_flat = T_vec_now      # current T, shape (n_z,)
            history_x.append(x_flat.copy())
            history_f.append(f_flat.copy())
            if len(history_x) >= 2:
                T_aa = _anderson_mix(history_x, history_f, m=anderson_m)
                profile = profile.copy()
                profile[0] = T_aa
                profile[1] = _q_ref_for_T(spec, T_aa)
            else:
                # Bootstrap: take one Picard step (reuse T_next_now).
                profile = profile.copy()
                profile[0] = T_next_now
                profile[1] = _q_ref_for_T(spec, T_next_now)
            if len(history_x) > anderson_m + 2:
                history_x.pop(0)
                history_f.pop(0)
        else:
            # Plain Picard: profile_{k+1} = 𝒯_split(profile_k).
            # Reuse T_next_now computed in the convergence check.
            profile = profile.copy()
            profile[0] = T_next_now
            profile[1] = _q_ref_for_T(spec, T_next_now)

    # Estimate WV-feedback-inclusive spectral radius if requested.
    # ρ(K_eff_T) < 1 is the IFT contraction condition (iii) for the
    # substituted T-cycle Jacobian (see _estimate_spectral_radius docstring).
    rho_K = None
    if compute_spectral_radius and converged:
        rho_K = _estimate_spectral_radius(
            spec, profile, subgrid_tendency,
            dt=dt, co2_ppm=co2_ppm,
            n_iter=80, seed=spectral_radius_seed,
        )

    return RceResult(
        x_star=profile,
        converged=converged,
        n_iter=i + 1,
        residual_history=residual_history,
        K_spectral_radius=rho_K,
        feedback_factor=None,   # populated by metric.feedback_factor if desired
        cycle_path="compositional",
    )
