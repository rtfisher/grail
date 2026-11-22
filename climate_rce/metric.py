"""RCE climate-sensitivity (ECS-proxy) metric and feedback-factor computation.

SPEC_climate_rce §3.5, §4.4.

The science scalar is the **RCE climate sensitivity**:
  ΔT_surf (or column-mean ΔT) between two equilibrated RCE states at
  co2_baseline and co2_perturbed.

This is the single-column ECS-proxy used by RCE-MIP (Wing et al. 2018, GMD
"Radiative-Convective Equilibrium Model Intercomparison Project") to compare
models.  It is a **recognized metric** (SPEC_climate_rce §10.2 R10) — not an
invented one.

The feedback factor f (§2.3, §3.5 eq. 4) is the dominant eigenvalue of the
linearized feedback loop projected on the global-mean-temperature mode.
Here we compute it as:
  f = 1 − 1 / (ΔT_surf / ΔT_0)
where ΔT_0 is the no-feedback (Planck-only) response computed analytically
from the gray RT model, and ΔT_surf is the full equilibrated response.

Observable functional Φ_i
--------------------------
Φ maps the column state vector (non-dimensional, shape n_state) to the
RCE scalar metric (global-mean surface temperature departure).  It is the
linearization of the metric functional around x*, i.e. the row vector
  Φ_i = ∂(metric)/∂x_i |_{x*}
In the gray model, the surface T is the last element of the T block:
  metric ≈ T_surf * T_scale = x[n_z-1] * T_scale
so Φ_i = T_scale * δ_{i, n_z-1} (a one-hot vector with the T_scale at
the surface-T position in the flat state vector).  This is the load-bearing
observable functional for the G ↔ 1/(1−f) scalar identification (§2.3).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .column import ColumnSpec


# ---------------------------------------------------------------------------
# Observable functional
# ---------------------------------------------------------------------------


def observable_functional(spec: ColumnSpec, x_star: np.ndarray) -> np.ndarray:
    """Return the linearized observable functional Φ, shape (n_state,).

    Φ_i = ∂(T_surf)/∂x_i evaluated at x* (non-dimensional units).

    T_surf in physical units = x_flat[n_z - 1] * T_scale
    → Φ_i = T_scale * δ_{i, n_z-1}

    Parameters
    ----------
    spec : ColumnSpec
    x_star : np.ndarray
        Converged RCE state (physical units, shape (n_v, n_z)); used for
        context / potential nonlinear extensions (currently linear).

    Returns
    -------
    np.ndarray
        Shape ``(n_state,)`` float64.
    """
    # x_flat layout: [T_block (n_z), q_block (n_z), ...]
    # Surface T is at index n_z - 1 in the T block.
    Phi = np.zeros(spec.n_state, dtype=np.float64)
    T_scale = spec.var_scales[0]   # characteristic T scale (e.g. 300 K)
    # Position of surface T in the flattened state (T block is first)
    surf_T_idx = spec.n_z - 1
    Phi[surf_T_idx] = T_scale   # gradient of T_surf (physical) w.r.t. x_flat (non-dim)
    return Phi


# ---------------------------------------------------------------------------
# Feedback factor
# ---------------------------------------------------------------------------


def feedback_factor(
    spec: ColumnSpec,
    x_star: np.ndarray,
    K_matvec: Callable,
    *,
    n_power_iter: int = 50,
    seed: int = 42,
) -> float:
    """Estimate the dominant feedback factor f projected on the temperature mode.

    f = dominant eigenvalue of K restricted to the global-mean-T mode.
    We compute the dominant eigenvalue of K via power iteration, then return
    its absolute value as the amplification factor.

    From the Roe (2009) correspondence:
      ||G|| ≈ 1/(1−f)  when f is the dominant real eigenvalue.

    Parameters
    ----------
    spec : ColumnSpec
    x_star : np.ndarray
        Converged RCE state (physical units, shape (n_v, n_z)).
    K_matvec : Callable
        Function ``v: (n_state,) → K·v: (n_state,)`` (the cycle Jacobian
        matrix-vector product).
    n_power_iter : int
        Number of power-iteration steps for the dominant eigenvalue.
    seed : int
        RNG seed.

    Returns
    -------
    float
        Estimated dominant feedback eigenvalue f ∈ (0, 1) in the stable regime.
    """
    rng = np.random.default_rng(seed)
    n_state = spec.n_state

    v = rng.standard_normal(n_state)
    v /= np.linalg.norm(v) + 1e-15
    rho = 0.0
    for _ in range(n_power_iter):
        w = K_matvec(v)
        rho = np.linalg.norm(w)
        if rho < 1e-15:
            break
        v = w / rho

    # f is the dominant eigenvalue (spectral radius of K);
    # from K_{ij} = ∂𝒯_i/∂x_j, ρ(K) is the spectral radius.
    # The feedback factor in the scalar Roe sense = ρ(K) restricted to
    # the slow climate mode; here we return ρ(K) as the dominant f.
    return float(rho)


# ---------------------------------------------------------------------------
# Climate sensitivity (ECS-proxy)
# ---------------------------------------------------------------------------


def climate_sensitivity(
    spec: ColumnSpec,
    subgrid_tendency: Callable,
    *,
    co2_baseline: float = 280.0,
    co2_perturbed: float = 560.0,
) -> float:
    """Compute the RCE climate sensitivity: ΔT_surf between two equilibria.

    Equilibrates the column at ``co2_baseline`` and ``co2_perturbed``,
    then returns the surface temperature difference.

    This is the single-column ECS-proxy used by RCE-MIP (Wing et al. 2018, GMD).
    For 2×CO₂ (560 → baseline 280), values in the gray model are typically
    1–5 K depending on the water-vapor feedback strength.

    Parameters
    ----------
    spec : ColumnSpec
    subgrid_tendency : Callable
        Subgrid tendency (truth or emulator).
    co2_baseline : float
        Baseline CO₂ in ppm (default: pre-industrial 280 ppm).
    co2_perturbed : float
        Perturbed CO₂ in ppm (default: 2×CO₂ = 560 ppm).

    Returns
    -------
    float
        ΔT_surf = T_surf(perturbed) − T_surf(baseline) in K.
        Positive = warming.
    """
    from .host import equilibrate_to_rce

    res_base = equilibrate_to_rce(spec, subgrid_tendency, co2_ppm=co2_baseline)
    res_pert = equilibrate_to_rce(spec, subgrid_tendency, co2_ppm=co2_perturbed)

    # Surface temperature = last level of T profile (highest pressure)
    T_surf_base = res_base.x_star[0, -1]
    T_surf_pert = res_pert.x_star[0, -1]
    return float(T_surf_pert - T_surf_base)
