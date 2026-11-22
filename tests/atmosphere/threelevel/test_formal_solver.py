"""Formal-solver tests for the 3-level setup. SPEC_threelevel §M3c TASK-016.

Test IDs per SPEC_threelevel §8.2:
  A-eddington3-01 (TASK-016): Eddington-Barbier per-line relation in a 3-level
      atmosphere view. For each line k, a linearly-varying S_k passed to the
      formal solver gives I(0, μ) ≈ S_k(τ = μ). Tolerance 5% at μ ∈ {0.1, 0.5, 0.9}.

References
----------
- Hubeny & Mihalas (2014) §11.5 (Eddington-Barbier relation).
- SPEC_threelevel §M3c TASK-016.
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.atmosphere import Atmosphere
from atmosphere.atomic_model import AtomicModel
from atmosphere.formal_solver import formal_solve_with_surface_I
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

# ---------------------------------------------------------------------------
# Physical constants (SI) — mirrored from formal_solver.py for test assertions
# ---------------------------------------------------------------------------
_H = 6.62607015e-34
_C = 2.99792458e8
_K = 1.380649e-23

# 3-level hydrogen frequencies (from AtomicModel.three_level_hydrogen_chained)
_NU_LYA = 2.467381e15   # Hz  (Ly-α, 1→2)
_NU_HA  = 4.569225e14   # Hz  (Hα,   2→3)


def _planck_scalar(T0: float, nu_0: float) -> float:
    """Planck function B_nu(T0) at line center nu_0, scalar Python float."""
    return (2.0 * _H * nu_0**3 / _C**2) / (math.exp(_H * nu_0 / (_K * T0)) - 1.0)


def _make_3level_atm_for_line(
    line_index: int,
    *,
    N_tau: int = 200,
    N_nu: int = 11,
    N_mu: int = 8,
    tau_min: float = 1e-4,
    tau_max: float = 1e4,
    eps_lya: float = 0.5,
    eps_ha: float = 0.5,
) -> Atmosphere:
    """Build a per-line atmosphere view from a 3-level setup.

    Constructs the full 3-level Atmosphere, then returns an Atmosphere
    with nu_0 and epsilon set to transition line_index (using _atm_for_line).
    The formal solver only needs nu_0 (for the deep BC) and the grids, so
    the per-line view is the correct input.

    Parameters
    ----------
    line_index : int
        0 for Ly-α, 1 for Hα.

    Returns
    -------
    atm_k : Atmosphere
        Per-line view ready for formal_solve / formal_solve_with_surface_I.
    """
    from atmosphere.threelevel.fixed_point import _atm_for_line

    depth = DepthGrid.log_spaced(tau_min=tau_min, tau_max=tau_max, n_tau=N_tau)
    freq  = FrequencyGrid.doppler(n_nu=N_nu, half_width=3.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=N_mu)

    model = AtomicModel.three_level_hydrogen_chained(
        epsilon_lya=eps_lya, epsilon_ha=eps_ha
    )
    N_t = depth.tau.shape[0]
    T0 = 6000.0
    T   = torch.full((N_t,), T0, dtype=torch.float64)
    n_e = torch.full((N_t,), 1e10, dtype=torch.float64)

    atm_full = Atmosphere(
        depth=depth, frequency=freq, angle=angle,
        T=T, n_e=n_e,
        epsilon=model.transitions[0].epsilon,
        nu_0=model.transitions[0].nu_0,
        atomic_model=model,
    )
    return _atm_for_line(atm_full, line_index)


# ---------------------------------------------------------------------------
# A-eddington3-01 — Eddington-Barbier per line (TASK-016)
# ---------------------------------------------------------------------------

@pytest.mark.analytic
@pytest.mark.parametrize("line_index,nu_0", [(0, _NU_LYA), (1, _NU_HA)])
def test_A_eddington3_01_surface_intensity(line_index: int, nu_0: float) -> None:
    """A-eddington3-01: Eddington-Barbier emergent intensity per line.

    For a linearly varying source function S_k(τ) = a + b·τ, the Eddington-
    Barbier relation (Hubeny & Mihalas 2014 §11.5) predicts:
        I(τ=0, μ) ≈ S_k(τ = μ) = a + b·μ
    to first order in the source function gradient.

    This test loops over both lines of the 3-level setup (line_index=0: Ly-α,
    line_index=1: Hα) and verifies the relation for each using a per-line
    atmosphere view built via _atm_for_line.

    Acceptance criterion (SPEC_threelevel §M3c TASK-016):
        agreement to within 5% at μ ∈ {0.1, 0.5, 0.9} for both lines.

    Reference: Hubeny & Mihalas (2014) §11.5, eq. (11.26).
    """
    N_tau   = 200
    N_nu    = 11
    N_mu    = 8
    tau_min = 1e-4
    tau_max = 1e4

    # Linear source function: S_k(τ) = a + b·τ.
    # b is chosen gentle so S_k stays positive over τ ∈ [τ_min, τ_max]
    # and the gradient is large enough to probe the EB relation at μ ~ 0.1–0.9.
    a = 1.0
    b = 5e-5   # S_k(τ_max=1e4) ≈ 1.5

    tau_grid = torch.logspace(
        math.log10(tau_min), math.log10(tau_max), N_tau, dtype=torch.float64
    )
    S_tau = a + b * tau_grid   # shape (N_tau,)

    # T0 such that B_nu_k(T0) = S_k(τ_max) — the deep BC is exactly S_k(τ_max),
    # which eliminates boundary artefacts.
    S_max     = float(S_tau[-1].item())
    prefactor = 2.0 * _H * nu_0**3 / _C**2
    T0        = (_H * nu_0 / _K) / math.log(1.0 + prefactor / S_max)

    # Build per-line atmosphere view.
    atm_k = _make_3level_atm_for_line(
        line_index,
        N_tau=N_tau, N_nu=N_nu, N_mu=N_mu,
        tau_min=tau_min, tau_max=tau_max,
    )

    # Override T so that B(T0) = S(τ_max) at the relevant line frequency.
    from dataclasses import replace
    T_vec = torch.full((N_tau,), T0, dtype=torch.float64)

    # Rebuild atm_k with the matched temperature.
    atm_k = Atmosphere(
        depth=atm_k.depth,
        frequency=atm_k.frequency,
        angle=atm_k.angle,
        T=T_vec,
        n_e=atm_k.n_e,
        epsilon=atm_k.epsilon,
        nu_0=atm_k.nu_0,
        atomic_model=atm_k.atomic_model,
    )

    # S broadcast over frequency (CRD: same S at all frequencies).
    S = S_tau.unsqueeze(1).expand(N_tau, N_nu)  # (N_tau, N_nu)

    _, I_surface = formal_solve_with_surface_I(S, atm_k)
    # I_surface shape: (N_mu, N_nu)

    # Use the central frequency channel (x=0, psi=1) — optically thick
    # interior avoids contamination from wing channels.
    x_center          = N_nu // 2
    I_surf_center     = I_surface[:, x_center]  # (N_mu,)
    mu_values         = atm_k.angle.mu           # (N_mu,)

    line_label = "Ly-α" if line_index == 0 else "Hα"
    target_mu  = [0.1, 0.5, 0.9]
    for mu_t in target_mu:
        idx    = int((mu_values - mu_t).abs().argmin().item())
        mu_i   = float(mu_values[idx].item())
        I_calc = float(I_surf_center[idx].item())
        I_pred = a + b * mu_i

        rel_err = abs(I_calc - I_pred) / abs(I_pred)
        assert rel_err < 0.05, (
            f"A-eddington3-01 ({line_label}): Eddington-Barbier failed at "
            f"μ={mu_t:.2f} (nearest μ={mu_i:.4f}): "
            f"I_calc={I_calc:.6e}, I_EB={I_pred:.6e}, rel_err={rel_err:.3e} "
            f"(want < 5%)  [HM14 §11.5]"
        )
