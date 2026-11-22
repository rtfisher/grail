"""Reference 2-level statistical-equilibrium solver. the technical reference §3.3.

This is the ground-truth rate-equation map T_SE: (J_bar, T, n_e) -> (n_1, n_2)
that the APNN emulator replaces.  Pure PyTorch, autograd-ready.

It plays three roles in the prototype:
  1. generates training labels for the APNN (LHS sampling).
  2. defines the residual delta_T = T_SE^APNN - T_SE^ref on the validation set.
  3. is the rate-equation solver in `cycle_truth`.

Keeping all three roles tied to the same function makes delta_T pure emulator
error rather than emulator error + reference mismatch.

Physics summary (reference guide §2.3, §2.5, §7; the technical reference §3.1):
  - Two-level atom under CRD (assumption A6).
  - Statistical equilibrium (assumption A2): at each depth, populations are
    determined instantaneously by the local radiation field and temperature.
  - Master-parameter epsilon (assumption A10): photon destruction probability
    collapses all explicit Einstein and collisional rate coefficients into one
    parameter.  n_e is accepted for API compatibility with Phase II but is
    unused here.

Closed-form derivation
----------------------
Line source function (CRD, A6, A10):
    S_l = (1 - epsilon) * J_bar + epsilon * B(T)

Planck function at line center (Wien+correction form):
    B(T) = B_star / (exp(h*nu_0 / (k*T)) - 1)
    B_star = 2*h*nu_0^3 / c^2

Population ratio from SE (see reference guide §7):
    n_2 / n_1 = (g_upper / g_lower) * S_l / (S_l + B_star)

LTE check: at epsilon=1, S_l = B(T), giving
    n_2/n_1 = (g_u/g_l) * [1/(exp(x)-1)] / [1/(exp(x)-1) + 1]
             = (g_u/g_l) * exp(-h*nu_0/(k*T))   [Boltzmann]

Closure: n_1 + n_2 = n_tot
    n_1 = n_tot / (1 + n_2/n_1)
    n_2 = n_tot - n_1
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Physical constants (SI; used consistently throughout — B and J_bar must
# share the same unit system when calling this function).
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant, J·s
_K = 1.380649e-23     # Boltzmann constant, J/K
_C = 2.99792458e8     # Speed of light, m/s


def rate_equation_solve(
    J_bar: torch.Tensor,
    T: torch.Tensor,
    n_e: torch.Tensor,
    *,
    epsilon: float,
    nu_0: float,
    g_lower: int = 1,
    g_upper: int = 1,
    n_tot: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Reference SE solve for two-level populations (n_1, n_2).

    Closed-form solution of the 2x2 statistical-equilibrium system under the
    master-epsilon parameterisation (the technical reference §3.3, assumption A10).

    Parameters
    ----------
    J_bar : Tensor of shape (N_tau,), float64
        Profile-weighted mean intensity in any consistent unit system.
        The same unit system must be used for B(T) — i.e., J_bar should be in
        SI (W/m²/Hz/sr) when using the SI constants in this module.
    T : Tensor of shape (N_tau,), float64
        Local temperature in Kelvin.
    n_e : Tensor of shape (N_tau,), float64
        Local electron density.  **Unused in the master-epsilon form.**
        Accepted for API compatibility; Phase II will replace this with
        explicit collisional rate coefficients.
    epsilon : float
        Photon destruction probability (collisional de-excitation / total
        de-excitation).  Must satisfy 0 < epsilon <= 1.
    nu_0 : float
        Line-center frequency in Hz.
    g_lower : int, default 1
        Statistical weight of the lower level.  Default 1 (minimum-viable atom).
    g_upper : int, default 1
        Statistical weight of the upper level.  Default 1 reflects the
        prototype's "minimum viable atom" framing (assumption A10): the
        master parameter ε absorbs all level-physics, and g_lower=g_upper=1
        keeps the Boltzmann ratio at LTE equal to exp(-hν/kT) without any
        unwarranted degeneracy factor.  Tests that require a specific
        degeneracy ratio pass g_lower/g_upper explicitly.
    n_tot : float or Tensor broadcastable to (N_tau,), default 1.0
        Total number density n_1 + n_2.  Scalar 1.0 gives normalised
        fractional populations; a depth-dependent array gives physical
        densities.

    Returns
    -------
    populations : Tensor of shape (N_tau, 2), float64
        Column 0: n_1 (lower-level population).
        Column 1: n_2 (upper-level population).

    Notes
    -----
    At epsilon=1 (LTE) the result reduces exactly to the Boltzmann ratio
    regardless of J_bar, as expected from assumptions A5+A6+A10 (HM14 §13.2).

    n_e is accepted but intentionally unused.  This is a deliberate design
    decision (assumption A10): all collisional physics is absorbed into the
    scalar epsilon.  A type-checker or linter warning about an unused argument
    is expected; the parameter exists for interface consistency with Phase II.
    """
    # -----------------------------------------------------------------------
    # Work in float64 throughout; cast inputs if needed.
    # -----------------------------------------------------------------------
    J_bar = J_bar.to(dtype=torch.float64)
    T = T.to(dtype=torch.float64)
    # n_e intentionally unused — see docstring.

    # -----------------------------------------------------------------------
    # Physical constants as scalar tensors so torch traces the graph cleanly.
    # -----------------------------------------------------------------------
    h = torch.tensor(_H, dtype=torch.float64, device=J_bar.device)
    k = torch.tensor(_K, dtype=torch.float64, device=J_bar.device)
    c = torch.tensor(_C, dtype=torch.float64, device=J_bar.device)

    g_ratio = torch.tensor(
        float(g_upper) / float(g_lower), dtype=torch.float64, device=J_bar.device
    )

    # -----------------------------------------------------------------------
    # Wien amplitude: B_star = 2*h*nu_0^3 / c^2  (units: same as J_bar * c^2
    # per steradian, so the caller must use consistent units).
    # -----------------------------------------------------------------------
    nu = torch.tensor(nu_0, dtype=torch.float64, device=J_bar.device)
    B_star = 2.0 * h * nu**3 / c**2

    # -----------------------------------------------------------------------
    # Planck function at line center: B = B_star / (exp(h*nu/(k*T)) - 1)
    # -----------------------------------------------------------------------
    x = h * nu / (k * T)          # dimensionless excitation parameter (N_tau,)
    B = B_star / (torch.exp(x) - 1.0)

    # -----------------------------------------------------------------------
    # Line source function (CRD, master-epsilon):
    #   S_l = (1 - epsilon) * J_bar + epsilon * B(T)
    # -----------------------------------------------------------------------
    eps = torch.as_tensor(epsilon, dtype=torch.float64, device=J_bar.device)
    S_l = (1.0 - eps) * J_bar + eps * B

    # -----------------------------------------------------------------------
    # Population ratio from SE (reference guide §7):
    #   n_2 / n_1 = (g_upper / g_lower) * S_l / (S_l + B_star)
    #
    # Derivation: balance of stimulated emission + spontaneous emission against
    # absorption, with the Wien amplitude appearing as the saturation scale.
    # -----------------------------------------------------------------------
    ratio = g_ratio * S_l / (S_l + B_star)   # (N_tau,)

    # -----------------------------------------------------------------------
    # Closure: n_1 + n_2 = n_tot
    # -----------------------------------------------------------------------
    if isinstance(n_tot, torch.Tensor):
        n_tot_t = n_tot.to(dtype=torch.float64, device=J_bar.device)
    else:
        n_tot_t = torch.tensor(n_tot, dtype=torch.float64, device=J_bar.device)

    # Compute the small excited fraction directly. Subtracting n_1 from
    # n_tot loses relative precision in cold slabs (n_2/n_tot ~ 1e-10).
    n_2 = n_tot_t * ratio / (1.0 + ratio)
    n_1 = n_tot_t - n_2

    return torch.stack([n_1, n_2], dim=-1)   # (N_tau, 2)
