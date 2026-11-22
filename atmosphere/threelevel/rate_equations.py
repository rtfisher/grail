"""3-level statistical-equilibrium solver. SPEC_threelevel §6.1, TASK-008.

Closed-form solution via torch.linalg.solve over the augmented (L x L) system,
vectorised over the N_tau depth dimension as a batched solve.

Physics summary
---------------
For each transition k connecting (lower=l_k, upper=u_k) with master-epsilon
parameterisation (assumption A10):

  Source function:
      S_k(tau) = (1 - eps_k) * J_bar_k + eps_k * B(T, nu_k)

  Wien amplitude:
      B_star_k = 2 h nu_k^3 / c^2

  Planck function at line center:
      B(T, nu_k) = B_star_k / (exp(h nu_k / k T) - 1)

Rate matrix construction
------------------------
Using the Einstein-relation form of the rate equations, the SE condition for
level i is: sum_k contributions from transition k that involve level i = 0.

For each transition k = (lower=l_k, upper=u_k):
  - Rate coefficient out of lower level per unit population:
        C_out_k = (g_u / g_l) * S_k
    (absorption: proportional to B_lu * J_bar ~ g_u/g_l * S when using
     the master-eps convention; see 2-level derivation in rate_equations.py)

  - Rate coefficient out of upper level per unit population:
        C_in_k = S_k + B_star_k
    (stimulated emission + spontaneous emission)

  Rate-matrix entries for transition k:
      R[l_k, l_k] -= C_out_k    (loss from lower: absorbed upward)
      R[l_k, u_k] += C_in_k     (gain to lower: emission from upper)
      R[u_k, u_k] -= C_in_k     (loss from upper: emits downward)
      R[u_k, l_k] += C_out_k    (gain to upper: absorbed from lower)

Augmented system
----------------
The homogeneous SE system R n = 0 has a non-trivial null vector (populations
are underdetermined by rate balance alone).  We replace the last row with
the normalisation constraint:
    [1, 1, ..., 1] n = n_tot

This gives a well-determined (L x L) system that is solved via
torch.linalg.solve, batched over tau.

LTE check (eps_k = 1 for all k)
---------------------------------
At eps_k = 1: S_k = B(T, nu_k).  The rate-matrix ratio for transition k is
    n_u / n_l = C_out_k / C_in_k = (g_u/g_l) * S_k / (S_k + B_star_k)
              = (g_u/g_l) / (1 + B_star_k/S_k)
At S_k = B(T, nu_k) = B_star_k / (exp(x_k) - 1):
    B_star_k / S_k = exp(x_k) - 1
    n_u / n_l = (g_u/g_l) / exp(x_k) = (g_u/g_l) * exp(-h nu_k / k T)  [Boltzmann]

References
----------
- Hubeny & Mihalas (2014), Theory of Stellar Atmospheres, §13.2 (SE equations)
- SPEC_threelevel §6.1 (algorithm specification)
"""

from __future__ import annotations

import torch

from ..atomic_model import AtomicModel

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant, J s
_K = 1.380649e-23     # Boltzmann constant, J/K
_C = 2.99792458e8     # speed of light, m/s


def rate_equation_solve_3level(
    J_bar_lines: torch.Tensor,
    T: torch.Tensor,
    n_e: torch.Tensor,
    *,
    atomic_model: AtomicModel,
    n_tot: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """3-level closed-form SE solver under master-eps per transition.

    Algorithm
    ---------
    For each depth tau:
      1. Build the (L x L) rate matrix R[tau] from all K transitions.
      2. Replace the last row with [1, 1, ..., 1] (sum n_l = n_tot).
      3. Right-hand side: [0, ..., 0, n_tot].
      4. Solve the augmented system via torch.linalg.solve (batched over tau).

    Parameters
    ----------
    J_bar_lines : Tensor of shape (N_tau, K), float64
        Profile-weighted mean intensity per line, SI units (W/m^2/Hz/sr).
        Column order matches atomic_model.transitions order.
    T : Tensor of shape (N_tau,), float64
        Temperature at each depth [K].
    n_e : Tensor of shape (N_tau,), float64
        Electron density [cm^-3].  UNUSED in master-eps form; kept for API
        compatibility with Phase II explicit collisional rates.
    atomic_model : AtomicModel
        Must have n_levels == 3 (Phase I restriction).  Supplies transition
        indices, frequencies, epsilons, and statistical weights.
    n_tot : float or Tensor broadcastable to (N_tau,), default 1.0
        Total bound population (sum over all levels).

    Returns
    -------
    populations : Tensor of shape (N_tau, L), float64
        Each row sums to n_tot within float64 rounding.

    Raises
    ------
    ValueError
        If atomic_model.n_levels != 3 (Phase I restriction).

    Notes
    -----
    n_e is accepted but unused — see docstring.  A type-checker may warn about
    an unused argument; this is expected per the master-eps design (A10).
    """
    if atomic_model.n_levels != 3:
        raise ValueError(
            f"rate_equation_solve_3level requires atomic_model.n_levels == 3, "
            f"got n_levels={atomic_model.n_levels!r} (Phase I restriction)."
        )

    # Ensure float64 throughout; n_e intentionally unused.
    J_bar_lines = J_bar_lines.to(dtype=torch.float64)
    T = T.to(dtype=torch.float64)

    N_tau = T.shape[0]
    L = atomic_model.n_levels
    K = atomic_model.K

    # Prepare n_tot as a (N_tau,) float64 tensor for the rhs.
    if isinstance(n_tot, torch.Tensor):
        n_tot_t = n_tot.to(dtype=torch.float64, device=T.device).expand(N_tau)
    else:
        n_tot_t = torch.full((N_tau,), float(n_tot), dtype=torch.float64, device=T.device)

    # Allocate batched rate matrix: shape (N_tau, L, L).
    R = torch.zeros(N_tau, L, L, dtype=torch.float64, device=T.device)

    # Build rate matrix contributions from each transition.
    for k in range(K):
        tr = atomic_model.transitions[k]
        l_k = tr.lower    # lower level index
        u_k = tr.upper    # upper level index
        eps_k = tr.epsilon
        nu_k = tr.nu_0

        # Statistical weight ratio g_u / g_l for this transition.
        g_l = float(atomic_model.statistical_weights[l_k])
        g_u = float(atomic_model.statistical_weights[u_k])
        g_ratio = g_u / g_l

        # Wien amplitude: B_star_k = 2 h nu_k^3 / c^2, scalar.
        B_star_k = 2.0 * _H * nu_k**3 / _C**2

        # Planck function at line center, shape (N_tau,).
        x_k = _H * nu_k / (_K * T)
        B_k = B_star_k / (torch.exp(x_k) - 1.0)

        # Source function: S_k = (1 - eps_k) * J_bar_k + eps_k * B_k.
        J_k = J_bar_lines[:, k]  # (N_tau,)
        S_k = (1.0 - eps_k) * J_k + eps_k * B_k  # (N_tau,)

        # Rate coefficients (per unit population of respective level).
        # C_out_k: effective absorption rate coefficient (lower -> upper).
        # C_in_k:  effective de-excitation rate coefficient (upper -> lower).
        C_out_k = g_ratio * S_k           # (N_tau,): rate out of lower level
        C_in_k = S_k + B_star_k           # (N_tau,): rate out of upper level

        # Accumulate into rate matrix R[tau, i, j]:
        #   R[l_k, l_k] -= C_out_k   (loss from lower)
        #   R[l_k, u_k] += C_in_k    (gain to lower from upper emission)
        #   R[u_k, u_k] -= C_in_k    (loss from upper)
        #   R[u_k, l_k] += C_out_k   (gain to upper from lower absorption)
        R[:, l_k, l_k] -= C_out_k
        R[:, l_k, u_k] += C_in_k
        R[:, u_k, u_k] -= C_in_k
        R[:, u_k, l_k] += C_out_k

    # Replace last row with normalisation constraint: sum n_l = n_tot.
    # R[:, L-1, :] = [1, 1, ..., 1].
    R[:, L - 1, :] = 0.0
    R[:, L - 1, :] = 1.0

    # Build right-hand side: [0, ..., 0, n_tot].
    b = torch.zeros(N_tau, L, dtype=torch.float64, device=T.device)
    b[:, L - 1] = n_tot_t

    # Batched linear solve: R n = b, shape (N_tau, L).
    # torch.linalg.solve expects (..., n, n) A and (..., n) or (..., n, k) b.
    populations = torch.linalg.solve(R, b)   # (N_tau, L)

    return populations
