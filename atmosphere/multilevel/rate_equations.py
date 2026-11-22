"""Generic N-level statistical-equilibrium solver. SPEC_nlte_scaleup §M.2.

Direct generalisation of atmosphere/threelevel/rate_equations.py to
arbitrary level count L.  The algorithm is identical — only the
precondition `n_levels != 3` is lifted and L is read dynamically from
the AtomicModel.

Physics summary
---------------
For each transition k connecting (lower=l_k, upper=u_k) with master-epsilon
parameterisation (assumption A10):

    Source function:
        S_k = (1 - eps_k) * J_bar_k + eps_k * B(T, nu_k)

    Wien amplitude:
        B_star_k = 2 h nu_k^3 / c^2

    Planck function at line center:
        B(T, nu_k) = B_star_k / (exp(h nu_k / k T) - 1)

Rate matrix construction
------------------------
For each transition k = (lower=l_k, upper=u_k):
    C_out_k = (g_u / g_l) * S_k     (absorption: lower -> upper per unit n_lower)
    C_in_k  = S_k + B_star_k        (de-excitation: upper -> lower per unit n_upper)

    R[tau, l_k, l_k] -= C_out_k
    R[tau, l_k, u_k] += C_in_k
    R[tau, u_k, u_k] -= C_in_k
    R[tau, u_k, l_k] += C_out_k

Augmented system
----------------
Replace the ground-state row for blocks with >=4 levels (the last row
for the legacy 2/3-level case) by the conservation constraint. Keeping
every excited-state balance avoids cancellation of very small populations.
Batched torch.linalg.solve over the N_tau dimension.

3-level equivalence
-------------------
When called with AtomicModel.three_level_hydrogen_chained, produces the
same populations as rate_equation_solve_3level to float64 precision (the
algorithms are algebraically identical; see test A-nlevel-3level-limit).

Multi-species block-diagonal path (SPEC_phase_s §2, v1.1)
----------------------------------------------------------
When atomic_model.species_blocks is not None, the solver branches to a
per-species block-diagonal solve.  Each species block (lo, hi) builds its
own sub-matrix and applies its own conservation constraint:

    Σ_{i ∈ [lo, hi)} n_i(τ) = species_fractions[species_label]

where species_fractions comes from atmosphere.abundance_fractions when
provided, and defaults to a uniform distribution otherwise.

The H-only and single-block paths use the same solve. For blocks with
>=4 levels the ground-state normalization row is essential to preserve
small excited populations; legacy 2/3-level arithmetic is retained.

n_e handling (Q3 resolved decision)
-------------------------------------
n_e is ACCEPTED as an argument but UNUSED: master-epsilon (A10) absorbs
all collisional physics into eps_k.  This preserves API compatibility with
the 3-level path and matches the resolved Q3 decision in SPEC §11.

References
----------
- Hubeny & Mihalas (2014), Theory of Stellar Atmospheres, §13.2
- SPEC_nlte_scaleup §M.2 (algorithm + acceptance criteria)
- SPEC_threelevel §6.1 (3-level ancestor)
- SPEC_phase_s §2 (multi-species block-diagonal extension)
"""

from __future__ import annotations

import torch

from ..atomic_model import AtomicModel

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_H = 6.62607015e-34   # Planck constant [J s]
_K = 1.380649e-23     # Boltzmann constant [J/K]
_C = 2.99792458e8     # speed of light [m/s]


def _build_rate_matrix_block(
    J_bar_lines: torch.Tensor,
    T: torch.Tensor,
    atomic_model: AtomicModel,
    level_indices: tuple[int, int],
    transition_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the rate matrix for a contiguous block of levels.

    This helper implements the same rate-matrix construction as
    rate_equation_solve_nlevel but on a sub-block of levels specified
    by ``level_indices = (lo, hi)`` (half-open).  Only the transitions
    listed in ``transition_indices`` contribute to this block.

    Parameters
    ----------
    J_bar_lines : Tensor of shape (N_tau, K), float64
        Profile-weighted mean intensities for ALL K transitions (global).
    T : Tensor of shape (N_tau,), float64
        Temperature at each depth.
    atomic_model : AtomicModel
        Full atomic model (used for statistical weights and transition data).
    level_indices : (lo, hi)
        Half-open range of level indices belonging to this species block.
    transition_indices : list of int
        Indices into atomic_model.transitions that belong to this block.

    Returns
    -------
    R_block : Tensor of shape (N_tau, B, B)
        Rate matrix for the block of size B = hi - lo.
    b_block : Tensor of shape (N_tau, B)
        Right-hand-side vector (zeroes except the conservation-row entry).
    norm_row : int
        Row index (within the block) where the conservation constraint is
        placed: ground row for B >= 4, otherwise B - 1.
    """
    N_tau = T.shape[0]
    lo, hi = level_indices
    B = hi - lo   # block size (number of levels in this species block)

    R_block = torch.zeros(N_tau, B, B, dtype=torch.float64, device=T.device)

    for k in transition_indices:
        tr = atomic_model.transitions[k]
        # Block-local level indices (0-based within this species block).
        l_k = tr.lower - lo
        u_k = tr.upper - lo
        eps_k = tr.epsilon
        nu_k = tr.nu_0

        g_l = float(atomic_model.statistical_weights[tr.lower])
        g_u = float(atomic_model.statistical_weights[tr.upper])
        g_ratio = g_u / g_l

        B_star_k = 2.0 * _H * nu_k**3 / _C**2
        x_k = _H * nu_k / (_K * T)
        B_k = B_star_k / (torch.exp(x_k) - 1.0)

        J_k = J_bar_lines[:, k]
        S_k = (1.0 - eps_k) * J_k + eps_k * B_k

        C_out_k = g_ratio * S_k
        C_in_k  = S_k + B_star_k

        R_block[:, l_k, l_k] -= C_out_k
        R_block[:, l_k, u_k] += C_in_k
        R_block[:, u_k, u_k] -= C_in_k
        R_block[:, u_k, l_k] += C_out_k

    # Eliminate the ground-state balance, keeping every small excited-state
    # balance. Replacing the highest-level row instead loses small populations
    # through cancellation against the unit conservation constraint.
    norm_row = 0 if B >= 4 else B - 1
    R_block[:, norm_row, :] = 1.0

    # RHS: all zeros except the last entry (filled by caller with n_species).
    b_block = torch.zeros(N_tau, B, dtype=torch.float64, device=T.device)

    return R_block, b_block, norm_row


def rate_equation_solve_nlevel(
    J_bar_lines: torch.Tensor,
    T: torch.Tensor,
    n_e: torch.Tensor,
    *,
    atomic_model: AtomicModel,
    n_tot: float | torch.Tensor = 1.0,
    species_fractions: dict | None = None,
) -> torch.Tensor:
    """Generic N-level statistical-equilibrium solver under master-epsilon (A10).

    Solves the augmented rate-balance linear system at each depth point,
    batched over the N_tau depth dimension.  L = atomic_model.n_levels;
    K = atomic_model.K.  No restriction on L (unlike rate_equation_solve_3level
    which requires L == 3).

    When ``atomic_model.species_blocks`` is not None, the solver uses a
    block-diagonal path: each species block is solved independently with its
    own conservation constraint ``Σ_{i ∈ block} n_i = f_species``.  This is
    the SPEC_phase_s §2 multi-species extension.  The single-species path
    (species_blocks=None) uses the same conservation-row precision rule.

    Parameters
    ----------
    J_bar_lines : Tensor of shape (N_tau, K), float64
        Profile-weighted mean intensity per line [W m^-2 Hz^-1 sr^-1].
        Column k corresponds to atomic_model.transitions[k].
    T : Tensor of shape (N_tau,), float64
        Temperature at each depth [K].
    n_e : Tensor of shape (N_tau,), float64
        Electron density [cm^-3].  ACCEPTED, UNUSED (master-eps A10).
        Kept for API compatibility with 3-level path and future extension.
    atomic_model : AtomicModel
        Provides n_levels (L), transitions (K), statistical_weights, and
        optionally species_blocks for the multi-species path.
    n_tot : float or Tensor broadcastable to (N_tau,), default 1.0
        Total bound population (sum over ALL levels at each depth).
        Used only in the single-species path; in the multi-species path the
        per-species fractions from species_fractions govern each block.
    species_fractions : dict[str, float] or None, default None
        Per-species number fractions (sum = 1).  Used only when
        atomic_model.species_blocks is not None.  When None in the
        multi-species path, defaults to uniform distribution across the
        blocks that sums to n_tot (= 1 by default).

    Returns
    -------
    populations : Tensor of shape (N_tau, L), float64
        In the single-species path, each row sums to n_tot.
        In the multi-species path, the slice ``[:,lo:hi]`` for each block
        sums to ``species_fractions[label]``.

    Raises
    ------
    ValueError
        If the rate matrix is singular at any depth (e.g., zero-frequency
        transition causing a degenerate row), or if a species block
        contains no transitions (disconnected levels).  Message includes
        the failing depth index and epsilon vector.

    Notes
    -----
    n_e is accepted but unused — master-epsilon (A10) absorbs collisional
    physics into eps_k per each transition.  Type-checkers may warn about
    the unused argument; this is expected.

    The n_levels == 3 precondition of rate_equation_solve_3level is lifted;
    calling this function with a 3-level AtomicModel is valid and produces
    the same result as rate_equation_solve_3level.
    """
    # Ensure float64 throughout.  n_e is unused after this point.
    J_bar_lines = J_bar_lines.to(dtype=torch.float64)
    T = T.to(dtype=torch.float64)
    if not torch.isfinite(J_bar_lines).all() or (J_bar_lines < 0).any():
        raise ValueError("mean intensities must be finite and nonnegative")
    if not torch.isfinite(T).all() or (T <= 0).any():
        raise ValueError("temperatures must be finite and positive")

    N_tau = T.shape[0]
    L = atomic_model.n_levels
    K = atomic_model.K

    # ------------------------------------------------------------------
    # Multi-species block-diagonal path (SPEC_phase_s §2).
    # Only entered when atomic_model.species_blocks is not None AND more
    # than one block exists (single-species blocks fall through to the
    # standard path below to guarantee byte-identical results).
    # ------------------------------------------------------------------
    if (
        atomic_model.species_blocks is not None
        and len(atomic_model.species_blocks) > 1
    ):
        return _rate_equation_solve_multispecies(
            J_bar_lines=J_bar_lines,
            T=T,
            atomic_model=atomic_model,
            N_tau=N_tau,
            species_fractions=species_fractions,
        )

    # ------------------------------------------------------------------
    # Single-species path (original M.2 code — byte-identical).
    # Also covers the case species_blocks=((0, L),) — a single block.
    # ------------------------------------------------------------------

    # Prepare n_tot as a (N_tau,) float64 tensor for the RHS.
    if isinstance(n_tot, torch.Tensor):
        n_tot_t = n_tot.to(dtype=torch.float64, device=T.device).expand(N_tau)
    else:
        n_tot_t = torch.full(
            (N_tau,), float(n_tot), dtype=torch.float64, device=T.device
        )

    # Allocate batched rate matrix: shape (N_tau, L, L).
    R = torch.zeros(N_tau, L, L, dtype=torch.float64, device=T.device)

    # Build rate matrix contributions from each transition k.
    for k in range(K):
        tr = atomic_model.transitions[k]
        l_k = tr.lower    # 0-based lower level index
        u_k = tr.upper    # 0-based upper level index
        eps_k = tr.epsilon
        nu_k = tr.nu_0

        # Statistical weight ratio g_u / g_l for this transition.
        g_l = float(atomic_model.statistical_weights[l_k])
        g_u = float(atomic_model.statistical_weights[u_k])
        g_ratio = g_u / g_l

        # Wien amplitude: B_star_k = 2 h nu_k^3 / c^2 (scalar float).
        B_star_k = 2.0 * _H * nu_k**3 / _C**2

        # Planck function at line center, shape (N_tau,).
        x_k = _H * nu_k / (_K * T)
        B_k = B_star_k / (torch.exp(x_k) - 1.0)

        # Source function: S_k = (1 - eps_k) * J_bar_k + eps_k * B_k.
        J_k = J_bar_lines[:, k]   # (N_tau,)
        S_k = (1.0 - eps_k) * J_k + eps_k * B_k   # (N_tau,)

        # Rate coefficients (per unit population of the respective level):
        #   C_out_k: effective absorption rate (lower -> upper per n_lower).
        #   C_in_k:  effective de-excitation rate (upper -> lower per n_upper).
        C_out_k = g_ratio * S_k        # (N_tau,)
        C_in_k = S_k + B_star_k        # (N_tau,)

        # Accumulate into rate matrix R[tau, i, j]:
        #   R[l_k, l_k] -= C_out_k   (loss from lower level)
        #   R[l_k, u_k] += C_in_k    (gain to lower from upper emission)
        #   R[u_k, u_k] -= C_in_k    (loss from upper level)
        #   R[u_k, l_k] += C_out_k   (gain to upper from lower absorption)
        R[:, l_k, l_k] -= C_out_k
        R[:, l_k, u_k] += C_in_k
        R[:, u_k, u_k] -= C_in_k
        R[:, u_k, l_k] += C_out_k

    norm_row = 0 if L >= 4 else L - 1
    R[:, norm_row, :] = 1.0

    # Build the right-hand side at the selected conservation row.
    b = torch.zeros(N_tau, L, dtype=torch.float64, device=T.device)
    b[:, norm_row] = n_tot_t

    # Singularity guard: check that column L-1 (the disconnected level column)
    # is non-zero in at least one of the rate-equation rows (rows 0..L-2).
    # A column that is entirely zero in the rate rows (before normalization
    # replacement) means one level has no transitions connecting it to the rest
    # of the system.  After the normalization row replacement the resulting
    # matrix has determinant zero, causing torch.linalg.solve to return
    # nonsensical results silently in the batched mode.
    #
    # More general check: compute singular values of the augmented matrix.
    # For efficiency use the rate matrix BEFORE normalization row replacement
    # to detect under-connected levels: if any column in rows 0..L-2 is
    # identically zero (no transition connects that level), the system is
    # structurally singular.
    #
    # We scan over the N_tau=1 representative slice (index 0) for speed.
    if N_tau > 0:
        rate_rows = [j for j in range(L) if j != norm_row]
        R0_rate = R[0, rate_rows, :]
        col_max = R0_rate.abs().max(dim=0).values   # (L,) — max abs per column
        if (col_max == 0.0).any():
            bad_levels = (col_max == 0.0).nonzero(as_tuple=True)[0].tolist()
            eps_vec = [tr.epsilon for tr in atomic_model.transitions]
            raise ValueError(
                f"rate_equation_solve_nlevel: singular rate matrix — "
                f"level indices {bad_levels} are disconnected (no transition "
                f"references them in rows 0..{L-2}). "
                f"L={L}, K={K}. Per-line epsilon values: {eps_vec}."
            )

    # Batched linear solve: R n = b, shape (N_tau, L).
    # Catches singular-matrix errors and re-raises with a clear message
    # including the failing depth index and per-line epsilon values.
    try:
        populations = torch.linalg.solve(R, b)   # (N_tau, L)
    except (torch._C._LinAlgError, RuntimeError) as exc:
        # Determine which depth index caused the problem by scanning
        # the condition number of each depth slice (expensive but informative).
        worst_idx = None
        for i in range(N_tau):
            try:
                torch.linalg.solve(R[i], b[i])
            except (torch._C._LinAlgError, RuntimeError):
                worst_idx = i
                break
        eps_vec = [
            tr.epsilon for tr in atomic_model.transitions
        ]
        raise ValueError(
            f"rate_equation_solve_nlevel: singular rate matrix at depth index "
            f"{worst_idx} (L={L}, K={K}). "
            f"Per-line epsilon values: {eps_vec}. "
            f"Original error: {exc}"
        ) from exc

    return populations


def _rate_equation_solve_multispecies(
    J_bar_lines: torch.Tensor,
    T: torch.Tensor,
    atomic_model: AtomicModel,
    N_tau: int,
    species_fractions: dict | None,
) -> torch.Tensor:
    """Block-diagonal multi-species SE solver (SPEC_phase_s §2).

    Solves each species block independently with its own conservation
    constraint.  Called only when atomic_model.species_blocks has more
    than one block.

    Returns populations : Tensor of shape (N_tau, L), float64.
    """
    L = atomic_model.n_levels
    K = atomic_model.K
    blocks = atomic_model.species_blocks  # tuple of (lo, hi) pairs
    species = atomic_model.species        # per-level species labels

    # Build a map from transition index -> species block (lo, hi).
    # Also build the list of transitions belonging to each block.
    block_transitions: dict[int, list[int]] = {i: [] for i in range(len(blocks))}
    for k in range(K):
        tr = atomic_model.transitions[k]
        # The transition's species block is determined by tr.lower's block.
        for b_idx, (lo, hi) in enumerate(blocks):
            if lo <= tr.lower < hi:
                block_transitions[b_idx].append(k)
                break

    # Resolve per-species fractions.
    # species_fractions is keyed by species label string.
    # When None, use uniform 1/n_blocks per block (for testing convenience).
    if species_fractions is None:
        n_blocks = len(blocks)
        sf: dict[str, float] = {}
        for lo, hi in blocks:
            label = species[lo]
            sf[label] = 1.0 / n_blocks
    else:
        sf = species_fractions

    # Allocate output.
    populations = torch.zeros(N_tau, L, dtype=torch.float64, device=T.device)

    for b_idx, (lo, hi) in enumerate(blocks):
        B = hi - lo
        sp_label = species[lo]
        frac = float(sf.get(sp_label, 1.0 / len(blocks)))

        tr_indices = block_transitions[b_idx]

        # Build the block rate matrix via the helper.
        R_block, b_vec, norm_row = _build_rate_matrix_block(
            J_bar_lines=J_bar_lines,
            T=T,
            atomic_model=atomic_model,
            level_indices=(lo, hi),
            transition_indices=tr_indices,
        )
        # Set the RHS conservation entry.
        b_vec[:, norm_row] = frac

        # Singularity guard for this block (analogous to single-species guard).
        if N_tau > 0:
            R0_rate = R_block[0, [j for j in range(B) if j != norm_row], :]
            col_max = R0_rate.abs().max(dim=0).values
            if (col_max == 0.0).any():
                bad_local = (col_max == 0.0).nonzero(as_tuple=True)[0].tolist()
                bad_global = [lo + j for j in bad_local]
                eps_block = [
                    atomic_model.transitions[k].epsilon for k in tr_indices
                ]
                raise ValueError(
                    f"rate_equation_solve_nlevel: singular rate matrix in "
                    f"species '{sp_label}' block (levels {lo}..{hi-1}). "
                    f"Disconnected block-local levels: {bad_local} "
                    f"(global: {bad_global}). "
                    f"Block transition epsilons: {eps_block}."
                )

        # Batched solve for this block.
        try:
            pops_block = torch.linalg.solve(R_block, b_vec)  # (N_tau, B)
        except (torch._C._LinAlgError, RuntimeError) as exc:
            worst_idx = None
            for i in range(N_tau):
                try:
                    torch.linalg.solve(R_block[i], b_vec[i])
                except (torch._C._LinAlgError, RuntimeError):
                    worst_idx = i
                    break
            raise ValueError(
                f"rate_equation_solve_nlevel: singular rate matrix in "
                f"species '{sp_label}' block at depth index {worst_idx}. "
                f"Block levels [{lo}, {hi}). "
                f"Original error: {exc}"
            ) from exc

        populations[:, lo:hi] = pops_block

    return populations
