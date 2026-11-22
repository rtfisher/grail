"""Λ*-preconditioner factory for the multilevel GMRES resolvent solve.

Classical ALI preconditioner (Rybicki & Hummer 1991; Olson, Auer & Buchler 1986)
adapted for the N-level NLTE cycle.

Theory
------
The cycle Jacobian K = ∂T/∂x at x*_truth chains two blocks:

    K = (∂SE/∂J̄) · Λ · (∂S/∂x)

where:
  - ∂S/∂x : (K × L) source-function Jacobian at each depth (S = S(x))
  - Λ      : (K × K) block-diagonal in depth, full off-diagonal (Λ[i,j] ≠ 0
             for i ≠ j because photons propagate)
  - ∂SE/∂J̄: (L × K) SE Jacobian at each depth

Replacing the full Λ with its diagonal Λ* = diag(Λ) gives the ALI-approximate
Jacobian K*:

    K*_ii = (∂SE/∂J̄)_i · diag(λ*_k[i]) · (∂S/∂x)_i    ∈ ℝ^{L×L}

K* is block-diagonal in depth (the local ∂S/∂x and ∂SE/∂J̄ involve only depth i).
(I − K*) is therefore block-diagonal in depth with L×L blocks.

Preconditioner construction
---------------------------
1. Compute Λ*_k diagonal for each line k via _compute_lambda_star_diagonal.
2. Compute (K × L) ∂S/∂x at each depth via torch.func.jacobian on
   _source_from_populations_nlevel at x_truth.
3. Compute (L × K) ∂SE/∂J̄ at each depth via torch.func.jacobian on
   rate_equation_solve_nlevel at J_bar_truth.
4. Assemble L×L blocks: M_ii = I_L - (∂SE/∂J̄)_i · diag(λ*[i]) · (∂S/∂x)_i.
5. Factor each M_ii (small L×L LU) and cache the L×L inverses.

Apply
-----
For v ∈ ℝ^{L·N_τ}: reshape to (L, N_τ), apply depth-i L×L inverse to column i,
reshape back.  Cost: O(L³ · N_τ) — effectively free.

References
----------
Rybicki, G. B. & Hummer, D. G. (1991), A&A 245, 171.
Olson, G. L., Auer, L. H., & Buchler, J. R. (1986), JQSRT 35, 431.
SPEC_nlte_scaleup §M.6 (follow-on task).
"""

from __future__ import annotations

from typing import Callable

import torch

from ..atmosphere import Atmosphere
from ..atomic_model import AtomicModel
from ..twolevel.fixed_point import _compute_lambda_star_diagonal
from .fixed_point import (
    _atm_for_line,
    _compute_lambda_full_nlevel,
    _source_from_populations_nlevel,
)
from .rate_equations import rate_equation_solve_nlevel


def make_lambda_star_preconditioner_multilevel(
    atm: Atmosphere,
    x_truth: torch.Tensor,
    J_bar_lines_truth: torch.Tensor | None = None,
    formal_solve_mode: str = "linear",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the ALI Λ*-preconditioner callable for multilevel GMRES.

    M = I - K* where K* is the ALI-approximated cycle Jacobian with each
    per-line Λ_k replaced by its diagonal Λ*_k.  K* is block-diagonal in
    depth (L×L block per depth point), so the apply cost is O(L³ · N_τ).

    Parameters
    ----------
    atm : Atmosphere
        Must have atomic_model with n_levels and K transitions.
    x_truth : Tensor of shape (N_tau, L)
        Level populations at the truth fixed point.  Jacobians are evaluated
        here (linearization point for the preconditioner).
    J_bar_lines_truth : Tensor of shape (N_tau, K) or None
        Profile-weighted mean intensities at the truth fixed point.
        If None, computed from x_truth via _source_from_populations_nlevel
        and the Λ matrices (first-order estimate).  Providing the actual
        J_bar_truth from ali_loop_nlevel is preferred for accuracy.
    formal_solve_mode : str, default 'linear'
        Passed to _compute_lambda_star_diagonal.

    Returns
    -------
    preconditioner : callable  v -> M^{-1} v
        Right preconditioner for grail.resolvent_solve's preconditioner slot.
        Accepts and returns float64 tensors of shape (L*N_tau,).
    """
    atomic_model: AtomicModel = atm.atomic_model
    L = atomic_model.n_levels
    K = atomic_model.K
    N_tau = int(atm.depth.tau.shape[0])

    # ------------------------------------------------------------------
    # Step 1: Λ*_k diagonal for each line k — shape (N_tau,) per line.
    # ------------------------------------------------------------------
    lambda_star: list[torch.Tensor] = []
    for k in range(K):
        atm_k = _atm_for_line(atm, k)
        lstar_k = _compute_lambda_star_diagonal(
            atm_k, formal_solve_mode=formal_solve_mode
        )   # (N_tau,)
        lambda_star.append(lstar_k.to(torch.float64))

    # ------------------------------------------------------------------
    # Step 2: (K × L) ∂S/∂x Jacobian at each depth.
    # Compute the full (N_tau, K, N_tau, L) Jacobian of _source_from_populations_nlevel
    # w.r.t. populations at x_truth, then extract the diagonal blocks [i, :, i, :].
    # ------------------------------------------------------------------
    x_truth_f64 = x_truth.detach().to(torch.float64)

    # We want ∂S_lines[i, k] / ∂x[i, l] for each depth i.
    # torch.func.jacobian gives shape (N_tau, K, N_tau, L) — extract diagonal in depth.
    def _S_fn(pops: torch.Tensor) -> torch.Tensor:
        return _source_from_populations_nlevel(pops, atomic_model)

    # Jacobian shape: (N_tau, K, N_tau, L)
    jac_S = torch.func.jacrev(_S_fn)(x_truth_f64)   # (N_tau, K, N_tau, L)
    # Extract depth-diagonal blocks: dS_dx[i, k, l] = ∂S_lines[i, k] / ∂x[i, l].
    # jac_S[i, k, i, l] for each depth i.  Shape: (N_tau, K, L).
    dS_dx = torch.stack(
        [jac_S[i, :, i, :] for i in range(N_tau)],
        dim=0,
    )   # (N_tau, K, L)

    # ------------------------------------------------------------------
    # Step 3: (L × K) ∂SE/∂J̄ Jacobian at each depth.
    # Compute at J_bar_lines_truth if provided, otherwise use LTE approximation.
    # ------------------------------------------------------------------
    if J_bar_lines_truth is not None:
        J_bar_f64 = J_bar_lines_truth.detach().to(torch.float64)
    else:
        # Fallback: compute J_bar from S(x_truth) via Λ (first-order approximation).
        # Use Λ* diagonal as a cheap J_bar estimate: J_bar_k ≈ Λ*_k * S_k + b_k.
        # For the preconditioner this doesn't need to be exact — we use x_truth
        # directly as the linearization point and set J_bar = SE^{-1}(x_truth).
        # Simplest fallback: J_bar from direct ali_loop result would be needed,
        # but since it's optional we use B(T) as the LTE estimate.
        from ..twolevel.fixed_point import _planck_nu
        J_bar_f64 = torch.stack(
            [_planck_nu(atm.atomic_model.transitions[k].nu_0, atm.T.to(torch.float64))
             for k in range(K)],
            dim=1,
        )   # (N_tau, K)

    T_f64 = atm.T.to(torch.float64)
    n_e_f64 = atm.n_e.to(torch.float64)

    def _SE_fn(jbar: torch.Tensor) -> torch.Tensor:
        return rate_equation_solve_nlevel(
            jbar, T_f64, n_e_f64, atomic_model=atomic_model
        )

    # Jacobian shape: (N_tau, L, N_tau, K)
    jac_SE = torch.func.jacrev(_SE_fn)(J_bar_f64)   # (N_tau, L, N_tau, K)
    # Extract depth-diagonal blocks: dSE_dJ[i] has shape (L, K)
    dSE_dJ = torch.stack(
        [jac_SE[i, :, i, :] for i in range(N_tau)],
        dim=0,
    )   # (N_tau, L, K)

    # ------------------------------------------------------------------
    # Step 4: Assemble and factor depth-local L×L matrices.
    # M_i = I - (∂SE/∂J̄)_i · diag(λ*[i]) · (∂S/∂x)_i
    # ------------------------------------------------------------------
    # Stack Λ* diagonals to shape (N_tau, K).
    lambda_star_mat = torch.stack(lambda_star, dim=1)   # (N_tau, K)

    # For each depth i:
    #   diag_i = lambda_star_mat[i, :]  (K,)
    #   (dSE_dJ)_i * diag_i * (dS_dx)_i
    # = dSE_dJ[i] (L,K) @ diag(diag_i) (K,K) @ dS_dx[i] (K,L)
    # = (dSE_dJ[i] * diag_i.unsqueeze(0)) @ dS_dx[i]   (L,L)
    # K_star_blocks[i] = dSE_dJ[i] @ diag(lambda_star_mat[i]) @ dS_dx[i]
    K_star_blocks = torch.stack([
        dSE_dJ[i] @ torch.diag(lambda_star_mat[i]) @ dS_dx[i]
        for i in range(N_tau)
    ], dim=0)   # (N_tau, L, L)

    I_L = torch.eye(L, dtype=torch.float64)
    M_blocks = I_L.unsqueeze(0) - K_star_blocks   # (N_tau, L, L)

    # Factor each block (small L×L).  Use torch.linalg.inv since L <= 10.
    M_inv_blocks = torch.linalg.inv(M_blocks)   # (N_tau, L, L)

    # ------------------------------------------------------------------
    # Step 5: Build the apply callable.
    # ------------------------------------------------------------------
    def preconditioner(v: torch.Tensor) -> torch.Tensor:
        """Apply M^{-1}: v ∈ ℝ^{L*N_tau} -> M^{-1} v ∈ ℝ^{L*N_tau}.

        Reshapes v to (L, N_tau), applies the per-depth L×L inverse,
        reshapes back to (L*N_tau,).
        """
        import numpy as np
        v_np = v if isinstance(v, np.ndarray) else (
            v.detach().numpy() if hasattr(v, "detach") else np.asarray(v)
        )
        v_t = torch.as_tensor(v_np, dtype=torch.float64)   # (L*N_tau,)
        # Reshape to (N_tau, L) in the GRAIL flat-blocked convention:
        # flat[l*N_tau + i] = populations[i, l]
        # -> reshape(L, N_tau).T gives (N_tau, L)
        v_pops = v_t.reshape(L, N_tau).T.contiguous()   # (N_tau, L)

        # Batched L×L solve: result[i] = M_inv_blocks[i] @ v_pops[i]
        # v_pops[i] is (L,); M_inv_blocks[i] is (L,L).
        # torch.bmm requires (batch, L, 1) for batched matvec.
        v_col = v_pops.unsqueeze(-1)   # (N_tau, L, 1)
        out_col = torch.bmm(M_inv_blocks, v_col)   # (N_tau, L, 1)
        out_pops = out_col.squeeze(-1)             # (N_tau, L)

        # Flatten back: (N_tau, L) -> (L*N_tau,) via populations.T.flatten()
        return out_pops.T.reshape(-1)   # (L*N_tau,)

    return preconditioner


def make_selective_lambda_preconditioner_multilevel(
    atm: Atmosphere,
    x_truth: torch.Tensor,
    full_lambda_lines: tuple[int, ...] = (0, 4),
    J_bar_lines_truth: torch.Tensor | None = None,
    formal_solve_mode: str = "linear",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Selective Λ*/Λ preconditioner: full Λ_k for stiff lines, diagonal Λ*_k for the rest.

    For each line k in ``full_lambda_lines``, the per-line Λ_k matrix (N_τ × N_τ)
    is used in full — cross-depth coupling preserved.  For all other lines,
    the diagonal Λ*_k is used (only the diagonal of the full N_τ × N_τ matrix
    is non-zero).  The full K* matrix (L·N_τ × L·N_τ) is assembled once,
    (I − K*) is LU-factored once via torch.linalg.lu_factor, and the
    preconditioner apply uses lu_solve.

    Classical "core/halo" ALI of Olson, Auer & Buchler (1986): use the full
    Λ_k for lines whose near-unit eigenvalues dominate (the stiffest scattering
    chains), and the cheap diagonal Λ*_k for the rest.  In 5-level Balmer H,
    Ly-α (k=0) and H-α (k=4) are the resonance lines with the deepest
    scattering chains; the other 8 transitions are sub-dominant.

    Transition ordering (AtomicModel.balmer_5level()):
      k=0 Ly-α (1→2, 2.467e15 Hz)    k=5 H-β  (2→4)
      k=1 Ly-β (1→3)                  k=6 H-γ  (2→5)
      k=2 Ly-γ (1→4)                  k=7 Pa-α (3→4)
      k=3 Ly-δ (1→5)                  k=8 Pa-β (3→5)
      k=4 H-α  (2→3, 4.569e14 Hz)    k=9 Br-α (4→5)

    Cost per apply: O((L·N_τ)²) — the LU back-substitution.  At N_τ=80 / L=5,
    that is 400-dim back-sub ≈ 1 ms; at N_τ=200 / L=5, 1000-dim ≈ 10 ms.
    Factorization cost (one-time): O((L·N_τ)³); at N_τ=80 ≈ tens of ms.

    Parameters
    ----------
    atm : Atmosphere
        Must have atomic_model with n_levels and K transitions.
    x_truth : Tensor of shape (N_tau, L)
        Level populations at the truth fixed point.  Jacobians are evaluated
        here (linearization point).
    full_lambda_lines : tuple of int, default (0, 4)
        Line indices k for which the full N_τ × N_τ Λ_k matrix is used.
        All other lines use only diag(Λ_k).  Default is Ly-α and H-α,
        the two resonance lines with the deepest scattering chains.
    J_bar_lines_truth : Tensor of shape (N_tau, K) or None
        Profile-weighted mean intensities at the truth fixed point.
        If None, LTE estimate B(T, ν_k) is used (same fallback as the
        diagonal-only version).
    formal_solve_mode : str, default 'linear'
        Passed to _compute_lambda_full_nlevel.

    Returns
    -------
    preconditioner : callable  v -> M^{-1} v
        Right preconditioner for grail.resolvent_solve's preconditioner slot.
        Accepts and returns float64 tensors of shape (L*N_tau,).

    References
    ----------
    Olson, G. L., Auer, L. H., & Buchler, J. R. (1986), JQSRT 35, 431.
    SPEC_nlte_scaleup §M.6 option 1b.
    """
    atomic_model: AtomicModel = atm.atomic_model
    L = atomic_model.n_levels
    K = atomic_model.K
    N_tau = int(atm.depth.tau.shape[0])
    full_set = set(full_lambda_lines)

    # ------------------------------------------------------------------
    # Step 1: Build per-line N_τ × N_τ Λ matrices.
    # For lines in full_set:  use the full Λ_k (N_τ × N_τ, all entries).
    # For other lines:        use torch.diag(diag(Λ_k)) — N_τ × N_τ,
    #                         only the diagonal non-zero (= diagonal-only ALI).
    # _compute_lambda_full_nlevel returns [(Λ_k, b_k)] for all K lines.
    # ------------------------------------------------------------------
    lambda_results = _compute_lambda_full_nlevel(atm, formal_solve_mode=formal_solve_mode)

    # Lambda_matrices[k] is always (N_tau, N_tau) — full for stiff lines,
    # diagonal-zeroed for the rest.
    Lambda_matrices: list[torch.Tensor] = []
    for k in range(K):
        Lambda_k, _ = lambda_results[k]
        Lambda_k = Lambda_k.to(torch.float64)
        if k in full_set:
            # Full Λ_k: keep all cross-depth entries.
            Lambda_matrices.append(Lambda_k)
        else:
            # Diagonal-only: zero out all off-diagonal entries.
            Lambda_matrices.append(torch.diag(Lambda_k.diag()))

    # ------------------------------------------------------------------
    # Step 2: ∂S/∂x Jacobian at each depth.
    # Same as the diagonal-only version: evaluate torch.func.jacrev at
    # x_truth and extract per-depth (K, L) blocks from the (N_tau, K, N_tau, L)
    # full Jacobian.
    # ------------------------------------------------------------------
    x_truth_f64 = x_truth.detach().to(torch.float64)

    def _S_fn(pops: torch.Tensor) -> torch.Tensor:
        return _source_from_populations_nlevel(pops, atomic_model)

    jac_S = torch.func.jacrev(_S_fn)(x_truth_f64)   # (N_tau, K, N_tau, L)
    # dS_dx[i] = ∂S_lines[i, :] / ∂x[i, :]  — shape (K, L) per depth.
    dS_dx = torch.stack(
        [jac_S[i, :, i, :] for i in range(N_tau)],
        dim=0,
    )   # (N_tau, K, L)

    # ------------------------------------------------------------------
    # Step 3: ∂SE/∂J̄ Jacobian at each depth.
    # ------------------------------------------------------------------
    if J_bar_lines_truth is not None:
        J_bar_f64 = J_bar_lines_truth.detach().to(torch.float64)
    else:
        from ..twolevel.fixed_point import _planck_nu
        J_bar_f64 = torch.stack(
            [_planck_nu(atm.atomic_model.transitions[k].nu_0, atm.T.to(torch.float64))
             for k in range(K)],
            dim=1,
        )   # (N_tau, K)

    T_f64 = atm.T.to(torch.float64)
    n_e_f64 = atm.n_e.to(torch.float64)

    def _SE_fn(jbar: torch.Tensor) -> torch.Tensor:
        return rate_equation_solve_nlevel(
            jbar, T_f64, n_e_f64, atomic_model=atomic_model
        )

    jac_SE = torch.func.jacrev(_SE_fn)(J_bar_f64)   # (N_tau, L, N_tau, K)
    # dSE_dJ[i] = ∂SE[i, :] / ∂J̄[i, :]  — shape (L, K) per depth.
    dSE_dJ = torch.stack(
        [jac_SE[i, :, i, :] for i in range(N_tau)],
        dim=0,
    )   # (N_tau, L, K)

    # ------------------------------------------------------------------
    # Step 4: Assemble the full (L·N_τ × L·N_τ) K* matrix.
    #
    # The per-line contribution to K* is:
    #   K*_k = (∂SE/∂J̄) · Λ̃_k · (∂S/∂x)
    # where Λ̃_k is either the full Λ_k or diag(λ*_k).
    #
    # In the state-vector flat-blocked layout the full-state indices are:
    #   flat index (l * N_tau + i)  <->  (depth i, level l)
    # (same convention as populations.T.flatten() / reshape(L, N_tau).T).
    #
    # For each line k, the (L*N_tau × L*N_tau) contribution K*_k satisfies:
    #   K*_k[l2*N_tau + j, l1*N_tau + i]
    #     = sum_{i2} (∂SE/∂J̄)[j, l2, k] · Λ̃_k[j, i2] · (∂S/∂x)[i2, k, l1]
    #              (where i2 index is summed)
    #   Rearranging: contribution at (j, l2) from (i, l1) via line k =
    #     (∂SE_l2/∂J̄_k)[j] · Λ̃_k[j, i] · (∂S_k/∂x_l1)[i]
    #
    # We build K* as the sum of per-line (L*N_tau × L*N_tau) contributions.
    # ------------------------------------------------------------------
    D = L * N_tau   # full state-space dimension

    K_star = torch.zeros(D, D, dtype=torch.float64)

    for k in range(K):
        # Λ̃_k : (N_tau, N_tau)  (full or diagonal)
        Lambda_k_tilde = Lambda_matrices[k]   # (N_tau, N_tau)

        # Per-depth ∂S_k/∂x_l (source Jacobian for line k w.r.t. all levels):
        # dS_dx[:, k, :] has shape (N_tau, L).
        # We treat it as a column selector: dSdx_k[i, l1] = ∂S_k[i] / ∂x[i, l1].
        dSdx_k = dS_dx[:, k, :]   # (N_tau, L)

        # Per-depth ∂SE_l/∂J̄_k (SE Jacobian for line k w.r.t. all levels):
        # dSE_dJ[:, :, k] has shape (N_tau, L).
        # dSEdJ_k[j, l2] = ∂SE[j, l2] / ∂J̄[j, k].
        dSEdJ_k = dSE_dJ[:, :, k]   # (N_tau, L)

        # Contribution to K*:
        #   K*_k[l2*N_tau + j, l1*N_tau + i]
        #     = dSEdJ_k[j, l2] · Lambda_k_tilde[j, i] · dSdx_k[i, l1]
        #
        # Vectorize: build an (N_tau, N_tau) coupling matrix C_k where
        #   C_k[j, i] = Lambda_k_tilde[j, i]   (already computed).
        # Then the (L*N_tau, L*N_tau) block for (l2, l1) is:
        #   B_{l2,l1}[j, i] = dSEdJ_k[j, l2] · C_k[j, i] · dSdx_k[i, l1]
        #                    = diag(dSEdJ_k[:, l2]) @ C_k @ diag(dSdx_k[:, l1])
        #
        # Build all L² blocks at once using outer products.
        # dSEdJ_k : (N_tau, L) — col l2 scales rows of C_k.
        # dSdx_k  : (N_tau, L) — col l1 scales cols of C_k (at the i side).
        #
        # B_{l2, l1} = dSEdJ_k[:, l2].unsqueeze(1) * C_k * dSdx_k[:, l1].unsqueeze(0)
        # Broadcast: (N_tau,1) * (N_tau, N_tau) * (1, N_tau) -> (N_tau, N_tau).
        for l2 in range(L):
            row_start = l2 * N_tau
            row_end   = row_start + N_tau
            # diag(dSEdJ_k[:, l2]) @ Lambda_k_tilde  — scale rows
            scaled_rows = dSEdJ_k[:, l2].unsqueeze(1) * Lambda_k_tilde   # (N_tau, N_tau)
            for l1 in range(L):
                col_start = l1 * N_tau
                col_end   = col_start + N_tau
                # scaled_rows @ diag(dSdx_k[:, l1])  — scale cols
                block = scaled_rows * dSdx_k[:, l1].unsqueeze(0)   # (N_tau, N_tau)
                K_star[row_start:row_end, col_start:col_end] += block

    # ------------------------------------------------------------------
    # Step 5: Assemble M = I - K*, LU-factor once.
    # ------------------------------------------------------------------
    I_D = torch.eye(D, dtype=torch.float64)
    M = I_D - K_star

    # torch.linalg.lu_factor returns (LU, pivots) for lu_solve.
    LU, pivots = torch.linalg.lu_factor(M)   # LU: (D, D), pivots: (D,)

    # ------------------------------------------------------------------
    # Step 6: Build the apply closure.
    # ------------------------------------------------------------------
    def preconditioner(v: torch.Tensor) -> torch.Tensor:
        """Apply M^{-1}: v ∈ ℝ^{L*N_tau} -> M^{-1} v ∈ ℝ^{L*N_tau}.

        Uses the pre-factored LU decomposition of (I − K*).
        """
        import numpy as np
        v_np = v if isinstance(v, np.ndarray) else (
            v.detach().numpy() if hasattr(v, "detach") else np.asarray(v)
        )
        v_t = torch.as_tensor(v_np, dtype=torch.float64)   # (D,)
        # torch.linalg.lu_solve expects right-hand-side shape (..., D, nrhs).
        # Unsqueeze to (D, 1), solve, squeeze back to (D,).
        result = torch.linalg.lu_solve(LU, pivots, v_t.unsqueeze(-1))   # (D, 1)
        return result.squeeze(-1)   # (D,)

    return preconditioner
