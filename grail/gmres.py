"""Matrix-free GMRES with restart and diagnostics. the technical reference §5.3.

Right-preconditioned restarted GMRES following Saad & Schultz (1986,
SIAM J. Sci. Stat. Comput. 7, 856).  The outer restart loop is the
standard GMRES(m) algorithm; within each restart cycle the Arnoldi process
builds an orthonormal Krylov basis and the upper Hessenberg H.

Givens rotations are applied incrementally to maintain a reduced QR
factorization of H, so the least-squares problem is solved in O(m^2)
work per outer iteration rather than O(m^3).

Diagnostics returned on every call (the technical reference §5.3):
  - residual_history: ‖r_k‖ / ‖b‖ at each Arnoldi step across all restarts.
  - condition_estimate: σ_max(H_m) / σ_min(H_m) at the end of the final
    restart, estimated from the singular values of the upper Hessenberg
    matrix.  If the Hessenberg has only one column, condition is 1.0.
  - iter_count: total Arnoldi steps consumed.

All arithmetic is in float64.

References
----------
Saad, Y. & Schultz, M. H. (1986), SIAM J. Sci. Stat. Comput. 7(3), 856–869.
Saad, Y. (2003), Iterative Methods for Sparse Linear Systems, 2nd ed., Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .backend import Backend, get_backend


@dataclass
class GmresOutput:
    """Output of a matrix-free GMRES solve.

    Attributes
    ----------
    x : array of shape (n,)
        Approximate solution (native backend type).
    iter_count : int
        Total Arnoldi steps performed across all restart cycles.
    residual_history : list[float]
        Relative residual ‖r_k‖/‖b‖ at each Arnoldi step (and at each
        restart boundary).
    condition_estimate : float
        Condition-number estimate of (I − DT) from the Arnoldi Hessenberg:
        σ_max / σ_min of the final m×(m+1) Hessenberg.  This is the
        condition number of the Hessenberg, a proxy for the full operator
        condition.  Returns 1.0 if the Hessenberg has fewer than 2 columns.
    """

    x: object  # native backend array (torch.Tensor or jax.Array)
    iter_count: int
    residual_history: list[float] = field(default_factory=list)
    condition_estimate: float = 1.0


def gmres(
    matvec: Callable,
    b,
    *,
    tol: float = 1e-8,
    maxiter: int = 200,
    restart: int = 30,
    preconditioner: Optional[Callable] = None,
    backend: Optional[Backend] = None,
) -> GmresOutput:
    """Right-preconditioned restarted GMRES: solve A x = b.

    Right preconditioning: the system solved is A M^{-1} y = b, x = M^{-1} y.
    If preconditioner is None, M = I.

    Parameters
    ----------
    matvec : callable v -> A v
        Matrix-free operator.  Must accept and return float64 arrays of shape
        (n,).  May also accept/return NumPy arrays as long as they are
        numerically equivalent; gmres will convert to the active backend type.
    b : float64 array of shape (n,)
        Right-hand side.  May be any backend type; converted internally.
    tol : float
        Convergence tolerance on the relative residual ‖r‖/‖b‖.
    maxiter : int
        Maximum total Arnoldi steps across all restart cycles.
    restart : int
        Krylov subspace dimension before each restart.  Effective restart is
        min(restart, maxiter).
    preconditioner : callable v -> M^{-1} v, optional
        Right preconditioner.  Same type conventions as matvec.
    backend : Backend or None
        Array-dispatch backend to use.  Defaults to the active default (torch).

    Returns
    -------
    GmresOutput

    Notes
    -----
    Implementation strategy:
      1. Start with initial guess x = 0 and residual r = b − A·x = b.
      2. In each restart cycle of length at most `restart`:
         a. Normalize r to get q_1 = r / ‖r‖.
         b. Arnoldi: expand the Krylov basis Q = [q_1, …, q_m], building
            the (m+1)×m upper Hessenberg H column by column.
         c. Apply Givens rotations to solve the least-squares problem
            ‖H y − e_1 β‖ in the (m+1)-dimensional space, giving the
            correction δx = Q_m y.
         d. Update x ← x + P δx (right preconditioner unwrap if used).
      3. Converge when ‖r‖/‖b‖ < tol, or when maxiter is exhausted.

    Givens rotation encoding follows Saad (2003) §6.5.3: the (i, i+1)
    rotation is encoded as (c_i, s_i) with c_i = cos θ, s_i = sin θ.
    """
    bk = backend if backend is not None else get_backend()

    # Ensure float64 via the backend's from_numpy path (handles any input type).
    import numpy as np
    if not isinstance(b, np.ndarray):
        try:
            b = bk.to_numpy(b)
        except Exception:
            b = np.asarray(b, dtype=np.float64)
    b = bk.from_numpy(np.asarray(b, dtype=np.float64))

    n = b.shape[0]
    b_norm = bk.norm(b)

    residual_history: list[float] = []
    iter_count = 0
    condition_estimate = 1.0

    # Handle the zero RHS trivially.
    if b_norm == 0.0:
        return GmresOutput(
            x=bk.zeros(n),
            iter_count=0,
            residual_history=[0.0],
            condition_estimate=1.0,
        )

    # Initial guess x = 0.
    x = bk.zeros(n)

    def _apply(op: Callable, v) -> object:
        """Apply a callback, converting output to the active backend type."""
        out = op(v)
        if not isinstance(out, np.ndarray):
            try:
                out = bk.to_numpy(out)
            except Exception:
                out = np.asarray(out, dtype=np.float64)
        return bk.from_numpy(np.asarray(out, dtype=np.float64))

    outer_iters = 0  # total Arnoldi steps performed

    while outer_iters < maxiter:
        # --- Compute current residual r = b - A x ---
        if outer_iters == 0:
            r = bk.clone(b)
        else:
            r = b - _apply(matvec, x)  # backend supports element-wise subtraction

        r_norm = bk.norm(r)
        rel_resid = r_norm / b_norm
        residual_history.append(rel_resid)

        if rel_resid < tol:
            break

        # Arnoldi basis: Q has shape (n, m+1) where m = current restart size.
        # H (upper Hessenberg) stored as a dense matrix for condition estimation.
        m_max = min(restart, maxiter - outer_iters)

        # Use backend zeros for Q and H so all arithmetic is backend-native.
        Q = bk.zeros(n, m_max + 1)
        H = bk.zeros(m_max + 1, m_max)

        # q_0 = r / ‖r‖  — stored in column 0 of Q.
        # We perform indexed assignment via numpy round-trip for portability
        # across backends (JAX arrays are immutable).
        # NOTE: bk.to_numpy() may return a read-only numpy view (JAX arrays
        # are immutable; np.asarray of a JAX array is non-writable by design).
        # .copy() ensures we get a mutable numpy array for in-place assignment.
        Q_np = bk.to_numpy(Q).copy()
        Q_np[:, 0] = bk.to_numpy(r).copy() / r_norm
        Q = bk.from_numpy(Q_np)

        H_np = bk.to_numpy(H).copy()

        # Givens rotation parameters (plain Python floats — scalar ops).
        cs = [0.0] * m_max  # cosines
        sn = [0.0] * m_max  # sines

        # g: right-hand side of the Hessenberg LS problem (size m_max+1).
        g = [0.0] * (m_max + 1)
        g[0] = r_norm

        j = 0  # steps within this restart
        for j in range(m_max):
            outer_iters += 1

            # --- Arnoldi step: compute A q_j (with preconditioning) ---
            q_j = bk.from_numpy(bk.to_numpy(Q)[:, j])
            if preconditioner is not None:
                p_j = _apply(preconditioner, q_j)
                w = _apply(matvec, p_j)
            else:
                w = _apply(matvec, q_j)

            w_np = bk.to_numpy(w).copy()
            Q_np = bk.to_numpy(Q).copy()  # refresh — mutable copy for assignment below

            # Modified Gram-Schmidt orthogonalization.
            for i in range(j + 1):
                q_i = Q_np[:, i]
                h_ij = float(np.dot(w_np, q_i))
                H_np[i, j] = h_ij
                w_np = w_np - h_ij * q_i

            h_jp1 = float(np.linalg.norm(w_np))
            H_np[j + 1, j] = h_jp1

            if h_jp1 > 0.0:
                Q_np[:, j + 1] = w_np / h_jp1
                Q = bk.from_numpy(Q_np)

            # --- Apply previous Givens rotations to the new column ---
            for i in range(j):
                tmp = cs[i] * H_np[i, j] + sn[i] * H_np[i + 1, j]
                H_np[i + 1, j] = -sn[i] * H_np[i, j] + cs[i] * H_np[i + 1, j]
                H_np[i, j] = tmp

            # --- Compute new Givens rotation to zero H[j+1, j] ---
            denom = math.hypot(H_np[j, j], H_np[j + 1, j])
            if denom == 0.0:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                cs[j] = H_np[j, j] / denom
                sn[j] = H_np[j + 1, j] / denom

            H_np[j, j] = cs[j] * H_np[j, j] + sn[j] * H_np[j + 1, j]
            H_np[j + 1, j] = 0.0

            # Apply to the RHS.
            g_jp1 = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            g[j + 1] = g_jp1

            # Residual after j+1 Arnoldi steps.
            r_norm_approx = abs(g[j + 1])
            rel_resid = r_norm_approx / b_norm
            residual_history.append(rel_resid)

            if outer_iters >= maxiter or rel_resid < tol:
                break

        # Number of completed Arnoldi steps in this restart.
        m = j + 1

        # --- Solve the upper triangular system H[0:m, 0:m] y = g[0:m] ---
        R_np = H_np[:m, :m].copy()
        rhs_g_np = np.array(g[:m], dtype=np.float64).reshape(-1, 1)

        R_bk = bk.from_numpy(R_np)
        rhs_g_bk = bk.from_numpy(rhs_g_np)

        # Back-substitution (R is upper triangular after Givens).
        y_bk = bk.solve_triangular(R_bk, rhs_g_bk, upper=True)
        y_np = bk.to_numpy(y_bk).flatten()

        # --- Compute condition estimate from the Hessenberg ---
        # Use the full (m+1) x m Hessenberg (before Givens were applied).
        if m >= 2:
            H_orig_bk = bk.from_numpy(H_np[:m + 1, :m].copy())
            sv = bk.svdvals(H_orig_bk)
            sv_np = bk.to_numpy(sv)
            sv_max = float(sv_np[0])
            sv_min = float(sv_np[-1])
            if sv_min > 0:
                condition_estimate = sv_max / sv_min
            else:
                condition_estimate = float("inf")

        # --- Update the solution x ← x + Q_m y ---
        # Q_m = Q[:, :m], shape (n, m).  Correction: Q_m @ y.
        Q_final_np = bk.to_numpy(Q)
        delta_x_np = Q_final_np[:, :m] @ y_np  # (n,)

        if preconditioner is not None:
            delta_x_bk = _apply(preconditioner, bk.from_numpy(delta_x_np))
            delta_x_np = bk.to_numpy(delta_x_bk)

        x_np = bk.to_numpy(x) + delta_x_np
        x = bk.from_numpy(x_np)

        # Arnoldi's projected residual can underestimate the actual defect
        # when callbacks or arithmetic are inexact. Re-evaluate the original
        # system before declaring convergence; restart if necessary.
        true_resid = bk.norm(b - _apply(matvec, x)) / b_norm
        residual_history.append(true_resid)
        if true_resid < tol:
            break

    return GmresOutput(
        x=x,
        iter_count=outer_iters,
        residual_history=residual_history,
        condition_estimate=condition_estimate,
    )
