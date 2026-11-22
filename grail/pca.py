"""SVD-based residual basis and per-component resolvent application. the technical reference §5.4.

Workflow
--------
1. fit_basis(residuals, ...)  →  PCABasis
   Compute the mean-subtracted SVD of the (N_val × 2N_tau) residual matrix.
   Truncate to rank r where the first r singular values explain >= 99% of the
   total variance (configurable via explained_variance or fixed_rank).

2. propagate(basis, apply_resolvent, projected_weights)  →  array
   For each principal component u_i (row of basis.components), solve the
   resolvent equation once to get the propagated correction w_i.  Then form
   the output as a weighted combination:

       δx*_band = Σ_i projected_weights[i] * w_i

   where projected_weights are the empirical projection coefficients of the
   validation residuals onto the PC axes, combined with the singular values
   to quantify amplitude.

Float64 throughout; RNG not used in this module (SVD is deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .backend import Backend, get_backend


@dataclass
class PCABasis:
    """Truncated SVD basis on validation residuals delta_T.  the technical reference §5.4.

    Attributes
    ----------
    components : array of shape (rank, dim)
        Rows are the right singular vectors of the mean-centered
        sample-by-component matrix (principal axes in residual space).
        Each row has unit norm. Native backend type.
    singular_values : array of shape (rank,)
        Singular values sigma_i in descending order.  sigma_i^2 is the
        quantity proportional to variance explained by the i-th PC.
        Native backend type.
    rank : int
        Truncation rank r.
    explained_variance_ratio : float
        Fraction of total variance explained by the retained r components,
        i.e.  sum(sigma_i^2, i<=r) / sum(sigma_i^2, all).
    mean : array of shape (dim,)
        Sample mean subtracted before SVD; needed to center new samples.
        Native backend type.
    """

    components: object      # (rank, dim)  native backend array
    singular_values: object  # (rank,)      native backend array
    rank: int
    explained_variance_ratio: float
    mean: object             # (dim,)       native backend array


def fit_basis(
    residuals,
    *,
    explained_variance: float = 0.99,
    fixed_rank: Optional[int] = None,
    backend: Optional[Backend] = None,
) -> PCABasis:
    """Mean-subtracted SVD of validation residuals; rank chosen by explained
    variance threshold (default 99%) or fixed (configurable). the technical reference §5.4.

    Parameters
    ----------
    residuals : array of shape (N, d), float64
        N validation residual vectors delta_T(x*_k), each of dimension d=2*N_tau.
        May be any backend type; converted internally via numpy.
    explained_variance : float in (0, 1], default 0.99
        Retain the fewest components that explain this fraction of total variance.
        Used only when fixed_rank is None.
    fixed_rank : int or None, default None
        If given, truncate to exactly this rank (clamped to min(N, d)).
    backend : Backend or None
        Active backend for returned arrays.  Defaults to the module-level
        default (torch).

    Returns
    -------
    PCABasis with components (rank, d), singular_values (rank,), and metadata.
    All array-valued fields are in the active backend's native type.

    Notes
    -----
    We compute the economy SVD of the mean-centered matrix:
        X_c = X - mean(X, dim=0)   shape (N, d)
        X_c = U S V^T               economy form: U (N×r), S (r,), V (d×r)

    The principal components (row vectors of V^T, i.e., columns of V) form
    the basis in residual space.  The singular values S encode amplitude.

    For N < d the economy SVD returns min(N, d) = N components, each of
    dimension d.  This is the standard PCA behaviour.
    """
    bk = backend if backend is not None else get_backend()

    # Convert to numpy for SVD (numpy's linalg.svd is reliable across backends).
    try:
        res_np = bk.to_numpy(residuals).astype(np.float64)
    except Exception:
        res_np = np.asarray(residuals, dtype=np.float64)

    N, d = res_np.shape

    # Mean-center.
    mean_np = res_np.mean(axis=0)  # (d,)
    X_c = res_np - mean_np         # (N, d)

    # Economy SVD: U (N, min(N,d)), S (min(N,d),), Vh (min(N,d), d).
    # numpy.linalg.svd is deterministic and backend-agnostic.
    U, S, Vh = np.linalg.svd(X_c, full_matrices=False)
    # S is in descending order (numpy guarantees this).

    total_var = float((S ** 2).sum())

    # Determine rank.
    if fixed_rank is not None:
        r = max(1, min(int(fixed_rank), S.shape[0]))
    else:
        # Smallest r such that sum(S_i^2, i<r) / total_var >= explained_variance.
        if total_var == 0.0:
            r = 1
        elif explained_variance >= 1.0:
            # Retain all components — avoid floating-point ambiguity where
            # cumsum reaches exactly 1.0 at r < S.shape[0] due to the last
            # singular value being ~machine epsilon.
            r = S.shape[0]
        else:
            cumvar = np.cumsum(S ** 2)
            mask = (cumvar / total_var) >= explained_variance
            indices = np.where(mask)[0]
            r = int(indices[0]) + 1 if len(indices) > 0 else S.shape[0]
        r = max(1, min(r, S.shape[0]))

    components_np = Vh[:r]   # (r, d): each row is a PC direction
    sv_np = S[:r]             # (r,)

    explained = float((sv_np ** 2).sum()) / total_var if total_var > 0.0 else 1.0

    return PCABasis(
        components=bk.from_numpy(components_np),
        singular_values=bk.from_numpy(sv_np),
        rank=r,
        explained_variance_ratio=float(explained),
        mean=bk.from_numpy(mean_np),
    )


def propagate(
    basis: PCABasis,
    apply_resolvent: Callable,
    projected_weights,
    backend: Optional[Backend] = None,
) -> object:
    """Apply the resolvent to each PC and combine with empirical weights.

    For each principal component u_i (a row of basis.components), solves the
    resolvent equation once:

        w_i = (I − DT)^{-1} u_i

    by calling apply_resolvent(u_i).  Then forms the propagated δx* band:

        δx*_band = Σ_i projected_weights[i] * w_i

    Parameters
    ----------
    basis : PCABasis
        Output of fit_basis.
    apply_resolvent : callable array(d,) -> array(d,)
        Solves (I − DT) w = v for w given v; this is one call to the GMRES
        driver.  The callback is called once per PC (basis.rank calls total).
    projected_weights : array of shape (rank,)
        Empirical weights for each PC — typically the projection of a specific
        residual vector onto the PC axes (u_i · δT), scaled by sigma_i, or
        any other user-supplied amplitude weighting.
    backend : Backend or None
        Active backend.  Defaults to the module-level default (torch).

    Returns
    -------
    delta_x_star_band : array of shape (d,) in the active backend's native type.
        Weighted superposition of resolvent-propagated PCs.

    Notes
    -----
    The caller is responsible for constructing projected_weights to reflect
    the residual amplitude in each PC direction.  A natural choice is

        projected_weights[i] = (delta_T - mean) · components[i]

    which is the projection of the centered residual onto the i-th PC axis.
    The resolvent then amplifies each component by (I − DT)^{-1}, and the
    weighted sum reconstructs the band on δx*.
    """
    bk = backend if backend is not None else get_backend()

    # Normalise projected_weights to numpy for scalar indexing.
    try:
        pw_np = bk.to_numpy(projected_weights).astype(np.float64)
    except Exception:
        pw_np = np.asarray(projected_weights, dtype=np.float64)

    r = basis.rank
    try:
        comp_np = bk.to_numpy(basis.components).astype(np.float64)  # (r, d)
    except Exception:
        comp_np = np.asarray(basis.components, dtype=np.float64)

    d = comp_np.shape[1]

    result_np = np.zeros(d, dtype=np.float64)
    for i in range(r):
        u_i = bk.from_numpy(comp_np[i])        # (d,): the i-th principal component
        w_i_raw = apply_resolvent(u_i)           # (d,): resolvent-propagated PC
        try:
            w_i_np = bk.to_numpy(w_i_raw).astype(np.float64)
        except Exception:
            w_i_np = np.asarray(w_i_raw, dtype=np.float64)
        result_np = result_np + float(pw_np[i]) * w_i_np

    return bk.from_numpy(result_np)
