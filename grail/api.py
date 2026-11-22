"""GRAIL public entry point: resolvent_solve.  the technical reference §5.2.

================================================================================
GRAIL Resolvent Equation — Derivation (the technical reference §5.1; reference guide §1.3)
================================================================================

Setup.  The coupled RT+SE cycle is T : R^{2N_tau} -> R^{2N_tau}, where the
state vector x = (n_1, n_2) stacked over depth is laid out in flat blocked
form (2*N_tau,) at the GRAIL seam.  Conversion: populations (N_tau, 2) →
flat via populations.T.flatten(); inverse: flat.reshape(2, N_tau).T.

    T = T_SE ∘ T_RT   (RT step first, then SE step)

The truth fixed point satisfies x_t* = T_t(x_t*).  The emulator cycle is
T_e = T_t + δT, where δT is the per-cycle emulator residual.

The emulator fixed point x_e* satisfies:

    x_e* = T_e(x_e*) = T_t(x_e*) + δT(x_e*)               [perturbed FP]

Subtracting x_t* = T_t(x_t*):

    δx* = T_t(x_e*) - T_t(x_t*) + δT(x_e*)                [exact]

Taylor-expanding T_t about x_t*:

    T_t(x_e*) = T_t(x_t*) + DT_t · δx* + R_2(δx*)

gives the exact equation

    (I - DT_t) δx* = δT(x_e*) + R_2(δx*)                  [exact resolvent]

Because I - DT_t is invertible (spectral radius of DT_t < 1 by ALI
convergence), and R_2 = O(||δT||^2) while δT(x_e*) = δT(x_t*) + O(||δT||^2),
the leading-order result is:

    δx* = (I - DT)^{-1} δT(x*) + O(||δT||^2)              [eq:resolvent]

where DT ≈ DT_t ≈ DT_e to O(||δT||) and x* is evaluated at the emulator
fixed point x_e* (the available approximation).

Practical evaluation.  GRAIL:
  1. Receives x_star ≡ x_e* and delta_T ≡ δT(x_e*) from the caller.
  2. Builds the matvec v ↦ (I - DT) v via the chain rule:
         DT · v = D(T_SE) · (D(T_RT) · v)
     using the emulator_jvp and host_jvp callbacks respectively.
  3. Solves (I - DT) δx* = δT via matrix-free GMRES.
  4. Returns δx* and full diagnostics (the technical reference §5.3).

Connection to ALI (§1.4 of reference guide).  The operator (I - DT)^{-1} is the
discrete Green's function of the scattering problem, analogous to the ALI
resolvent Λ^{-1} (I - (1-ε)Λ)^{-1}.  GMRES with the DT matvec is thus
equivalent to a Krylov-accelerated Lambda iteration on the emulator-residual
right-hand side.

References
----------
- Rybicki, G. B. & Hummer, D. G. (1991), A&A 245, 171.  ALI Green's function.
- Hubeny, I. (2003), in Stellar Atmosphere Modeling, eds. Hubeny et al., ASPC
  288, 17.  ALI preconditioning.
- Bai, S. et al. (2019), NeurIPS 32.  DEQ implicit differentiation.
- Saad, Y. & Schultz, M. H. (1986), SIAM J. Sci. Stat. Comput. 7, 856.  GMRES.

================================================================================
Numpy-Callback Boundary Contract                              SPEC_jax.md §4.3
================================================================================

GRAIL's public callbacks (emulator_jvp, host_jvp, and the PCA
apply_resolvent closure) **may return any of numpy.ndarray,
torch.Tensor, or jax.Array** of shape (N_state,) with float64-equivalent
dtype.  GRAIL converts each callback output to the active backend's tensor
type via that backend's from_numpy adapter (going through numpy for cross-
library transfers).  The conversion is a no-op when the callback already
returns the active backend's native type.

    ResolventResult.delta_x_star is the active backend's native type.
    Callers should np.asarray(result.delta_x_star) to obtain a numpy array
    in a backend-independent way.

Hosts written in Fortran (bridged via f2py), C++ (via pybind11), or any
other language that emits numpy interop work *identically* on either backend.
The host-agnostic claim is preserved by this numpy bridge, not by GRAIL
internals being numpy-typed.

    Slot-naming note (SPEC_jax.md §3.3 / the callback contract "Slot-naming wart").
    The parameter names emulator_jvp / host_jvp reflect the NLTE chain-rule
    convention where the emulator lives in the *second*-applied JVP (SE step
    after RT).  The SCF testbed have the NN in the *first*-applied
    slot (opposite convention).  The parameter names are not renamed in this
    spec; a future grail/ v0.2 sprint will rename to first_jvp / second_jvp.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .backend import Backend, get_backend
from .gmres import gmres
from .jacobian import resolvent_matvec


@dataclass
class ResolventResult:
    """Output of resolvent_solve.  Diagnostics per the technical reference §5.3.

    Attributes
    ----------
    delta_x_star : array (native backend type)
        The propagated displacement for the supplied signed residual.
        For emulator-minus-reference residuals, subtract it from x*_emul
        to estimate the reference state.  Shape (2*N_tau,) flat
        blocked: flat.reshape(2, N_tau).T gives (N_tau, 2) populations.
        Type is torch.Tensor when backend="torch", jax.Array when backend="jax".
        Use np.asarray(result.delta_x_star) for a backend-independent copy.
    iter_count : int
        GMRES iterations consumed.
    residual_history : list[float]
        GMRES relative residual norm ‖r_k‖/‖b‖ at each iteration.
    condition_estimate : float
        Condition-number estimate from the Arnoldi Hessenberg.
    pca_rank : int
        Rank of the PCA truncation used in residual decomposition.
        0 if PCA propagation was not run (default resolvent-only mode).
    conformal_intervals : optional array
        (low, high) calibrated bands in observable space; None if not requested.
    """

    delta_x_star: object    # native backend type
    iter_count: int
    residual_history: list[float]
    condition_estimate: float
    pca_rank: int
    conformal_intervals: object = None  # Optional native backend array


def resolvent_solve(
    x_star,
    delta_T,
    emulator_jvp: Callable,
    host_jvp: Callable,
    *,
    gmres_tol: float = 1e-8,
    gmres_maxiter: int = 200,
    gmres_restart: int = 30,
    preconditioner: Optional[Callable] = None,
    backend: Optional[str] = None,
) -> ResolventResult:
    """Solve (I - DT) δx* = δT via matrix-free GMRES.

    Parameters
    ----------
    x_star : array of shape (2*N_tau,), float64
        Converged emulator fixed point x*_emul.  Flat blocked layout:
        populations (N_tau, 2) → x_star via populations.T.flatten().
        Linearization is taken at this point.  May be numpy, torch, or jax.
    delta_T : array of shape (2*N_tau,), float64
        Emulator residual δT(x_star) = T^APNN(x_star) - T^ref(x_star).
        Must be pre-evaluated at x_star before calling resolvent_solve.
        May be numpy, torch, or jax.
    emulator_jvp : callable v -> D(T_SE) @ v
        JVP of the SE block (the emulator).  Opaque callback; may be
        PyTorch autograd, NumPy finite difference, or a Fortran adjoint.
        Accepts and returns vectors of shape (2*N_tau,).  See boundary
        contract in module docstring.
    host_jvp : callable v -> D(T_RT) @ v
        JVP of the RT block (the formal solver + line-profile closure).
        Treated as opaque; host-agnosticism is gated by I-callback-isolation.
        Accepts and returns vectors of shape (2*N_tau,).
    gmres_tol : float, default 1e-8
        GMRES convergence tolerance on the relative residual ‖r‖/‖b‖.
    gmres_maxiter : int, default 200
        Maximum GMRES iterations.
    gmres_restart : int, default 30
        Krylov-subspace dimension at which restarted GMRES restarts.
        Raise this when the cycle Jacobian has many near-marginal-modulus
        eigenvalues clustered near unity (the documented "restart-30
        stall" regime); a Krylov subspace smaller than the dimension of
        that cluster fills before deflation and convergence stagnates.
        Typical bumps: 60 or 90.
    preconditioner : callable v -> M^{-1} v, optional
        Right preconditioner for GMRES.  None = identity.
    backend : str or None, default None
        Internal array library.  One of "torch" (default) or "jax".
        If None, uses the module-level default set by set_default_backend().
        Existing callers omitting this kwarg continue to use "torch"
        with zero behaviour change.  See SPEC_jax.md §4.1.

    Returns
    -------
    ResolventResult with delta_x_star and full diagnostics.
    delta_x_star is the active backend's native type (torch.Tensor for
    "torch", jax.Array for "jax").  Use np.asarray() for a portable copy.

    Notes
    -----
    State-vector seam: GRAIL's state vector is (2*N_tau,) flat blocked,
    where the first N_tau entries are n_1(τ) and the next N_tau are n_2(τ).
    Convention:
        populations (N_tau, 2) → flat: populations.T.flatten()
        flat (2*N_tau,) → populations: flat.reshape(2, N_tau).T
    Both emulator_jvp and host_jvp must operate in this same flat-blocked
    coordinate system.
    """
    bk: Backend = get_backend(backend)

    # Normalize inputs to the active backend type via numpy.
    def _to_bk(v) -> object:
        try:
            arr = bk.to_numpy(v)
        except Exception:
            arr = np.asarray(v, dtype=np.float64)
        return bk.from_numpy(np.asarray(arr, dtype=np.float64))

    x_star_bk = _to_bk(x_star)
    delta_T_bk = _to_bk(delta_T)

    # Build the (I - DT) matrix-vector product from the two callbacks.
    # This is the only operation GRAIL performs on the callbacks: it composes
    # them via the chain rule DT·v = D(T_SE)·(D(T_RT)·v) and subtracts from v.
    matvec = resolvent_matvec(emulator_jvp, host_jvp, backend=bk)

    # Solve (I - DT) δx* = δT via matrix-free GMRES.
    result = gmres(
        matvec,
        delta_T_bk,
        tol=gmres_tol,
        maxiter=gmres_maxiter,
        restart=gmres_restart,
        preconditioner=preconditioner,
        backend=bk,
    )

    return ResolventResult(
        delta_x_star=result.x,
        iter_count=result.iter_count,
        residual_history=result.residual_history,
        condition_estimate=result.condition_estimate,
        pca_rank=0,          # PCA propagation not run in the direct API path
        conformal_intervals=None,
    )
