'Neural population-cycle derivative utilities. Differentiate the actual adapter, preprocessing and forward model, and check equation defects independently. The positive-gap and rate-relaxed models support automatic differentiation. Finite-difference utilities differentiate their declared forward map in the same state coordinates.'

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..apnn import APNN

# Minimum positive value for safe log10 conversion.
_LOG_CLAMP_MIN = 1e-300


def make_cycle_adapter_5level(
    model: APNN,
    atomic_model: "object | None" = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a callable bridging the 5-level APNN to the SE-solver contract.

    The SE-solver contract:
        se_solver(J_bar_lines: Tensor[N_tau, K=10],
                  T:           Tensor[N_tau],
                  n_e:         Tensor[N_tau])
            -> Tensor[N_tau, L=5]

    This adapter:
        1. Applies log10 scaling to each of the K J_bar line columns, T, n_e.
           Clamps to _LOG_CLAMP_MIN before log to avoid log(0).
        2. Stacks into (N_tau, K+2) float32 for the APNN.
        3. Runs model.forward() -> (N_tau, L) float64 under no_grad.

    Parameters
    ----------
    model : APNN
        Trained APNN.  Must satisfy model.n_in >= 3 (K+2 == model.n_in).
        For the canonical 5-level APNN: model.n_in == 12, model.n_out == 5.
    atomic_model : optional
        Accepted for API completeness (mirrors 3-level adapter); unused at
        call time.  If provided and has a `.K` attribute, validates
        model.n_in == K + 2.

    Returns
    -------
    adapter : Callable
        Satisfies the SE-solver contract above.

    Raises
    ------
    ValueError
        If model.n_in / model.n_out mismatch when atomic_model is provided.
    """
    # Versioned positive-opacity models have a shared epsilon-conditioned
    # encoder rather than the legacy raw log10(J,T,ne) representation.
    if getattr(model, "schema", None) in {"positive-opacity-se-v2", "rate-relaxed-positive-se-v3"}:
        if atomic_model is None:
            raise ValueError("v2 adapter requires atomic_model for epsilon conditioning")
        from .positive_apnn import make_positive_cycle_adapter
        return make_positive_cycle_adapter(model, atomic_model)
    # Validate against atomic_model if provided.
    if atomic_model is not None:
        K = getattr(atomic_model, "K", None)
        L = getattr(atomic_model, "n_levels", None)
        if K is not None and model.n_in != K + 2:
            raise ValueError(
                f"make_cycle_adapter_5level: model.n_in={model.n_in!r} but "
                f"atomic_model.K={K} requires model.n_in == {K + 2}."
            )
        if L is not None and model.n_out != L:
            raise ValueError(
                f"make_cycle_adapter_5level: model.n_out={model.n_out!r} but "
                f"atomic_model.n_levels={L} requires model.n_out == {L}."
            )

    K_model = model.n_in - 2   # number of J_bar lines

    def adapter(
        J_bar_lines: torch.Tensor,   # (N_tau, K), float64
        T: torch.Tensor,             # (N_tau,),   float64
        n_e: torch.Tensor,           # (N_tau,),   float64
    ) -> torch.Tensor:               # (N_tau, L), float64
        """Log10-scale (J_bar_lines, T, n_e) and run APNN forward.

        Parameters
        ----------
        J_bar_lines : Tensor of shape (N_tau, K), float64
            Profile-weighted mean intensities, one column per transition.
        T     : Tensor of shape (N_tau,), float64
        n_e   : Tensor of shape (N_tau,), float64

        Returns
        -------
        populations : Tensor of shape (N_tau, L), float64
        """
        # Safe log10 for each J_bar column.
        log_J_cols = [
            torch.log10(J_bar_lines[:, k].clamp(min=_LOG_CLAMP_MIN))
            for k in range(K_model)
        ]
        log_T  = torch.log10(T.clamp(min=_LOG_CLAMP_MIN))
        log_ne = torch.log10(n_e.clamp(min=_LOG_CLAMP_MIN))

        # Stack into (N_tau, K+2) float32 for the APNN.
        x = torch.stack(log_J_cols + [log_T, log_ne], dim=-1)  # (N_tau, K+2)

        with torch.no_grad():
            return model(x)  # (N_tau, L), float64

    return adapter


def _apnn_fd_jvp_multilevel(
    adapter: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    J_bar_lines_base: torch.Tensor,    # (N_tau, K), float64
    T: torch.Tensor,                   # (N_tau,),   float64
    n_e: torch.Tensor,                 # (N_tau,),   float64
    jbar_lines_tangent: torch.Tensor,  # (N_tau, K), float64
    h: float = 1e-3,
) -> torch.Tensor:                     # (N_tau, L), float64
    """Central-difference FD-JVP of the APNN cycle adapter along a J_bar tangent.

    Computes the directional derivative:
        d/dt adapter(J_bar + t * v, T, n_e) |_{t=0}  =  ∂APNN/∂J̄ · v.

    Implementation: central differences in LOG-J space, which is the APNN's
    native input (the adapter applies log10 to each J̄ column).  At
    deep-stellar operating points where J̄_truth has magnitudes 10⁻¹⁵–10⁻⁷
    across the ten Balmer transitions, an absolute-step FD
    (``J_fwd = J + h·v`` with ``h = 10⁻⁴``) shifts log10(J̄) by ~5–7
    orders of magnitude and pushes the APNN evaluation outside its
    training distribution.  The resulting "derivative" is OOD noise:
    the M.5 Born-remainder integration test caught this with measured
    ρ(K_emul) ≈ 3582 (vs ρ(K_truth) ≈ 1.0001) via Arnoldi on the composed
    host∘emul JVP — physically impossible for a converging cycle and
    a pure FD-step artifact.

    Log-space chain rule.  Set ``y = log10(J̄)``; the APNN evaluates
    ``f(y, T, n_e)``.  The directional derivative in J̄ along a tangent v is

        ∂(f∘log10)/∂J̄ · v = (∂f/∂y) · v / (J̄ · ln10),

    so picking the log-tangent ``v_log = v / (J̄ · ln10)`` and central-
    differencing in y by ±h gives exactly ``∂APNN/∂J̄ · v``.  In J space
    this is the multiplicative perturbation
    ``J_fwd = J̄ · 10^(h·v_log)``, which is bounded relative to J̄.

    This satisfies TASK-035 architectural rule: the emulator's own
    Jacobian is used (not the truth-SE autograd Jacobian).

    Cost: 2 × adapter forward passes, independent of K and L.

    Parameters
    ----------
    adapter : Callable
        The adapter returned by make_cycle_adapter_5level.
    J_bar_lines_base : Tensor of shape (N_tau, K), float64
        J_bar at the operating point.
    T : Tensor of shape (N_tau,), float64
    n_e : Tensor of shape (N_tau,), float64
    jbar_lines_tangent : Tensor of shape (N_tau, K), float64
        Tangent direction in J_bar space.
    h : float, default 1e-3
        FD step size in log10(J̄) space.  Replaces the prior absolute-step
        h=1e-4 default; the log-space step is regime-invariant
        (independent of |J̄| at the operating point).

    Returns
    -------
    jvp_val : Tensor of shape (N_tau, L), float64
        Directional derivative of the adapter output in the J̄ tangent direction.
    """
    # Adaptive step in J space.  Constrain the relative perturbation
    # |α · v_ik / J_ik|  ≤  h  componentwise; the actual step taken is
    # α = h / max_ik (|v_ik| / J_ik).  This keeps the FD evaluation inside
    # the APNN's log-J training distribution at every depth and line.
    eps = 1e-300
    J_base = J_bar_lines_base.clamp(min=eps)
    rel = jbar_lines_tangent.abs() / J_base
    rel_max = rel.max().clamp(min=eps).item()
    alpha = h / rel_max
    # Floor at float-noise safety; if ||v|| ≈ 0 (rel_max tiny), the JVP is
    # 0 to leading order and the step floor avoids underflow rather than
    # affecting the answer.
    alpha = max(alpha, 1e-30)

    J_fwd = J_base + alpha * jbar_lines_tangent
    J_bwd = J_base - alpha * jbar_lines_tangent

    f_fwd = adapter(J_fwd, T, n_e)   # (N_tau, L), float64
    f_bwd = adapter(J_bwd, T, n_e)   # (N_tau, L), float64

    return (f_fwd - f_bwd) / (2.0 * alpha)


# ---------------------------------------------------------------------------
# M.5 GRAIL callback factories: make_emul_jvp_5level, make_host_jvp_5level
# ---------------------------------------------------------------------------

def make_emul_jvp_5level(
    adapter: Callable,
    J_bar_lines_truth: torch.Tensor,    # (N_tau, K), float64 — base point
    T: torch.Tensor,                    # (N_tau,), float64
    n_e: torch.Tensor,                  # (N_tau,), float64
    n_tau: int,
    n_lines: int,                       # K — number of J_bar lines
    n_levels: int,                      # L — number of population levels
    h: float = 1e-4,
) -> Callable:
    """Build the emul_jvp callback for use by grail.resolvent_solve (M.5).

    The emulator JVP maps a J_bar tangent vector (K*N_tau,) -> populations
    tangent vector (L*N_tau,) via central-difference FD-JVP through the APNN
    adapter.  This satisfies the TASK-035 architectural rule: the Jacobian
    is the emulator's own Jacobian, not the truth SE Jacobian.

    Intermediate-space note (5-level K > L)
    ----------------------------------------
    At 5-level L=5, K=10.  GRAIL's composed_jvp passes the output of host_jvp
    directly as input to emul_jvp.  host_jvp returns (K*N_tau,); this function
    accepts (K*N_tau,) and returns (L*N_tau,), so the composed result has the
    correct state-vector size for GMRES.

    Parameters
    ----------
    adapter : Callable
        (J_bar_lines: Tensor[N_tau, K], T: Tensor[N_tau], n_e: Tensor[N_tau])
            -> Tensor[N_tau, L].
        Returned by make_cycle_adapter_5level.
    J_bar_lines_truth : Tensor of shape (N_tau, K), float64
        J_bar at the truth fixed point x*_truth.  Base point for the FD-JVP.
    T : Tensor of shape (N_tau,), float64
        Temperature (held fixed at the base point).
    n_e : Tensor of shape (N_tau,), float64
        Electron density (held fixed at the base point).
    n_tau : int
        Number of depth points N_tau.
    n_lines : int
        Number of transitions K.
    n_levels : int
        Number of population levels L.
    h : float, default 1e-4
        Central-difference step size.

    Returns
    -------
    emul_jvp : Callable
        Accepts a J_bar tangent (K*N_tau,) tensor (or numpy array) and
        returns a populations tangent (L*N_tau,) torch.Tensor float64.
        The layout is flat-blocked: first N_tau entries are line 0 tangent,
        next N_tau entries are line 1 tangent, etc.
    """
    # Freeze base-point tensors to avoid accidental in-place mutation.
    J_base = J_bar_lines_truth.detach().clone()
    T_f64 = T.to(torch.float64)
    n_e_f64 = n_e.to(torch.float64)

    def emul_jvp(w: torch.Tensor) -> torch.Tensor:
        """D(T_SE_emul) · w: J_bar tangent (K*N_tau,) -> populations tangent (L*N_tau,).

        Receives w of shape (K*N_tau,) — flat-blocked J_bar tangent from host_jvp.
        Applies the APNN central-difference FD-JVP and returns populations tangent.
        """
        w = torch.as_tensor(w, dtype=torch.float64).reshape(-1)
        # Unpack (K*N_tau,) flat-blocked -> (N_tau, K).
        jbar_tangent = w.reshape(n_lines, n_tau).T.clone()   # (N_tau, K)

        # Central-difference FD-JVP through the APNN (TASK-035 rule).
        pops_tangent = _apnn_fd_jvp_multilevel(
            adapter, J_base, T_f64, n_e_f64, jbar_tangent, h=h,
        )   # (N_tau, L)

        # Pack to (L*N_tau,) flat-blocked.
        return pops_tangent.T.reshape(-1)   # (L*N_tau,), float64

    return emul_jvp


def make_host_jvp_5level(
    atmosphere: "object",
    x_truth: torch.Tensor,             # (N_tau, L), float64 — base point
    n_tau: int,
    n_lines: int,                       # K
    n_levels: int,                      # L
) -> Callable:
    """Build the host_jvp callback for use by grail.resolvent_solve (M.5).

    The host JVP maps a populations tangent (L*N_tau,) -> J_bar tangent
    (K*N_tau,).  This is the D(T_RT) · v directional derivative: it linearizes
    the K formal solves and line-profile averages at x*_truth.

    Intermediate-space note (5-level K > L)
    ----------------------------------------
    At 5-level L=5, K=10.  This callback returns (K*N_tau,), not (L*N_tau,).
    emul_jvp accepts (K*N_tau,) so the chain is self-consistent.  The resolvent
    matvec v - emul_jvp(host_jvp(v)) has v: (L*N_tau,) and out: (L*N_tau,).

    Implementation — precomputed-Λ (fast)
    --------------------------------------
    The chain rule is: dJ_bar_k = Λ_k · dS_k  where:
      - Λ_k is the (N_tau, N_tau) Λ matrix for line k, precomputed once.
      - dS_k is the tangent of the source function S_k wrt populations,
        computed cheaply via torch.func.jvp through only
        _source_from_populations_nlevel (no formal solves at JVP call time).

    This reduces each host_jvp call from O(K × formal_solve) to O(N_tau^2)
    (K matrix-vector products), making GMRES practical at N_tau=80, K=10.

    The Λ matrices are precomputed at factory-build time (one-time cost of
    K × _compute_lambda_full, each O(N_tau^2 × N_nu × N_mu)).

    Parameters
    ----------
    atmosphere : Atmosphere
        Must have atomic_model with K transitions and frequency grid.
    x_truth : Tensor of shape (N_tau, L), float64
        Converged truth populations.  Base point for the JVP.
    n_tau : int
    n_lines : int
        K — number of transitions.
    n_levels : int
        L — number of population levels.

    Returns
    -------
    host_jvp : Callable
        Accepts a populations tangent (L*N_tau,) and returns a J_bar tangent
        (K*N_tau,), both float64.  Flat-blocked layout in both cases.
    """
    from atmosphere.multilevel.fixed_point import (
        _source_from_populations_nlevel,
        _compute_lambda_full_nlevel,
    )

    K = n_lines
    L = n_levels
    atomic_model = atmosphere.atomic_model

    # Freeze base-point tensor.
    x_base = x_truth.detach().clone().to(torch.float64)

    # Pre-compute the K Λ matrices once at factory-build time.
    # Each Lambda_k is (N_tau, N_tau); we only need Λ, not the boundary vector b.
    lambda_results = _compute_lambda_full_nlevel(atmosphere)
    # Extract only the Λ matrices; b_k is not needed for the JVP.
    Lambda_list = [lr[0] for lr in lambda_results]   # list of K (N_tau, N_tau) tensors

    # Define a stripped source-function function for JVP tracing.
    # _source_from_populations_nlevel is cheap (no formal solves).
    def _S_from_pops(pops: torch.Tensor) -> torch.Tensor:
        return _source_from_populations_nlevel(pops, atomic_model)   # (N_tau, K)

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        """D(T_RT) · v: populations tangent (L*N_tau,) -> J_bar tangent (K*N_tau,).

        Uses the precomputed Λ matrices:  dJ_bar_k = Λ_k · dS_k,
        where dS_k is computed via JVP through _source_from_populations_nlevel.
        """
        v = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
        # Unpack (L*N_tau,) flat-blocked -> (N_tau, L).
        v_pops = v.reshape(L, n_tau).T.clone()   # (N_tau, L)

        # Compute tangent of S_lines wrt populations via JVP.
        # This is cheap: _source_from_populations_nlevel has no formal solves.
        _, dS_lines = torch.func.jvp(
            _S_from_pops,
            (x_base,),
            (v_pops,),
        )   # dS_lines: (N_tau, K)

        # Apply Λ_k to dS_k for each line: dJ_bar_k = Λ_k @ dS_k.
        dJ_bar_list = []
        for k in range(K):
            dS_k = dS_lines[:, k]                   # (N_tau,)
            dJ_bar_k = Lambda_list[k] @ dS_k        # (N_tau,)
            dJ_bar_list.append(dJ_bar_k)

        dJ_bar = torch.stack(dJ_bar_list, dim=1)    # (N_tau, K)

        # Pack to (K*N_tau,) flat-blocked.
        return dJ_bar.T.reshape(-1)   # (K*N_tau,), float64

    return host_jvp
