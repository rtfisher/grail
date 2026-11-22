'Electronic-structure cycle derivatives. The response differentiates the actual neural Fock contribution and the density reconstruction at consistent intermediate states. Automatic differentiation supplies the default neural derivative; finite differences provide an independent cross-check. Reference-functional derivatives may serve verification targets or preconditioners, not substitutes for the surrogate operator.'

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .emulator.network import XCNet
from .fock_build import _build_mf, _build_mol
from .state import (
    MoleculeState,
    n_unique,
    pack_density_matrix,
    unpack_density_matrix,
)


# ---------------------------------------------------------------------------
# Constants — matching SPEC_scf §7.
# ---------------------------------------------------------------------------

_FD_STEP = 1e-4   # FD-JVP step in flat-P space (SPEC_scf §7, TASK-035 precedent).


# ---------------------------------------------------------------------------
# Fock-builder with NN-substituted V_xc.
# ---------------------------------------------------------------------------

def make_fock_builder_with_nn(
    xc_net: XCNet,
    state: MoleculeState,
) -> Callable[[MoleculeState, tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """Return a Fock builder that substitutes ``XCNet``-predicted ``V_xc``.

    The returned closure matches the ``(state, (P_α, P_β)) → (F_α, F_β)``
    signature of :func:`scf.fock_build.build_fock_truth` for the UHF/UKS
    branch — it slots directly into the ``fock_builder=`` parameter of
    :func:`scf.host.picard_iterate_scf`.  For each call:

      1. Symmetrize the input per-spin densities defensively.
      2. Pack ``(P_α, P_β)`` to the doubled-seam flat with
         :func:`scf.state.pack_density_matrix`.
      3. Forward the packed flat through the trained ``xc_net`` to obtain
         the doubled-seam ``V_xc`` prediction; unpack back to a per-spin
         ``(V_xc^α, V_xc^β)`` matrix pair (symmetry mirrored within each
         spin block by :func:`scf.state.unpack_density_matrix`).
      4. Build the shared Hartree-Coulomb piece via PySCF's
         ``mf.get_j(P_α + P_β)`` (cheap, analytic, exact).
      5. Return ``(F_α, F_β) = (h_core + J + V_xc^α_NN,
         h_core + J + V_xc^β_NN)``.

    Per R3 hand-off (c): the Hartree-Coulomb piece is shared across both
    spins and computed analytically — only the spin-resolved XC piece
    runs through the network.  This mirrors the training-data convention
    in :func:`scf.emulator.training.generate_training_data_uhf` where
    the network's targets are ``V_eff − J`` per spin.

    Parameters
    ----------
    xc_net : XCNet
        Trained UHF network with ``n_unique`` matching the doubled-seam
        length for ``state.basis`` (6 for STO-3G H₂, 20 for 6-31G H₂).
        Loaded via :func:`scf.emulator.load_xc_emulator`.
    state : MoleculeState
        Must have ``method ∈ {"uhf", "uks"}``.  Used to instantiate the
        PySCF Mole / mean-field once at closure-creation time so the
        per-iteration Fock builds amortize the Mole construction cost.

    Returns
    -------
    fock_builder : callable
        Closure with signature
        ``(state, (P_α, P_β)) → (F_α, F_β)`` returning a 2-tuple of
        symmetric ``(M, M)`` float64 ndarrays.

    Notes
    -----
    - The returned closure ignores the ``state`` argument passed at call
      time (it captures the closure-creation-time ``state`` instead).
      ``picard_iterate_scf`` always passes the same state through the
      Fock-builder so this is a no-op, but the API surface matches the
      truth Fock builder for drop-in substitution.
    - ``XCNet.forward`` casts internally to float32 and back to float64
      on output; no caller-side dtype management needed.
    - Symmetrization on the output is defensive: each spin's V_xc^NN
      is mirrored by the unpack helper, and J is symmetric by
      construction, but PySCF's get_j can leak ~1e-14 antisymmetric
      noise so we re-symmetrize before returning F.
    """
    if state.method not in ("uhf", "uks"):
        raise ValueError(
            f"make_fock_builder_with_nn requires state.method in "
            f"{{'uhf', 'uks'}}; got method={state.method!r}.  The "
            "closed-shell RHF/RKS path is obsolete per SPEC_scf §8.10."
        )
    # Build the PySCF Mole and mean-field once at closure-creation time.
    # The Mole and mf objects are stateless w.r.t. density (we always
    # pass the current P explicitly to get_j), so a single instance can
    # service every iteration of picard_iterate_scf.
    mol_cached = _build_mol(state)
    mf_cached = _build_mf(state, mol=mol_cached)
    h_core_cached = np.asarray(mf_cached.get_hcore(), dtype=np.float64)
    M_cached = int(mol_cached.nao_nr())

    # Sanity check: network seam matches the basis seam.
    expected_n_unique = 2 * n_unique(M_cached)
    if xc_net.n_unique != expected_n_unique:
        raise ValueError(
            f"xc_net.n_unique={xc_net.n_unique} does not match the "
            f"doubled-seam length for state.basis={state.basis!r} "
            f"(expected {expected_n_unique} for M={M_cached})."
        )

    # Ensure the network is in eval mode (we are not training here).
    xc_net.eval()

    def _fock_builder_nn(
        state_arg: MoleculeState,
        P_pair: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        # The state_arg is provided by picard_iterate_scf for API symmetry
        # with build_fock_truth; we use the closure-captured state.
        del state_arg  # explicit no-op so static analyzers don't flag it.
        if not (isinstance(P_pair, tuple) and len(P_pair) == 2):
            raise ValueError(
                f"NN Fock builder requires a 2-tuple (P_α, P_β); got input "
                f"of type {type(P_pair).__name__}."
            )
        P_alpha = np.asarray(P_pair[0], dtype=np.float64)
        P_beta = np.asarray(P_pair[1], dtype=np.float64)
        # Defensive symmetrize each spin block (mirrors build_fock_truth).
        P_alpha_sym = 0.5 * (P_alpha + P_alpha.T)
        P_beta_sym = 0.5 * (P_beta + P_beta.T)

        # Pack to the doubled-seam flat for the network input.
        P_flat = pack_density_matrix(
            (torch.from_numpy(P_alpha_sym), torch.from_numpy(P_beta_sym)),
            method=state.method,
        )                                                # (2*n_unique,), float64

        # NN forward pass: doubled-flat P → doubled-flat V_xc.
        with torch.no_grad():
            V_xc_flat = xc_net(P_flat.unsqueeze(0)).squeeze(0)   # (2*n_unique,)

        # Unpack to per-spin (V_xc^α, V_xc^β) — symmetrized within each block.
        V_alpha_t, V_beta_t = unpack_density_matrix(
            V_xc_flat, M=M_cached, method=state.method,
        )
        V_xc_alpha = V_alpha_t.numpy()
        V_xc_beta = V_beta_t.numpy()

        # Shared Hartree-Coulomb piece: analytic via PySCF.
        P_total = P_alpha_sym + P_beta_sym
        J = np.asarray(mf_cached.get_j(mol_cached, dm=P_total), dtype=np.float64)

        # Assemble per-spin Fock and re-symmetrize defensively.
        F_alpha = h_core_cached + J + V_xc_alpha
        F_beta = h_core_cached + J + V_xc_beta
        F_alpha = 0.5 * (F_alpha + F_alpha.T)
        F_beta = 0.5 * (F_beta + F_beta.T)
        return F_alpha, F_beta

    return _fock_builder_nn


# ---------------------------------------------------------------------------
# host_jvp: Fock-build differential D𝐅|_{P*}.  Contains the NN — FD-JVP.
# ---------------------------------------------------------------------------
#
# This slot is applied FIRST in GRAIL's composed_jvp(emulator_jvp, host_jvp).
# Per the slot-naming inversion (SPEC §8.1 / module docstring above), this
# is the Fock-build differential, which evaluates the trained XCNet.  The
# FD-JVP rule applies here.
# ---------------------------------------------------------------------------

def make_host_jvp_scf(
    state: MoleculeState,
    P_star_tuple: tuple[np.ndarray, np.ndarray],
    fock_builder: Callable[
        [MoleculeState, tuple[np.ndarray, np.ndarray]],
        tuple[np.ndarray, np.ndarray],
    ],
    *,
    h: float = _FD_STEP,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Central-FD JVP of the Fock-build block ``D𝐅|_{P*}``.

    **Slot-naming inversion (SPEC §8.1).**  This corresponds to GRAIL's
    ``host_jvp`` slot — the first-applied differential in the chain rule
    ``D𝒯 · v = D𝐃 · (D𝐅 · v)``.  Per SPEC §8.1 this is also the slot
    that contains the NN-XC emulator, so the FD-JVP requirement
    (the callback contract hard rule + SPEC_scf §5) bites here.

    The JVP is computed by materializing the dense
    ``(2*n_unique, 2*n_unique)`` Jacobian via per-column central
    differences at the truth-converged ``(P_α^*, P_β^*)``::

        column_k = (𝐅(P^* + h E_k) − 𝐅(P^* − h E_k)) / (2 h)

    where ``E_k`` is the symmetric unit perturbation in the k-th
    packed-flat slot of the doubled UHF seam.  The full Jacobian is
    small (6 × 6 for STO-3G H₂, 20 × 20 for 6-31G), so materializing it
    once and reusing across GMRES iterations is cheaper than re-running
    FD per matvec.

    Parameters
    ----------
    state : MoleculeState
        UHF / UKS state; carries basis and method.  Used to set the
        unpacking ``M`` and the doubled-seam length.
    P_star_tuple : tuple ``(P_α^*, P_β^*)``
        Truth-converged density-matrix pair, each ``(M, M)`` ndarray.
        Linearization is taken at this point.
    fock_builder : callable
        The NN-substituted Fock builder
        (:func:`make_fock_builder_with_nn`'s output).  This is the slot
        whose evaluation queries the NN, so the FD-JVP is taken through
        this callable, NOT through the truth builder.
    h : float, default 1e-4
        Central-difference step in packed-flat-P space (SPEC §7).

    Returns
    -------
    host_jvp : callable
        ``host_jvp(v: Tensor) → Tensor`` mapping a doubled-flat
        perturbation ``v ∈ R^{2 n_unique}`` to the doubled-flat
        Fock-build differential ``D𝐅 · v ∈ R^{2 n_unique}``.  Float64.

    Notes
    -----
    - The unpack helper symmetrizes within each spin block, so the
      perturbed densities ``P^* ± h E_k`` are guaranteed symmetric per
      spin when re-built from the flat representation.  Pre-symmetrize
      per R2 advisory #3.
    - Wall-clock: each FD column costs 2 Fock builds → 2 NN forwards +
      2 PySCF ``get_j`` calls.  At STO-3G the doubled seam is length 6
      → 12 Fock builds total per ``make_host_jvp_scf`` call.  Sub-second
      on the conda env.
    """
    if state.method not in ("uhf", "uks"):
        raise ValueError(
            f"make_host_jvp_scf requires state.method in "
            f"{{'uhf', 'uks'}}; got method={state.method!r}."
        )
    if not (isinstance(P_star_tuple, tuple) and len(P_star_tuple) == 2):
        raise ValueError(
            f"P_star_tuple must be a 2-tuple (P_α^*, P_β^*); got type "
            f"{type(P_star_tuple).__name__}."
        )
    P_alpha_star = np.asarray(P_star_tuple[0], dtype=np.float64)
    P_beta_star = np.asarray(P_star_tuple[1], dtype=np.float64)
    if P_alpha_star.shape != P_beta_star.shape:
        raise ValueError(
            f"P_α^* and P_β^* must share shape; got "
            f"alpha={P_alpha_star.shape!r}, beta={P_beta_star.shape!r}."
        )
    if h <= 0:
        raise ValueError(f"h must be positive, got h={h!r}.")

    M = P_alpha_star.shape[0]
    nu_single = n_unique(M)
    nu_total = 2 * nu_single

    # Pack P^* to the doubled-flat seam once.
    P_star_flat = pack_density_matrix(
        (torch.from_numpy(P_alpha_star), torch.from_numpy(P_beta_star)),
        method=state.method,
    ).numpy()                                       # (2*n_unique,) float64

    # Materialize the dense (2 n_unique, 2 n_unique) D𝐅 Jacobian by
    # per-column central differences.  Columns of D𝐅 = pack(F) act on the
    # full Fock (NOT V_xc only) packed in the same doubled-seam layout —
    # the chain-rule composition v ↦ D𝐃 · (D𝐅 · v) on the density seam
    # treats v and D𝐅 · v as living in the SAME doubled-flat space (the
    # GRAIL boundary), so we pack the perturbed Fock the same way we pack
    # P.  Symmetry of F per spin is enforced by the Fock builder's
    # output-side symmetrize.
    D_F = np.zeros((nu_total, nu_total), dtype=np.float64)
    for k in range(nu_total):
        # E_k as a doubled-flat unit vector.
        e_flat = np.zeros(nu_total, dtype=np.float64)
        e_flat[k] = 1.0

        # P^* + h E_k → unpacked (P_α^+, P_β^+).
        flat_plus = P_star_flat + h * e_flat
        P_alpha_p_t, P_beta_p_t = unpack_density_matrix(
            torch.from_numpy(flat_plus), M=M, method=state.method,
        )
        # P^* − h E_k → unpacked (P_α^−, P_β^−).
        flat_minus = P_star_flat - h * e_flat
        P_alpha_m_t, P_beta_m_t = unpack_density_matrix(
            torch.from_numpy(flat_minus), M=M, method=state.method,
        )

        # Two Fock builds (NN evaluations).
        F_alpha_p, F_beta_p = fock_builder(
            state, (P_alpha_p_t.numpy(), P_beta_p_t.numpy()),
        )
        F_alpha_m, F_beta_m = fock_builder(
            state, (P_alpha_m_t.numpy(), P_beta_m_t.numpy()),
        )

        # Central difference per spin.
        dF_alpha = (F_alpha_p - F_alpha_m) / (2.0 * h)
        dF_beta = (F_beta_p - F_beta_m) / (2.0 * h)
        # Symmetrize per spin (defensive — should already be symmetric).
        dF_alpha = 0.5 * (dF_alpha + dF_alpha.T)
        dF_beta = 0.5 * (dF_beta + dF_beta.T)

        # Pack the per-spin pair to the doubled-flat column.
        D_F[:, k] = pack_density_matrix(
            (torch.from_numpy(dF_alpha), torch.from_numpy(dF_beta)),
            method=state.method,
        ).numpy()

    # Bind the dense Jacobian into the returned closure.  The closure is
    # the canonical "apply a precomputed dense Jacobian" matvec.
    D_F_tensor = torch.from_numpy(D_F)              # (2*n_unique, 2*n_unique)

    def _host_jvp(v: torch.Tensor) -> torch.Tensor:
        # Accept any input dtype; cast to float64 to match D_F's dtype.
        v_64 = v.to(dtype=torch.float64)
        if v_64.shape != (nu_total,):
            raise ValueError(
                f"host_jvp_scf: input shape must be ({nu_total},); got "
                f"shape={tuple(v_64.shape)!r}."
            )
        return D_F_tensor @ v_64

    return _host_jvp


# ---------------------------------------------------------------------------
# emulator_jvp: density-update differential D𝐃|_{F[P*]}.  No NN.
# ---------------------------------------------------------------------------
#
# This slot is applied SECOND in GRAIL's composed_jvp(emulator_jvp, host_jvp).
# Per the slot-naming inversion (SPEC §8.1 / module docstring above), this
# is the diag-and-rebuild differential — emulator-free.  The FD-JVP rule
# does NOT bite here (no NN evaluation); we use FD purely for numerical
# convenience (SPEC §6 P4 option 1).  The analytic χ_0 alternative lives
# in scf/jacobian.py if FD discretization noise becomes load-bearing.
# ---------------------------------------------------------------------------

def make_emulator_jvp_scf(
    state: MoleculeState,
    mo_energy_tuple: tuple[np.ndarray, np.ndarray] | None = None,
    mo_coeff_tuple: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    fock_tuple: tuple[np.ndarray, np.ndarray] | None = None,
    h: float = _FD_STEP,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Central-FD JVP of the diagonalize+rebuild block ``D𝐃|_{F[P*]}``.

    **Slot-naming inversion (SPEC §8.1).**  This corresponds to GRAIL's
    ``emulator_jvp`` slot — the second-applied differential in the chain
    rule ``D𝒯 · v = D𝐃 · (D𝐅 · v)``.  Despite the slot name, the NN-XC
    surrogate does NOT live here in the SCF testbed — this is the
    emulator-free density-update block.  See module docstring.

    The differential ``D𝐃[F] · v = δP_new(F + v) / δv`` is the standard
    CPHF / CPKS response of the closed-shell occupied-orbital projector
    to a Fock perturbation, generalized to UHF per-spin.  Per SPEC §6 P4
    option 1 we compute it by per-column central differences over the
    per-spin generalized-eigenproblem solve + density rebuild::

        column_k = (𝐃(F^* + h E_k) − 𝐃(F^* − h E_k)) / (2 h)

    For an emulator cycle, pass ``fock_tuple=F_NN(P_base)``.  This is
    the actual intermediate point in the chain rule.  The positional
    orbital arguments remain available for reference-cycle derivatives;
    using truth orbitals for a substituted cycle gives the wrong operator.

    Per R2 advisory #3, the per-spin Fock perturbation is pre-symmetrized
    before applying ``𝐃``: PySCF's eigensolver expects exact Hermiticity
    and a ~1e-14 antisymmetric leak from the unpack→matvec→pack round-trip
    would otherwise propagate as orbital-rotation noise.

    Parameters
    ----------
    state : MoleculeState
        UHF / UKS state; carries basis and method.
    mo_energy_tuple : tuple ``(ε_α, ε_β)``
        Per-spin orbital energies at the truth fixed point, from
        :class:`scf.host.SCFResult`.  Sorted ascending.
    mo_coeff_tuple : tuple ``(C_α, C_β)``
        Per-spin orbital coefficient matrices at the truth fixed point.
        Column ``i`` corresponds to ``mo_energy[i]``.
    fock_tuple : tuple ``(F_α, F_β)``, optional
        Actual Fock intermediate to differentiate. For an NN cycle use
        ``fock_builder(state, P_base)``. Mutually exclusive with orbital
        arguments, which serve reference-cycle derivatives.
    h : float, default 1e-4
        Central-difference step in packed-flat-F space (SPEC §7).

    Returns
    -------
    emulator_jvp : callable
        ``emulator_jvp(w: Tensor) → Tensor`` mapping a doubled-flat
        Fock-tangent ``w ∈ R^{2 n_unique}`` to the doubled-flat
        density-update tangent ``D𝐃 · w ∈ R^{2 n_unique}``.  Float64.

    Notes
    -----
    - We materialize the dense Jacobian once and reuse via matvec; the
      6 × 6 (STO-3G) or 20 × 20 (6-31G) cost is negligible.
    - PySCF's ``scf.hf.eig`` is used for the generalized eigenproblem
      (matching ``scf.host._solve_generalized_eig``); per-spin unit
      occupation per UHF/UKS.
    - The χ_0 alternative (analytic via PySCF response functions) is
      implemented in :func:`scf.jacobian.cpks_jacobian_analytic_uhf`; the
      ``χ_0`` piece is the same operator.  We use FD here for simplicity
      and round-trip consistency with the FD-side ``cpks_jacobian_fd_uhf``
      reference in R2.
    """
    if state.method not in ("uhf", "uks"):
        raise ValueError(
            f"make_emulator_jvp_scf requires state.method in "
            f"{{'uhf', 'uks'}}; got method={state.method!r}."
        )
    if fock_tuple is not None:
        if mo_energy_tuple is not None or mo_coeff_tuple is not None:
            raise ValueError("pass fock_tuple or orbitals, not both")
        if not (isinstance(fock_tuple, tuple) and len(fock_tuple) == 2):
            raise ValueError("fock_tuple must be a 2-tuple")
        # Recover orbitals of the supplied Fock only to reuse the reference
        # API validation/reconstruction below.  No truth Fock is evaluated.
        from pyscf.scf import hf as scf_hf
        _mol = _build_mol(state)
        _mf = _build_mf(state, mol=_mol)
        _S = np.asarray(_mf.get_ovlp(), dtype=np.float64)
        pairs = [scf_hf.eig(0.5 * (np.asarray(F) + np.asarray(F).T), _S)
                 for F in fock_tuple]
        mo_energy_tuple = tuple(pair[0] for pair in pairs)
        mo_coeff_tuple = tuple(pair[1] for pair in pairs)
    if not (isinstance(mo_energy_tuple, tuple) and len(mo_energy_tuple) == 2):
        raise ValueError(
            f"mo_energy_tuple must be a 2-tuple (ε_α, ε_β); got type "
            f"{type(mo_energy_tuple).__name__}."
        )
    if not (isinstance(mo_coeff_tuple, tuple) and len(mo_coeff_tuple) == 2):
        raise ValueError(
            f"mo_coeff_tuple must be a 2-tuple (C_α, C_β); got type "
            f"{type(mo_coeff_tuple).__name__}."
        )
    if h <= 0:
        raise ValueError(f"h must be positive, got h={h!r}.")

    C_alpha = np.asarray(mo_coeff_tuple[0], dtype=np.float64)
    C_beta = np.asarray(mo_coeff_tuple[1], dtype=np.float64)
    eps_alpha = np.asarray(mo_energy_tuple[0], dtype=np.float64)
    eps_beta = np.asarray(mo_energy_tuple[1], dtype=np.float64)
    M = C_alpha.shape[0]
    if C_alpha.shape != (M, M) or C_beta.shape != (M, M):
        raise ValueError(
            f"mo_coeff arrays must be (M, M)={M, M}; got "
            f"alpha={C_alpha.shape!r}, beta={C_beta.shape!r}."
        )
    if eps_alpha.shape != (M,) or eps_beta.shape != (M,):
        raise ValueError(
            f"mo_energy arrays must be (M,)={(M,)}; got "
            f"alpha={eps_alpha.shape!r}, beta={eps_beta.shape!r}."
        )

    nu_single = n_unique(M)
    nu_total = 2 * nu_single

    # PySCF integral machinery: overlap S and per-spin occupation pattern.
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    n_alpha, n_beta = mol.nelec

    # Reconstruct the selected per-spin Fock matrices from their
    # (C_σ, ε_σ) without invoking the truth Fock builder.  The Roothaan–Hall
    # eigenproblem at the fixed point satisfies ``F_σ C_σ = S C_σ diag(ε_σ)``,
    # so ``F_σ = S C_σ diag(ε_σ) C_σ^{-1}``.  This avoids paying for an
    # extra truth Fock build, which would also be redundant — we only need
    # F_σ as the linearization base point for FD.
    F_alpha_star = S @ C_alpha @ np.diag(eps_alpha) @ np.linalg.inv(C_alpha)
    F_beta_star = S @ C_beta @ np.diag(eps_beta) @ np.linalg.inv(C_beta)
    # Symmetrize (these come from a generalized eigenproblem; small
    # numerical asymmetry is expected and removed defensively).
    F_alpha_star = 0.5 * (F_alpha_star + F_alpha_star.T)
    F_beta_star = 0.5 * (F_beta_star + F_beta_star.T)

    # Build the dense (2 n_unique, 2 n_unique) D𝐃 Jacobian via per-column
    # central differences over the per-spin diag+rebuild map.
    from pyscf.scf import hf as scf_hf

    def _diag_rebuild_uhf(
        F_alpha: np.ndarray, F_beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One per-spin diagonalize + density-rebuild step (UHF/UKS)."""
        _e_a, C_a = scf_hf.eig(F_alpha, S)
        _e_b, C_b = scf_hf.eig(F_beta, S)
        C_occ_a = C_a[:, :n_alpha]
        C_occ_b = C_b[:, :n_beta]
        P_a = C_occ_a @ C_occ_a.T
        P_b = C_occ_b @ C_occ_b.T
        P_a = 0.5 * (P_a + P_a.T)
        P_b = 0.5 * (P_b + P_b.T)
        return P_a, P_b

    D_D = np.zeros((nu_total, nu_total), dtype=np.float64)

    for k in range(nu_total):
        e_flat = np.zeros(nu_total, dtype=np.float64)
        e_flat[k] = 1.0

        # E_k as a per-spin (M, M) symmetric pair via the same unpack as
        # the density seam.  This is the load-bearing detail: the input
        # tangent ``w`` from the host_jvp lives in the SAME doubled-flat
        # space the density does, packed with the same convention.
        E_alpha_t, E_beta_t = unpack_density_matrix(
            torch.from_numpy(e_flat), M=M, method=state.method,
        )
        E_alpha = E_alpha_t.numpy()
        E_beta = E_beta_t.numpy()
        # Pre-symmetrize (R2 advisory #3).  The unpack helper already
        # mirrors strict-upper into strict-lower, so this is defensive.
        E_alpha = 0.5 * (E_alpha + E_alpha.T)
        E_beta = 0.5 * (E_beta + E_beta.T)

        F_alpha_p = F_alpha_star + h * E_alpha
        F_beta_p = F_beta_star + h * E_beta
        F_alpha_m = F_alpha_star - h * E_alpha
        F_beta_m = F_beta_star - h * E_beta

        P_a_p, P_b_p = _diag_rebuild_uhf(F_alpha_p, F_beta_p)
        P_a_m, P_b_m = _diag_rebuild_uhf(F_alpha_m, F_beta_m)

        dP_a = (P_a_p - P_a_m) / (2.0 * h)
        dP_b = (P_b_p - P_b_m) / (2.0 * h)
        # Symmetrize per spin (defensive).
        dP_a = 0.5 * (dP_a + dP_a.T)
        dP_b = 0.5 * (dP_b + dP_b.T)

        D_D[:, k] = pack_density_matrix(
            (torch.from_numpy(dP_a), torch.from_numpy(dP_b)),
            method=state.method,
        ).numpy()

    D_D_tensor = torch.from_numpy(D_D)

    def _emulator_jvp(w: torch.Tensor) -> torch.Tensor:
        w_64 = w.to(dtype=torch.float64)
        if w_64.shape != (nu_total,):
            raise ValueError(
                f"emulator_jvp_scf: input shape must be ({nu_total},); "
                f"got shape={tuple(w_64.shape)!r}."
            )
        return D_D_tensor @ w_64

    return _emulator_jvp


def make_cycle_jvps_scf(state, P_base, fock_builder, *, h=_FD_STEP):
    """Return the two callbacks for the actual cycle ``D ∘ F`` at P_base.

    Central differences through the supplied Fock builder include the
    actual NN.  The second differential is evaluated at F(P_base), even
    when P_base is a truth equilibrium.  Prefer this factory for response
    calculations so the intermediate evaluation point cannot drift.
    """
    first = make_host_jvp_scf(state, P_base, fock_builder, h=h)
    second = make_emulator_jvp_scf(
        state, fock_tuple=fock_builder(state, P_base), h=h,
    )
    return first, second
