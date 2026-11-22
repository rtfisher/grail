"""Analytic and finite-difference CPKS Jacobians for the SCF cycle.

SPEC_scf §3.2, §6 P2, §8.10 R2.  This module exposes the following public
entry points:

- :func:`cpks_jacobian_analytic`  — closed-form ``D𝒯 = χ_0 · 𝐊`` matrix at
  ``P*``, built by composing PySCF's response-function infrastructure
  (the ``𝐊``-side: ``pyscf.scf._response_functions._gen_*_response``) with
  the explicit occupied-virtual orbital-rotation formula for the density-
  matrix response ``χ_0``.  Closed-shell (RHF/RKS) and open-shell (UHF/UKS)
  are dispatched based on ``state.method`` — see the function docstring.
  Returns a dense matrix acting on the packed-flat density seam defined by
  :func:`scf.state.pack_density_matrix`: ``(n_unique, n_unique)`` for the
  closed-shell case, ``(2*n_unique, 2*n_unique)`` for UHF/UKS (R1's
  doubled α | β block layout).
- :func:`cpks_jacobian_analytic_uhf` — UHF/UKS-only direct entry point for
  callers that already have the per-spin orbital data in hand.  Mirrors
  ``cpks_jacobian_analytic`` for the open-shell branch (SPEC_scf §8.10 R2).
- :func:`cpks_jacobian_fd`        — central finite differences over the
  full cycle map ``𝒯(P) = 𝐃[𝐅[P]]`` per packed direction (closed-shell);
  used as the brute-force reference for cross-validation.  Default
  ``h=1e-4`` matches TASK-035 / SPEC v1.6 (``the callback contract`` Engineering
  Conventions).
- :func:`cpks_jacobian_fd_uhf`     — central-FD reference Jacobian for the
  UHF/UKS branch (SPEC_scf §8.10 R2).  Same step convention as the
  closed-shell FD path.
- :func:`A_op_analytic`            — operator-norm amplification factor
  ``A_op = ||(I − D𝒯)^{-1}||_2`` computed by direct dense inversion (the
  matrix is small — at most ``2 * n_unique ≤ 42`` for the H₂ STO-3G /
  6-31G UHF testbed — so no Krylov is needed).
- :func:`spectral_radius`         — ``ρ(D𝒯) = max_k |λ_k(D𝒯)|`` via
  ``numpy.linalg.eigvals``.

PySCF intermediates used (all from the public + ``_response_functions``
namespace per SPEC §8.2 option 1+2 blend):

- ``mf.get_ovlp()``, ``mf.get_hcore()``                       — integrals.
- ``pyscf.scf._response_functions._gen_rhf_response`` /
  ``_gen_rks_response`` (dispatched by ``KohnShamDFT``)        — returns a
  closure ``vind(dm1) → 𝐊 · dm1`` (the AO-basis Fock perturbation
  produced by a density-matrix perturbation ``dm1``).  Internally PySCF
  handles XC kernel, Hartree, exact-exchange-mix, and NLC terms.
- ``MoleculeState`` ``mo_coeff``, ``mo_energy``                — orbital
  basis used to build the ``χ_0`` orbital-rotation transform explicitly.
  We do NOT call ``pyscf.scf.cphf.solve`` because we need the Jacobian
  matrix, not its inverse action on a single right-hand side.

Chain-rule labeling (SPEC_scf §4.2): ``D𝒯 = D𝐃 · D𝐅``.  The PySCF
response closure gives ``𝐊 = ∂𝐅/∂P`` (the Fock-build differential).
The orbital-rotation formula gives ``χ_0 = D𝐃``.

State-vector convention (SPEC_scf §3.2, §4.1):
upper-triangular row-major packed-flat (length ``n_unique = M·(M+1)/2``).
The returned Jacobian acts in this representation: column ``k`` is
``T(P* + h · E_k) − T(P* − h · E_k)`` mapped back to the flat unique
entries, where ``E_k`` is the symmetric matrix corresponding to flat
index ``k`` (its only off-diagonal pair, or its diagonal, is unity; all
other entries are zero).
"""

from __future__ import annotations

import numpy as np
import torch

from .fock_build import _build_mf, _build_mol, build_fock_truth
from .state import (
    MoleculeState,
    n_unique,
    pack_density_matrix,
    unpack_density_matrix,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unit_perturbation_matrix(k: int, M: int) -> np.ndarray:
    """Return ``E_k`` — the symmetric (M, M) matrix corresponding to flat index k.

    For flat index ``k`` of the upper-triangular row-major packing, ``E_k`` has
    ``1`` at the corresponding (i, j) position and at (j, i) (its mirror), and
    ``0`` everywhere else.  Used to build the canonical perturbation basis for
    finite-difference Jacobian columns and analytic Jacobian columns.

    Note: there is no factor of 1/2 applied to off-diagonal entries — both
    upper and lower triangular entries take the value 1.  This matches the
    convention that ``pack_density_matrix`` reads ONLY the upper triangle, so
    pack→unpack→pack is identity by construction.
    """
    flat = np.zeros(n_unique(M), dtype=np.float64)
    flat[k] = 1.0
    # unpack_density_matrix mirrors strict upper into strict lower → E_k is
    # symmetric with the (i, j) and (j, i) entries both 1 (or diagonal entry 1).
    E = unpack_density_matrix(torch.from_numpy(flat), M).numpy()
    return E


def _diag_and_rebuild(F: np.ndarray, S: np.ndarray, n_occ: int) -> np.ndarray:
    """Solve ``F C = S C ε`` and rebuild ``P = 2 C_occ C_occᵀ``.

    Used by :func:`cpks_jacobian_fd` to apply the second half of the cycle
    map ``𝐃[F]``.  Matches the eigenproblem solver chosen in
    ``scf.host._solve_generalized_eig`` (PySCF's ``scf.hf.eig``) so the FD
    Jacobian is consistent with the actual cycle iterated by
    ``picard_iterate_scf``.
    """
    from pyscf.scf import hf as scf_hf

    _, c = scf_hf.eig(F, S)
    C_occ = c[:, :n_occ]
    P = 2.0 * (C_occ @ C_occ.T)
    # Force exact symmetry to match scf.host._build_density_from_orbitals.
    P = 0.5 * (P + P.T)
    return np.asarray(P, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public: analytic CPKS Jacobian (dispatcher + closed-shell implementation)
# ---------------------------------------------------------------------------

def cpks_jacobian_analytic(
    state: MoleculeState,
    P_star: np.ndarray | tuple,
    mo_energy: np.ndarray | tuple,
    mo_coeff: np.ndarray | tuple,
) -> np.ndarray:
    """Analytic ``D𝒯 = χ_0 · 𝐊`` at the converged SCF density.

    Dispatches on ``state.method`` (SPEC_scf §8.10 R2):

    - Closed-shell (``"rhf"``, ``"rks"``): ``P_star``, ``mo_energy``,
      ``mo_coeff`` are single arrays of shape ``(M, M)``, ``(M,)``,
      ``(M, M)`` respectively.  Returns ``(n_unique, n_unique)``.
    - Open-shell (``"uhf"``, ``"uks"``): the three arguments are 2-tuples
      of the per-spin arrays — ``(P_α, P_β)``, ``(ε_α, ε_β)``,
      ``(C_α, C_β)``.  Returns ``(2 * n_unique, 2 * n_unique)`` packed in
      R1's α | β block layout (first ``n_unique`` rows/columns are α,
      last ``n_unique`` are β).  See :func:`cpks_jacobian_analytic_uhf`
      for the UHF-only direct entry point.

    Builds the dense Jacobian column-by-column in the packed-flat
    representation.  For each unit perturbation ``E_k`` (symmetric AO
    density direction), the column is

        column_k = pack( χ_0 · 𝐊 · E_k )

    where:

    - ``𝐊 · E_k = vind(E_k)`` is the AO-basis Fock perturbation returned
      by ``pyscf.scf._response_functions._gen_*_response`` — internally
      this evaluates ``∂𝐉/∂P + ∂𝐕_xc/∂P`` (and the hybrid-exchange piece
      for B3LYP / RKS) acting on the perturbation density matrix.
    - ``χ_0 · F_1`` (first-order density response to a perturbation Fock
      ``F_1``) is computed explicitly via the closed-shell occupied-virtual
      orbital-rotation formula:

          U_{ai} = -F_1^{ai}/(ε_a − ε_i)             (MO basis)
          δP^{mo}_{ai} = δP^{mo}_{ia} = 2 U_{ai}     (closed-shell)
          δP^{ao} = C · δP^{mo} · Cᵀ

      ``C`` are the MO coefficients in ``state.mo_coeff``, ``ε`` the
      orbital energies in ``state.mo_energy``.

    For UHF/UKS the same construction is applied per spin with **unit**
    (not factor-2) occupation; the ``𝐊``-side comes from PySCF's
    ``_gen_uhf_response`` which returns a closure accepting and returning
    spin-stacked ``(2, M, M)`` density perturbations.

    Parameters
    ----------
    state : MoleculeState
        Molecular configuration (geometry, basis, method, XC functional).
    P_star : np.ndarray (M, M) or 2-tuple of such for UHF/UKS
        Converged density matrix (or per-spin pair) at which the Jacobian
        is evaluated.  Used to seed the PySCF mean-field object so that
        ``vind`` knows the reference density.  Need not be exactly the SCF
        fixed point — the response closure depends on it only through the
        XC kernel (cached from ``mo_coeff, mo_occ``).
    mo_energy : np.ndarray (M,) or 2-tuple of such
        Orbital energies at the fixed point, sorted ascending (per spin
        for UHF/UKS).  From :class:`scf.host.SCFResult`.
    mo_coeff : np.ndarray (M, M) or 2-tuple of such
        Orbital coefficients; column ``i`` corresponds to ``mo_energy[i]``
        (per spin for UHF/UKS).  From :class:`scf.host.SCFResult`.

    Returns
    -------
    DT : np.ndarray, float64
        Shape ``(n_unique, n_unique)`` for closed-shell;
        ``(2 * n_unique, 2 * n_unique)`` for UHF/UKS.

    Notes
    -----
    The Jacobian is not in general symmetric: the cycle map is on a
    nonlinear manifold (symmetric matrices with ``tr(SP) = 2``), and the
    flat-P representation is unconstrained.  For the H₂/STO-3G closed-
    shell test case it happens to be symmetric to float precision because
    of the two-dimensional, single-occupied-orbital structure; UHF/UKS
    and larger systems are generally non-symmetric.
    """
    # Dispatcher: UHF/UKS path goes through cpks_jacobian_analytic_uhf so
    # the open-shell implementation lives in one place.
    if state.method in ("uhf", "uks"):
        if not (isinstance(P_star, tuple) and len(P_star) == 2):
            raise ValueError(
                f"cpks_jacobian_analytic(method={state.method!r}) requires "
                f"P_star as a 2-tuple (P_alpha, P_beta); got type "
                f"{type(P_star).__name__}."
            )
        if not (isinstance(mo_energy, tuple) and len(mo_energy) == 2):
            raise ValueError(
                f"cpks_jacobian_analytic(method={state.method!r}) requires "
                f"mo_energy as a 2-tuple (eps_alpha, eps_beta); got type "
                f"{type(mo_energy).__name__}."
            )
        if not (isinstance(mo_coeff, tuple) and len(mo_coeff) == 2):
            raise ValueError(
                f"cpks_jacobian_analytic(method={state.method!r}) requires "
                f"mo_coeff as a 2-tuple (C_alpha, C_beta); got type "
                f"{type(mo_coeff).__name__}."
            )
        return cpks_jacobian_analytic_uhf(
            state,
            P_star_alpha=P_star[0], P_star_beta=P_star[1],
            mo_energy_alpha=mo_energy[0], mo_energy_beta=mo_energy[1],
            mo_coeff_alpha=mo_coeff[0], mo_coeff_beta=mo_coeff[1],
        )

    # Closed-shell branch (RHF/RKS).
    if isinstance(P_star, tuple):
        raise ValueError(
            f"cpks_jacobian_analytic(method={state.method!r}) requires a "
            f"single P_star ndarray; got a 2-tuple (open-shell input "
            f"supplied to closed-shell method)."
        )
    P_star = np.asarray(P_star, dtype=np.float64)
    if P_star.ndim != 2 or P_star.shape[0] != P_star.shape[1]:
        raise ValueError(
            f"P_star must be square (M, M), got shape={P_star.shape!r}."
        )
    M = P_star.shape[0]
    nu = n_unique(M)
    if mo_coeff.shape != (M, M) or mo_energy.shape != (M,):
        raise ValueError(
            f"mo_coeff must be (M, M) and mo_energy (M,) for M={M}; "
            f"got mo_coeff.shape={mo_coeff.shape}, mo_energy.shape={mo_energy.shape}."
        )

    n_occ = state.n_electrons // 2

    # Build PySCF mean-field and prime it with the converged orbitals so that
    # the response closure caches the XC kernel at the right reference state.
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    mf.mo_coeff = np.asarray(mo_coeff, dtype=np.float64)
    mo_occ = np.zeros(M, dtype=np.float64)
    mo_occ[:n_occ] = 2.0
    mf.mo_occ = mo_occ

    # Dispatch to the right response builder.  ``_gen_rhf_response`` handles
    # both RHF and RKS automatically (the RKS branch is activated by the
    # ``isinstance(mf, hf.KohnShamDFT)`` check inside that function).
    from pyscf.scf._response_functions import _gen_rhf_response

    vind = _gen_rhf_response(
        mf, mo_coeff=mf.mo_coeff, mo_occ=mo_occ, hermi=1, singlet=None,
    )

    # Precompute MO-basis quantities used in the χ_0 transform.  We avoid
    # rebuilding C inside the loop.
    C = np.asarray(mo_coeff, dtype=np.float64)
    eps = np.asarray(mo_energy, dtype=np.float64)
    occ_idx = np.arange(n_occ)
    virt_idx = np.arange(n_occ, M)

    # χ_0 is rank-(n_occ * n_virt); we can vectorize the orbital-rotation
    # formula by forming the per-(i, a) reciprocal energy denominators once.
    # de[i, a] = ε_i − ε_a    (negative; occ below virt).
    de = eps[occ_idx][:, None] - eps[virt_idx][None, :]   # (n_occ, n_virt)

    DT = np.zeros((nu, nu), dtype=np.float64)
    for k in range(nu):
        # 𝐊 · E_k : Fock-side response to a unit density perturbation.
        E_k = _unit_perturbation_matrix(k, M)
        F_1 = np.asarray(vind(E_k), dtype=np.float64)
        # The response closure may break exact symmetry by ~1e-14 due to the
        # XC quadrature and the J/K bookkeeping; restore Hermiticity to make
        # the χ_0 transform stable.
        F_1 = 0.5 * (F_1 + F_1.T)

        # χ_0 · F_1: transform to MO basis, apply orbital-rotation formula,
        # transform back.
        F_1_mo = C.T @ F_1 @ C                                 # (M, M)
        F_1_ai = F_1_mo[np.ix_(virt_idx, occ_idx)]             # (n_virt, n_occ)
        # P_1^{mo}_{ai} = P_1^{mo}_{ia} = 2 F_1_{ai}/(ε_i − ε_a)
        # The factor 2 is the closed-shell occupation.
        # Note: de.T has shape (n_virt, n_occ) matching F_1_ai.
        U_ai = F_1_ai / de.T                                   # (n_virt, n_occ)
        # 2 U_ai goes into the (a, i) MO-basis density slot AND its transpose.
        P_1_mo = np.zeros((M, M), dtype=np.float64)
        P_1_mo[np.ix_(virt_idx, occ_idx)] = 2.0 * U_ai
        P_1_mo[np.ix_(occ_idx, virt_idx)] = 2.0 * U_ai.T
        # Back to AO.
        P_1 = C @ P_1_mo @ C.T
        # Defensive symmetrize on output side (the orbital-rotation formula
        # produces a symmetric result analytically, but float noise leaks).
        P_1 = 0.5 * (P_1 + P_1.T)

        # Pack into flat column.
        DT[:, k] = pack_density_matrix(torch.from_numpy(P_1)).numpy()

    return DT


# ---------------------------------------------------------------------------
# Public: finite-difference reference Jacobian
# ---------------------------------------------------------------------------

def cpks_jacobian_fd(
    state: MoleculeState,
    P_star: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference Jacobian of the full cycle map ``𝒯(P)``.

    Builds the dense Jacobian column-by-column by evaluating the cycle at
    ``P* ± h · E_k`` and packing the central difference.  Used as the
    brute-force reference for cross-validating
    :func:`cpks_jacobian_analytic`.

    The cycle evaluated is exactly the one Picard iterates:

        F = build_fock_truth(state, P)
        F = (F + Fᵀ)/2
        F C = S C ε       (generalized eigenproblem, PySCF's scf.hf.eig)
        P_new = 2 C_occ C_occᵀ                (closed-shell, symmetrized)

    Parameters
    ----------
    state : MoleculeState
    P_star : np.ndarray of shape (M, M)
        Density matrix at which to evaluate ``D𝒯``.  Typically the converged
        SCF density from :func:`scf.host.picard_iterate_scf`.
    h : float, default 1e-4
        Finite-difference step in flat-P space.  Matches the TASK-035 /
        SPEC v1.6 convention.

    Returns
    -------
    DT_fd : np.ndarray of shape (n_unique, n_unique), float64
        Central-FD approximation to ``D𝒯|_{P*}`` in the packed-flat
        representation.

    Notes
    -----
    The truth Fock builder is used (``build_fock_truth``); the FD Jacobian
    is the reference *for the truth cycle*.  A separate FD path through an
    NN-replaced Fock builder is in scope for P4 (``scf/jvp.py``) and is NOT
    provided here — P2's role is the analytic anchor at the truth fixed
    point.
    """
    P_star = np.asarray(P_star, dtype=np.float64)
    if P_star.ndim != 2 or P_star.shape[0] != P_star.shape[1]:
        raise ValueError(
            f"P_star must be square (M, M), got shape={P_star.shape!r}."
        )
    if h <= 0:
        raise ValueError(f"h must be positive, got h={h!r}.")
    M = P_star.shape[0]
    nu = n_unique(M)

    # Constant geometry / basis quantities.  Reuse one Mole + mean-field per
    # call to avoid rebuilding them 2·n_unique times below; build_fock_truth
    # rebuilds its own internally but we still need ``S`` here for the
    # generalized eigenproblem on the FD perturbed Fock matrices.
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    n_occ = state.n_electrons // 2

    def cycle_T(P: np.ndarray) -> np.ndarray:
        F = build_fock_truth(state, P)
        F = 0.5 * (F + F.T)
        return _diag_and_rebuild(F, S, n_occ)

    DT_fd = np.zeros((nu, nu), dtype=np.float64)
    for k in range(nu):
        E_k = _unit_perturbation_matrix(k, M)
        P_plus = P_star + h * E_k
        P_minus = P_star - h * E_k
        dT = (cycle_T(P_plus) - cycle_T(P_minus)) / (2.0 * h)
        DT_fd[:, k] = pack_density_matrix(torch.from_numpy(dT)).numpy()

    return DT_fd


# ---------------------------------------------------------------------------
# Internal helpers for the UHF / UKS branch (SPEC_scf §8.10 R2).
# ---------------------------------------------------------------------------

def _chi0_sigma_uhf(
    F1_sigma: np.ndarray,
    C_sigma: np.ndarray,
    eps_sigma: np.ndarray,
    occ_idx: np.ndarray,
    virt_idx: np.ndarray,
) -> np.ndarray:
    """Per-spin first-order AO density response (UHF/UKS unit occupation).

    UHF/UKS spin orbitals are singly occupied — the MO-basis density
    perturbation carries unit (not factor-2) weight per orbital, in
    contrast to the closed-shell factor-2 path in
    :func:`cpks_jacobian_analytic`.  Mirrors the diagnostic helper
    ``examples/diagnose_uhf_amplification.py::_chi0_sigma``.

    Standard CPKS formula per spin σ:

        U^σ_{ai} = -F^σ_{1,ai}/(ε^σ_a − ε^σ_i)
        δP^{σ,mo}_{ai} = δP^{σ,mo}_{ia} = U^σ_{ai} · 1   (occupation = 1)
                                       = F^σ_{1,ai}/(ε^σ_i − ε^σ_a)

    Back-transform to AO: ``δP^σ = C^σ · δP^{σ,mo} · C^{σ,T}``.

    Parameters
    ----------
    F1_sigma : (M, M) Fock perturbation for spin σ (AO basis).
    C_sigma : (M, M) MO coefficients for spin σ.
    eps_sigma : (M,) MO energies for spin σ.
    occ_idx : array of occupied MO indices for spin σ.
    virt_idx : array of virtual MO indices for spin σ.

    Returns
    -------
    P1 : (M, M) AO-basis density-matrix response, symmetric.
    """
    F1_mo = C_sigma.T @ F1_sigma @ C_sigma            # (M, M)
    # de[i, a] = ε_i − ε_a  (occ minus virt, negative).
    de = eps_sigma[occ_idx][:, None] - eps_sigma[virt_idx][None, :]
    F1_ai = F1_mo[np.ix_(virt_idx, occ_idx)]          # (n_virt, n_occ)
    # U_ai with σ occupation 1 (no factor 2 vs closed-shell).
    U_ai = F1_ai / de.T                               # (n_virt, n_occ)
    M = C_sigma.shape[0]
    P1_mo = np.zeros((M, M), dtype=np.float64)
    P1_mo[np.ix_(virt_idx, occ_idx)] = U_ai
    P1_mo[np.ix_(occ_idx, virt_idx)] = U_ai.T
    P1 = C_sigma @ P1_mo @ C_sigma.T
    # Defensive symmetrize on output (orbital-rotation formula is
    # analytically symmetric; float noise leaks).
    return 0.5 * (P1 + P1.T)


# ---------------------------------------------------------------------------
# Public: analytic CPKS Jacobian — UHF / UKS branch
# ---------------------------------------------------------------------------

def cpks_jacobian_analytic_uhf(
    state: MoleculeState,
    P_star_alpha: np.ndarray,
    P_star_beta: np.ndarray,
    mo_energy_alpha: np.ndarray,
    mo_energy_beta: np.ndarray,
    mo_coeff_alpha: np.ndarray,
    mo_coeff_beta: np.ndarray,
) -> np.ndarray:
    """UHF / UKS analytic CPKS Jacobian.  SPEC_scf §8.10 R2.

    Builds the ``(2 * n_unique, 2 * n_unique)`` block CPKS Jacobian in
    R1's doubled state-vector seam: the first ``n_unique`` rows and
    columns are the α-spin block (upper-triangular row-major packing of
    ``P_α``); the last ``n_unique`` rows and columns are the β-spin block.
    Mirrors the structure validated in
    ``examples/diagnose_uhf_amplification.py::_build_uhf_cpks_jacobian``.

    For each unit perturbation ``E_k`` placed in the α (or β) channel:

      dm1 = (E_k, 0)  for α-channel perturbations, or  (0, E_k)  for β
      v1 = vind(dm1)                       # (2, M, M) — PySCF UHF closure
      δP_α = χ_0^α · v1[0]                 # per-spin unit-occupation χ_0
      δP_β = χ_0^β · v1[1]
      column_k = [pack(δP_α); pack(δP_β)]  # length 2*n_unique

    The PySCF response closure ``_gen_uhf_response`` returns a callable
    ``vind(dm1)`` where ``dm1`` has shape ``(2, M, M)`` (α and β stacked)
    and the output has the same shape.  Internally PySCF handles Hartree,
    spin-resolved XC (for UKS), and exact-exchange-mix bookkeeping with
    full spin-block structure.

    Parameters
    ----------
    state : MoleculeState
        Must have ``state.method in {"uhf", "uks"}``.  Carries the
        molecule, basis, and (for UKS) XC functional.
    P_star_alpha, P_star_beta : np.ndarray of shape (M, M)
        Converged per-spin density matrices.  Used to seed the PySCF
        mean-field's reference state so the response closure caches the
        right XC kernel.
    mo_energy_alpha, mo_energy_beta : np.ndarray of shape (M,)
        Per-spin orbital energies, sorted ascending.
    mo_coeff_alpha, mo_coeff_beta : np.ndarray of shape (M, M)
        Per-spin orbital coefficients.

    Returns
    -------
    DT : np.ndarray of shape (2 * n_unique, 2 * n_unique), float64
        Block CPKS Jacobian in the R1 α | β packed-flat convention.

    Notes
    -----
    The α–α and β–β diagonal blocks reflect the same-spin response.  The
    α–β / β–α off-diagonal blocks arise from the Hartree-J coupling (which
    is spin-independent: both spins see ``J[P_α + P_β]``) and, for UKS,
    from cross-spin XC kernel mixing.  In the symmetric (RHF-like) regime
    these off-diagonal blocks are small; in the broken-symmetry regime
    (above the Coulson–Fischer point) they grow to comparable magnitude
    with the diagonals — see the test
    ``test_alpha_beta_block_zero_below_R_CF_nonzero_above`` for the
    quantitative statement.
    """
    if state.method not in ("uhf", "uks"):
        raise ValueError(
            f"cpks_jacobian_analytic_uhf requires state.method in "
            f"{{'uhf', 'uks'}}; got method={state.method!r}."
        )
    P_alpha = np.asarray(P_star_alpha, dtype=np.float64)
    P_beta = np.asarray(P_star_beta, dtype=np.float64)
    for label, mat in (("P_star_alpha", P_alpha), ("P_star_beta", P_beta)):
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"{label} must be square (M, M), got shape={mat.shape!r}."
            )
    if P_alpha.shape != P_beta.shape:
        raise ValueError(
            f"P_star_alpha and P_star_beta must share shape; got "
            f"alpha={P_alpha.shape!r}, beta={P_beta.shape!r}."
        )
    M = P_alpha.shape[0]
    nu = n_unique(M)
    eps_a = np.asarray(mo_energy_alpha, dtype=np.float64)
    eps_b = np.asarray(mo_energy_beta, dtype=np.float64)
    C_a = np.asarray(mo_coeff_alpha, dtype=np.float64)
    C_b = np.asarray(mo_coeff_beta, dtype=np.float64)
    if C_a.shape != (M, M) or C_b.shape != (M, M):
        raise ValueError(
            f"mo_coeff arrays must be (M, M)={M, M}; got alpha={C_a.shape!r}, "
            f"beta={C_b.shape!r}."
        )
    if eps_a.shape != (M,) or eps_b.shape != (M,):
        raise ValueError(
            f"mo_energy arrays must be (M,)={(M,)}; got alpha={eps_a.shape!r}, "
            f"beta={eps_b.shape!r}."
        )

    # Determine per-spin occupation pattern from the molecule.  PySCF stores
    # nelec as (n_alpha, n_beta) on the Mole.
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    n_alpha, n_beta = mol.nelec

    # Prime the mean-field with the converged orbitals so the response
    # closure caches the XC kernel at the correct reference state.
    mo_coeff_stack = np.array([C_a, C_b], dtype=np.float64)
    mo_occ_a = np.zeros(M, dtype=np.float64)
    mo_occ_a[:n_alpha] = 1.0
    mo_occ_b = np.zeros(M, dtype=np.float64)
    mo_occ_b[:n_beta] = 1.0
    mo_occ_stack = np.array([mo_occ_a, mo_occ_b], dtype=np.float64)
    mf.mo_coeff = mo_coeff_stack
    mf.mo_occ = mo_occ_stack

    from pyscf.scf._response_functions import _gen_uhf_response

    vind = _gen_uhf_response(
        mf, mo_coeff=mo_coeff_stack, mo_occ=mo_occ_stack, hermi=1,
    )

    occ_idx_a = np.where(mo_occ_a > 0.5)[0]
    virt_idx_a = np.where(mo_occ_a <= 0.5)[0]
    occ_idx_b = np.where(mo_occ_b > 0.5)[0]
    virt_idx_b = np.where(mo_occ_b <= 0.5)[0]

    n_state = 2 * nu
    DT = np.zeros((n_state, n_state), dtype=np.float64)

    for spin in (0, 1):
        for k in range(nu):
            E_k = _unit_perturbation_matrix(k, M)
            dm1 = np.zeros((2, M, M), dtype=np.float64)
            dm1[spin] = E_k
            v1 = np.asarray(vind(dm1), dtype=np.float64)   # (2, M, M)
            # Symmetrize each spin block (XC quadrature ~1e-14 noise).
            v1[0] = 0.5 * (v1[0] + v1[0].T)
            v1[1] = 0.5 * (v1[1] + v1[1].T)

            # χ₀ acts independently per spin.
            dP_a = _chi0_sigma_uhf(v1[0], C_a, eps_a, occ_idx_a, virt_idx_a)
            dP_b = _chi0_sigma_uhf(v1[1], C_b, eps_b, occ_idx_b, virt_idx_b)

            col = k + (0 if spin == 0 else nu)
            # Pack each spin block into its half of the column.
            DT[:nu, col] = pack_density_matrix(torch.from_numpy(dP_a)).numpy()
            DT[nu:, col] = pack_density_matrix(torch.from_numpy(dP_b)).numpy()

    return DT


# ---------------------------------------------------------------------------
# Public: finite-difference reference Jacobian — UHF / UKS branch
# ---------------------------------------------------------------------------

def cpks_jacobian_fd_uhf(
    state: MoleculeState,
    P_star_tuple: tuple[np.ndarray, np.ndarray],
    h: float = 1e-4,
) -> np.ndarray:
    """Central-FD reference CPKS Jacobian for UHF/UKS.  SPEC_scf §8.10 R2.

    Central finite differences over the full UHF cycle map
    ``𝒯(P_α, P_β)`` per packed direction in the doubled state-vector seam.
    For each flat column index ``k ∈ [0, 2*n_unique)``: identify whether
    ``k`` lies in the α-block ``[0, n_unique)`` or the β-block; build the
    corresponding ``(E_k, 0)`` or ``(0, E_k)`` AO perturbation pair;
    evaluate ``𝒯`` at ``(P_α, P_β) ± h · δ`` and pack the central
    difference.  Used as the brute-force reference for cross-validating
    :func:`cpks_jacobian_analytic_uhf`.

    The cycle evaluated is exactly the one iterated by
    :func:`scf.host.picard_iterate_scf` in the UHF branch:

        F_α, F_β = build_fock_truth(state, (P_α, P_β))
        F_σ ← (F_σ + F_σᵀ)/2  per σ
        F_σ C_σ = S C_σ ε_σ           (per-spin generalized eigenproblem)
        P_σ_new = C_σ,occ · C_σ,occᵀ  (unit occupation per σ)

    Parameters
    ----------
    state : MoleculeState
        Must have ``state.method in {"uhf", "uks"}``.
    P_star_tuple : 2-tuple of np.ndarray, each (M, M)
        Per-spin density matrix pair at which to evaluate ``D𝒯``.
        Typically the converged pair from :func:`scf.host.picard_iterate_scf`.
    h : float, default 1e-4
        Finite-difference step in flat-P space.  Matches the closed-shell
        FD path and the TASK-035 / SPEC v1.6 convention.

    Returns
    -------
    DT_fd : np.ndarray of shape (2 * n_unique, 2 * n_unique), float64
        Central-FD approximation to ``D𝒯|_{(P_α*, P_β*)}`` in R1's
        doubled packed-flat representation.

    Notes
    -----
    Truth Fock builder (``build_fock_truth``) is used; the NN-replaced
    builder lands in P4 (``scf/jvp.py``).  R2's role is the analytic
    anchor at the truth fixed point.
    """
    if state.method not in ("uhf", "uks"):
        raise ValueError(
            f"cpks_jacobian_fd_uhf requires state.method in "
            f"{{'uhf', 'uks'}}; got method={state.method!r}."
        )
    if not (isinstance(P_star_tuple, tuple) and len(P_star_tuple) == 2):
        raise ValueError(
            f"cpks_jacobian_fd_uhf requires P_star_tuple as a 2-tuple "
            f"(P_alpha, P_beta); got type {type(P_star_tuple).__name__}."
        )
    if h <= 0:
        raise ValueError(f"h must be positive, got h={h!r}.")

    P_alpha = np.asarray(P_star_tuple[0], dtype=np.float64)
    P_beta = np.asarray(P_star_tuple[1], dtype=np.float64)
    for label, mat in (("P_alpha", P_alpha), ("P_beta", P_beta)):
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"{label} must be square (M, M), got shape={mat.shape!r}."
            )
    if P_alpha.shape != P_beta.shape:
        raise ValueError(
            f"P_alpha and P_beta must share shape; got alpha={P_alpha.shape!r}, "
            f"beta={P_beta.shape!r}."
        )
    M = P_alpha.shape[0]
    nu = n_unique(M)

    # Reuse a single Mole + mean-field for the overlap and per-spin
    # eigenproblems.  build_fock_truth builds its own internally; we
    # cannot easily share that without rewriting it, so accept the cost
    # — for H₂/STO-3G it is microseconds per FD evaluation.
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    n_alpha, n_beta = mol.nelec

    def cycle_T(P_a: np.ndarray, P_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One UHF/UKS cycle: Fock-build, per-spin diag, density rebuild."""
        F_pair = build_fock_truth(state, (P_a, P_b))
        F_a, F_b = F_pair
        F_a = 0.5 * (F_a + F_a.T)
        F_b = 0.5 * (F_b + F_b.T)
        from pyscf.scf import hf as scf_hf
        _e_a, C_a = scf_hf.eig(F_a, S)
        _e_b, C_b = scf_hf.eig(F_b, S)
        C_occ_a = C_a[:, :n_alpha]
        C_occ_b = C_b[:, :n_beta]
        # Unit occupation per spin (UHF/UKS), in contrast to factor-2 RHF.
        P_a_new = C_occ_a @ C_occ_a.T
        P_b_new = C_occ_b @ C_occ_b.T
        P_a_new = 0.5 * (P_a_new + P_a_new.T)
        P_b_new = 0.5 * (P_b_new + P_b_new.T)
        return (
            np.asarray(P_a_new, dtype=np.float64),
            np.asarray(P_b_new, dtype=np.float64),
        )

    n_state = 2 * nu
    DT_fd = np.zeros((n_state, n_state), dtype=np.float64)

    for spin in (0, 1):
        for k in range(nu):
            E_k = _unit_perturbation_matrix(k, M)
            if spin == 0:
                P_a_p, P_b_p = P_alpha + h * E_k, P_beta.copy()
                P_a_m, P_b_m = P_alpha - h * E_k, P_beta.copy()
            else:
                P_a_p, P_b_p = P_alpha.copy(), P_beta + h * E_k
                P_a_m, P_b_m = P_alpha.copy(), P_beta - h * E_k
            P_a_T_p, P_b_T_p = cycle_T(P_a_p, P_b_p)
            P_a_T_m, P_b_T_m = cycle_T(P_a_m, P_b_m)
            dP_a = (P_a_T_p - P_a_T_m) / (2.0 * h)
            dP_b = (P_b_T_p - P_b_T_m) / (2.0 * h)
            col = k + (0 if spin == 0 else nu)
            DT_fd[:nu, col] = pack_density_matrix(torch.from_numpy(dP_a)).numpy()
            DT_fd[nu:, col] = pack_density_matrix(torch.from_numpy(dP_b)).numpy()

    return DT_fd


# ---------------------------------------------------------------------------
# Public: amplification + spectral diagnostics
# ---------------------------------------------------------------------------

def A_op_analytic(jacobian: np.ndarray) -> float:
    """Ambient packed-coordinate inverse norm ``||(I − D𝒯)^{-1}||_2``.

    This uses the ordinary Euclidean norm of the supplied packed seam
    (off-diagonal weight one). It is not generally a positive-eigenvalue
    proxy or a norm restricted to physical density tangents. See
    ``scf.diagnostics.resolvent_metrics_scf`` for those distinctions.

    Solves the small dense system directly (the Jacobian has at most
    ``n_unique ≤ 21`` rows/cols for the H₂ STO-3G / 6-31G testbed; no
    Krylov is needed).  Returns the spectral (2-norm) operator norm via
    ``numpy.linalg.norm(..., ord=2)``.

    Parameters
    ----------
    jacobian : np.ndarray of shape (n_unique, n_unique)
        The CPKS Jacobian ``D𝒯`` from
        :func:`cpks_jacobian_analytic` or :func:`cpks_jacobian_fd`.

    Returns
    -------
    A_op : float
        ``||(I − D𝒯)^{-1}||_2``, positive and finite when ``(I − D𝒯)`` is
        non-singular.  Returns ``+inf`` if ``(I − D𝒯)`` is singular to
        within float precision (no resolvent exists; cycle is exactly
        marginally stable, which is non-physical at the SCF fixed point
        but possible in pathological tests).

    Notes
    -----
    SPEC §2.4 framing assumes positive eigenvalues approaching 1 (the
    photon-scattering analogy).  At stretched H₂ the dominant CPKS
    eigenvalue can go strongly negative; in that case ``|1 − λ|`` is
    large and the operator-norm amplification is small.  This is
    consistent with the local linear analysis but inverts the SPEC's
    monotonicity prediction — the empirical sweep in
    :file:`examples/measure_scf_amplification.py` documents the actual
    behavior.  See BLOCKERS in the P2 completion report.
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError(
            f"jacobian must be square, got shape={jacobian.shape!r}."
        )
    n = jacobian.shape[0]
    M_inv_target = np.eye(n, dtype=np.float64) - jacobian
    try:
        # Invert directly; the matrix is small enough that LAPACK ``gesv``
        # is far cheaper than any iterative method.
        resolvent = np.linalg.inv(M_inv_target)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.linalg.norm(resolvent, ord=2))


def spectral_radius(jacobian: np.ndarray) -> float:
    """Spectral radius ``ρ(D𝒯) = max_k |λ_k(D𝒯)|``.

    Computes all eigenvalues via :func:`numpy.linalg.eigvals` (no
    symmetry assumption — the CPKS Jacobian is in general not symmetric)
    and returns the maximum absolute value.

    Parameters
    ----------
    jacobian : np.ndarray of shape (n, n)

    Returns
    -------
    rho : float
        Largest absolute eigenvalue.
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError(
            f"jacobian must be square, got shape={jacobian.shape!r}."
        )
    eigs = np.linalg.eigvals(jacobian)
    return float(np.max(np.abs(eigs)))
