"""Truth Fock-matrix construction via PySCF.  SPEC_scf §3.2, §4.3, §8.10.

The truth Fock matrix is

    F[P] = h_core + J[P] + V_xc^B3LYP[P]    (RKS, the production default)
    F[P] = h_core + J[P] +     K[P]         (RHF; V_xc = 0, K is Hartree–Fock
                                              exchange, only the Szabo–Ostlund
                                              analytic anchor uses this)

For UHF/UKS (R1, SPEC_scf §8.10) the input density is the pair
``(P_alpha, P_beta)`` and the output is the pair ``(F_alpha, F_beta)``:

    F_σ[P_α, P_β] = h_core + J[P_α + P_β] + V_xc^σ[P_α, P_β]  (UKS)
    F_σ[P_α, P_β] = h_core + J[P_α + P_β] − K[P_σ]            (UHF)

In both cases the spins share the Hartree (Coulomb J) potential — built
from the total density ``P_α + P_β`` — but each receives its own exchange
or XC contribution.  PySCF's ``mf.get_veff(mol, dm=(P_α, P_β))`` returns
the spin-resolved ``V_eff`` of shape ``(2, M, M)``.

The truth Fock-build is a thin adapter over PySCF's machinery: we use
``mol.intor`` for h_core integrals and ``mf.get_veff`` to obtain the
relevant ``V_eff`` in one call.  Reusing PySCF avoids re-implementing
electron-repulsion integral contraction and the XC grid quadrature.

PySCF is imported inside functions, not at module top, to avoid OpenMP
load-order surprises at package-collection time (SPEC_scf §8.5).  The
``KMP_DUPLICATE_LIB_OK=TRUE`` guard in ``tests/conftest.py`` covers the
triple-init case (PyTorch + h5py + PySCF).
"""

from __future__ import annotations

import numpy as np

from .state import MoleculeState


# ---------------------------------------------------------------------------
# PySCF Mole / mean-field-object construction (private helpers).
# ---------------------------------------------------------------------------

def _build_mol(state: MoleculeState):
    """Construct the PySCF :class:`pyscf.gto.Mole` for ``state``.

    The Mole is built fresh every call: it is cheap (microseconds for H₂
    minimal-basis) and avoids mutable cache state at the package boundary.
    Callers that need to amortize this cost should cache the result.

    Coordinates are passed in Angstrom (PySCF's ``unit='Angstrom'``); the
    spin/charge fields are forwarded from ``state``.

    Returns
    -------
    mol : pyscf.gto.Mole
        Built Mole object (``mol.build()`` already invoked).
    """
    # Inline import: keep PySCF off the package-collection import path
    # (OpenMP load-order, SPEC_scf §8.5).
    from pyscf import gto

    atom_lines = [
        (sym, tuple(float(c) for c in xyz))
        for sym, xyz in zip(state.atom_symbols, state.coords)
    ]
    mol = gto.Mole()
    mol.atom = atom_lines
    mol.basis = state.basis
    mol.charge = state.charge
    mol.spin = state.spin
    mol.unit = "Angstrom"
    mol.verbose = 0   # quiet PySCF; production code chooses verbosity locally
    mol.build()
    return mol


def _build_mf(state: MoleculeState, mol=None):
    """Construct a PySCF mean-field object matching ``state.method``.

    Returns an :class:`pyscf.scf.RKS` for ``method="rks"`` (with
    ``xc=state.xc_functional``), an :class:`pyscf.scf.RHF` for
    ``method="rhf"``, an :class:`pyscf.scf.UHF` for ``method="uhf"``, or
    an :class:`pyscf.dft.UKS` for ``method="uks"`` (with
    ``xc=state.xc_functional``).  The object is NOT run (no ``kernel()``
    call here); it carries the integral machinery and the get_veff method
    we need.

    Parameters
    ----------
    state : MoleculeState
    mol : pyscf.gto.Mole or None
        Pre-built Mole; if None, a fresh one is built.
    """
    from pyscf import scf, dft

    if mol is None:
        mol = _build_mol(state)

    if state.method == "rks":
        mf = dft.RKS(mol)
        mf.xc = state.xc_functional
    elif state.method == "rhf":
        mf = scf.RHF(mol)
    elif state.method == "uhf":
        mf = scf.UHF(mol)
    elif state.method == "uks":
        mf = dft.UKS(mol)
        mf.xc = state.xc_functional
    else:
        # Should be unreachable: MoleculeState.__post_init__ validates this.
        raise ValueError(f"unknown method {state.method!r}.")
    mf.verbose = 0
    return mf


# ---------------------------------------------------------------------------
# Public Fock-build entry point.
# ---------------------------------------------------------------------------

def build_fock_truth(
    state: MoleculeState,
    P: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build the truth Fock matrix at density ``P``.

    Closed-shell (RHF/RKS): ``P`` is a single ``(M, M)`` symmetric matrix
    and the return is ``F = h_core + J[P] + V_xc^B3LYP[P]`` (RKS) or
    ``F = h_core + J[P] + K[P]`` (RHF), a single symmetric ``(M, M)``
    ndarray.

    Open-shell (UHF/UKS, R1; SPEC_scf §8.10): ``P`` is a 2-tuple
    ``(P_alpha, P_beta)`` of symmetric ``(M, M)`` matrices and the return
    is the pair ``(F_alpha, F_beta)``.  Each spin sees the same Hartree
    potential ``J[P_α + P_β]`` plus its own exchange (UHF: ``−K[P_σ]``) or
    its own spin component of the XC potential (UKS).  PySCF's
    ``mf.get_veff`` with a tuple ``dm`` returns the spin-resolved ``V_eff``
    of shape ``(2, M, M)``.

    Parameters
    ----------
    state : MoleculeState
        Carries the molecule, basis, method, and (for RKS/UKS) the XC
        functional name.
    P : np.ndarray of shape (M, M), or 2-tuple of two such for UHF/UKS
        AO-basis density matrix.  Should be symmetric; symmetry is gently
        re-imposed before passing to PySCF by symmetrizing ``(P+P.T)/2``,
        because PySCF's ``get_veff`` keys some XC paths off Hermiticity and
        a numerically lopsided input can return a numerically lopsided
        output (we then re-symmetrize on the output side too).

    Returns
    -------
    F : np.ndarray of shape (M, M), float64 (closed-shell), OR
        2-tuple ``(F_alpha, F_beta)`` of symmetric ``(M, M)`` ndarrays
        (open-shell).  Output type mirrors input type.

    Notes
    -----
    Float64 throughout.  This is the canonical truth Fock-builder; the P3
    NN-replaced builder substitutes ``V_xc^NN`` for the XC piece and lives
    in ``scf/fock_build.py::make_fock_builder_with_nn`` (P3 / P4).
    """
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)

    # h_core = kinetic + nuclear-attraction.  PySCF computes both via the
    # libcint engine; ``get_hcore`` returns the sum.
    h_core = np.asarray(mf.get_hcore(), dtype=np.float64)

    if state.method in ("uhf", "uks"):
        # Open-shell branch.  Accept a 2-tuple input only; reject single-
        # matrix inputs because the seam is unambiguously different.
        if not (isinstance(P, tuple) and len(P) == 2):
            raise ValueError(
                f"build_fock_truth(method={state.method!r}) requires a 2-tuple "
                f"(P_alpha, P_beta); got input of type {type(P).__name__}."
            )
        P_alpha = np.asarray(P[0], dtype=np.float64)
        P_beta = np.asarray(P[1], dtype=np.float64)
        for label, mat in (("P_alpha", P_alpha), ("P_beta", P_beta)):
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"build_fock_truth: {label} must be square (M, M), "
                    f"got shape={mat.shape!r}."
                )
        # Defensive symmetrize each spin block.
        P_alpha_sym = 0.5 * (P_alpha + P_alpha.T)
        P_beta_sym = 0.5 * (P_beta + P_beta.T)
        dm = np.array([P_alpha_sym, P_beta_sym], dtype=np.float64)

        # PySCF UHF/UKS get_veff returns spin-resolved (2, M, M) V_eff.
        v_eff = np.asarray(mf.get_veff(mol, dm=dm), dtype=np.float64)
        F_alpha = h_core + v_eff[0]
        F_beta = h_core + v_eff[1]
        # Final symmetrize per spin (XC quadrature can leak ~1e-14 noise).
        F_alpha = 0.5 * (F_alpha + F_alpha.T)
        F_beta = 0.5 * (F_beta + F_beta.T)
        return F_alpha, F_beta

    # Closed-shell branch.
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(
            f"build_fock_truth requires square (M, M) P, got shape={P.shape!r}."
        )
    # Re-symmetrize defensively (see docstring).
    P_sym = 0.5 * (P + P.T)

    # V_eff = J[P] + V_xc[P] for RKS, or J[P] + K[P] for RHF.  PySCF's
    # ``get_veff`` dispatches on the mean-field type.
    v_eff = np.asarray(mf.get_veff(mol, dm=P_sym), dtype=np.float64)

    F = h_core + v_eff
    # Final symmetrize: a few thousand quadrature-grid evaluations of the
    # exchange–correlation potential can leak ~1e-14 antisymmetric noise;
    # the Roothaan–Hall step calls torch.linalg.eigh which expects
    # exact Hermiticity.
    F = 0.5 * (F + F.T)
    return F
