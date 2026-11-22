"""SCF physical seams and equation diagnostics.

All norms here use the ordinary Euclidean norm of the unweighted packed
upper triangles (alpha then beta).  Off-diagonal entries have weight one;
this is different from the AO Frobenius norm.  Constraint bases are
orthonormal in these same packed coordinates.
"""
from __future__ import annotations

import numpy as np
import torch

from .fock_build import _build_mf, _build_mol
from .state import pack_density_matrix, unpack_density_matrix


def pack_pair(pair):
    return pack_density_matrix(tuple(torch.as_tensor(p) for p in pair), method="uhf").numpy()


def make_cycle_scf(state, fock_builder):
    """Return the actual unaccelerated cycle in packed coordinates."""
    from pyscf.scf import hf
    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    M = S.shape[0]

    def cycle(x):
        pair = tuple(p.numpy() for p in unpack_density_matrix(
            torch.as_tensor(x, dtype=torch.float64), M=M, method=state.method))
        result = []
        for F, n_occ in zip(fock_builder(state, pair), mol.nelec):
            _, C = hf.eig(0.5 * (F + F.T), S)
            result.append(C[:, :n_occ] @ C[:, :n_occ].T)
        return pack_pair(result)
    return cycle


def tangent_bases_scf(state, mo_coeff_tuple):
    """Return electron-count and occupied–virtual tangent bases.

    Idempotency means P S P=P per spin, with Tr(P S)=N_sigma.  The
    occupied–virtual basis differentiates those projectors.  This is a
    linear tangent at the given reference orbitals, not a finite-distance
    projection between distinct branches.
    """
    mol = _build_mol(state)
    S = np.asarray(_build_mf(state, mol=mol).get_ovlp(), dtype=np.float64)
    M = len(S)
    rows, cols = np.triu_indices(M)
    nu = len(rows)
    weights = np.where(rows == cols, 1., 2.)
    constraints = np.zeros((2, 2 * nu))
    for spin in range(2):
        constraints[spin, spin * nu:(spin + 1) * nu] = S[rows, cols] * weights
    _, _, Vh = np.linalg.svd(constraints, full_matrices=True)
    Q_count = Vh[2:].T
    columns = []
    for spin, (C, n_occ) in enumerate(zip(mo_coeff_tuple, mol.nelec)):
        for i in range(n_occ):
            for a in range(n_occ, M):
                ov = np.outer(C[:, a], C[:, i])
                col = np.zeros(2 * nu)
                col[spin * nu:(spin + 1) * nu] = (ov + ov.T)[rows, cols]
                columns.append(col)
    Q_projector = (np.linalg.qr(np.column_stack(columns), mode="reduced")[0]
                   if columns else np.empty((2 * nu, 0)))
    return Q_count, Q_projector


def resolvent_metrics_scf(K, Q_count, Q_projector):
    """Return explicitly labeled ambient and restricted inverse 2-norms."""
    def inverse_norm(A):
        return float(1. / np.linalg.svd(A, compute_uv=False)[-1])
    dim = len(K)
    values = {"ambient_packed_inverse_norm": inverse_norm(np.eye(dim) - K)}
    for name, Q in (("electron_count", Q_count), ("projector", Q_projector)):
        values[f"{name}_tangent_inverse_norm"] = inverse_norm(np.eye(Q.shape[1]) - Q.T @ K @ Q)
        values[f"{name}_tangent_invariance_defect"] = float(np.linalg.norm((np.eye(dim) - Q @ Q.T) @ K @ Q))
    positive = [float(z.real) for z in np.linalg.eigvals(K)
                if abs(z.imag) < 1e-9 and z.real > 0]
    values["positive_eigenvalue_proxy"] = 1. / (1. - max(positive)) if positive else None
    return values
