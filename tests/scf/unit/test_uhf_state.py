"""Tier U tests for the UHF/UKS extension of scf/state.py.  SPEC_scf §8.10 R1.

Covers:
- ``MoleculeState(method="uhf")`` constructs cleanly with the broken-symmetry
  initial-guess field set.
- ``pack_density_matrix((P_alpha, P_beta))`` round-trips through
  ``unpack_density_matrix`` for STO-3G H₂ (M=2, doubled-flat length 6).
- Symmetry of each spin block separately is preserved through pack/unpack.
- The closed-shell pack/unpack signature is back-compatible (no method
  keyword required for RHF/RKS callers).

These tests do not require PySCF; state.py is pure numpy/torch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scf.state import (
    MoleculeState,
    n_unique,
    pack_density_matrix,
    unpack_density_matrix,
)


# ---------------------------------------------------------------------------
# MoleculeState construction with UHF / UKS
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_molecule_state_uhf_constructs() -> None:
    """A minimal H₂ UHF MoleculeState constructs without error."""
    state = MoleculeState(
        coords=np.array([[0.0, 0.0, -0.65], [0.0, 0.0, 0.65]], dtype=np.float64),
        atom_symbols=("H", "H"),
        basis="sto-3g",
        n_electrons=2,
        method="uhf",
        init_guess="breaksym",
    )
    assert state.method == "uhf"
    assert state.init_guess == "breaksym"
    assert state.spin == 0
    assert state.coords.shape == (2, 3)


@pytest.mark.unit
def test_molecule_state_uks_constructs() -> None:
    """A minimal H₂ UKS MoleculeState constructs without error."""
    state = MoleculeState(
        coords=np.array([[0.0, 0.0, -0.65], [0.0, 0.0, 0.65]], dtype=np.float64),
        atom_symbols=("H", "H"),
        basis="sto-3g",
        n_electrons=2,
        method="uks",
        xc_functional="B3LYP",
        init_guess="breaksym",
    )
    assert state.method == "uks"
    assert state.xc_functional == "B3LYP"
    assert state.init_guess == "breaksym"


@pytest.mark.unit
def test_molecule_state_rejects_bad_init_guess() -> None:
    """init_guess must be 'minao' or 'breaksym'."""
    with pytest.raises(ValueError, match="init_guess must be"):
        MoleculeState(
            coords=np.zeros((2, 3), dtype=np.float64),
            atom_symbols=("H", "H"),
            basis="sto-3g",
            n_electrons=2,
            method="uhf",
            init_guess="atom",
        )


@pytest.mark.unit
def test_molecule_state_uhf_default_init_guess_is_minao() -> None:
    """UHF MoleculeState without an explicit init_guess defaults to 'minao'.

    The 'breaksym' default is applied by ``make_h2_state`` (the convenience
    constructor), not by ``MoleculeState`` itself — the dataclass stays
    PySCF-naïve.
    """
    state = MoleculeState(
        coords=np.zeros((2, 3), dtype=np.float64),
        atom_symbols=("H", "H"),
        basis="sto-3g",
        n_electrons=2,
        method="uhf",
    )
    assert state.init_guess == "minao"


# ---------------------------------------------------------------------------
# pack / unpack round-trip for UHF doubled seam
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pack_uhf_returns_doubled_seam_length_M2() -> None:
    """For UHF with M=2, pack outputs length 6 (2 * M*(M+1)/2)."""
    P_alpha = torch.tensor([[1.0, 0.3], [0.3, 0.7]], dtype=torch.float64)
    P_beta = torch.tensor([[0.7, 0.3], [0.3, 1.0]], dtype=torch.float64)
    flat = pack_density_matrix((P_alpha, P_beta), method="uhf")
    assert flat.shape == (6,)
    assert flat.dtype == torch.float64
    # α block is the first 3 entries.
    assert torch.allclose(
        flat[:3], torch.tensor([1.0, 0.3, 0.7], dtype=torch.float64)
    )
    # β block is the last 3 entries.
    assert torch.allclose(
        flat[3:], torch.tensor([0.7, 0.3, 1.0], dtype=torch.float64)
    )


@pytest.mark.unit
@pytest.mark.parametrize("M", [2, 4])
@pytest.mark.parametrize("method", ["uhf", "uks"])
def test_pack_unpack_uhf_roundtrip(M: int, method: str) -> None:
    """pack/unpack round-trip for UHF/UKS produces α and β separately."""
    rng = np.random.default_rng(7 + M)
    A = rng.standard_normal((M, M))
    B = rng.standard_normal((M, M))
    P_alpha = 0.5 * (A + A.T)
    P_beta = 0.5 * (B + B.T)
    P_alpha_t = torch.from_numpy(P_alpha)
    P_beta_t = torch.from_numpy(P_beta)

    flat = pack_density_matrix((P_alpha_t, P_beta_t), method=method)
    assert flat.shape == (2 * n_unique(M),)

    out = unpack_density_matrix(flat, M=M, method=method)
    assert isinstance(out, tuple)
    assert len(out) == 2
    P_alpha_back, P_beta_back = out
    assert P_alpha_back.shape == (M, M)
    assert P_beta_back.shape == (M, M)

    # Round-trip identity to machine precision per spin.
    assert torch.allclose(P_alpha_back, P_alpha_t, atol=1e-15, rtol=0.0)
    assert torch.allclose(P_beta_back, P_beta_t, atol=1e-15, rtol=0.0)

    # Each spin block must be exactly symmetric.
    assert torch.allclose(P_alpha_back, P_alpha_back.T, atol=0.0, rtol=0.0)
    assert torch.allclose(P_beta_back, P_beta_back.T, atol=0.0, rtol=0.0)


@pytest.mark.unit
def test_pack_uhf_rejects_single_matrix_input() -> None:
    """pack(method='uhf') with a single matrix instead of a tuple raises."""
    P = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="2-tuple"):
        pack_density_matrix(P, method="uhf")


@pytest.mark.unit
def test_pack_rhf_rejects_tuple_input() -> None:
    """pack(method='rhf') with a 2-tuple instead of a matrix raises."""
    P_alpha = torch.eye(2, dtype=torch.float64)
    P_beta = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="single"):
        pack_density_matrix((P_alpha, P_beta), method="rhf")


@pytest.mark.unit
def test_unpack_uhf_rejects_wrong_length() -> None:
    """unpack(method='uhf') with the closed-shell length raises."""
    flat = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="p_flat must have shape"):
        unpack_density_matrix(flat, M=2, method="uhf")


@pytest.mark.unit
def test_pack_unpack_uhf_accepts_numpy_input() -> None:
    """UHF pack/unpack accept numpy arrays as well as torch tensors."""
    P_alpha = np.array([[1.2, 0.1], [0.1, 0.8]], dtype=np.float64)
    P_beta = np.array([[0.8, 0.05], [0.05, 1.2]], dtype=np.float64)
    flat = pack_density_matrix((P_alpha, P_beta), method="uhf")
    out = unpack_density_matrix(flat.numpy(), M=2, method="uhf")
    P_alpha_back, P_beta_back = out
    assert torch.allclose(P_alpha_back, torch.from_numpy(P_alpha), atol=1e-15)
    assert torch.allclose(P_beta_back, torch.from_numpy(P_beta), atol=1e-15)


@pytest.mark.unit
def test_pack_rhf_back_compat_no_method_keyword() -> None:
    """Closed-shell pack still works without specifying ``method`` (P1 callers)."""
    P = torch.tensor([[1.5, 0.25], [0.25, 0.75]], dtype=torch.float64)
    p_flat = pack_density_matrix(P)  # method="rhf" (default)
    assert p_flat.shape == (3,)
    assert torch.allclose(
        p_flat, torch.tensor([1.5, 0.25, 0.75], dtype=torch.float64)
    )
    # Unpack also back-compat.
    P_back = unpack_density_matrix(p_flat, M=2)
    assert P_back.shape == (2, 2)
    assert torch.allclose(P_back, P, atol=1e-15)
