"""Tier U tests for scf/state.py.  SPEC_scf §6 P1.

Covers:
- MoleculeState construction and validation.
- pack_density_matrix ∘ unpack_density_matrix round-trip on random symmetric
  matrices for M=2 (STO-3G H₂) and M=10 (representative larger basis).

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
# MoleculeState construction
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_molecule_state_h2_default() -> None:
    """A minimal H₂ MoleculeState constructs without error."""
    state = MoleculeState(
        coords=np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]], dtype=np.float64),
        atom_symbols=("H", "H"),
        basis="sto-3g",
        n_electrons=2,
    )
    assert state.method == "rks"
    assert state.xc_functional == "B3LYP"
    assert state.charge == 0
    assert state.spin == 0
    assert state.coords.shape == (2, 3)


@pytest.mark.unit
def test_molecule_state_validation_rejects_bad_method() -> None:
    """method must be one of the supported values.

    After R1 (SPEC_scf §8.10) the supported set is {'rks', 'rhf', 'uhf', 'uks'};
    this test checks that a genuinely-unknown method is rejected.
    """
    with pytest.raises(ValueError, match="method must be one of"):
        MoleculeState(
            coords=np.zeros((2, 3), dtype=np.float64),
            atom_symbols=("H", "H"),
            basis="sto-3g",
            n_electrons=2,
            method="cassscf",
        )


@pytest.mark.unit
def test_molecule_state_validation_rejects_odd_closed_shell() -> None:
    """Closed shell (spin=0) requires even n_electrons."""
    with pytest.raises(ValueError, match="closed-shell"):
        MoleculeState(
            coords=np.zeros((1, 3), dtype=np.float64),
            atom_symbols=("H",),
            basis="sto-3g",
            n_electrons=1,
            spin=0,
        )


@pytest.mark.unit
def test_molecule_state_validation_rejects_bad_coords_shape() -> None:
    """coords must be (n_atoms, 3)."""
    with pytest.raises(ValueError, match="coords must have shape"):
        MoleculeState(
            coords=np.zeros((2, 2), dtype=np.float64),
            atom_symbols=("H", "H"),
            basis="sto-3g",
            n_electrons=2,
        )


@pytest.mark.unit
def test_molecule_state_validation_rejects_symbol_count_mismatch() -> None:
    """len(atom_symbols) must equal n_atoms."""
    with pytest.raises(ValueError, match="len\\(atom_symbols\\)"):
        MoleculeState(
            coords=np.zeros((2, 3), dtype=np.float64),
            atom_symbols=("H",),
            basis="sto-3g",
            n_electrons=2,
        )


@pytest.mark.unit
def test_molecule_state_hash_and_equality() -> None:
    """Two MoleculeStates with identical fields hash and compare equal."""
    coords = np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]], dtype=np.float64)
    s1 = MoleculeState(
        coords=coords, atom_symbols=("H", "H"),
        basis="sto-3g", n_electrons=2,
    )
    s2 = MoleculeState(
        coords=coords.copy(), atom_symbols=("H", "H"),
        basis="sto-3g", n_electrons=2,
    )
    assert s1 == s2
    assert hash(s1) == hash(s2)


# ---------------------------------------------------------------------------
# pack / unpack round-trip
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_n_unique_helper() -> None:
    """n_unique(M) returns M*(M+1)/2."""
    assert n_unique(2) == 3
    assert n_unique(4) == 10
    assert n_unique(10) == 55


@pytest.mark.unit
@pytest.mark.parametrize("M", [2, 10])
def test_pack_unpack_roundtrip_random_symmetric(M: int) -> None:
    """pack(unpack(p)) and unpack(pack(P)) are identities to float64 precision.

    Builds a random symmetric (M, M) tensor, packs it, unpacks it, and checks
    equality to within 1e-15 (essentially machine precision for float64).
    """
    rng = np.random.default_rng(42 + M)
    A = rng.standard_normal((M, M)).astype(np.float64)
    P = 0.5 * (A + A.T)
    P_t = torch.from_numpy(P)

    p_flat = pack_density_matrix(P_t)
    assert p_flat.shape == (M * (M + 1) // 2,)
    assert p_flat.dtype == torch.float64

    P_back = unpack_density_matrix(p_flat, M=M)
    assert P_back.shape == (M, M)
    assert P_back.dtype == torch.float64

    # Round-trip identity to machine precision.
    assert torch.allclose(P_back, P_t, atol=1e-15, rtol=0.0)

    # The reconstructed P must be exactly symmetric.
    assert torch.allclose(P_back, P_back.T, atol=0.0, rtol=0.0)


@pytest.mark.unit
def test_pack_specific_M2_convention() -> None:
    """Explicit check that M=2 packs to [P[0,0], P[0,1], P[1,1]]."""
    P = torch.tensor([[1.5, 0.25], [0.25, 0.75]], dtype=torch.float64)
    p_flat = pack_density_matrix(P)
    assert torch.allclose(
        p_flat, torch.tensor([1.5, 0.25, 0.75], dtype=torch.float64)
    )


@pytest.mark.unit
def test_unpack_rejects_wrong_length() -> None:
    """unpack raises if p_flat length is inconsistent with M."""
    p_flat = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="p_flat must have shape"):
        unpack_density_matrix(p_flat, M=2)


@pytest.mark.unit
def test_pack_rejects_nonsquare() -> None:
    """pack raises on a non-square input."""
    bad = torch.zeros((2, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="requires a square"):
        pack_density_matrix(bad)


@pytest.mark.unit
def test_pack_accepts_numpy_input() -> None:
    """pack/unpack accept numpy arrays in addition to torch Tensors."""
    P = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    p_flat = pack_density_matrix(P)
    P_back = unpack_density_matrix(p_flat.numpy(), M=2)
    assert torch.allclose(
        P_back, torch.from_numpy(P), atol=1e-15, rtol=0.0
    )
