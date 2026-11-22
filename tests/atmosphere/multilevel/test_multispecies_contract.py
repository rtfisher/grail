"""Phase S tests for multi-species rate equations and AtomicModel. SPEC_phase_s §6.

Test IDs:
  U-hhe-instantiation
  U-hhe-cross-species-rejected
  U-hhe-noncontiguous-rejected
  U-hhe-eps-intercomb-floor
  U-multispecies-rate-eq-singleblock-equivalence  (LOAD-BEARING)
  U-multispecies-block-detection
  U-multispecies-singular-block
  A-hhe-frequencies
  A-hhe-statistical-weights
  A-multispecies-block-conservation
  A-multispecies-noncoupling
  A-hhe-lte-boltzmann
  A-hhe-positivity
  A-hhe-2level-limit
  A-hhe-5level-limit  (degenerate He block)

SPEC_phase_s §6.1 (Tier U) and §6.2 (Tier A).
"""

from __future__ import annotations

import math

import pytest
import torch

from atmosphere.atomic_model import AtomicModel, Transition
from atmosphere.multilevel.rate_equations import rate_equation_solve_nlevel, _H, _K, _C


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_KB = 1.380649e-23     # Boltzmann constant [J/K]
_h_eV_s = 4.135667696e-15   # Planck constant [eV·s]
_R_H = 1.0973731568e7       # Rydberg constant [m^-1]
_C_SI = 2.99792458e8        # speed of light [m/s]

# NIST ASD 2024 He I level energies (eV, ground-relative)
_E_He_ground  = 0.0
_E_He_1s2s_1S = 20.616
_E_He_1s2p_1P = 21.218
_E_He_1s2s_3S = 19.820
_E_He_1s2p_3P = 20.964

_T_TEST = 8000.0   # K — canonical test temperature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rydberg_nu(n_l: int, n_u: int) -> float:
    return _R_H * _C_SI * (1.0 / n_l**2 - 1.0 / n_u**2)


def _he_nu_expected(E_upper_eV: float, E_lower_eV: float) -> float:
    return (E_upper_eV - E_lower_eV) / _h_eV_s


def _make_T_n_e(N_tau: int = 50, T: float = _T_TEST):
    T_t = torch.full((N_tau,), T, dtype=torch.float64)
    n_e_t = torch.full((N_tau,), 1e12, dtype=torch.float64)
    return T_t, n_e_t


def _make_J_bar(N_tau: int, K: int, J_val: float = 1e-20) -> torch.Tensor:
    return torch.full((N_tau, K), J_val, dtype=torch.float64)


# ---------------------------------------------------------------------------
# U-hhe-instantiation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# U-hhe-cross-species-rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_hhe_cross_species_rejected() -> None:
    """U-hhe-cross-species-rejected: cross-species Transition raises ValueError."""
    # H level 4 (index 4) <-> He level 5 (index 5) — across species boundary.
    cross_tr = Transition(lower=4, upper=5, nu_0=1e15, epsilon=0.1, label="cross")
    h_transitions = [
        Transition(0, 1, _rydberg_nu(1, 2), 1e-4, "Ly_alpha"),
        Transition(1, 2, _rydberg_nu(2, 3), 1e-4, "H_alpha"),
    ]
    with pytest.raises(ValueError, match="cross"):
        AtomicModel(
            n_levels=6,
            transitions=tuple(h_transitions + [cross_tr]),
            statistical_weights=(2, 8, 18, 1, 1, 1),
            label="bad_model",
            species=("H", "H", "H", "H", "H", "He"),
        )


# ---------------------------------------------------------------------------
# U-hhe-noncontiguous-rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_hhe_noncontiguous_rejected() -> None:
    """U-hhe-noncontiguous-rejected: non-contiguous species tuple raises ValueError."""
    # Species H, He, H — non-contiguous.
    with pytest.raises(ValueError, match="non-contiguous|contiguous"):
        AtomicModel(
            n_levels=3,
            transitions=(
                Transition(0, 1, _rydberg_nu(1, 2), 1e-4, "Ly_alpha"),
                Transition(1, 2, 1e15, 1e-4, "fake"),
            ),
            statistical_weights=(1, 1, 1),
            label="bad_noncontiguous",
            species=("H", "He", "H"),
        )


# ---------------------------------------------------------------------------
# U-hhe-eps-intercomb-floor
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# U-multispecies-rate-eq-singleblock-equivalence  (LOAD-BEARING)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_multispecies_singleblock_equivalence() -> None:
    """U-multispecies-rate-eq-singleblock-equivalence: species=('H',)*5 is byte-identical to species=None.

    LOAD-BEARING: guarantees the multi-species code path has zero impact on
    the existing 5-level H-only path.  Tolerance: 1e-12 relative.
    """
    N_tau = 50
    K = 10

    # Species=None (original M.2 path).
    model_none = AtomicModel.balmer_5level(epsilon=1e-4)
    assert model_none.species is None
    assert model_none.species_blocks is None

    # Species=("H",)*5 (single block, multi-species code path not triggered
    # because len(species_blocks) == 1, falls through to single-species).
    model_h5 = AtomicModel(
        n_levels=model_none.n_levels,
        transitions=model_none.transitions,
        statistical_weights=model_none.statistical_weights,
        label=model_none.label,
        species=("H",) * 5,
    )
    # species_blocks should be ((0, 5),) — one block.
    assert model_h5.species_blocks == ((0, 5),), (
        f"species_blocks={model_h5.species_blocks}"
    )

    T_t, n_e_t = _make_T_n_e(N_tau)
    J_bar = _make_J_bar(N_tau, K, J_val=1e-18)

    pops_none = rate_equation_solve_nlevel(J_bar, T_t, n_e_t, atomic_model=model_none)
    pops_h5   = rate_equation_solve_nlevel(J_bar, T_t, n_e_t, atomic_model=model_h5)

    max_rel_err = (
        (pops_none - pops_h5).abs()
        / pops_none.abs().clamp(min=1e-300)
    ).max().item()

    assert max_rel_err < 1e-12, (
        f"Single-block equivalence violated: max relative error = {max_rel_err:.2e} "
        f"(limit 1e-12). The multi-species code path diverges from the M.2 path."
    )


# ---------------------------------------------------------------------------
# U-multispecies-block-detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_multispecies_block_detection() -> None:
    """U-multispecies-block-detection: species=('H','H','H','He','He') -> blocks=((0,3),(3,5))."""
    # Build a small 5-level H+He model (3 H + 2 He).
    # Transitions: H: 0->1, 1->2; He: 3->4.
    nu_h01 = _rydberg_nu(1, 2)
    nu_h12 = _rydberg_nu(2, 3)
    nu_he  = _he_nu_expected(_E_He_1s2p_1P, _E_He_ground)

    model = AtomicModel(
        n_levels=5,
        transitions=(
            Transition(0, 1, nu_h01, 1e-4, "H01"),
            Transition(1, 2, nu_h12, 1e-4, "H12"),
            Transition(3, 4, nu_he,  1e-4, "He34"),
        ),
        statistical_weights=(2, 8, 18, 1, 3),
        label="test_5level_hhe",
        species=("H", "H", "H", "He", "He"),
    )

    assert model.species_blocks == ((0, 3), (3, 5)), (
        f"species_blocks={model.species_blocks}"
    )

    # Solve and verify block-diagonal structure: perturbing He J_bar
    # should not change H populations.
    N_tau = 20
    T_t, n_e_t = _make_T_n_e(N_tau)
    J_bar_base = _make_J_bar(N_tau, 3, J_val=1e-18)

    pops_base = rate_equation_solve_nlevel(
        J_bar_base, T_t, n_e_t, atomic_model=model,
        species_fractions={"H": 0.7, "He": 0.3},
    )

    # Perturb only the He transition (k=2).
    J_bar_perturb = J_bar_base.clone()
    J_bar_perturb[:, 2] = 1e-15   # large perturbation on He line

    pops_perturb = rate_equation_solve_nlevel(
        J_bar_perturb, T_t, n_e_t, atomic_model=model,
        species_fractions={"H": 0.7, "He": 0.3},
    )

    # H populations (indices 0..2) must be unchanged.
    h_diff = (pops_base[:, :3] - pops_perturb[:, :3]).abs().max().item()
    assert h_diff < 1e-12, (
        f"H populations changed when He J_bar was perturbed: max diff = {h_diff:.2e}"
    )

    # He populations (indices 3..4) must have changed.
    he_diff = (pops_base[:, 3:] - pops_perturb[:, 3:]).abs().max().item()
    assert he_diff > 1e-20, (
        f"He populations did not change when He J_bar was perturbed: "
        f"max diff = {he_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# U-multispecies-singular-block
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_multispecies_singular_block() -> None:
    """U-multispecies-singular-block: disconnected level in a species block raises ValueError."""
    # 4-level model: H(0,1) + He(2,3).
    # Only one He transition (2->3) but He has a disconnected level if we
    # add a 5th level in the He block with no transitions.
    nu_h = _rydberg_nu(1, 2)
    nu_he = _he_nu_expected(_E_He_1s2p_1P, _E_He_ground)

    model = AtomicModel(
        n_levels=5,
        transitions=(
            Transition(0, 1, nu_h,  1e-4, "H01"),
            Transition(2, 3, nu_he, 1e-4, "He23"),
            # He level 4 has NO transitions — disconnected within the He block.
        ),
        statistical_weights=(2, 8, 1, 3, 1),
        label="test_singular_he_block",
        species=("H", "H", "He", "He", "He"),
    )

    N_tau = 10
    T_t, n_e_t = _make_T_n_e(N_tau)
    J_bar = _make_J_bar(N_tau, 2, J_val=1e-18)

    with pytest.raises(ValueError, match="singular"):
        rate_equation_solve_nlevel(
            J_bar, T_t, n_e_t, atomic_model=model,
            species_fractions={"H": 0.9, "He": 0.1},
        )


# ---------------------------------------------------------------------------
# A-hhe-frequencies
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# A-hhe-statistical-weights
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# A-multispecies-block-conservation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# A-multispecies-noncoupling
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# A-hhe-lte-boltzmann
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# A-hhe-positivity
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# A-hhe-2level-limit
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# A-hhe-5level-limit
# ---------------------------------------------------------------------------
