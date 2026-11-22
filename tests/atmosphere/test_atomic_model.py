"""Tests for AtomicModel.balmer_5level() constructor. M.1 acceptance gates.

Test IDs: A-balmer-frequencies, A-balmer-statistical-weights,
          A-balmer-instantiation, U-balmer-per-line-epsilon.

SPEC_nlte_scaleup §M.1 acceptance criteria.
"""

from __future__ import annotations

import math

import pytest

from atmosphere.atomic_model import AtomicModel, Transition


# ---------------------------------------------------------------------------
# Physical constants for Rydberg reference computation
# ---------------------------------------------------------------------------

_R_H = 1.0973731568e7   # Rydberg constant [m^-1]
_C = 2.99792458e8       # speed of light [m/s]


def _rydberg_nu(n_l: int, n_u: int) -> float:
    """Rydberg formula for transition frequency between levels n_l and n_u."""
    return _R_H * _C * (1.0 / n_l**2 - 1.0 / n_u**2)


# ---------------------------------------------------------------------------
# Canonical transition list for reference comparison
# ---------------------------------------------------------------------------

# (lower_n, upper_n) in physical level numbering (1-based)
_CANONICAL_TRANSITIONS = [
    (1, 2),  # Ly_alpha
    (1, 3),  # Ly_beta
    (1, 4),  # Ly_gamma
    (1, 5),  # Ly_delta
    (2, 3),  # H_alpha
    (2, 4),  # H_beta
    (2, 5),  # H_gamma
    (3, 4),  # Pa_alpha
    (3, 5),  # Pa_beta
    (4, 5),  # Br_alpha
]

_CANONICAL_LABELS = [
    "Ly_alpha", "Ly_beta", "Ly_gamma", "Ly_delta",
    "H_alpha", "H_beta", "H_gamma",
    "Pa_alpha", "Pa_beta",
    "Br_alpha",
]


# ---------------------------------------------------------------------------
# A-balmer-frequencies
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_balmer_frequencies() -> None:
    """A-balmer-frequencies: all 10 nu_0 values match Rydberg to < 1e-5 relative.

    SPEC_nlte_scaleup §M.1 acceptance criterion.
    """
    model = AtomicModel.balmer_5level(epsilon=1e-4)

    assert len(model.transitions) == 10, (
        f"Expected 10 transitions, got {len(model.transitions)}"
    )

    for k, ((n_l, n_u), tr) in enumerate(
        zip(_CANONICAL_TRANSITIONS, model.transitions)
    ):
        nu_expected = _rydberg_nu(n_l, n_u)
        nu_actual = tr.nu_0
        rel_err = abs(nu_actual - nu_expected) / nu_expected
        assert rel_err < 1e-5, (
            f"Transition k={k} ({n_l}->{n_u}): "
            f"nu_actual={nu_actual:.6e} Hz, nu_expected={nu_expected:.6e} Hz, "
            f"rel_err={rel_err:.2e} (limit 1e-5)"
        )


# ---------------------------------------------------------------------------
# A-balmer-statistical-weights
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_balmer_statistical_weights() -> None:
    """A-balmer-statistical-weights: g_n = 2 n^2 for n = 1..5.

    Expected: (2, 8, 18, 32, 50).
    SPEC_nlte_scaleup §M.1 acceptance criterion.
    """
    model = AtomicModel.balmer_5level(epsilon=1e-4)

    expected_weights = (2, 8, 18, 32, 50)
    assert model.statistical_weights == expected_weights, (
        f"Statistical weights: got {model.statistical_weights}, "
        f"expected {expected_weights}"
    )

    # Also verify the formula g_n = 2 n^2 explicitly.
    for idx, g in enumerate(model.statistical_weights):
        n = idx + 1  # physical level number (1-based)
        g_expected = 2 * n**2
        assert g == g_expected, (
            f"Level n={n} (index {idx}): g={g}, expected 2*{n}^2={g_expected}"
        )


# ---------------------------------------------------------------------------
# A-balmer-instantiation
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_balmer_instantiation() -> None:
    """A-balmer-instantiation: balmer_5level(epsilon=1e-4) properties.

    Verifies n_levels == 5, n_transitions == 10, frozen (hashable),
    and correct label.
    SPEC_nlte_scaleup §M.1 acceptance criterion.
    """
    model = AtomicModel.balmer_5level(epsilon=1e-4)

    assert model.n_levels == 5, f"n_levels={model.n_levels}, expected 5"
    assert model.K == 10, f"K (n_transitions)={model.K}, expected 10"
    assert model.label == "balmer_5level", f"label={model.label!r}"

    # Frozen dataclass -> hashable
    h = hash(model)
    assert isinstance(h, int), "hash(model) must return an int"

    # Level indices are 0-based: check bounds
    for tr in model.transitions:
        assert 0 <= tr.lower < model.n_levels, (
            f"Transition lower={tr.lower} out of [0, {model.n_levels})"
        )
        assert 0 <= tr.upper < model.n_levels, (
            f"Transition upper={tr.upper} out of [0, {model.n_levels})"
        )
        assert tr.upper > tr.lower, (
            f"Transition upper={tr.upper} must be > lower={tr.lower}"
        )


# ---------------------------------------------------------------------------
# U-balmer-per-line-epsilon
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_balmer_per_line_epsilon_accepts_tuple() -> None:
    """U-balmer-per-line-epsilon (part 1): per-line epsilon tuple of length 10 accepted.

    Each transition should carry its own epsilon value.
    SPEC_nlte_scaleup §M.1 acceptance criterion.
    """
    eps_values = tuple(1e-2 * (k + 1) / 10 for k in range(10))
    # Make sure all are in (0, 1]
    eps_values = tuple(min(e, 1.0) for e in eps_values)
    model = AtomicModel.balmer_5level(epsilon=eps_values)

    for k, (tr, expected_eps) in enumerate(
        zip(model.transitions, eps_values)
    ):
        assert abs(tr.epsilon - expected_eps) < 1e-15, (
            f"Transition k={k}: stored epsilon={tr.epsilon}, "
            f"expected {expected_eps}"
        )


@pytest.mark.unit
def test_balmer_per_line_epsilon_wrong_length_raises() -> None:
    """U-balmer-per-line-epsilon (part 2): wrong-length sequence raises ValueError.

    SPEC_nlte_scaleup §M.1 acceptance criterion.
    """
    with pytest.raises(ValueError, match="length"):
        AtomicModel.balmer_5level(epsilon=(1e-2,) * 9)  # 9 != 10


@pytest.mark.unit
def test_balmer_per_line_epsilon_scalar_broadcast() -> None:
    """U-balmer-per-line-epsilon (part 3): scalar epsilon broadcast to all 10 lines.

    SPEC_nlte_scaleup §M.1 acceptance criterion.
    """
    eps = 5e-3
    model = AtomicModel.balmer_5level(epsilon=eps)
    for k, tr in enumerate(model.transitions):
        assert abs(tr.epsilon - eps) < 1e-15, (
            f"Transition k={k}: stored epsilon={tr.epsilon}, expected {eps}"
        )
