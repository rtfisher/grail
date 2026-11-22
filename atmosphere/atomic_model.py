"""Atomic-model abstraction: Transition and AtomicModel dataclasses. SPEC_threelevel §6.4.

These frozen dataclasses carry the atomic data (transitions, statistical weights,
photon destruction probabilities) independently of the thermodynamic profiles (T, n_e)
that live on Atmosphere.  Immutability and hashability are important: AtomicModel
instances are used as dictionary keys and compared for equality in tests.

Design rationale (SPEC_threelevel §4.2):
  - A single generic Atmosphere class stores an AtomicModel field.  Per-setup
    constructors (two_level_*, three_level_*) build the appropriate model internally.
  - This avoids a TwoLevelAtmosphere / ThreeLevelAtmosphere class hierarchy that
    would duplicate grid plumbing and force GRAIL-side isinstance branching.
  - The AtomicModel does NOT carry T or n_e; those belong to Atmosphere.

Multi-species extension (SPEC_phase_s §2, v1.1):
  - Optional ``species`` field (tuple of length n_levels) labels each level with
    its species string (e.g. "H" or "He").
  - Species labels must form contiguous blocks (no interleaving).
  - Cross-species transitions are rejected in __post_init__.
  - ``species_blocks`` is derived automatically from ``species``.
  - The single-species path (species=None) is byte-identical to the original code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    """One bound-bound radiative transition.

    Parameters
    ----------
    lower : int
        Index of the lower level (0-based; 0 = ground state).
    upper : int
        Index of the upper level.  Must satisfy upper > lower.
    nu_0 : float
        Line-center frequency in Hz (rest frame).  Must be positive.
    epsilon : float
        Photon destruction probability for THIS line (master-ε per line,
        assumption A10).  Must satisfy 0 < epsilon <= 1.
    label : str
        Human-readable label, e.g. "Ly_alpha", "H_alpha".  Used in
        diagnostic output and HDF5 regression-file keys.
    """

    lower: int
    upper: int
    nu_0: float
    epsilon: float
    label: str


def _derive_blocks(
    species: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    """Derive contiguous species blocks from a species label tuple.

    Each block is a half-open (lo, hi) index pair such that
    ``species[lo:hi]`` is a homogeneous run of the same species label.
    If the same species label appears in more than one non-contiguous
    run, ValueError is raised (block-diagonality requires contiguity).

    Parameters
    ----------
    species : tuple of str
        Per-level species labels, length L.

    Returns
    -------
    blocks : tuple of (lo, hi) pairs
        Half-open index slices in ascending order.

    Raises
    ------
    ValueError
        If any species label is split across non-contiguous level index
        ranges (e.g., ("H", "He", "H", ...)).
    """
    if not species:
        return ()

    blocks: list[tuple[int, int]] = []
    seen_species: set[str] = set()

    lo = 0
    current = species[0]
    seen_species.add(current)

    for i in range(1, len(species)):
        if species[i] != current:
            # End of current run.
            blocks.append((lo, i))
            # Check that the new species has not appeared before (non-contiguous).
            if species[i] in seen_species:
                raise ValueError(
                    f"Species '{species[i]}' appears in non-contiguous blocks "
                    f"(seen at indices before {lo}, now reappearing at index {i}). "
                    f"Species labels must form contiguous blocks."
                )
            seen_species.add(species[i])
            current = species[i]
            lo = i

    # Close the final block.
    blocks.append((lo, len(species)))
    return tuple(blocks)


@dataclass(frozen=True)
class AtomicModel:
    """Atomic data for a multi-level NLTE setup.

    Parameters
    ----------
    n_levels : int
        Number of bound levels L.  Must satisfy n_levels >= 2.
    transitions : tuple[Transition, ...]
        K transitions.  Each must reference valid level indices in
        [0, n_levels).  Tuple (not list) for hashability.
    statistical_weights : tuple[int, ...]
        Per-level statistical weights g_l.  Length == n_levels.
        Default (1, 1, ..., 1) matches the 2-level prototype convention
        (assumption A10 absorbs level-structure detail into epsilon).
    label : str
        Human-readable label, e.g. "two_level_master_epsilon",
        "three_level_hydrogen_chained".
    species : tuple[str, ...] or None, default None
        Optional per-level species labels (e.g. ``("H","H","H","He","He")``).
        Length must equal n_levels when not None.  Species must be
        contiguous (no interleaving).  Cross-species transitions raise
        ValueError.  When None, single-species behaviour applies and the
        code path is byte-identical to the pre-Phase-S implementation.
    species_blocks : tuple[tuple[int, int], ...] or None
        Derived automatically from ``species`` in __post_init__.
        Each entry is a half-open (lo, hi) index pair covering a
        homogeneous species run.  Do not pass manually — the field is
        set by __post_init__ via ``object.__setattr__``.

    Notes
    -----
    The AtomicModel does not carry T or n_e; those live on Atmosphere.
    The model is purely the atomic data plus the transition list.

    Validation (raised in __post_init__):
      - n_levels >= 2.
      - All (lower, upper) in [0, n_levels) with upper > lower.
      - All 0 < epsilon <= 1 for each transition.
      - len(statistical_weights) == n_levels.
      - K = len(transitions) >= 1.
      - When species is not None: len(species) == n_levels, labels are
        contiguous, and no transition spans two different species.
    """

    n_levels: int
    transitions: tuple[Transition, ...]
    statistical_weights: tuple[int, ...]
    label: str
    species: tuple[str, ...] | None = None
    species_blocks: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        """Validate all fields; raise ValueError on any out-of-range value."""
        if self.n_levels < 2:
            raise ValueError(
                f"n_levels must be >= 2, got n_levels={self.n_levels!r}."
            )
        if len(self.transitions) < 1:
            raise ValueError(
                "transitions must contain at least one Transition, got empty tuple."
            )
        if len(self.statistical_weights) != self.n_levels:
            raise ValueError(
                f"len(statistical_weights) must equal n_levels={self.n_levels}, "
                f"got {len(self.statistical_weights)}."
            )
        for tr in self.transitions:
            if not (0 <= tr.lower < self.n_levels):
                raise ValueError(
                    f"Transition.lower={tr.lower!r} out of range [0, {self.n_levels})."
                )
            if not (0 <= tr.upper < self.n_levels):
                raise ValueError(
                    f"Transition.upper={tr.upper!r} out of range [0, {self.n_levels})."
                )
            if tr.upper <= tr.lower:
                raise ValueError(
                    f"Transition.upper={tr.upper!r} must be > lower={tr.lower!r}."
                )
            if not (0.0 < tr.epsilon <= 1.0):
                raise ValueError(
                    f"Transition.epsilon={tr.epsilon!r} must satisfy 0 < epsilon <= 1 "
                    f"(label={tr.label!r})."
                )

        # ------------------------------------------------------------------
        # Multi-species validation (SPEC_phase_s §2.4).
        # Only runs when species is not None; otherwise single-species path
        # is byte-identical to the pre-Phase-S code.
        # ------------------------------------------------------------------
        if self.species is not None:
            if len(self.species) != self.n_levels:
                raise ValueError(
                    f"len(species)={len(self.species)} must equal "
                    f"n_levels={self.n_levels}."
                )
            # Derive contiguous blocks; raises ValueError on non-contiguous species.
            blocks = _derive_blocks(self.species)
            # Build level -> species-label mapping for cross-species transition check.
            level_species = {i: self.species[i] for i in range(self.n_levels)}
            for tr in self.transitions:
                if level_species[tr.lower] != level_species[tr.upper]:
                    raise ValueError(
                        f"Transition '{tr.label}' (lower={tr.lower}, upper={tr.upper}) "
                        f"spans two species: '{level_species[tr.lower]}' and "
                        f"'{level_species[tr.upper]}'. Cross-species transitions are "
                        f"not allowed (SPEC_phase_s §2.1 block-diagonal precondition)."
                    )
            # Store the derived blocks on the frozen dataclass.
            object.__setattr__(self, "species_blocks", blocks)
        else:
            # Ensure species_blocks is None for the single-species path.
            object.__setattr__(self, "species_blocks", None)

    @property
    def K(self) -> int:
        """Number of transitions."""
        return len(self.transitions)

    @classmethod
    def two_level_master_epsilon(
        cls,
        nu_0: float,
        epsilon: float,
    ) -> "AtomicModel":
        """The 2-level prototype atomic model (one line, master-ε).

        Matches the existing 2-level NLTE setup: a single transition between
        levels 0 and 1 with a single photon destruction probability epsilon
        (assumption A10 — all collisional rate detail absorbed into epsilon).
        Statistical weights are (1, 1) per the minimum-viable-atom convention.

        Parameters
        ----------
        nu_0 : float
            Line-center frequency in Hz.
        epsilon : float
            Photon destruction probability in (0, 1].

        Returns
        -------
        AtomicModel with label="two_level_master_epsilon"
        """
        return cls(
            n_levels=2,
            transitions=(Transition(0, 1, nu_0, epsilon, "line_0_1"),),
            statistical_weights=(1, 1),
            label="two_level_master_epsilon",
        )

    @classmethod
    def balmer_5level(
        cls,
        epsilon: float | tuple = 1e-4,
    ) -> "AtomicModel":
        """5-level hydrogen Balmer atom with 10 bound-bound transitions.

        Levels n = 1..5 (0-based indices 0..4).  All 5-choose-2 = 10 pairs.
        Frequencies from the Rydberg formula:
            nu(l, u) = R_H * c * (1/l^2 - 1/u^2)
        with R_H = 1.0973731568e7 m^-1, c = 2.99792458e8 m/s.
        Statistical weights g_n = 2 n^2 for n = 1..5 -> (2, 8, 18, 32, 50).

        Parameters
        ----------
        epsilon : float or tuple/list of length 10
            Per-line photon destruction probability.  If scalar, applied to
            all 10 transitions.  If sequence of length 10, applied per
            transition in canonical order (see Notes).

        Returns
        -------
        AtomicModel with n_levels=5, K=10, label="balmer_5level".

        Notes
        -----
        Canonical transition ordering (k=0..9):
            k=0  Ly_alpha   1->2   nu ~ 2.467381e15 Hz
            k=1  Ly_beta    1->3   nu ~ 2.923252e15 Hz
            k=2  Ly_gamma   1->4   nu ~ 3.082608e15 Hz
            k=3  Ly_delta   1->5   nu ~ 3.155855e15 Hz
            k=4  H_alpha    2->3   nu ~ 4.569225e14 Hz
            k=5  H_beta     2->4   nu ~ 6.166528e14 Hz
            k=6  H_gamma    2->5   nu ~ 6.907464e14 Hz
            k=7  Pa_alpha   3->4   nu ~ 1.597303e14 Hz
            k=8  Pa_beta    3->5   nu ~ 2.338239e14 Hz
            k=9  Br_alpha   4->5   nu ~ 7.409364e13 Hz

        Level indices are 0-based (n-1); e.g. n=1 -> index 0.
        Statistical weights follow g_n = 2 n^2 (fine structure ignored).
        """
        # Physical constants for Rydberg formula
        _R_H = 1.0973731568e7   # Rydberg constant [m^-1]
        _C_local = 2.99792458e8       # speed of light [m/s]

        # Canonical ordering: (lower_n, upper_n, label)
        # Ordered as: Lyman series (1->2,3,4,5), Balmer (2->3,4,5),
        # Paschen (3->4,5), Brackett (4->5).
        _transitions_spec = [
            (1, 2, "Ly_alpha"),
            (1, 3, "Ly_beta"),
            (1, 4, "Ly_gamma"),
            (1, 5, "Ly_delta"),
            (2, 3, "H_alpha"),
            (2, 4, "H_beta"),
            (2, 5, "H_gamma"),
            (3, 4, "Pa_alpha"),
            (3, 5, "Pa_beta"),
            (4, 5, "Br_alpha"),
        ]

        K = len(_transitions_spec)  # = 10

        # Parse epsilon: scalar or per-line sequence.
        if isinstance(epsilon, (int, float)):
            eps_list = [float(epsilon)] * K
        else:
            eps_seq = tuple(epsilon)
            if len(eps_seq) != K:
                raise ValueError(
                    f"epsilon must be a scalar or a sequence of length {K}, "
                    f"got length {len(eps_seq)}."
                )
            eps_list = [float(e) for e in eps_seq]

        # Build Transition objects (0-based level indices = n - 1).
        transitions = []
        for idx, (n_l, n_u, label) in enumerate(_transitions_spec):
            nu = _R_H * _C_local * (1.0 / n_l**2 - 1.0 / n_u**2)
            transitions.append(
                Transition(
                    lower=n_l - 1,
                    upper=n_u - 1,
                    nu_0=nu,
                    epsilon=eps_list[idx],
                    label=label,
                )
            )

        # Statistical weights: g_n = 2 n^2 for n = 1..5.
        stat_weights = tuple(2 * n**2 for n in range(1, 6))  # (2, 8, 18, 32, 50)

        return cls(
            n_levels=5,
            transitions=tuple(transitions),
            statistical_weights=stat_weights,
            label="balmer_5level",
        )

    @classmethod
    def three_level_hydrogen_chained(
        cls,
        epsilon_lya: float,
        epsilon_ha: float,
    ) -> "AtomicModel":
        """Toy hydrogen 3-level model: Ly-alpha (1↔2) + H-alpha (2↔3).

        Level topology (0-based): n=1 (ground), n=2 (first excited), n=3.
        Transitions share level 1 (n=2): a perturbation in J̄_12 shifts n_2,
        which directly drives the J̄_23 cycle through the formal solve.
        This chained topology populates the off-diagonal blocks of DT that
        the amplification-factor gate needs (SPEC_threelevel §4.2, decision 2).

        Frequencies from Rydberg formula (R_H * c * Δ(1/n²)):
          Ly-alpha: nu = R_H · c · (1 - 1/4)   = 2.467381e15 Hz
          H-alpha : nu = R_H · c · (1/4 - 1/9) = 4.569225e14 Hz

        Both lines use master-ε per assumption A10.  Statistical weights
        are (1, 1, 1) for the minimum-viable-atom convention.

        Parameters
        ----------
        epsilon_lya : float
            Photon destruction probability for Lyman-alpha in (0, 1].
        epsilon_ha : float
            Photon destruction probability for H-alpha in (0, 1].

        Returns
        -------
        AtomicModel with label="three_level_hydrogen_chained"
        """
        return cls(
            n_levels=3,
            transitions=(
                Transition(0, 1, 2.467381e15, epsilon_lya, "Ly_alpha"),
                Transition(1, 2, 4.569225e14, epsilon_ha, "H_alpha"),
            ),
            statistical_weights=(1, 1, 1),
            label="three_level_hydrogen_chained",
        )
