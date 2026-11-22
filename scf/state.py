"""MoleculeState dataclass and the GRAIL state-vector seam.  SPEC_scf §3.2, §4.1, §8.10.

The state for the SCF cycle is the AO-basis density matrix ``P`` of shape
``(M, M)``, where ``M`` is the number of basis functions.  By spatial symmetry
``P`` is symmetric, so only ``M*(M+1)/2`` unique entries carry information.
At the GRAIL boundary we flatten ``P`` to its upper-triangular unique entries
using **row-major** ordering — see :func:`pack_density_matrix` for the exact
convention.  H₂/STO-3G gives ``M=2`` ⇒ ``n_unique=3``; H₂/6-31G gives
``M=4`` ⇒ ``n_unique=10``.

For UHF/UKS (R1, SPEC_scf §8.10) the state is the pair ``(P_alpha, P_beta)``
and the packed-flat seam is *doubled* to ``2 * M*(M+1)/2`` entries: the
α-spin upper-triangular packing concatenated with the β-spin upper-triangular
packing.  The pack/unpack helpers dispatch on input type — a single ``(M, M)``
tensor packs to the singlet seam, a 2-tuple of tensors packs to the doubled
seam.  The same applies on unpack via a ``method`` keyword.

Float64 everywhere at the GRAIL boundary, matching the NLTE testbed
convention (the callback contract Engineering Conventions).

The dataclass is ``frozen=True`` for hashability and to prevent accidental
mutation; downstream code constructs new instances rather than editing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MoleculeState:
    """Molecular configuration for the SCF testbed.

    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates in Angstrom, shape ``(n_atoms, 3)``, float64.
    atom_symbols : tuple[str, ...]
        Element symbols matching the rows of ``coords``, length ``n_atoms``.
        Stored as a tuple (not list) for hashability under ``frozen=True``.
    basis : str
        PySCF basis-set label (e.g. ``"sto-3g"``, ``"6-31g"``).
    n_electrons : int
        Total number of electrons.  Closed-shell methods (``"rhf"``, ``"rks"``)
        require even ``n_electrons``; open-shell methods (``"uhf"``, ``"uks"``)
        accept any positive integer (≥ 1).
    method : str
        ``"rks"`` (default; Kohn–Sham with XC functional), ``"rhf"`` (Hartree–
        Fock; ``V_xc = 0``, used for the Szabo–Ostlund anchor), ``"uhf"``
        (unrestricted Hartree–Fock, R1), or ``"uks"`` (unrestricted Kohn–
        Sham, R1).  UHF/UKS expose the spin-localizing mode of the cycle
        Jacobian that drives the Coulson–Fischer instability (SPEC_scf §8.10).
    xc_functional : str
        XC functional name passed to PySCF for ``method="rks"`` or
        ``method="uks"`` (default ``"B3LYP"``).  Ignored for HF methods.
    charge : int
        Net molecular charge.  Default 0.
    spin : int
        2S (number of unpaired electrons).  Default 0.  For UHF/UKS this
        sets the (n_α − n_β) electron-count split that PySCF's UHF kernel
        consumes; the broken-symmetry initial guess (``init_guess="breaksym"``)
        operates on top of that occupation pattern.
    init_guess : str
        Initial-guess strategy for ``picard_iterate_scf``.  ``"minao"``
        (default for RHF/RKS) defers to PySCF's ``get_init_guess`` (SAD for
        atoms, MINAO for molecules); ``"breaksym"`` (default for UHF/UKS;
        SPEC_scf §8.10) applies the 1.5×/0.5× asymmetric scaling on the
        first two diagonal basis-function entries of the α / β density
        guesses (mirrored: α gets 1.5× on bf 0, 0.5× on bf 1; β reverses).
        This is the load-bearing prescription that selects the broken-
        symmetry branch above the Coulson–Fischer point; the symmetric
        MINAO guess would collapse to RHF and miss the amplification.

    Notes
    -----
    The dataclass is frozen for hashability; the coords array is exposed as
    a numpy ndarray because PySCF wants numerical coordinates downstream.
    Equality/hashing on a numpy array is non-trivial under ``frozen=True``;
    we override ``__hash__`` and ``__eq__`` to use ``tuple(coords.flatten())``.
    """

    coords: np.ndarray            # (n_atoms, 3), float64, Angstrom
    atom_symbols: tuple[str, ...]
    basis: str
    n_electrons: int
    method: str = "rks"           # "rks", "rhf", "uhf", or "uks"
    xc_functional: str = "B3LYP"
    charge: int = 0
    spin: int = 0
    init_guess: str = "minao"     # "minao" (default RHF/RKS) or "breaksym" (UHF/UKS)

    def __post_init__(self) -> None:
        """Validate fields; raise ValueError on any out-of-range value."""
        if self.method not in ("rks", "rhf", "uhf", "uks"):
            raise ValueError(
                f"method must be one of 'rks', 'rhf', 'uhf', 'uks'; "
                f"got method={self.method!r}."
            )
        if self.n_electrons < 1:
            raise ValueError(
                f"n_electrons must be >= 1, got n_electrons={self.n_electrons!r}."
            )
        # Closed-shell methods (RHF/RKS) require even electron count at spin=0.
        # UHF/UKS permit any electron count — spin field carries the imbalance.
        if self.method in ("rhf", "rks") and self.spin == 0 and (self.n_electrons % 2 != 0):
            raise ValueError(
                f"closed-shell (method={self.method!r}, spin=0) requires "
                f"even n_electrons, got n_electrons={self.n_electrons!r}."
            )
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(
                f"coords must have shape (n_atoms, 3), got {self.coords.shape!r}."
            )
        if len(self.atom_symbols) != self.coords.shape[0]:
            raise ValueError(
                f"len(atom_symbols)={len(self.atom_symbols)} must match "
                f"coords.shape[0]={self.coords.shape[0]}."
            )
        if self.init_guess not in ("minao", "breaksym"):
            raise ValueError(
                f"init_guess must be 'minao' or 'breaksym', got "
                f"init_guess={self.init_guess!r}."
            )

    # ------------------------------------------------------------------
    # Hash and equality: ndarray fields need explicit handling under frozen.
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        return hash((
            tuple(self.coords.flatten().tolist()),
            self.atom_symbols,
            self.basis,
            self.n_electrons,
            self.method,
            self.xc_functional,
            self.charge,
            self.spin,
            self.init_guess,
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MoleculeState):
            return NotImplemented
        return (
            np.array_equal(self.coords, other.coords)
            and self.atom_symbols == other.atom_symbols
            and self.basis == other.basis
            and self.n_electrons == other.n_electrons
            and self.method == other.method
            and self.xc_functional == other.xc_functional
            and self.charge == other.charge
            and self.spin == other.spin
            and self.init_guess == other.init_guess
        )


# ---------------------------------------------------------------------------
# Pack / unpack: GRAIL state-vector seam.
# ---------------------------------------------------------------------------
# Convention (SPEC_scf §3.2, §4.1): upper-triangular row-major flattening.
# For an (M, M) symmetric P, the packed vector has length M*(M+1)/2 with
# entries ordered by (row=i, col=j>=i) lexicographic on (i, j):
#   [P[0,0], P[0,1], ..., P[0,M-1], P[1,1], P[1,2], ..., P[M-1,M-1]]
# This matches numpy's row-major (C-order) traversal of triu_indices(M).
#
# UHF / UKS extension (SPEC_scf §8.10): the state is the pair
# ``(P_alpha, P_beta)``.  The packed-flat seam concatenates the α and β
# upper-triangular packings, giving total length ``2 * M*(M+1)/2``.  For
# M=2 the length is 6.  The α block occupies indices [0, n_unique), the
# β block [n_unique, 2*n_unique).  This matches the convention used by
# the standalone diagnostic ``examples/diagnose_uhf_amplification.py``.
# ---------------------------------------------------------------------------

def _is_uhf_method(method: str) -> bool:
    """Return True if ``method`` is an open-shell (UHF/UKS) method."""
    return method in ("uhf", "uks")


def _pack_single(P: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Internal: upper-triangular row-major pack of a single (M, M) matrix."""
    if isinstance(P, np.ndarray):
        P_t = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float64))
    else:
        P_t = P.to(torch.float64)
    if P_t.ndim != 2 or P_t.shape[0] != P_t.shape[1]:
        raise ValueError(
            f"pack_density_matrix requires a square (M, M) input, "
            f"got shape={tuple(P_t.shape)!r}."
        )
    M = P_t.shape[0]
    iu = torch.triu_indices(M, M)
    return P_t[iu[0], iu[1]].clone()


def _unpack_single(flat_t: torch.Tensor, M: int) -> torch.Tensor:
    """Internal: invert :func:`_pack_single` into a symmetric (M, M) tensor."""
    P = torch.zeros((M, M), dtype=torch.float64)
    iu = torch.triu_indices(M, M)
    P[iu[0], iu[1]] = flat_t
    # Mirror the strict upper triangle into the strict lower triangle.
    P = P + P.T - torch.diag(torch.diagonal(P))
    return P


def pack_density_matrix(
    P: torch.Tensor | np.ndarray | tuple,
    *,
    method: str = "rhf",
) -> torch.Tensor:
    """Flatten a (closed- or open-shell) density matrix to its unique entries.

    Convention: upper-triangular row-major.  For closed-shell (RHF/RKS),
    a single ``(M, M)`` matrix packs to length ``M*(M+1)/2``; for M=2 the
    output is ``[P[0,0], P[0,1], P[1,1]]``.  For open-shell (UHF/UKS), a
    tuple ``(P_alpha, P_beta)`` packs to length ``2 * M*(M+1)/2`` — the α
    upper-triangular packing concatenated with the β upper-triangular
    packing (SPEC_scf §8.10).

    Parameters
    ----------
    P : Tensor or ndarray of shape (M, M), or tuple of two such for UHF/UKS
        Symmetric density matrix or pair of density matrices.  Symmetry is
        NOT enforced here — the caller is responsible for passing a
        symmetric input.  We read only the upper-triangular entries, so any
        antisymmetric residue in the input is silently discarded.
    method : str, default ``"rhf"``
        Either a closed-shell method (``"rhf"``, ``"rks"``) — in which case
        ``P`` must be a single matrix — or an open-shell method (``"uhf"``,
        ``"uks"``) — in which case ``P`` must be a 2-tuple.  Provided for
        explicitness; the function will reject mismatched inputs.

    Returns
    -------
    p_flat : torch.Tensor of shape ``(n_unique,)`` or ``(2*n_unique,)``, float64
        Packed unique entries (closed-shell) or concatenated α/β unique
        entries (open-shell).

    See Also
    --------
    unpack_density_matrix : the inverse operation.
    """
    if _is_uhf_method(method):
        if not (isinstance(P, tuple) and len(P) == 2):
            raise ValueError(
                f"pack_density_matrix for method={method!r} requires a "
                f"2-tuple (P_alpha, P_beta); got input of type {type(P).__name__}."
            )
        P_alpha, P_beta = P
        flat_a = _pack_single(P_alpha)
        flat_b = _pack_single(P_beta)
        if flat_a.shape != flat_b.shape:
            raise ValueError(
                f"pack_density_matrix: P_alpha and P_beta must have matching "
                f"shapes; got alpha-flat shape={tuple(flat_a.shape)!r}, "
                f"beta-flat shape={tuple(flat_b.shape)!r}."
            )
        return torch.cat([flat_a, flat_b]).clone()
    # Closed-shell branch.
    if isinstance(P, tuple):
        raise ValueError(
            f"pack_density_matrix for method={method!r} requires a single "
            f"(M, M) matrix; got a 2-tuple."
        )
    return _pack_single(P)


def unpack_density_matrix(
    p_flat: torch.Tensor | np.ndarray,
    M: int,
    *,
    method: str = "rhf",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_density_matrix`.

    For closed-shell methods (``"rhf"``, ``"rks"``) returns a single
    symmetric ``(M, M)`` matrix.  For open-shell methods (``"uhf"``,
    ``"uks"``) returns a tuple ``(P_alpha, P_beta)`` of symmetric ``(M, M)``
    matrices.  Strict-upper entries are mirrored to the strict-lower
    triangle to guarantee exact symmetry within each spin block.

    Parameters
    ----------
    p_flat : Tensor or ndarray
        Packed unique entries from :func:`pack_density_matrix`.  Length
        ``M*(M+1)/2`` for closed-shell methods, ``2*M*(M+1)/2`` for
        open-shell methods.
    M : int
        Original matrix size.  Required because the flat length alone is
        ambiguous; we ask the caller to be explicit.
    method : str, default ``"rhf"``
        Closed-shell (``"rhf"``, ``"rks"``) returns a single matrix;
        open-shell (``"uhf"``, ``"uks"``) returns a 2-tuple.

    Returns
    -------
    P : torch.Tensor of shape ``(M, M)`` (closed-shell) or
        tuple of two ``(M, M)`` torch.Tensors (open-shell), all float64.
    """
    if isinstance(p_flat, np.ndarray):
        flat_t = torch.from_numpy(np.ascontiguousarray(p_flat, dtype=np.float64))
    else:
        flat_t = p_flat.to(torch.float64)

    n_unique_single = M * (M + 1) // 2
    if _is_uhf_method(method):
        n_unique_expected = 2 * n_unique_single
        if flat_t.shape != (n_unique_expected,):
            raise ValueError(
                f"unpack_density_matrix(method={method!r}): p_flat must have "
                f"shape ({n_unique_expected},) for M={M}, got "
                f"shape={tuple(flat_t.shape)!r}."
            )
        P_alpha = _unpack_single(flat_t[:n_unique_single], M)
        P_beta = _unpack_single(flat_t[n_unique_single:], M)
        return P_alpha, P_beta

    # Closed-shell branch.
    if flat_t.shape != (n_unique_single,):
        raise ValueError(
            f"unpack_density_matrix: p_flat must have shape ({n_unique_single},) "
            f"for M={M}, got shape={tuple(flat_t.shape)!r}."
        )
    return _unpack_single(flat_t, M)


def n_unique(M: int) -> int:
    """Number of unique entries in a symmetric (M, M) matrix.

    Returns ``M*(M+1)/2``.  Provided as a named helper for readability at
    call sites that build state-vector buffers without constructing a
    matrix first.
    """
    return M * (M + 1) // 2
