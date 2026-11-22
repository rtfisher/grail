"""Convenience constructors for SCF testbed states.  SPEC_scf §P1.

Currently provides ``make_h2_state(R, ...)`` for diatomic H₂ along the
internuclear axis; further constructors land in P5+ if/when larger
molecules join the testbed.
"""

from __future__ import annotations

import numpy as np

from .state import MoleculeState


def make_h2_state(
    R: float,
    *,
    basis: str = "sto-3g",
    method: str = "rks",
    xc: str = "B3LYP",
    init_guess: str | None = None,
) -> MoleculeState:
    """Construct an H₂ :class:`MoleculeState`.

    The two H atoms are placed on the z-axis at ``(0, 0, ±R/2)`` so the
    centre of mass is at the origin.  Coordinates are in Angstrom (PySCF
    accepts both Angstrom and Bohr; we use the package-wide convention of
    Angstrom at the public boundary, set ``unit='Angstrom'`` when building
    the PySCF Mole — see ``fock_build._build_mol``).

    Parameters
    ----------
    R : float
        H–H bond length in Angstrom.  Must be positive.  Typical operating
        range is ``[0.5, 2.5] Å`` (equilibrium ≈ 0.74 Å).
    basis : str, default "sto-3g"
        PySCF basis-set label.  ``"sto-3g"`` gives M=2, ``"6-31g"`` gives M=4.
    method : str, default "rks"
        ``"rks"`` (Kohn–Sham), ``"rhf"`` (Hartree–Fock), ``"uhf"``
        (unrestricted Hartree–Fock, R1), or ``"uks"`` (unrestricted
        Kohn–Sham, R1).
    xc : str, default "B3LYP"
        XC functional name for ``method="rks"`` or ``method="uks"``.
        Ignored for HF methods.
    init_guess : str or None, default None
        ``"minao"`` or ``"breaksym"``.  When ``None``, defaults to
        ``"breaksym"`` for UHF/UKS (selects the Coulson–Fischer-relevant
        broken-symmetry branch via the 1.5×/0.5× density scaling, see
        ``examples/diagnose_uhf_amplification.py``) and to ``"minao"`` for
        closed-shell methods.

    Returns
    -------
    MoleculeState
    """
    if R <= 0.0:
        raise ValueError(f"R must be positive (Angstrom), got R={R!r}.")
    # Resolve init_guess default per method, per SPEC_scf §8.10.  The UHF/
    # UKS default of ``"breaksym"`` is the load-bearing prescription that
    # picks the broken-symmetry branch above R_CF; ``"minao"`` (PySCF stock
    # symmetric guess) would collapse to RHF and miss the amplification.
    if init_guess is None:
        init_guess = "breaksym" if method in ("uhf", "uks") else "minao"
    coords = np.array(
        [[0.0, 0.0, -R / 2.0], [0.0, 0.0, +R / 2.0]],
        dtype=np.float64,
    )
    return MoleculeState(
        coords=coords,
        atom_symbols=("H", "H"),
        basis=basis,
        n_electrons=2,
        method=method,
        xc_functional=xc,
        charge=0,
        spin=0,
        init_guess=init_guess,
    )
