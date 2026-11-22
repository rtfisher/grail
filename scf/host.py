"""Roothaan–Hall SCF fixed-point iteration with optional DIIS.  SPEC_scf §3.2, §4.3, §8.10.

The SCF cycle is the composition 𝒯 = 𝐃 ∘ 𝐅 (SPEC_scf §2.1):

1. Build the Fock matrix ``F = build_fock(state, P)``.
2. Solve the generalized eigenproblem ``F C = S C ε`` for orbital coefficients
   ``C`` and orbital energies ``ε``.
3. Form ``P_new = 2 · C_occ · C_occᵀ`` (closed-shell, RHF/RKS) or
   ``P_σ_new = C_σ,occ · C_σ,occᵀ`` per spin (open-shell, UHF/UKS).
4. Optionally extrapolate ``F`` via DIIS (Pulay 1980) using the
   ``[F, P, S]`` commutator residual.  For UHF/UKS the same CDIIS engine
   handles a stacked ``(F_α, F_β)`` Fock representation.

For UHF/UKS stopping requires both ``max |ΔP| < tol`` and the actual
unaccelerated equation defect below tol.  A stale DIIS history is restarted
when its tiny updates hide a larger equation defect.  Explicit
``verify_equation=False`` reproduces the legacy step-only stopping rule.
For UHF/UKS the convergence criterion is taken over both spin blocks
(``max(|ΔP_α|, |ΔP_β|)``).

Implementation notes:
- The eigenproblem uses PySCF's eigensolver (``pyscf.scf.hf.eig``), a thin
  wrapper around LAPACK ``dsygv``.  The conda env forbids scipy.
- DIIS uses PySCF's ``pyscf.scf.diis.CDIIS`` so we inherit its battle-tested
  numerics.  The CDIIS subspace size defaults to 8 (SPEC_scf §7).  PySCF's
  CDIIS handles the UHF stacked-Fock case transparently: when ``F`` has
  shape ``(2, M, M)`` it stacks the spin blocks internally.
- Float64 everywhere.
- For UHF/UKS the broken-symmetry initial-guess prescription (1.5×/0.5×
  asymmetry on the first two basis-function diagonals of α and β) is the
  load-bearing API contract that selects the broken-symmetry branch above
  the Coulson–Fischer point (SPEC_scf §8.10).  The symmetric MINAO guess
  would collapse to RHF and miss the amplification regime.

PySCF is imported inside functions, not at module top (OpenMP, SPEC_scf §8.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .fock_build import _build_mf, _build_mol, build_fock_truth
from .state import MoleculeState


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SCFResult:
    """Converged output from :func:`picard_iterate_scf`.

    Field shapes depend on the ``method`` of the input
    :class:`~scf.state.MoleculeState`:

    - Closed-shell (RHF/RKS): ``P_star`` is a single symmetric ``(M, M)``
      ndarray; ``mo_energy`` has shape ``(M,)``; ``mo_coeff`` has shape
      ``(M, M)``.
    - Open-shell (UHF/UKS, R1; SPEC_scf §8.10): ``P_star`` is a 2-tuple
      ``(P_alpha, P_beta)`` of symmetric ``(M, M)`` ndarrays;
      ``mo_energy`` is a 2-tuple ``(eps_alpha, eps_beta)`` of ``(M,)``
      arrays; ``mo_coeff`` is a 2-tuple ``(C_alpha, C_beta)`` of
      ``(M, M)`` arrays.  ``s_squared`` is populated with the converged
      ⟨S²⟩ from PySCF's ``spin_square``; ``None`` for closed-shell.

    Attributes
    ----------
    P_star : np.ndarray (M, M) or 2-tuple of such
        Converged AO density matrix (symmetric, float64), one per spin
        block for UHF/UKS.
    mo_energy : np.ndarray (M,) or 2-tuple of such
        Orbital energies in Hartree.
    mo_coeff : np.ndarray (M, M) or 2-tuple of such
        Orbital coefficients; column ``i`` is the MO with energy
        ``mo_energy[i]`` (per spin for UHF/UKS).
    iter_count : int
        Number of Roothaan–Hall iterates performed.
    converged : bool
        True if ``max|ΔP| < tol`` was reached before ``max_iter`` (taken
        over both spin blocks for UHF/UKS).  With the UHF/UKS default
        ``verify_equation=True``, the unaccelerated equation defect must
        also satisfy tol.
    energy_total : float
        Total electronic + nuclear-repulsion energy at the converged P,
        in Hartree.
    residual_history : list[float]
        ``max|ΔP|`` per iteration; useful for diagnosing stalls.
    diis_engaged : bool
        True if DIIS was active (``accelerator="diis"`` and at least one
        DIIS extrapolation step was applied).
    s_squared : float or None
        ``⟨S²⟩`` from PySCF's ``spin_square`` at the converged density;
        populated only for UHF/UKS.  ``None`` for closed-shell.
    """

    P_star: np.ndarray | tuple
    mo_energy: np.ndarray | tuple
    mo_coeff: np.ndarray | tuple
    iter_count: int
    converged: bool
    energy_total: float
    residual_history: list[float] = field(default_factory=list)
    diis_engaged: bool = False
    s_squared: float | None = None
    cycle_defect_max: float | None = None
    cycle_defect_l2: float | None = None
    equation_converged: bool | None = None
    polish_iterations: int = 0


# ---------------------------------------------------------------------------
# Roothaan–Hall driver
# ---------------------------------------------------------------------------

def _polish_uhf_equation(P_pair, S, nelec, state, fock_builder, tol, max_steps=8):
    """Bounded Newton polish of the actual small UHF cycle equation.

    Used after DIIS stagnation only.  The actual supplied builder is
    finite-differenced; no reference derivative substitutes for a NN.
    Backtracking accepts only smaller true defects.  Returned densities
    are occupied-orbital projectors, checked again after projection.
    """
    M = len(S)
    rows, cols = np.triu_indices(M)
    nu = len(rows)

    def pack(pair):
        return np.concatenate([p[rows, cols] for p in pair])

    def unpack(x):
        result = []
        for spin in range(2):
            p = np.zeros((M, M))
            p[rows, cols] = x[spin * nu:(spin + 1) * nu]
            p[cols, rows] = p[rows, cols]
            result.append(p)
        return tuple(result)

    def cycle(x):
        orbitals = []
        for F, n_occ in zip(fock_builder(state, unpack(x)), nelec):
            e, C = _solve_generalized_eig(0.5 * (F + F.T), S)
            orbitals.append((e, C, _build_density_from_orbitals_uhf(C, n_occ)))
        return pack([o[2] for o in orbitals]), orbitals

    x = pack(P_pair)
    basis = np.eye(len(x))
    for iteration in range(max_steps + 1):
        tx, orbitals = cycle(x)
        defect = tx - x
        # Check the returned physical projectors themselves, not merely
        # the unconstrained Newton iterate.
        ttx, _ = cycle(tx)
        if np.max(np.abs(ttx - tx)) < tol:
            return orbitals, iteration, True
        if iteration == max_steps:
            break
        h = 1e-5
        K = np.column_stack([(cycle(x + h * v)[0] - cycle(x - h * v)[0]) / (2 * h)
                             for v in basis])
        try:
            step = np.linalg.solve(np.eye(len(x)) - K, defect)
        except np.linalg.LinAlgError:
            break
        old_norm = np.linalg.norm(defect)
        for power in range(16):
            trial = x + 2. ** (-power) * step
            if np.linalg.norm(cycle(trial)[0] - trial) < old_norm:
                x = trial
                break
        else:
            break
    return None, iteration, False

def _initial_density_guess(state: MoleculeState, mf) -> np.ndarray:
    """Compute an initial density-matrix guess.

    Delegates to PySCF's ``get_init_guess`` (SAD by default for atoms, MINAO
    for molecules), then returns the result as a float64 ndarray.  Used only
    for closed-shell methods (RHF/RKS); the UHF/UKS guess is produced by
    :func:`_broken_symmetry_init`.

    Parameters
    ----------
    state : MoleculeState
    mf : pyscf mean-field object (RKS or RHF)
        Pre-built via :func:`fock_build._build_mf`.

    Returns
    -------
    P0 : np.ndarray of shape (M, M), float64
    """
    P0 = mf.get_init_guess(mf.mol)
    return np.asarray(P0, dtype=np.float64)


def _broken_symmetry_init(
    state: MoleculeState, mf,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a broken-symmetry UHF/UKS initial guess.

    Implements the prescription validated in
    ``examples/diagnose_uhf_amplification.py`` (SPEC_scf §8.10): start from
    PySCF's UHF default guess (a ``(2, M, M)`` array with α and β blocks)
    and scale the first two diagonal basis-function entries asymmetrically.
    α gets 1.5× on basis function 0 and 0.5× on basis function 1; β gets
    the reverse.  This localizes α-density on one atom and β-density on the
    other prior to SCF, encouraging descent into the broken-symmetry
    (UHF) branch above the Coulson–Fischer point.

    Parameters
    ----------
    state : MoleculeState
        Must have ``method in {"uhf", "uks"}``; not enforced here (caller
        responsibility).
    mf : pyscf UHF or UKS mean-field object

    Returns
    -------
    (P_alpha, P_beta) : tuple of np.ndarray, each shape (M, M), float64
        Broken-symmetry initial density guess for the two spin blocks.

    Notes
    -----
    The 1.5×/0.5× factor is more aggressive than a 1.1/0.9 fallback —
    preferable here because we want broken symmetry to take above R_CF
    without further coaxing (per the diagnostic's empirical finding).  If
    M < 2 the scaling is applied only to basis function 0; the asymmetry
    pattern still differs between α and β (1.5× vs 0.5×).
    """
    dm0 = np.asarray(mf.get_init_guess(mf.mol), dtype=np.float64)
    # PySCF UHF/UKS returns shape (2, M, M); legacy shapes are extremely
    # rare here but we accept a single (M, M) by replicating both spins.
    if dm0.ndim == 3:
        dm_a = dm0[0].copy()
        dm_b = dm0[1].copy()
    elif dm0.ndim == 2:
        # Some PySCF UHF init paths give a single matrix; split it evenly.
        dm_a = 0.5 * dm0.copy()
        dm_b = 0.5 * dm0.copy()
    else:
        raise ValueError(
            f"_broken_symmetry_init: unexpected init-guess shape "
            f"{dm0.shape!r} from PySCF; expected (2, M, M) or (M, M)."
        )
    M = dm_a.shape[0]
    # Apply the load-bearing 1.5×/0.5× asymmetric scaling.  The diagnostic's
    # convention: α on basis function 0, β on basis function 1.
    if M >= 1:
        dm_a[0, 0] *= 1.5
        dm_b[0, 0] *= 0.5
    if M >= 2:
        dm_a[1, 1] *= 0.5
        dm_b[1, 1] *= 1.5
    return dm_a, dm_b


def _build_density_from_orbitals_uhf(
    mo_coeff_sigma: np.ndarray, n_occ_sigma: int,
) -> np.ndarray:
    """Form the open-shell AO density for one spin block: ``P_σ = C_σ,occ C_σ,occᵀ``.

    Mirrors :func:`_build_density_from_orbitals` for the closed-shell case
    but with unit (not factor-2) occupation since UHF/UKS spin orbitals
    are singly occupied.

    Parameters
    ----------
    mo_coeff_sigma : np.ndarray of shape (M, M)
        Full MO coefficient matrix for spin σ.
    n_occ_sigma : int
        Number of occupied orbitals for spin σ.

    Returns
    -------
    P_sigma : np.ndarray of shape (M, M), float64
    """
    C_occ = mo_coeff_sigma[:, :n_occ_sigma]
    P = C_occ @ C_occ.T
    # Force exact symmetry.
    P = 0.5 * (P + P.T)
    return np.asarray(P, dtype=np.float64)


def _solve_generalized_eig(F: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``F C = S C ε`` for orbital energies and coefficients.

    Uses PySCF's eigensolver (``pyscf.scf.hf.eig``), which is a thin wrapper
    around LAPACK's symmetric-definite generalized eigenproblem routine.
    Returns eigenvalues sorted ascending.

    Parameters
    ----------
    F : np.ndarray of shape (M, M)
        Symmetric Fock matrix.
    S : np.ndarray of shape (M, M)
        Symmetric positive-definite overlap matrix.

    Returns
    -------
    mo_energy : np.ndarray of shape (M,), float64
        Orbital energies, sorted ascending.
    mo_coeff : np.ndarray of shape (M, M), float64
        Orbital coefficients; column ``i`` corresponds to ``mo_energy[i]``.
    """
    from pyscf.scf import hf as scf_hf

    e, c = scf_hf.eig(F, S)
    return np.asarray(e, dtype=np.float64), np.asarray(c, dtype=np.float64)


def _build_density_from_orbitals(
    mo_coeff: np.ndarray, n_occ: int,
) -> np.ndarray:
    """Form the closed-shell AO density ``P = 2 C_occ C_occᵀ``.

    Parameters
    ----------
    mo_coeff : np.ndarray of shape (M, M)
        Full MO coefficient matrix.
    n_occ : int
        Number of doubly-occupied orbitals (``n_electrons // 2``).

    Returns
    -------
    P : np.ndarray of shape (M, M), float64
    """
    C_occ = mo_coeff[:, :n_occ]
    P = 2.0 * (C_occ @ C_occ.T)
    # Force exact symmetry.
    P = 0.5 * (P + P.T)
    return np.asarray(P, dtype=np.float64)


def _total_energy(
    mf, F: np.ndarray, P: np.ndarray, h_core: np.ndarray,
) -> float:
    """Compute the total electronic+nuclear energy at the converged P.

    Uses PySCF's ``energy_tot`` which knows how to handle the XC double-
    counting correction for RKS.  Equivalent to::

        E_elec = 0.5 * tr(P · (h_core + F)) + E_xc - 0.5 * tr(P · V_xc)

    PySCF wraps that bookkeeping inside ``mf.energy_tot(dm=P)`` (or
    ``mf.energy_tot(dm=P, h1e=h_core, vhf=F-h_core)`` for the explicit form).
    We use the simple one-argument call; PySCF rebuilds the necessary
    intermediates from the supplied density.

    Parameters
    ----------
    mf : pyscf mean-field object
    F : np.ndarray
        Current Fock matrix (unused in the simple call but accepted for
        API symmetry).
    P : np.ndarray
        Converged density matrix.
    h_core : np.ndarray
        Core Hamiltonian (unused in the simple call; see above).

    Returns
    -------
    energy_total : float
        E_elec + E_nuc in Hartree.
    """
    # PySCF docs: energy_tot(dm=...) computes E[ρ] + E_nuc; this is the
    # right entry point because it handles the RKS XC double-counting
    # correctly without us having to reproduce it.
    return float(mf.energy_tot(dm=np.asarray(P, dtype=np.float64)))


def picard_iterate_scf(
    state: MoleculeState,
    P_init: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    fock_builder: Callable | None = None,
    *,
    tol: float = 1e-9,
    max_iter: int = 200,
    accelerator: str = "diis",
    diis_space: int = 8,
    verify_equation: bool = True,
) -> SCFResult:
    """Roothaan–Hall SCF iteration with optional DIIS.

    Closed-shell (RHF/RKS): iterates the cycle 𝒯 = 𝐃 ∘ 𝐅 on the AO density
    matrix until ``max|ΔP| < tol`` or ``iter_count >= max_iter``.

    Open-shell (UHF/UKS, R1; SPEC_scf §8.10): iterates the same cycle but
    on the pair ``(P_α, P_β)`` independently per spin, with convergence
    checked on ``max(|ΔP_α|, |ΔP_β|)``.  The default initial guess for
    UHF/UKS (when ``state.init_guess == "breaksym"``, also the default the
    convenience constructor sets for these methods) is the 1.5×/0.5×
    broken-symmetry prescription validated in
    ``examples/diagnose_uhf_amplification.py``.

    Parameters
    ----------
    state : MoleculeState
        Molecular configuration (geometry, basis, method, XC functional,
        init_guess).
    P_init : np.ndarray, 2-tuple of np.ndarray, or None, default None
        Initial density-matrix guess.  If None, the closed-shell branch
        uses PySCF's ``get_init_guess``; the UHF/UKS branch uses the
        broken-symmetry initial guess when ``state.init_guess ==
        "breaksym"``, otherwise PySCF's stock UHF default.  Pass a
        ``(M, M)`` ndarray for closed-shell or a 2-tuple ``(P_α, P_β)``
        for open-shell.
    fock_builder : callable or None, default None
        Closed-shell signature ``(state, P) -> F``; open-shell signature
        ``(state, (P_α, P_β)) -> (F_α, F_β)``.  If None, defaults to
        :func:`fock_build.build_fock_truth` (which dispatches on
        ``state.method``).  The emulator-replaced builder
        (``make_fock_builder_with_nn``) lands in P3/P4 and slots in here.
    tol : float, default 1e-9
        Convergence tolerance on ``max|ΔP|`` per element.  Matches PySCF's
        ``conv_tol_density`` default and is tighter than the GMRES tolerance
        downstream (SPEC_scf §7).
    max_iter : int, default 200
        Maximum number of Roothaan–Hall iterates.  Set to 200 to allow
        DIIS-accelerated stretched-bond convergence (SPEC_scf §7).
    accelerator : {"none", "diis"}, default "diis"
        ``"none"`` is plain Roothaan–Hall; ``"diis"`` engages PySCF's CDIIS
        (Pulay 1980) on the ``[F, P, S]`` commutator residual after a short
        warmup (one plain iteration to seed the history).  For UHF/UKS the
        same CDIIS engine accepts stacked ``(2, M, M)`` Fock and density
        inputs and extrapolates each spin block transparently.
    diis_space : int, default 8
        DIIS subspace size (CDIIS history depth).  Standard CDIIS choice;
        matches PySCF's default.
    verify_equation : bool, default True
        For UHF/UKS require the actual unaccelerated cycle defect to
        meet tol, restarting stale DIIS history when needed.  False
        exists for explicit legacy-protocol replay.  The result reports
        equation defects and ``equation_converged`` in either mode.

    Returns
    -------
    SCFResult

    Notes
    -----
    The reference value of ``energy_total`` for H₂/STO-3G/RHF at R=0.74 Å
    is ``-1.1167593073964 Ha`` in the independent matching-geometry RHF
    solve (approximately -1.117 Ha at textbook precision).  The
    ``test_szabo_ostlund_h2`` analytic anchor exercises this.  Diagnostic
    UHF reference values at H₂/STO-3G (from
    ``examples/uhf_amplification_diagnostic.md``): R=1.00 Å → ⟨S²⟩=0,
    E=-1.06611; R=1.18 Å → ⟨S²⟩=0.087, E=-1.01190; R=1.30 Å → ⟨S²⟩=0.394,
    E=-0.98403.
    """
    if accelerator not in ("none", "diis"):
        raise ValueError(
            f"accelerator must be 'none' or 'diis', got accelerator={accelerator!r}."
        )
    if fock_builder is None:
        fock_builder = build_fock_truth

    # Dispatch on the method.  UHF/UKS take an entirely different code path
    # (doubled state, per-spin diagonalization, per-spin density build); we
    # split the implementation rather than overload the closed-shell loop
    # to keep both paths readable.
    if state.method in ("uhf", "uks"):
        return _picard_iterate_uhf(
            state, P_init=P_init, fock_builder=fock_builder,
            tol=tol, max_iter=max_iter,
            accelerator=accelerator, diis_space=diis_space,
            verify_equation=verify_equation,
        )

    # Inline import: keep PySCF off the module-load OpenMP path.
    from pyscf.scf import diis as pyscf_diis

    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)

    # Overlap and core-Hamiltonian are constants of the geometry/basis.
    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    h_core = np.asarray(mf.get_hcore(), dtype=np.float64)

    n_occ = state.n_electrons // 2

    # Initial density guess.
    if P_init is None:
        P = _initial_density_guess(state, mf)
    else:
        P = np.asarray(P_init, dtype=np.float64).copy()

    # DIIS engine (only used if accelerator == "diis").
    diis_engine = None
    diis_engaged = False
    if accelerator == "diis":
        diis_engine = pyscf_diis.CDIIS(mf)
        diis_engine.space = diis_space

    residual_history: list[float] = []
    converged = False
    mo_energy = np.zeros(S.shape[0], dtype=np.float64)
    mo_coeff = np.eye(S.shape[0], dtype=np.float64)

    for iteration in range(max_iter):
        # Step 1: Fock-build.
        F = fock_builder(state, P)
        # Force symmetry (mirrors build_fock_truth's defensive symmetrize).
        F = 0.5 * (F + F.T)

        # Step 2 (optional): DIIS extrapolation.  PySCF's CDIIS.update takes
        # (S, P, F) and returns an extrapolated F.  Engage after the first
        # iteration so the history has a seed; this matches PySCF's own
        # ``scf.kernel`` warmup.
        if diis_engine is not None and iteration >= 1:
            F = diis_engine.update(S, P, F)
            diis_engaged = True

        # Step 3: solve F C = S C ε for orbitals.
        mo_energy, mo_coeff = _solve_generalized_eig(F, S)

        # Step 4: form new density from occupied orbitals.
        P_new = _build_density_from_orbitals(mo_coeff, n_occ)

        # Convergence on max|ΔP|.
        residual = float(np.max(np.abs(P_new - P)))
        residual_history.append(residual)

        P = P_new

        if residual < tol:
            converged = True
            break

    # Total energy at the converged P (or last iterate if not converged).
    energy_total = _total_energy(mf, F, P, h_core)

    return SCFResult(
        P_star=P,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        iter_count=len(residual_history),
        converged=converged,
        energy_total=energy_total,
        residual_history=residual_history,
        diis_engaged=diis_engaged,
        s_squared=None,
    )


def _picard_iterate_uhf(
    state: MoleculeState,
    P_init: tuple[np.ndarray, np.ndarray] | None,
    fock_builder: Callable,
    *,
    tol: float,
    max_iter: int,
    accelerator: str,
    diis_space: int,
    verify_equation: bool,
) -> SCFResult:
    """UHF / UKS Roothaan–Hall driver.  SPEC_scf §8.10.

    Parallel of the closed-shell loop but stepping the pair
    ``(P_α, P_β)`` independently per spin.  Convergence is on
    ``max(|ΔP_α|, |ΔP_β|)``.  See :func:`picard_iterate_scf` for the
    user-facing semantics.
    """
    from pyscf.scf import diis as pyscf_diis

    mol = _build_mol(state)
    mf = _build_mf(state, mol=mol)

    S = np.asarray(mf.get_ovlp(), dtype=np.float64)
    h_core = np.asarray(mf.get_hcore(), dtype=np.float64)

    # n_α and n_β from spin convention: PySCF defines nelec as a 2-tuple
    # (n_α, n_β) on the Mole.  This handles both spin=0 (n_α = n_β = N/2)
    # and unpaired-electron systems generically.
    n_alpha, n_beta = mol.nelec

    # Initial density guess: 2-tuple in all paths through this function.
    if P_init is None:
        if state.init_guess == "breaksym":
            P_alpha, P_beta = _broken_symmetry_init(state, mf)
        else:
            # Plain MINAO/SAD UHF guess from PySCF (shape (2, M, M)).
            dm0 = np.asarray(mf.get_init_guess(mf.mol), dtype=np.float64)
            if dm0.ndim == 3:
                P_alpha = dm0[0].copy()
                P_beta = dm0[1].copy()
            elif dm0.ndim == 2:
                P_alpha = 0.5 * dm0.copy()
                P_beta = 0.5 * dm0.copy()
            else:
                raise ValueError(
                    f"Unexpected UHF init-guess shape {dm0.shape!r}."
                )
    else:
        if not (isinstance(P_init, tuple) and len(P_init) == 2):
            raise ValueError(
                f"picard_iterate_scf(method={state.method!r}) requires P_init "
                f"as a 2-tuple (P_alpha, P_beta) or None; got type "
                f"{type(P_init).__name__}."
            )
        P_alpha = np.asarray(P_init[0], dtype=np.float64).copy()
        P_beta = np.asarray(P_init[1], dtype=np.float64).copy()

    M = P_alpha.shape[0]

    # DIIS engine: PySCF's CDIIS on a UHF/UKS mf transparently handles the
    # stacked (2, M, M) Fock/density representation (per PySCF 2.11
    # behavior — see the diagnostic's mf.kernel(dm0) path which exercises
    # this internally).  We feed (S, dm_stack, F_stack) where dm_stack and
    # F_stack are both (2, M, M).
    diis_engine = None
    diis_engaged = False
    if accelerator == "diis":
        diis_engine = pyscf_diis.CDIIS(mf)
        diis_engine.space = diis_space

    residual_history: list[float] = []
    converged = False
    mo_energy_a = np.zeros(M, dtype=np.float64)
    mo_energy_b = np.zeros(M, dtype=np.float64)
    mo_coeff_a = np.eye(M, dtype=np.float64)
    mo_coeff_b = np.eye(M, dtype=np.float64)
    F_alpha = np.zeros((M, M), dtype=np.float64)
    F_beta = np.zeros((M, M), dtype=np.float64)
    polish_iterations = 0
    polish_attempted = False

    for iteration in range(max_iter):
        # Step 1: Fock-build (returns a 2-tuple for UHF/UKS).
        F_pair = fock_builder(state, (P_alpha, P_beta))
        F_alpha, F_beta = F_pair
        # Defensive symmetrization per spin.
        F_alpha = 0.5 * (F_alpha + F_alpha.T)
        F_beta = 0.5 * (F_beta + F_beta.T)

        # Step 2 (optional): DIIS on the stacked (2, M, M) representation.
        if diis_engine is not None and iteration >= 1:
            F_stack = np.array([F_alpha, F_beta], dtype=np.float64)
            P_stack = np.array([P_alpha, P_beta], dtype=np.float64)
            F_extrap = diis_engine.update(S, P_stack, F_stack)
            # CDIIS.update for UHF returns a (2, M, M) extrapolated Fock.
            F_extrap = np.asarray(F_extrap, dtype=np.float64)
            if F_extrap.ndim == 3:
                F_alpha = F_extrap[0]
                F_beta = F_extrap[1]
            else:
                # Defensive fallback: if PySCF flattens to (2*M, M) (older
                # versions), reshape.  Shouldn't fire on 2.11 but cheap.
                F_extrap = F_extrap.reshape(2, M, M)
                F_alpha = F_extrap[0]
                F_beta = F_extrap[1]
            diis_engaged = True

        # Step 3: solve F_σ C_σ = S C_σ ε_σ for each spin independently.
        mo_energy_a, mo_coeff_a = _solve_generalized_eig(F_alpha, S)
        mo_energy_b, mo_coeff_b = _solve_generalized_eig(F_beta, S)

        # Step 4: form new densities from occupied orbitals (unit occupation
        # per spin orbital).
        P_alpha_new = _build_density_from_orbitals_uhf(mo_coeff_a, n_alpha)
        P_beta_new = _build_density_from_orbitals_uhf(mo_coeff_b, n_beta)

        # Convergence on max|ΔP_α| and max|ΔP_β|.
        residual = float(max(
            np.max(np.abs(P_alpha_new - P_alpha)),
            np.max(np.abs(P_beta_new - P_beta)),
        ))
        residual_history.append(residual)

        P_alpha = P_alpha_new
        P_beta = P_beta_new

        if residual < tol:
            if verify_equation:
                F_check = fock_builder(state, (P_alpha, P_beta))
                checks = []
                for F_sigma, n_sigma in zip(F_check, (n_alpha, n_beta)):
                    e_sigma, C_sigma = _solve_generalized_eig(
                        0.5 * (F_sigma + F_sigma.T), S)
                    checks.append((e_sigma, C_sigma,
                                   _build_density_from_orbitals_uhf(C_sigma, n_sigma)))
                true_defect = max(np.max(np.abs(checks[0][2] - P_alpha)),
                                  np.max(np.abs(checks[1][2] - P_beta)))
                if true_defect >= tol:
                    if not polish_attempted and M * (M + 1) <= 40:
                        polish_attempted = True
                        polished, polish_iterations, success = _polish_uhf_equation(
                            (P_alpha, P_beta), S, (n_alpha, n_beta), state,
                            fock_builder, tol)
                        if success:
                            mo_energy_a, mo_coeff_a, P_alpha = polished[0]
                            mo_energy_b, mo_coeff_b, P_beta = polished[1]
                            converged = True
                            break
                    # A tiny DIIS update can hide a nonzero equation
                    # defect. Restart its stale history and take the
                    # actual cycle, retaining the requested tolerance.
                    if diis_engine is not None:
                        diis_engine = pyscf_diis.CDIIS(mf)
                        diis_engine.space = diis_space
                    mo_energy_a, mo_coeff_a, P_alpha = checks[0]
                    mo_energy_b, mo_coeff_b, P_beta = checks[1]
                    continue
            converged = True
            break

    # A slow/stagnating accelerated sequence may never reach the tiny-step
    # trigger above.  The same bounded small-system polish is attempted
    # once at the iteration cap; a failed polish leaves converged=False.
    if verify_equation and not converged and not polish_attempted and M * (M + 1) <= 40:
        polished, polish_iterations, success = _polish_uhf_equation(
            (P_alpha, P_beta), S, (n_alpha, n_beta), state, fock_builder, tol)
        if success:
            mo_energy_a, mo_coeff_a, P_alpha = polished[0]
            mo_energy_b, mo_coeff_b, P_beta = polished[1]
            converged = True

    # Total energy at the converged density.  PySCF's energy_tot for UHF/UKS
    # accepts the (2, M, M) stacked DM.
    dm_final = np.array([P_alpha, P_beta], dtype=np.float64)
    energy_total = float(mf.energy_tot(dm=dm_final))

    # ⟨S²⟩ from PySCF.  spin_square returns (S², 2S+1); we want the first.
    # Provide the converged mo_coeff/mo_occ to the helper.  Build mo_occ
    # from the per-spin occupation pattern (first n_σ orbitals occupied).
    mo_occ_a = np.zeros(M, dtype=np.float64)
    mo_occ_a[:n_alpha] = 1.0
    mo_occ_b = np.zeros(M, dtype=np.float64)
    mo_occ_b[:n_beta] = 1.0
    mo_coeff_stack = np.array([mo_coeff_a, mo_coeff_b], dtype=np.float64)
    mo_occ_stack = np.array([mo_occ_a, mo_occ_b], dtype=np.float64)
    # Set the converged orbitals on the mf object so spin_square can read
    # them (PySCF's helper takes mo_coeff explicitly but uses the mf's
    # Mole for the overlap and basis dimensions).
    mf.mo_coeff = mo_coeff_stack
    mf.mo_occ = mo_occ_stack
    s2_tuple = mf.spin_square()
    s_squared = float(s2_tuple[0])

    # Check the equation actually solved, independently of the DIIS step.
    # No extrapolation is applied here.  Float32 NN roundoff can leave
    # this defect above the requested tolerance after tiny DIIS steps.
    F_actual = fock_builder(state, (P_alpha, P_beta))
    P_check = []
    for F_sigma, n_sigma in zip(F_actual, (n_alpha, n_beta)):
        _, C_sigma = _solve_generalized_eig(0.5 * (F_sigma + F_sigma.T), S)
        P_check.append(_build_density_from_orbitals_uhf(C_sigma, n_sigma))
    defects = [P_check[0] - P_alpha, P_check[1] - P_beta]
    cycle_defect_max = float(max(np.max(np.abs(d)) for d in defects))
    # Unweighted upper-triangle Euclidean norm, matching the GRAIL seam.
    cycle_defect_l2 = float(np.linalg.norm(np.concatenate([
        d[np.triu_indices(M)] for d in defects
    ])))

    return SCFResult(
        P_star=(P_alpha, P_beta),
        mo_energy=(mo_energy_a, mo_energy_b),
        mo_coeff=(mo_coeff_a, mo_coeff_b),
        iter_count=len(residual_history),
        converged=converged,
        energy_total=energy_total,
        residual_history=residual_history,
        diis_engaged=diis_engaged,
        s_squared=s_squared,
        cycle_defect_max=cycle_defect_max,
        cycle_defect_l2=cycle_defect_l2,
        equation_converged=bool(cycle_defect_max < tol),
        polish_iterations=polish_iterations,
    )
