"""Training-data generation and supervised regression for the UHF :class:`XCNet`.

SPEC_scf §3.2, §6 P3, §8.10 R3.

The XC surrogate learns the truth map
``(P_α, P_β) → (V_xc^α[P_α, P_β], V_xc^β[P_α, P_β])``
at the converged broken-symmetry UHF operating distribution.  We approximate
that distribution by:

  1. For each LHS-sampled bond length ``R ∈ [R_lo, R_hi]`` (default
     ``[1.10, 1.25]`` Å for the R3 focused sweep around the Coulson–Fischer
     point R_CF ≈ 1.16 Å, per SPEC_scf §8.10), converge the truth UHF SCF
     to obtain the broken-symmetry ``(P_α*(R), P_β*(R))``.
  2. Perturb each spin block in packed-flat space:
     ``P_σ*(R) → P_σ*(R) + scale · ξ_σ`` with ``ξ_α, ξ_β`` LHS-sampled
     symmetric directions and ``scale`` an LHS-sampled amplitude in
     ``[0, perturb_max]``.  ``perturb_max = 0.03`` per P3's pre-stop
     empirical finding (rel_mae ~1.35 % on closed-shell at this amplitude).
  3. Evaluate the truth spin-resolved B3LYP ``V_xc^(α,β)(P_α, P_β)`` via
     PySCF's ``mf.get_veff(dm=(P_α, P_β)) - mf.get_j(dm=P_α + P_β)`` —
     the Hartree-Coulomb piece (shared across spins, exactly computable)
     is removed so the NN learns only the XC piece (SPEC_scf §8.3).

Determinism: every entry point seeds Python random, numpy, and torch at
the top, so different seeds in the P6 conformal ensemble produce
reproducible perturbations (SPEC_scf §6 P6).

LHS implementation: numpy-only, pattern from
``emulator/twolevel/training.py::_lhs_sample``.  Conda env forbids scipy.

PySCF is imported lazily inside :func:`generate_training_data_uhf` (NOT at
module top) to keep the OpenMP triple-init off the package-collection
import path (SPEC_scf §8.5).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from ..state import MoleculeState, n_unique, pack_density_matrix
from .network import XCNet


# ---------------------------------------------------------------------------
# LHS sampler (numpy-only; pattern from emulator/twolevel/training.py).
# ---------------------------------------------------------------------------

def _lhs_sample(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin hypercube sample on ``[0, 1]^n_dims``.

    Each dimension has exactly one sample per stratum of width
    ``1 / n_samples``.  Replacement for ``scipy.stats.qmc.LatinHypercube``
    that uses only numpy.  Copied from
    ``emulator/twolevel/training.py::_lhs_sample``.
    """
    samples = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples).astype(float)
        u = rng.uniform(size=n_samples)
        samples[:, d] = (perm + u) / n_samples
    return samples


# ---------------------------------------------------------------------------
# Training data generation (UHF / UKS).
# ---------------------------------------------------------------------------

def generate_training_data_uhf(
    state_template: MoleculeState,
    n_samples: int = 8000,
    seed: int = 42,
    *,
    R_lo: float = 1.10,
    R_hi: float = 1.25,
    perturb_max: float = 0.03,
    n_R_anchors: int = 8,
) -> dict[str, np.ndarray]:
    """LHS-sampled training set for the UHF/UKS XC-net.

    For each sample ``i ∈ {0, …, n_samples-1}``:
      1. Pick a bond length ``R_i`` LHS-uniform on ``[R_lo, R_hi]``.
      2. Look up (with anchor caching) the truth-converged broken-symmetry
         ``(P_α*(R_i), P_β*(R_i))`` via
         :func:`scf.host.picard_iterate_scf` with DIIS.
      3. Draw a packed-flat perturbation direction
         ``ξ_i ∈ R^{2 n_unique}`` from LHS on ``[-1, 1]^{2 n_unique}``,
         and an amplitude ``α_i ∈ [0, perturb_max]`` LHS-uniform.
      4. Form ``P_i = unpack(pack((P_α*, P_β*)) + α_i ξ_i)`` — the
         ``unpack`` step automatically re-symmetrizes within each spin
         block, so the perturbed density pair is symmetric per spin
         by construction.
      5. Evaluate the truth spin-resolved
         ``V_xc^(α, β)(P_α_i, P_β_i)`` via PySCF and pack to
         ``V_xc_flat_i ∈ R^{2 n_unique}``.

    To keep the truth-SCF cost manageable, the ``R`` axis is discretized
    to ``n_R_anchors`` anchor points (default 8 across the narrow focused
    sweep [1.10, 1.25] Å) and ``P*(R)`` is cached once per anchor.

    Parameters
    ----------
    state_template : MoleculeState
        Template defining basis, method (must be ``"uhf"`` or ``"uks"``),
        XC functional.  The geometry's ``R`` is overridden per sample via
        :func:`scf._make_state.make_h2_state`.  Only H₂ is supported in R3.
    n_samples : int, default 8000
        Total LHS sample count.  SPEC_scf §7 recommends 8000 for STO-3G,
        16000 for 6-31G; the caller is responsible for scaling.
    seed : int, default 42
        Random seed for the LHS draws.  Forwarded to ``random.seed``,
        ``np.random.seed``, and ``torch.manual_seed`` for reproducibility.
    R_lo, R_hi : float
        Bond-length range in Angstrom.  Default ``[1.10, 1.25]`` Å is the
        SPEC_scf §8.10 focused R3 sweep centered on the Coulson–Fischer
        point.  Both extremes lie within the broken-symmetry branch above
        R_CF when ``init_guess="breaksym"``.
    perturb_max : float, default 0.03
        Maximum perturbation amplitude per packed-flat entry.  Matches
        the P3 pre-stop empirical sweet spot (rel_mae ~ 1.35 % at this
        amplitude on closed-shell) — the same heuristic applies for UHF
        per the task brief.
    n_R_anchors : int, default 8
        Number of distinct R values at which the truth UHF SCF is
        pre-converged.  Each anchor amortizes its SCF cost over
        roughly ``n_samples / n_R_anchors`` perturbed samples.

    Returns
    -------
    data : dict with keys
        ``"P_flat"`` : ndarray of shape ``(n_samples, 2*n_unique)``, float64
            Packed-flat doubled density inputs (α | β concatenated).
        ``"Vxc_flat"`` : ndarray of shape ``(n_samples, 2*n_unique)``, float64
            Packed-flat doubled B3LYP V_xc targets (V_α | V_β concatenated).
        ``"R"`` : ndarray of shape ``(n_samples,)``, float64
            Bond length for each sample (for diagnostics / stratified
            splits).
        ``"P_star_anchors"`` : ndarray of shape ``(n_R_anchors, 2, M, M)``, float64
            The converged truth (P_α, P_β) at each anchor (kept for
            verification / re-use).
        ``"R_anchors"`` : ndarray of shape ``(n_R_anchors,)``, float64
            The anchor bond lengths.

    Notes
    -----
    Wall-clock note for the SPEC_scf §7 default ``n_samples=8000`` on
    STO-3G H₂: most of the cost is the per-sample PySCF ``get_veff`` call
    (~10–15 ms on the conda env, modestly slower than closed-shell because
    of the spin-resolved kernel).  Expect ~100–150 s for training-data
    generation.

    PySCF is imported lazily inside this function (SPEC_scf §8.5).
    """
    # Deterministic seeding.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- Lazy PySCF imports (OpenMP triple-init guard) ------------------
    from .._make_state import make_h2_state
    from ..fock_build import _build_mf, _build_mol
    from ..host import picard_iterate_scf

    # ----- Sanity checks --------------------------------------------------
    if state_template.atom_symbols != ("H", "H"):
        raise NotImplementedError(
            "generate_training_data_uhf currently supports H₂ only; "
            f"got atom_symbols={state_template.atom_symbols!r}."
        )
    if state_template.method not in ("uhf", "uks"):
        raise ValueError(
            f"generate_training_data_uhf requires method ∈ {{'uhf', 'uks'}}; "
            f"got method={state_template.method!r}.  "
            "For closed-shell training use the (obsolete) P3 code path."
        )
    if R_lo <= 0.0 or R_hi <= R_lo:
        raise ValueError(
            f"require 0 < R_lo < R_hi; got R_lo={R_lo!r}, R_hi={R_hi!r}."
        )
    if perturb_max <= 0.0:
        raise ValueError(f"perturb_max must be positive, got {perturb_max!r}.")
    if n_samples < 1 or n_R_anchors < 1:
        raise ValueError(
            f"n_samples and n_R_anchors must be >= 1, got "
            f"n_samples={n_samples!r}, n_R_anchors={n_R_anchors!r}."
        )

    # Determine n_unique from the template geometry by building one Mole.
    template_mol = _build_mol(state_template)
    M = int(template_mol.nao_nr())
    nu_single = n_unique(M)            # per-spin packed length
    nu_total = 2 * nu_single           # doubled-seam length

    # ----- Precompute the R-anchor SCF solutions (broken-symmetry UHF) ----
    # Linearly spaced anchors across [R_lo, R_hi].  Even on the narrow
    # R3 sweep range, linear spacing is fine — the broken-symmetry branch
    # exists continuously across the entire interval once R > R_CF.
    R_anchors = np.linspace(R_lo, R_hi, n_R_anchors)
    P_star_anchors = np.empty((n_R_anchors, 2, M, M), dtype=np.float64)
    for k, R_k in enumerate(R_anchors):
        # Use the template's method/XC; force the breaksym initial guess so
        # the broken-symmetry branch is taken even very close to R_CF.
        state_k = make_h2_state(
            R=float(R_k),
            basis=state_template.basis,
            method=state_template.method,
            xc=state_template.xc_functional,
            init_guess="breaksym",
        )
        res_k = picard_iterate_scf(
            state_k,
            accelerator="diis",
            tol=1e-9,
            max_iter=200,
            diis_space=8,
        )
        if not res_k.converged:
            raise RuntimeError(
                f"truth UHF SCF failed to converge at R_anchor={R_k:.3f} Å "
                f"(iter_count={res_k.iter_count}, "
                f"last_resid={res_k.residual_history[-1]:.3e}); "
                "shrink the R range or raise max_iter."
            )
        P_alpha_star, P_beta_star = res_k.P_star
        P_star_anchors[k, 0] = np.asarray(P_alpha_star, dtype=np.float64)
        P_star_anchors[k, 1] = np.asarray(P_beta_star, dtype=np.float64)

    # ----- LHS draws ------------------------------------------------------
    # Dim layout: [R_unit, alpha_unit, ξ_1, …, ξ_{2*n_unique}].
    rng = np.random.default_rng(seed)
    n_dim = 2 + nu_total
    unit_cube = _lhs_sample(n_samples, n_dim, rng)  # (N, 2 + 2*n_unique) in [0,1]

    R_samples = R_lo + (R_hi - R_lo) * unit_cube[:, 0]                  # (N,)
    alpha_samples = perturb_max * unit_cube[:, 1]                       # (N,)
    xi_samples = 2.0 * unit_cube[:, 2:] - 1.0                           # (N, 2*n_unique)

    # ----- Build per-sample (P, V_xc) pairs -------------------------------
    P_flat_arr = np.empty((n_samples, nu_total), dtype=np.float64)
    Vxc_flat_arr = np.empty((n_samples, nu_total), dtype=np.float64)

    # Build one PySCF mf object per anchor R; reuse across all
    # perturbations of that anchor.
    mf_anchors: list[Any] = []
    mol_anchors: list[Any] = []
    for k, R_k in enumerate(R_anchors):
        state_k = make_h2_state(
            R=float(R_k),
            basis=state_template.basis,
            method=state_template.method,
            xc=state_template.xc_functional,
            init_guess="breaksym",
        )
        mol_k = _build_mol(state_k)
        mf_k = _build_mf(state_k, mol=mol_k)
        mol_anchors.append(mol_k)
        mf_anchors.append(mf_k)

    # Map each sample to its nearest anchor by R.  Anchor spacing on the
    # R3 sweep is ~ 0.02 Å (8 anchors across 0.15 Å) — well inside the
    # smooth-V_xc regime.
    anchor_idx = np.argmin(
        np.abs(R_samples[:, None] - R_anchors[None, :]), axis=1
    )  # (N,)

    from ..state import unpack_density_matrix

    for i in range(n_samples):
        k = int(anchor_idx[i])
        # Build the perturbed P in packed-flat space, then unpack to the
        # (P_α, P_β) pair for PySCF.  ``pack_density_matrix`` with
        # ``method="uhf"`` returns the doubled-seam packing.
        P_star_tuple = (
            torch.from_numpy(P_star_anchors[k, 0]),
            torch.from_numpy(P_star_anchors[k, 1]),
        )
        P_star_flat = pack_density_matrix(
            P_star_tuple, method=state_template.method,
        ).numpy()  # (2*n_unique,)
        P_perturbed_flat = P_star_flat + alpha_samples[i] * xi_samples[i]

        # Unpack to a (P_α, P_β) tuple → ndarrays for PySCF.
        P_alpha_t, P_beta_t = unpack_density_matrix(
            torch.as_tensor(P_perturbed_flat, dtype=torch.float64),
            M=M,
            method=state_template.method,
        )
        P_alpha_mat = P_alpha_t.numpy()
        P_beta_mat = P_beta_t.numpy()

        mol_k = mol_anchors[k]
        mf_k = mf_anchors[k]

        # Truth V_xc^(α,β) = V_eff^(α,β) − J[P_α + P_β].  PySCF's
        # ``mf.get_veff(dm=(P_α, P_β))`` returns the (2, M, M) spin-resolved
        # effective potential (Hartree + XC + exact-exchange-mix in B3LYP);
        # ``mf.get_j(dm=P_total)`` returns the spin-independent Hartree
        # potential.  Subtracting J isolates V_xc^σ per spin (SPEC_scf
        # §8.3 generalized to the open-shell case).
        dm_pair = np.array([P_alpha_mat, P_beta_mat], dtype=np.float64)
        v_eff = np.asarray(mf_k.get_veff(mol_k, dm=dm_pair), dtype=np.float64)
        # v_eff shape: (2, M, M).  Hartree-Coulomb is the same matrix for
        # both spins, so subtract a single J once per spin.
        P_total = P_alpha_mat + P_beta_mat
        J = np.asarray(mf_k.get_j(mol_k, dm=P_total), dtype=np.float64)
        V_xc_alpha = v_eff[0] - J
        V_xc_beta = v_eff[1] - J
        # Defensive symmetrize: PySCF's XC quadrature can leak ~1e-14
        # antisymmetric noise.
        V_xc_alpha = 0.5 * (V_xc_alpha + V_xc_alpha.T)
        V_xc_beta = 0.5 * (V_xc_beta + V_xc_beta.T)

        P_flat_arr[i, :] = P_perturbed_flat
        Vxc_flat_arr[i, :] = pack_density_matrix(
            (torch.from_numpy(V_xc_alpha), torch.from_numpy(V_xc_beta)),
            method=state_template.method,
        ).numpy()

    return {
        "P_flat": P_flat_arr,
        "Vxc_flat": Vxc_flat_arr,
        "R": R_samples,
        "P_star_anchors": P_star_anchors,
        "R_anchors": R_anchors,
    }


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------

def train_xc_emulator_uhf(
    xc_net: XCNet,
    data: dict[str, np.ndarray],
    *,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[float] | float]:
    """Supervised regression: MSE on packed-flat doubled ``V_xc``.

    Trains ``xc_net`` in-place on the LHS-sampled ``(P, V_xc)`` data
    produced by :func:`generate_training_data_uhf`.  The training loop is
    deliberately simple — Adam + per-column-rescaled MSE — mirroring
    ``emulator/twolevel/training.py::train`` at smaller scale.

    Parameters
    ----------
    xc_net : XCNet
        Network to train in-place.  Its ``n_unique`` attribute carries the
        *total* (doubled) seam length 2*M(M+1)/2 — for STO-3G H₂ that's 6.
    data : dict
        Output of :func:`generate_training_data_uhf`.  Only ``"P_flat"`` and
        ``"Vxc_flat"`` are consumed by the loop; the other keys are
        carried through for diagnostics.
    epochs : int, default 300
        Number of full passes over the training set.
    batch_size : int, default 64
        Mini-batch size.
    lr : float, default 1e-3
        Adam learning rate.
    val_fraction : float, default 0.2
        Fraction held out for validation.
    seed : int, default 42
        Seed for the train/val shuffle and mini-batch ordering.  Forwarded
        to Python ``random``, numpy, and torch.

    Returns
    -------
    history : dict with keys
        ``"train_loss"``       : list[float] — per-epoch normalized MSE on the train split.
        ``"val_loss"``         : list[float] — per-epoch normalized MSE on the val split.
        ``"val_rel_mae"``      : list[float] — per-epoch rel MAE on the val split.
        ``"final_val_rel_mae"`` : float — last epoch's val rel_mae; gated below 3%.
        ``"val_indices"``      : ndarray — indices into ``data["P_flat"]``.

    Notes
    -----
    Loss is per-column-rescaled MSE: each packed-flat entry is normalized
    by the training-set RMS of |V_xc| for that column.  This is the same
    pattern used in the P3 closed-shell training — the per-column scale
    equalizes the gradient signal across packed-flat entries (so the
    off-diagonal V_xc[α, 0, 1], which is small in the broken-symmetry
    regime, gets a learning signal commensurate with the larger diagonals).
    """
    # Determinism.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    P_flat_np = np.ascontiguousarray(data["P_flat"], dtype=np.float64)
    Vxc_flat_np = np.ascontiguousarray(data["Vxc_flat"], dtype=np.float64)
    if P_flat_np.shape != Vxc_flat_np.shape:
        raise ValueError(
            f"P_flat and Vxc_flat shape mismatch: "
            f"{P_flat_np.shape!r} vs {Vxc_flat_np.shape!r}."
        )
    n_total = P_flat_np.shape[0]
    if P_flat_np.shape[1] != xc_net.n_unique:
        raise ValueError(
            f"data n_unique={P_flat_np.shape[1]} does not match "
            f"xc_net.n_unique={xc_net.n_unique}."
        )

    # Train/val split (deterministic).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_val = int(n_total * val_fraction)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    # Tensorize.
    P_t = torch.from_numpy(P_flat_np)
    V_t = torch.from_numpy(Vxc_flat_np)
    P_train = P_t[train_idx]
    V_train = V_t[train_idx]
    P_val = P_t[val_idx]
    V_val = V_t[val_idx]

    optimizer = torch.optim.Adam(xc_net.parameters(), lr=lr)
    # Step-LR schedule: drop the rate by 10x at 50% and 80% of total
    # epochs.  Same schedule pattern as the P3 closed-shell training, which
    # found the bare Adam(lr=1e-3) loss plateaus around epoch ~50 of 300.
    sched_milestones = [int(epochs * 0.5), int(epochs * 0.8)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched_milestones, gamma=0.1,
    )

    # Per-entry-scaled MSE loss.  Same per-column-rescaling trick as in
    # the P3 closed-shell training: normalize each packed-flat *column* by
    # the training-set RMS of |V_xc| for that column.  This equalizes the
    # loss contribution across packed-flat entries.
    col_rms = V_train.abs().pow(2).mean(dim=0).sqrt() + 1e-8       # (n_unique,)
    col_rms_f32 = col_rms.to(torch.float32)

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred / target: (B, n_unique), float32 by the time we get here.
        return ((pred - target) / col_rms_f32).pow(2).mean()

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_rel_mae": [],
        "val_indices": val_idx,
    }

    n_train = P_train.shape[0]
    for _epoch in range(epochs):
        # ---- train pass ----
        xc_net.train()
        epoch_perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = epoch_perm[start : start + batch_size]
            P_batch = P_train[idx]              # (B, n_unique), float64
            V_batch = V_train[idx]              # (B, n_unique), float64
            optimizer.zero_grad()
            pred = xc_net(P_batch)              # (B, n_unique), float64
            loss = loss_fn(
                pred.to(torch.float32),
                V_batch.to(torch.float32),
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ---- val pass ----
        xc_net.eval()
        with torch.no_grad():
            val_pred = xc_net(P_val)
            val_loss = float(
                loss_fn(
                    val_pred.to(torch.float32),
                    V_val.to(torch.float32),
                ).item()
            )
            denom = V_val.abs() + 1e-8
            rel_mae = float(
                ((val_pred - V_val).abs() / denom).mean().item()
            )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_rel_mae"].append(rel_mae)

    history["final_val_rel_mae"] = float(history["val_rel_mae"][-1])
    return history
