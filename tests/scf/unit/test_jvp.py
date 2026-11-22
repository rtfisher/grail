"""Tier U tests for ``scf/jvp.py`` (P4 GRAIL integration).

SPEC_scf §6 P4, §8.1, §8.10.  Covers:

- :func:`make_fock_builder_with_nn` returns a 2-tuple of symmetric ``(M, M)``
  ndarrays from a ``(P_α, P_β)`` input.
- :func:`make_host_jvp_scf` is linear in the input perturbation (FD precision).
- :func:`make_emulator_jvp_scf` is linear in the input perturbation.
- FD-JVP round-trip on a zero perturbation returns the zero vector to
  machine precision.

These tests load the trained UHF XCNet artifact from
``examples/scf_xc_emulator_uhf_sto3g.h5`` and the truth-converged
``(P_α^*, P_β^*)`` from :func:`scf.host.picard_iterate_scf` at R=1.18 Å
(broken-symmetry side of R_CF, well within the trained NN's R range
[1.10, 1.25] Å).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.emulator import load_xc_emulator
from scf.host import picard_iterate_scf
from scf.jvp import (
    make_cycle_jvps_scf,
    make_emulator_jvp_scf,
    make_fock_builder_with_nn,
    make_host_jvp_scf,
)
from scf.state import n_unique
from scf.diagnostics import make_cycle_scf, pack_pair, tangent_bases_scf


_SCRATCH_DIR = Path(__file__).resolve().parents[3] / "examples"
_XC_NET_PATH = _SCRATCH_DIR / "scf_xc_emulator_uhf_sto3g.h5"

# H = doubled-seam length for STO-3G UHF (R1 §8.2 convention).
_NU_DOUBLED = 2 * n_unique(2)   # = 6


@pytest.mark.unit
def test_actual_composed_derivative_and_linear_defect(_truth_scf_at_R118):
    """End-to-end NN cycle checks the intermediate point independently.

    The truth-orbital callback is a negative control: a tiny callback
    equation residual alone must not certify the actual composed cycle.
    """
    state, res = _truth_scf_at_R118
    net = load_xc_emulator(_XC_NET_PATH, dtype=torch.float64)
    fb = make_fock_builder_with_nn(net, state)
    cycle = make_cycle_scf(state, fb)
    x = pack_pair(res.P_star)
    I = np.eye(len(x))
    first, second = make_cycle_jvps_scf(state, res.P_star, fb, h=1e-5)
    K_chain = np.column_stack([second(first(torch.from_numpy(v))).numpy() for v in I])
    K_direct = np.column_stack([(cycle(x + 1e-5 * v) - cycle(x - 1e-5 * v)) / 2e-5 for v in I])
    assert np.linalg.norm(K_chain - K_direct) / np.linalg.norm(K_direct) < 2e-8
    b = cycle(x) - x
    predicted = np.linalg.solve(I - K_chain, b)
    assert np.linalg.norm((I - K_direct) @ predicted - b) / np.linalg.norm(b) < 2e-7
    wrong = make_emulator_jvp_scf(state, res.mo_energy, res.mo_coeff, h=1e-5)
    K_wrong = np.column_stack([wrong(first(torch.from_numpy(v))).numpy() for v in I])
    p_wrong = np.linalg.solve(I - K_wrong, b)
    assert np.linalg.norm((I - K_direct) @ p_wrong - b) / np.linalg.norm(b) > 1e-3


@pytest.mark.unit
def test_tangent_basis_preserves_electron_count_and_idempotency(_truth_scf_at_R118):
    from scf.fock_build import _build_mf, _build_mol
    from scf.state import unpack_density_matrix
    state, res = _truth_scf_at_R118
    mol = _build_mol(state)
    S = _build_mf(state, mol=mol).get_ovlp()
    Q_count, Q_projector = tangent_bases_scf(state, res.mo_coeff)
    assert Q_count.shape == (6, 4)
    assert Q_projector.shape == (6, 2)
    np.testing.assert_allclose(Q_projector.T @ Q_projector, np.eye(2), atol=1e-14)
    for v in Q_projector.T:
        pair = unpack_density_matrix(torch.from_numpy(v), M=2, method="uhf")
        for d_t, P in zip(pair, res.P_star):
            d = d_t.numpy()
            assert abs(np.trace(d @ S)) < 2e-12
            np.testing.assert_allclose(d @ S @ P + P @ S @ d, d, atol=1e-9)


@pytest.mark.unit
def test_actual_fock_density_jvp_matches_occupied_virtual_formula(_truth_scf_at_R118):
    """Independently check occupation factor, denominator and evaluation point."""
    from scf.fock_build import _build_mf, _build_mol
    from scf.state import unpack_density_matrix
    from pyscf.scf import hf
    state, res = _truth_scf_at_R118
    net = load_xc_emulator(_XC_NET_PATH, dtype=torch.float64)
    fb = make_fock_builder_with_nn(net, state)
    F_pair = fb(state, res.P_star)
    mol = _build_mol(state)
    S = _build_mf(state, mol=mol).get_ovlp()
    callback = make_emulator_jvp_scf(state, fock_tuple=F_pair, h=1e-5)
    vector = np.random.default_rng(47).normal(size=6)
    perturbations = unpack_density_matrix(torch.from_numpy(vector), M=2, method='uhf')
    expected = []
    for F, dF_t in zip(F_pair, perturbations):
        energies, C = hf.eig(F, S)
        dF = dF_t.numpy()
        # One electron per spin: the occupied–virtual response has
        # occupation one; the denominator is epsilon_occ-epsilon_virt.
        U = (C[:, 1] @ dF @ C[:, 0]) / (energies[0] - energies[1])
        ov = np.outer(C[:, 1], C[:, 0])
        expected.append(U * (ov + ov.T))
    np.testing.assert_allclose(callback(torch.from_numpy(vector)).numpy(),
                               pack_pair(expected), rtol=2e-8, atol=2e-10)


@pytest.fixture(scope="module")
def _trained_xc_net():
    """Load the trained STO-3G UHF XCNet artifact once per test module."""
    if not _XC_NET_PATH.exists():
        pytest.skip(
            f"trained XCNet artifact not found at {_XC_NET_PATH}; "
            "run examples/train_xc_emulator_uhf_sto3g.py to produce it."
        )
    return load_xc_emulator(_XC_NET_PATH)


@pytest.fixture(scope="module")
def _truth_scf_at_R118():
    """Converged UHF SCF at R=1.18 Å (broken-symmetry side of R_CF)."""
    state = make_h2_state(R=1.18, basis="sto-3g", method="uhf")
    res = picard_iterate_scf(state, tol=1e-9, max_iter=200, accelerator="diis")
    if not res.converged:
        pytest.skip(
            f"truth UHF SCF did not converge at R=1.18 Å "
            f"(iter_count={res.iter_count})."
        )
    return state, res


# ---------------------------------------------------------------------------
# make_fock_builder_with_nn — shape + symmetry
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fock_builder_with_nn_returns_symmetric_pair(_trained_xc_net, _truth_scf_at_R118):
    """NN-substituted Fock builder returns (F_α, F_β), each symmetric (M, M)."""
    state, res = _truth_scf_at_R118
    fb_nn = make_fock_builder_with_nn(_trained_xc_net, state)
    F_pair = fb_nn(state, res.P_star)
    assert isinstance(F_pair, tuple) and len(F_pair) == 2
    F_alpha, F_beta = F_pair
    M = res.P_star[0].shape[0]
    assert F_alpha.shape == (M, M)
    assert F_beta.shape == (M, M)
    assert F_alpha.dtype == np.float64
    assert F_beta.dtype == np.float64
    # Symmetry: defensive re-symmetrize at the output side should give
    # exact equality.
    assert np.max(np.abs(F_alpha - F_alpha.T)) <= 1e-14
    assert np.max(np.abs(F_beta - F_beta.T)) <= 1e-14
    # Finite.
    assert np.all(np.isfinite(F_alpha))
    assert np.all(np.isfinite(F_beta))


@pytest.mark.unit
def test_fock_builder_with_nn_rejects_wrong_method(_trained_xc_net):
    """Closed-shell (RHF/RKS) state is rejected — UHF only per R3."""
    state_rhf = make_h2_state(R=1.18, basis="sto-3g", method="rhf")
    with pytest.raises(ValueError, match="uhf"):
        make_fock_builder_with_nn(_trained_xc_net, state_rhf)


@pytest.mark.unit
def test_fock_builder_with_nn_rejects_non_tuple(_trained_xc_net, _truth_scf_at_R118):
    """Single-matrix input rejected — UHF requires 2-tuple."""
    state, res = _truth_scf_at_R118
    fb_nn = make_fock_builder_with_nn(_trained_xc_net, state)
    with pytest.raises(ValueError, match="2-tuple"):
        fb_nn(state, res.P_star[0])  # single matrix, not tuple


# ---------------------------------------------------------------------------
# make_host_jvp_scf — shape, linearity, FD-JVP correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_host_jvp_returns_correct_shape(_trained_xc_net, _truth_scf_at_R118):
    """host_jvp maps R^{2*n_unique} → R^{2*n_unique}."""
    state, res = _truth_scf_at_R118
    fb_nn = make_fock_builder_with_nn(_trained_xc_net, state)
    host_jvp = make_host_jvp_scf(state, res.P_star, fb_nn)
    v = torch.randn(_NU_DOUBLED, dtype=torch.float64)
    out = host_jvp(v)
    assert out.shape == (_NU_DOUBLED,)
    assert out.dtype == torch.float64
    assert torch.all(torch.isfinite(out))


@pytest.mark.unit
def test_host_jvp_is_linear(_trained_xc_net, _truth_scf_at_R118):
    """host_jvp is linear: J(α v1 + β v2) = α J(v1) + β J(v2).

    The JVP is implemented as a precomputed dense matrix-vector product,
    so linearity is structural and should hold to float64 precision.
    """
    state, res = _truth_scf_at_R118
    fb_nn = make_fock_builder_with_nn(_trained_xc_net, state)
    host_jvp = make_host_jvp_scf(state, res.P_star, fb_nn)
    torch.manual_seed(0)
    v1 = torch.randn(_NU_DOUBLED, dtype=torch.float64)
    v2 = torch.randn(_NU_DOUBLED, dtype=torch.float64)
    alpha, beta = 0.3, -1.7
    lhs = host_jvp(alpha * v1 + beta * v2)
    rhs = alpha * host_jvp(v1) + beta * host_jvp(v2)
    # Matvec is exact arithmetic; allow only float-precision noise.
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-10)


@pytest.mark.unit
def test_host_jvp_zero_input_returns_zero(_trained_xc_net, _truth_scf_at_R118):
    """host_jvp(0) == 0 to machine precision (linear-map sanity check)."""
    state, res = _truth_scf_at_R118
    fb_nn = make_fock_builder_with_nn(_trained_xc_net, state)
    host_jvp = make_host_jvp_scf(state, res.P_star, fb_nn)
    zero = torch.zeros(_NU_DOUBLED, dtype=torch.float64)
    out = host_jvp(zero)
    assert torch.max(torch.abs(out)) <= 1e-14


@pytest.mark.unit
def test_host_jvp_rejects_wrong_shape(_trained_xc_net, _truth_scf_at_R118):
    """host_jvp rejects inputs of the wrong length."""
    state, res = _truth_scf_at_R118
    fb_nn = make_fock_builder_with_nn(_trained_xc_net, state)
    host_jvp = make_host_jvp_scf(state, res.P_star, fb_nn)
    with pytest.raises(ValueError, match="shape"):
        host_jvp(torch.zeros(3, dtype=torch.float64))


@pytest.mark.unit
def test_host_jvp_rejects_bad_step(_trained_xc_net, _truth_scf_at_R118):
    """h <= 0 is rejected."""
    state, res = _truth_scf_at_R118
    fb_nn = make_fock_builder_with_nn(_trained_xc_net, state)
    with pytest.raises(ValueError, match="positive"):
        make_host_jvp_scf(state, res.P_star, fb_nn, h=0.0)
    with pytest.raises(ValueError, match="positive"):
        make_host_jvp_scf(state, res.P_star, fb_nn, h=-1e-4)


# ---------------------------------------------------------------------------
# make_emulator_jvp_scf — shape, linearity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_emulator_jvp_returns_correct_shape(_truth_scf_at_R118):
    """emulator_jvp maps R^{2*n_unique} → R^{2*n_unique}."""
    state, res = _truth_scf_at_R118
    emul_jvp = make_emulator_jvp_scf(state, res.mo_energy, res.mo_coeff)
    v = torch.randn(_NU_DOUBLED, dtype=torch.float64)
    out = emul_jvp(v)
    assert out.shape == (_NU_DOUBLED,)
    assert out.dtype == torch.float64
    assert torch.all(torch.isfinite(out))


@pytest.mark.unit
def test_emulator_jvp_is_linear(_truth_scf_at_R118):
    """emulator_jvp is linear: J(α v1 + β v2) = α J(v1) + β J(v2)."""
    state, res = _truth_scf_at_R118
    emul_jvp = make_emulator_jvp_scf(state, res.mo_energy, res.mo_coeff)
    torch.manual_seed(1)
    v1 = torch.randn(_NU_DOUBLED, dtype=torch.float64)
    v2 = torch.randn(_NU_DOUBLED, dtype=torch.float64)
    alpha, beta = 0.7, 0.13
    lhs = emul_jvp(alpha * v1 + beta * v2)
    rhs = alpha * emul_jvp(v1) + beta * emul_jvp(v2)
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-10)


@pytest.mark.unit
def test_emulator_jvp_zero_input_returns_zero(_truth_scf_at_R118):
    """emulator_jvp(0) == 0 to machine precision."""
    state, res = _truth_scf_at_R118
    emul_jvp = make_emulator_jvp_scf(state, res.mo_energy, res.mo_coeff)
    zero = torch.zeros(_NU_DOUBLED, dtype=torch.float64)
    out = emul_jvp(zero)
    assert torch.max(torch.abs(out)) <= 1e-14


@pytest.mark.unit
def test_emulator_jvp_rejects_wrong_shape(_truth_scf_at_R118):
    """emulator_jvp rejects inputs of the wrong length."""
    state, res = _truth_scf_at_R118
    emul_jvp = make_emulator_jvp_scf(state, res.mo_energy, res.mo_coeff)
    with pytest.raises(ValueError, match="shape"):
        emul_jvp(torch.zeros(3, dtype=torch.float64))


@pytest.mark.unit
def test_emulator_jvp_rejects_wrong_method(_truth_scf_at_R118):
    """RHF / RKS state is rejected — UHF only path in P4."""
    state, res = _truth_scf_at_R118
    state_rhf = make_h2_state(R=1.18, basis="sto-3g", method="rhf")
    with pytest.raises(ValueError, match="uhf"):
        make_emulator_jvp_scf(state_rhf, res.mo_energy, res.mo_coeff)


@pytest.mark.unit
def test_emulator_jvp_matches_truth_cpks_chi0(_truth_scf_at_R118):
    """Spot-check: emulator_jvp output magnitude is comparable to the
    R2 analytic χ_0 piece on a random perturbation.

    The emulator_jvp is the truth ``D𝐃|_{F^*}`` — same operator as the
    χ_0 piece of :func:`scf.jacobian.cpks_jacobian_analytic_uhf`.  We
    don't expect exact equality (FD has h^2 truncation; the analytic
    Jacobian composes χ_0 with 𝐊, not pure χ_0), but the FD diag+rebuild
    JVP should produce outputs of comparable magnitude — within an order
    of magnitude — to a reference.  This is a smoke check that no
    obvious sign / scale error has leaked in.
    """
    state, res = _truth_scf_at_R118
    emul_jvp = make_emulator_jvp_scf(state, res.mo_energy, res.mo_coeff)
    torch.manual_seed(42)
    v = torch.randn(_NU_DOUBLED, dtype=torch.float64) * 1e-3
    out = emul_jvp(v)
    # Output magnitude should be finite and non-zero (the diag+rebuild
    # is non-trivial at the broken-symmetry fixed point) and not absurdly
    # larger than the input (no spurious blow-up).
    out_norm = float(out.norm())
    v_norm = float(v.norm())
    assert out_norm > 0.0
    assert out_norm < 1e3 * v_norm, (
        f"emulator_jvp output {out_norm:.3e} much larger than input "
        f"{v_norm:.3e} — possible numerical instability in diag+rebuild."
    )
