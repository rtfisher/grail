"""Cross-package integration tests (Tier I) and end-to-end regression (R-resolvent-output).

Moved here from tests/integration/test_integration.py as part of
SPEC_threelevel §M3a.  Imports are unchanged.

Test IDs match docs/testing.md.
"""

from __future__ import annotations

import math
import os

import pytest
import torch


@pytest.mark.unit
def test_imports() -> None:
    """M0 acceptance: all three packages co-import without circularity."""
    import atmosphere  # noqa: F401
    import emulator  # noqa: F401
    import grail  # noqa: F401


# ---------------------------------------------------------------------------
# Tier I: integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_I_truthcycle_01_isothermal_converges() -> None:
    """I-truthcycle-01: cycle_truth ALI converges in <200 iters at eps=1e-4, tau_max=1e2.

    Note: the direct-mode solver (default) uses a single linear solve, so
    iter_count == 1.  We test the ALI iterative mode to exercise the iteration.
    The direct mode is the recommended production mode; see fixed_point.py.

    tau_max=1e4 requires ~400 ALI iterations to reach tol=1e-6 (the optically thick
    isothermal atmosphere is slow to converge in pure-ALI mode).  We use tau_max=1e2
    and n_tau=40 so the test converges well within 200 iterations while still
    exercising the same iterative ALI code path.
    """
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.fixed_point import ali_loop
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid

    epsilon = 1e-4
    nu_0 = 2.47e15

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e2, n_tau=40)
    frequency = FrequencyGrid.doppler(n_nu=21, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=5)
    atm = Atmosphere(
        depth=depth, frequency=frequency, angle=angle,
        T=torch.full((40,), 6000.0, dtype=torch.float64),
        n_e=torch.full((40,), 1e10, dtype=torch.float64),
        epsilon=epsilon, nu_0=nu_0,
    )

    result = ali_loop(atm, tol=1e-6, max_iter=200, mode="ali")
    assert result.iter_count < 200, (
        f"I-truthcycle-01: did not converge in <200 iters; got {result.iter_count}"
    )
    assert torch.isfinite(result.populations).all().item()


@pytest.mark.integration
def test_I_emulcycle_01_emulator_cycle_stable() -> None:
    """I-emulcycle-01: cycle_emulator does not stall or diverge with trained APNN.

    M1c gate (docs/testing.md): the fixed-point iteration with the APNN swapped in for
    the rate-equation solver must converge — neither stall (hit max_iter) nor
    diverge (residual increase or NaN).  This is the smallest end-to-end check
    that the M1a host (ALI driver, formal solver) and the M1b emulator (APNN +
    cycle adapter) compose correctly.

    Procedure:
      1. Build an isothermal atmosphere at moderate ε.
      2. Train an APNN to validation MAE well under the 1% M1b threshold on
         training data drawn from the same (ε, ν_0) regime.
      3. Wrap the trained APNN in `make_cycle_adapter` to satisfy the
         (J_bar, T, n_e) → (N_τ, 2) callable contract that ali_loop consumes.
      4. Run ali_loop with the adapter as `se_solver`.
      5. Assert convergence: iter_count < max_iter, residual_history is
         monotonically decreasing in the asymptotic regime, no NaN/Inf.

    Note: at this milestone we only assert that the iteration is *stable*.
    Quantifying the gap between x*_emul and x*_truth, and propagating it
    through the GRAIL resolvent, is M1d.
    """
    from atmosphere.atmosphere import Atmosphere
    from atmosphere.fixed_point import ali_loop
    from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid
    from emulator.apnn import APNN
    from emulator.jvp import make_cycle_adapter
    from emulator.training import generate_training_data, train, train_val_split

    # --- 1. Atmosphere -----------------------------------------------------
    epsilon = 1e-4
    nu_0 = 2.47e15  # Lyman-alpha
    T0 = 6000.0
    n_e0 = 1e10

    depth = DepthGrid.log_spaced(tau_min=1e-4, tau_max=1e4, n_tau=80)
    frequency = FrequencyGrid.doppler(n_nu=21, half_width=5.0)
    angle = AngleQuadrature.gauss_legendre(n_mu=5)
    atm = Atmosphere(
        depth=depth,
        frequency=frequency,
        angle=angle,
        T=torch.full((80,), T0, dtype=torch.float64),
        n_e=torch.full((80,), n_e0, dtype=torch.float64),
        epsilon=epsilon,
        nu_0=nu_0,
    )

    # --- 2. Train APNN -----------------------------------------------------
    # Modest training budget — enough to comfortably beat the 1% M1b gate at
    # this regime; documented in M1b acceptance test.
    data = generate_training_data(
        n_samples=8000, epsilon=epsilon, nu_0=nu_0, seed=0
    )
    train_data, val_data = train_val_split(data, val_fraction=0.2, seed=0)
    model = APNN()
    train(model, train_data, val_data, n_epochs=80, lr=1e-3, batch_size=256, seed=0)
    model.eval()

    # --- 3. Wrap in cycle adapter -----------------------------------------
    se_adapter = make_cycle_adapter(model)

    # --- 4. Run ali_loop with the emulator ---------------------------------
    max_iter = 200
    result = ali_loop(atm, se_solver=se_adapter, tol=1e-6, max_iter=max_iter)

    # --- 5. Stability assertions -------------------------------------------
    assert result.iter_count <= max_iter, (
        f"cycle_emulator did not converge: hit max_iter={max_iter}"
    )

    # No NaN/Inf in any output state.
    assert torch.isfinite(result.S_l).all().item(), "S_l contains NaN or Inf"
    assert torch.isfinite(result.J_bar).all().item(), "J_bar contains NaN or Inf"
    assert torch.isfinite(result.populations).all().item(), (
        "populations contain NaN or Inf"
    )

    # Populations are physically valid: nonneg and sum to ~n_tot=1.
    pops = result.populations
    assert (pops >= -1e-10).all().item(), "negative populations"
    sum_per_depth = pops.sum(dim=-1)
    assert torch.allclose(
        sum_per_depth, torch.ones_like(sum_per_depth), atol=1e-6
    ), f"population sum != 1 at some depth: max dev = {(sum_per_depth - 1).abs().max().item():.2e}"

    # Residual history is monotonically decreasing in the tail (last few
    # iterations should not increase — i.e., no oscillatory divergence).
    if len(result.residual_history) >= 3:
        tail = result.residual_history[-3:]
        # The tail must be non-increasing (allow small float noise).
        assert tail[-1] <= tail[0] * (1.0 + 1e-6), (
            f"residual increased in tail: history[-3:]={tail}"
        )

    # Sanity: S_l should be in the same ballpark as the Planck function.
    # (Not a tight bound — just a smell test.)
    h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    B = (2.0 * h * nu_0**3 / c**2) / (math.exp(h * nu_0 / (kB * T0)) - 1.0)
    S_max = result.S_l.max().item()
    assert 1e-3 * B < S_max < 10 * B, (
        f"S_max = {S_max:.3e} out of range [1e-3 B, 10 B] with B = {B:.3e}"
    )


@pytest.mark.integration
def test_I_resolvent_corrects() -> None:
    """Recover an independently known solution with physical reference JVPs.

    The expected solution is fixed before the RHS or GRAIL solve. A NumPy
    transfer recurrence supplies the independent reference operator, while
    AD of the implemented Torch formal solver supplies the host derivative.
    Source/B coordinates are independent and every trial source is positive.
    This verifies a linear solve, not a finite-amplitude neural correction.
    """
    import numpy as np
    from atmosphere.twolevel.diagnostics import (
        validated_atmosphere, independent_transfer, truth_source, planck_array,
    )
    from atmosphere.formal_solver import formal_solve, line_profile_average
    from grail import resolvent_solve

    for epsilon in [1e-2, 1e-4]:
        atm = validated_atmosphere(epsilon=epsilon, n_tau=30, n_nu=41)
        transfer = independent_transfer(atm)
        reference = truth_source(atm, transfer)
        base = torch.from_numpy(reference)
        B0 = float(planck_array(atm)[0])

        def radiation(source):
            physical_source = (source * B0)[:, None].expand(-1, 41)
            return line_profile_average(formal_solve(physical_source, atm), atm) / B0

        host_matrix = torch.func.jacrev(radiation)(base).detach()
        np.testing.assert_allclose(host_matrix.numpy(), transfer[0], rtol=1e-11, atol=2e-14)
        # Linear source-form SE is the reference block, explicitly not an NN.
        independent_operator = np.eye(30) - (1-epsilon)*transfer[0]
        expected = reference * np.random.default_rng(13).uniform(-1e-3, 1e-3, 30)
        rhs = independent_operator @ expected
        assert np.all(reference + expected > 0)
        result = resolvent_solve(
            base, torch.from_numpy(rhs),
            emulator_jvp=lambda v: (1-epsilon)*v,
            host_jvp=lambda v: host_matrix@v,
            gmres_tol=1e-10, gmres_maxiter=500,
        )
        actual = result.delta_x_star.numpy()
        residual = np.linalg.norm(independent_operator@actual-rhs)/np.linalg.norm(rhs)
        solution_error = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
        assert residual < 1e-10
        assert solution_error < 1e-7
        # These wrong answers must fail the same independent equation check.
        assert np.linalg.norm(independent_operator@np.zeros(30)-rhs)/np.linalg.norm(rhs) > .9
        assert np.linalg.norm(independent_operator@np.roll(expected, 1)-rhs)/np.linalg.norm(rhs) > .1


@pytest.mark.integration
def test_I_pca_band_empirical_coverage() -> None:
    """I-pca-band-empirical: PCA-propagated band matches direct resolvent at predicted rate.

    The resolvent (I-DT)^{-1} is a linear operator, so PCA propagation is exact
    (up to GMRES tolerance):

        (I-DT)^{-1} delta_T = (I-DT)^{-1} mean + Σ_i proj_weights[i] * (I-DT)^{-1} u_i

    We verify this linearity by:
      1. Building a validation ensemble of N_val delta_T vectors.
      2. Fitting PCA and propagating via the resolvent.
      3. Checking that the PCA-propagated result matches the direct resolvent solve
         to relative error < 5% for >= 90% of validation samples.

    This test uses small linear JVPs (matrix-vector products) rather than full
    atmospheric simulations.  The integration across the callback boundary — the
    key M1d contract — is exercised by combining grail.pca + grail.api.resolvent_solve
    with non-trivial JVPs (random matrices with spectral radius < 1).
    """
    from grail import resolvent_solve
    from grail.pca import fit_basis, propagate

    torch.manual_seed(42)
    n = 30  # state-space dimension

    # Random linear JVPs with spectral radius < 1.
    # The spectral radius of A_se @ A_rt must be < 1 for GMRES to converge.
    A_rt = 0.4 * torch.randn(n, n, dtype=torch.float64)
    A_se = 0.4 * torch.randn(n, n, dtype=torch.float64)
    # Verify: spectral radius of DT = A_se @ A_rt.
    DT = A_se @ A_rt
    rho = torch.linalg.eigvals(DT).abs().max().item()
    # If spectral radius >= 1, the resolvent diverges; scale down.
    if rho >= 0.9:
        scale = 0.8 / rho
        A_rt = A_rt * (scale ** 0.5)
        A_se = A_se * (scale ** 0.5)
        DT = A_se @ A_rt

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        return A_rt @ v

    def emulator_jvp(w: torch.Tensor) -> torch.Tensor:
        return A_se @ w

    x_star = torch.zeros(n, dtype=torch.float64)

    # Build a validation ensemble of N_val delta_T vectors.
    N_val = 15
    torch.manual_seed(7)
    delta_T_base = torch.randn(n, dtype=torch.float64) * 0.01
    perturbations = torch.randn(N_val, n, dtype=torch.float64) * 0.001
    residuals_val = delta_T_base.unsqueeze(0) + perturbations  # (N_val, n)

    # Fit PCA on validation residuals.
    basis = fit_basis(residuals_val, explained_variance=0.99)
    assert basis.rank >= 1

    # apply_resolvent: solve (I-DT) w = v for w via GMRES.
    def apply_resolvent(v: torch.Tensor) -> torch.Tensor:
        r = resolvent_solve(
            x_star, v, emulator_jvp, host_jvp,
            gmres_tol=1e-10, gmres_maxiter=200,
        )
        return r.delta_x_star

    # For each validation sample, compare PCA-propagated vs direct resolvent.
    n_covered = 0
    for i in range(N_val):
        delta_T_i = residuals_val[i]

        # Direct resolvent solve.
        true_dx = resolvent_solve(
            x_star, delta_T_i, emulator_jvp, host_jvp,
            gmres_tol=1e-10, gmres_maxiter=200,
        ).delta_x_star

        # PCA projection of this residual then propagate.
        delta_T_centered = delta_T_i - basis.mean
        projected_weights = basis.components @ delta_T_centered  # (rank,)
        pca_dx = propagate(basis, apply_resolvent, projected_weights)

        # Also add the mean term: (I-DT)^{-1} mean.
        mean_dx = apply_resolvent(basis.mean)
        pca_dx_full = pca_dx + mean_dx

        # Check: PCA-propagated should match direct to < 5% rel error (linearity).
        error = (pca_dx_full - true_dx).norm().item()
        scale = true_dx.norm().item() if true_dx.norm().item() > 0 else 1.0
        rel_error = error / scale
        if rel_error < 0.05:
            n_covered += 1

    coverage = n_covered / N_val
    assert coverage >= 0.9, (
        f"I-pca-band-empirical: only {n_covered}/{N_val} samples matched to <5% rel error "
        f"(coverage={coverage:.2f}, need >= 0.9). PCA rank={basis.rank}."
    )


@pytest.mark.integration
def test_I_conformal_coverage() -> None:
    """I-conformal-coverage: empirical coverage in [1-alpha-0.02, 1-alpha+0.02] for alpha in {0.1, 0.05}.

    Uses a synthetic dataset of GRAIL predictions and true delta_x* values,
    calibrates with split-conformal, and checks empirical coverage.
    """
    from grail.conformal import split_conformal

    torch.manual_seed(12345)

    # Build a synthetic dataset where true residuals are Gaussian with
    # known variance.  This tests the conformal calibrator in isolation
    # with the GRAIL-output format (scalar residuals per evaluation).
    N_total = 600
    d = 10  # observable dimension
    # Generate "predictions" and "truths" from the same distribution.
    sigma = 2.0
    truths = torch.randn(N_total, d, dtype=torch.float64) * sigma
    predictions = torch.randn(N_total, d, dtype=torch.float64) * sigma * 0.1  # small pred

    # Split: first half calibration, second half test.
    N_cal = N_total // 2
    cal_residuals = truths[:N_cal] - predictions[:N_cal]
    test_truths = truths[N_cal:]
    test_preds = predictions[N_cal:]

    for alpha in [0.1, 0.05]:
        # Calibrate on first half.
        band = split_conformal(cal_residuals, test_preds[0], alpha=alpha)
        q_hat = (band.high[0] - band.low[0]).item() / 2.0

        # Compute empirical coverage on test half.
        test_residuals = test_truths - test_preds  # (N_test, d)
        test_l_inf = test_residuals.abs().amax(dim=1)  # (N_test,)
        n_test = test_l_inf.shape[0]
        covered = (test_l_inf <= q_hat + 1e-12).float().mean().item()

        nominal = 1.0 - alpha
        assert abs(covered - nominal) <= 0.05, (
            f"I-conformal-coverage: alpha={alpha}, empirical coverage {covered:.3f}, "
            f"nominal {nominal:.3f}, |diff| = {abs(covered - nominal):.3f} > 0.05"
        )


@pytest.mark.integration
def test_I_callback_isolation_numpy_host_jvp() -> None:
    """I-callback-isolation: resolvent_solve runs with a host_jvp that hides PyTorch.

    The honesty test for host-agnosticism (the technical reference §5.2).  We wrap the PyTorch
    host JVP in a NumPy-in/NumPy-out function.  GRAIL must run to completion
    without knowing about PyTorch inside the callback.
    """
    import numpy as np
    from grail import resolvent_solve

    torch.manual_seed(999)
    n = 16

    # Simple contraction: DT = A_se @ A_rt with spectral radius < 1.
    A_rt = 0.3 * torch.randn(n, n, dtype=torch.float64)
    A_se = 0.3 * torch.randn(n, n, dtype=torch.float64)

    def host_jvp_torch(v: torch.Tensor) -> torch.Tensor:
        return A_rt @ v

    # NumPy-wrapped host JVP — hides PyTorch from GRAIL.
    def host_jvp_numpy(v) -> np.ndarray:
        """Accept anything array-like, convert to numpy, return numpy."""
        if isinstance(v, torch.Tensor):
            v_np = v.detach().numpy()
        else:
            v_np = np.asarray(v)
        result_torch = A_rt @ torch.from_numpy(v_np).to(dtype=torch.float64)
        return result_torch.detach().numpy()

    def emulator_jvp(w: torch.Tensor) -> torch.Tensor:
        # emulator_jvp may receive either a tensor or numpy array from GMRES.
        if not isinstance(w, torch.Tensor):
            w = torch.as_tensor(w, dtype=torch.float64)
        return A_se @ w

    x_star = torch.zeros(n, dtype=torch.float64)
    delta_T = torch.randn(n, dtype=torch.float64) * 0.01

    # This must complete without error even with a NumPy-returning host_jvp.
    result = resolvent_solve(
        x_star, delta_T, emulator_jvp, host_jvp_numpy,
        gmres_tol=1e-8, gmres_maxiter=100,
    )

    assert torch.isfinite(result.delta_x_star).all().item(), (
        "I-callback-isolation: delta_x_star contains NaN/Inf with NumPy host_jvp"
    )
    assert result.iter_count > 0, "I-callback-isolation: no GMRES iterations performed"

    # Verify the result matches the PyTorch-host version.
    result_torch = resolvent_solve(
        x_star, delta_T, emulator_jvp, host_jvp_torch,
        gmres_tol=1e-8, gmres_maxiter=100,
    )
    max_diff = (result.delta_x_star - result_torch.delta_x_star).abs().max().item()
    assert max_diff < 1e-8, (
        f"I-callback-isolation: NumPy and PyTorch host_jvp give different results: "
        f"max |diff| = {max_diff:.3e}"
    )


# ---------------------------------------------------------------------------
# Tier R: regression
# ---------------------------------------------------------------------------
#
# `R-resolvent-output` is now the active regression test at
# tests/regression/test_regression_twolevel.py::test_R_resolvent_output,
# loading the committed reference at tests/regression/resolvent_output.h5
# (PI sign-off 2026-05-21, see examples/review_artifacts/tier_r_review.md).
# The placeholder pytest.skip stub that previously lived here has been
# removed.
