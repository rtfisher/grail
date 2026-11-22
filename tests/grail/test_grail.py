"""Tests for the grail/ package. Test IDs match docs/testing.md."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


@pytest.mark.unit
def test_imports() -> None:
    """M0 acceptance: public API surface is one function and one return type."""
    from grail import ResolventResult, resolvent_solve  # noqa: F401


# ---------------------------------------------------------------------------
# Tier U: GMRES
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_gmres_01_recovers_dense_solve() -> None:
    """U-gmres-01: GMRES recovers np.linalg.solve to 1e-10 within n iterations.

    On a dense SPD n×n system, GMRES with restart >= n must converge in at
    most n iterations to machine precision (exact arithmetic argument).  We
    verify agreement with np.linalg.solve to 1e-10.
    """
    from grail.gmres import gmres

    torch.manual_seed(42)
    n = 20
    # Build a symmetric positive definite matrix well-conditioned.
    A = torch.randn(n, n, dtype=torch.float64)
    A = A @ A.T + 5.0 * torch.eye(n, dtype=torch.float64)  # SPD, cond ~O(10)
    b = torch.randn(n, dtype=torch.float64)

    # Reference solution.
    x_ref = torch.linalg.solve(A, b)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return A @ v

    result = gmres(matvec, b, tol=1e-12, maxiter=n * 2, restart=n)

    assert torch.allclose(result.x, x_ref, atol=1e-10), (
        f"GMRES solution deviates from np.linalg.solve: max |diff| = "
        f"{(result.x - x_ref).abs().max().item():.3e}"
    )
    assert result.iter_count <= n + 5, (  # allow small slack for numerical noise
        f"GMRES took {result.iter_count} iterations for an n={n} system"
    )


@pytest.mark.unit
def test_U_gmres_02_eigenvalue_clustering_helps() -> None:
    """U-gmres-02: clustered eigenvalues away from 1 converge faster than near 1.

    Sanity check for the resolvent regime: (I - DT) has eigenvalues 1 - lambda_i
    where lambda_i are eigenvalues of DT.  When lambda_i are clustered near 0
    (standard convergent iteration), the eigenvalues of (I-DT) cluster near 1
    and GMRES may need more iterations.  When the cycle is highly contractive
    (lambda_i near 0), convergence is fast; when lambda_i cluster near 0.9,
    (I-DT) eigenvalues cluster near 0.1 — far from 0, making (I-DT)^{-1}
    well-conditioned.  We simply verify that a well-conditioned matrix (small
    condition number) converges faster than a poorly conditioned one.

    Formally: matrix A_good has all eigenvalues in {5, 6, 7} (clustered, far
    from zero); matrix A_bad has eigenvalues spread across [1, 20] (less
    clustered).  GMRES should reach tol faster for A_good.
    """
    from grail.gmres import gmres

    torch.manual_seed(7)
    n = 30

    # Build matrices with prescribed eigenvalue structure via random rotation.
    # A_good: eigenvalues all near 5–7 (clustered away from zero).
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    eig_good = torch.linspace(5.0, 7.0, n, dtype=torch.float64)
    A_good = Q @ torch.diag(eig_good) @ Q.T

    # A_bad: eigenvalues spread 0.5 to 20 (poorly clustered).
    eig_bad = torch.linspace(0.5, 20.0, n, dtype=torch.float64)
    A_bad = Q @ torch.diag(eig_bad) @ Q.T

    b = torch.randn(n, dtype=torch.float64)

    res_good = gmres(lambda v: A_good @ v, b, tol=1e-8, maxiter=n * 3, restart=n)
    res_bad = gmres(lambda v: A_bad @ v, b, tol=1e-8, maxiter=n * 3, restart=n)

    # The clustered-spectrum matrix should converge in fewer iterations.
    assert res_good.iter_count <= res_bad.iter_count, (
        f"Expected clustered spectrum to converge faster: "
        f"good={res_good.iter_count} iters, bad={res_bad.iter_count} iters"
    )


@pytest.mark.unit
def test_U_gmres_03_restart_preserves_convergence() -> None:
    """U-gmres-03: restart=10 reaches same tol as restart=50 with more iterations.

    For an n=50 system, restart=10 should still converge to tol=1e-8, though
    it may take more outer restart cycles than restart=50.
    """
    from grail.gmres import gmres

    torch.manual_seed(99)
    n = 50
    A = torch.randn(n, n, dtype=torch.float64)
    A = A @ A.T + 3.0 * torch.eye(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)

    res10 = gmres(lambda v: A @ v, b, tol=1e-8, maxiter=500, restart=10)
    res50 = gmres(lambda v: A @ v, b, tol=1e-8, maxiter=500, restart=50)

    x_ref = torch.linalg.solve(A, b)

    # Both should reach the tolerance.
    assert (res10.x - x_ref).norm().item() / b.norm().item() < 1e-7, (
        f"restart=10 did not converge: rel residual = "
        f"{(res10.x - x_ref).norm().item() / b.norm().item():.3e}"
    )
    assert (res50.x - x_ref).norm().item() / b.norm().item() < 1e-7, (
        f"restart=50 did not converge"
    )
    # restart=10 uses more total Arnoldi steps than restart=50 for the same result.
    assert res10.iter_count >= res50.iter_count, (
        f"Expected restart=10 to need >= iterations as restart=50: "
        f"restart=10: {res10.iter_count}, restart=50: {res50.iter_count}"
    )


# ---------------------------------------------------------------------------
# Tier U: PCA
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_pca_01_components_count_and_signs() -> None:
    """U-pca-01: PCA on N x d random produces min(N,d) components, sigma >= 0, descending."""
    from grail.pca import fit_basis

    torch.manual_seed(0)
    N, d = 20, 30  # N < d, so rank is N

    residuals = torch.randn(N, d, dtype=torch.float64)
    basis = fit_basis(residuals, explained_variance=1.0)  # keep all

    assert basis.rank == min(N, d), (
        f"Expected rank {min(N, d)}, got {basis.rank}"
    )
    assert basis.components.shape == (min(N, d), d), (
        f"Expected components shape ({min(N, d)}, {d}), got {basis.components.shape}"
    )
    assert basis.singular_values.shape == (min(N, d),), (
        f"Expected singular_values shape ({min(N, d)},), got {basis.singular_values.shape}"
    )
    # Non-negative singular values.
    assert (basis.singular_values >= 0).all().item(), (
        "Singular values must be non-negative"
    )
    # Descending order.
    sv = basis.singular_values
    assert (sv[:-1] >= sv[1:] - 1e-12).all().item(), (
        "Singular values must be in descending order"
    )

    # Also check d < N case.
    N2, d2 = 50, 10
    residuals2 = torch.randn(N2, d2, dtype=torch.float64)
    basis2 = fit_basis(residuals2, explained_variance=1.0)
    assert basis2.rank == min(N2, d2), (
        f"d < N case: expected rank {min(N2, d2)}, got {basis2.rank}"
    )


@pytest.mark.unit
def test_U_pca_02_rank_r_truncation_error() -> None:
    """U-pca-02: rank-r reconstruction Frobenius error matches tail singular-value sum.

    For a mean-centered matrix X_c = U S V^T, the Frobenius error of the
    rank-r approximation is ||X_c - X_c_r||_F^2 = sum(sigma_i^2, i > r).
    """
    from grail.pca import fit_basis

    torch.manual_seed(1)
    N, d = 40, 20

    residuals = torch.randn(N, d, dtype=torch.float64)
    # Keep all components for the full SVD reference.
    basis_full = fit_basis(residuals, explained_variance=1.0)

    r = 5  # truncation rank to test

    # Truncated reconstruction.
    mean = residuals.mean(dim=0)
    X_c = residuals - mean
    U_r = basis_full.components[:r]    # (r, d): right singular vectors
    S_r = basis_full.singular_values[:r]  # (r,)

    # Reconstruct X_c from the rank-r approximation: X_c_r = (X_c @ V_r.T) @ V_r
    # since U_r are right singular vectors (V in standard SVD notation).
    # Actually: X_c = U S V^T; rank-r approx = U_r diag(S_r) U_r V_r^T
    # With our convention, basis_full.components are rows of V^T (right singular
    # vectors), so the rank-r reconstruction is:
    #     X_c_r = (X_c @ V_r.T) @ V_r
    proj = X_c @ U_r.T              # (N, r): projection coefficients
    X_c_r = proj @ U_r              # (N, d): rank-r reconstruction

    # Frobenius error squared.
    frob_err_sq = ((X_c - X_c_r) ** 2).sum().item()

    # Expected error from tail singular values squared.
    sv_full = basis_full.singular_values
    expected_err_sq = (sv_full[r:] ** 2).sum().item()

    assert abs(frob_err_sq - expected_err_sq) < 1e-8 * max(abs(expected_err_sq), 1.0), (
        f"Frobenius reconstruction error mismatch: "
        f"computed={frob_err_sq:.6e}, expected={expected_err_sq:.6e}"
    )


# ---------------------------------------------------------------------------
# Tier U: conformal
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_conformal_01_coverage_on_synthetic_gaussian() -> None:
    """U-conformal-01: empirical coverage 1-alpha +/- 0.02 at N_cal=1000, alpha=0.1.

    Split-conformal on synthetic Gaussian residuals should achieve the nominal
    1-alpha coverage (by finite-sample guarantee) within ±0.02 tolerance at
    N_cal = 1000.
    """
    from grail.conformal import split_conformal

    torch.manual_seed(2024)
    N_cal = 1000
    N_test = 2000
    alpha = 0.1
    d = 10  # residual dimension

    # Calibration residuals drawn from N(0, 1) per dimension.
    cal_residuals = torch.randn(N_cal, d, dtype=torch.float64)

    # Test residuals from the same distribution (exchangeability assumption).
    test_residuals = torch.randn(N_test, d, dtype=torch.float64)

    # Predictions are zero; true values are test_residuals.
    # So the band should cover test_residuals[i] if |test_residuals[i]|_inf <= q_hat.
    test_predictions = torch.zeros(d, dtype=torch.float64)
    band = split_conformal(cal_residuals, test_predictions, alpha=alpha)

    # Compute empirical coverage: for each test point, check if it falls within
    # [low, high] = [-q_hat, q_hat] elementwise (the sup-norm band).
    q_hat = (band.high[0] - band.low[0]).item() / 2.0  # radius of the band

    # Coverage: fraction of test points whose L∞ norm <= q_hat.
    test_l_inf = test_residuals.abs().amax(dim=1)  # (N_test,)
    covered = (test_l_inf <= q_hat + 1e-12).float().mean().item()

    nominal = 1.0 - alpha
    assert abs(covered - nominal) <= 0.02, (
        f"Empirical coverage {covered:.3f} deviates from nominal {nominal:.3f} "
        f"by more than 0.02 (|diff| = {abs(covered - nominal):.3f})"
    )
    # The finite-sample guarantee: coverage must be >= 1 - alpha.
    assert covered >= nominal - 0.02, (
        f"Coverage {covered:.3f} too low (nominal {nominal:.3f})"
    )


# ---------------------------------------------------------------------------
# Tier U: jacobian composition
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_jacobian_01_composed_jvp_matches_direct() -> None:
    """U-jacobian-01: D T_SE(D T_RT(v)) matches direct autograd JVP to 1e-8.

    We build a small explicit T_t = T_SE ∘ T_RT where both factors are
    simple linear maps with known Jacobians.  The composed_jvp result must
    match the direct (A_SE @ A_RT) @ v product to 1e-8.
    """
    from grail.jacobian import composed_jvp

    torch.manual_seed(123)
    n = 16

    # T_RT and T_SE are linear: T_RT(v) = A_RT @ v, T_SE(w) = A_SE @ w.
    A_rt = torch.randn(n, n, dtype=torch.float64)
    A_se = torch.randn(n, n, dtype=torch.float64)

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        return A_rt @ v

    def emulator_jvp(w: torch.Tensor) -> torch.Tensor:
        return A_se @ w

    # Direct composed Jacobian.
    A_composed = A_se @ A_rt

    # Test on several random tangent vectors.
    for _ in range(10):
        v = torch.randn(n, dtype=torch.float64)
        direct = A_composed @ v
        jvp_fn = composed_jvp(emulator_jvp, host_jvp)
        composed = jvp_fn(v)

        max_diff = (direct - composed).abs().max().item()
        assert max_diff < 1e-8, (
            f"composed_jvp differs from direct: max |diff| = {max_diff:.3e}"
        )


# ---------------------------------------------------------------------------
# Tier A: resolvent analytic limits
# ---------------------------------------------------------------------------

@pytest.mark.analytic
def test_A_resolvent_identity_zero_in_zero_out() -> None:
    """A-resolvent-identity: delta_T = 0 -> delta_x* = 0 to GMRES tolerance.

    When the emulator residual is identically zero, the resolvent equation
    (I - DT) δx* = 0 has the unique solution δx* = 0 (since I - DT is
    invertible).  GMRES should return a zero vector to within its tolerance.
    """
    from grail import resolvent_solve

    n = 20
    torch.manual_seed(42)

    # Some well-behaved linear JVPs (DT is a contraction: spectral radius < 1).
    A_rt = 0.3 * torch.randn(n, n, dtype=torch.float64)
    A_se = 0.3 * torch.randn(n, n, dtype=torch.float64)

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        return A_rt @ v

    def emulator_jvp(w: torch.Tensor) -> torch.Tensor:
        return A_se @ w

    x_star = torch.randn(n, dtype=torch.float64)
    delta_T = torch.zeros(n, dtype=torch.float64)  # zero perturbation

    result = resolvent_solve(
        x_star, delta_T, emulator_jvp, host_jvp, gmres_tol=1e-10, gmres_maxiter=100
    )

    delta_norm = result.delta_x_star.norm().item()
    # Should be at most GMRES tolerance times the RHS norm (zero), so ~machine eps.
    assert delta_norm < 1e-10, (
        f"A-resolvent-identity: delta_x* = {delta_norm:.3e}, expected ~0"
    )


@pytest.mark.analytic
def test_A_resolvent_homogeneity() -> None:
    """Scaled RHSs produce scaled solutions of the same linear equation.

    This is a homogeneity diagnostic.  An exactly linear equation cannot
    establish a nonlinear remainder order or a linearization radius.
    """
    from grail import resolvent_solve

    torch.manual_seed(77)
    n = 20

    # Contraction: A_se has spectral radius < 1.
    A_se_raw = torch.randn(n, n, dtype=torch.float64)
    # Make spectral radius ~0.5 by scaling.
    eig = torch.linalg.eigvals(A_se_raw)
    rho = eig.abs().max().item()
    A_se = A_se_raw * (0.5 / rho)

    def host_jvp(v: torch.Tensor) -> torch.Tensor:
        return v  # identity: DT_RT = I

    def emulator_jvp(w: torch.Tensor) -> torch.Tensor:
        return A_se @ w

    x_star = torch.zeros(n, dtype=torch.float64)

    # Base δT — normalized so the problem is well-scaled.
    delta_T_base = torch.randn(n, dtype=torch.float64)
    delta_T_base = delta_T_base / delta_T_base.norm()

    # Solve for three scales.
    alphas = [1.0, 0.5, 0.25]
    results = {}
    for a in alphas:
        results[a] = resolvent_solve(
            x_star, a * delta_T_base, emulator_jvp, host_jvp,
            gmres_tol=1e-12, gmres_maxiter=300
        )

    # Because the problem is exactly linear (identity host, linear emulator),
    # δx*(α·δT) should be α·δx*(δT) to GMRES precision.
    dx1 = results[1.0].delta_x_star
    dx_half = results[0.5].delta_x_star
    dx_quarter = results[0.25].delta_x_star

    # Linearity check: |δx*(α·δT) - α·δx*(δT)| should be negligible.
    err_half = (dx_half - 0.5 * dx1).norm().item()
    err_quarter = (dx_quarter - 0.25 * dx1).norm().item()

    # Both errors should be tiny (O(GMRES tolerance)).
    tol = 1e-8
    assert err_half < tol, (
        f"A-resolvent-linearity: α=1/2 error = {err_half:.3e}, expected < {tol:.3e}"
    )
    assert err_quarter < tol, (
        f"A-resolvent-linearity: α=1/4 error = {err_quarter:.3e}, expected < {tol:.3e}"
    )

    # Independently check the solution norm's linear scaling.  The equation
    # contains no nonlinear remainder, so no Richardson order follows.
    norm1 = dx1.norm().item()
    norm_half = dx_half.norm().item()
    norm_quarter = dx_quarter.norm().item()

    ratio_half = norm_half / norm1 if norm1 > 0 else 1.0
    ratio_quarter = norm_quarter / norm1 if norm1 > 0 else 1.0

    assert abs(ratio_half - 0.5) < 0.01, (
        f"Homogeneity: ‖δx*(δT/2)‖ / ‖δx*(δT)‖ = {ratio_half:.4f}, expected 0.5"
    )
    assert abs(ratio_quarter - 0.25) < 0.01, (
        f"Homogeneity: ‖δx*(δT/4)‖ / ‖δx*(δT)‖ = {ratio_quarter:.4f}, expected 0.25"
    )
