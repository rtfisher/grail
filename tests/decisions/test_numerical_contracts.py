"""Negative controls for the repaired amplification/scaling/linearity gates."""
from __future__ import annotations

import pytest
import torch

from decisions._numerical_contracts import (
    LinearDiagnostics, conserving_probe, diagnose_linear_solve,
    population_tangent_basis,
)
from decisions import gate2_amplification as gate2
from decisions import gate3_gmres_scaling as gate3
from decisions import gate5_linearity as gate5
from grail import resolvent_solve


def _case(gain=2.0, norm=3.0, iterations=10, cap=300, defect=1e-12):
    return LinearDiagnostics(
        gain, defect, iterations, cap, iterations >= cap, defect <= 1e-8,
        0.01, norm, 1e-14 if norm is not None else None,
        1e-13 if norm is not None else None, 1e-15 if norm is not None else None,
    )


@pytest.mark.unit
@pytest.mark.parametrize("levels", [2, 3, 5])
def test_probe_and_basis_conserve_population(levels):
    n = 4
    q = population_tangent_basis(n, levels)
    torch.testing.assert_close(q.T @ q, torch.eye((levels - 1)*n, dtype=torch.float64))
    rhs = conserving_probe(n, levels, 42)
    assert rhs.reshape(levels, n).sum(0).abs().max() < 1e-18


@pytest.mark.unit
def test_dense_inverse_norm_is_separate_from_directional_gain_and_hessenberg():
    q = population_tangent_basis(2, 2)
    k_reduced = torch.tensor([[0.7, 0.5], [0.0, 0.2]], dtype=torch.float64)
    k = q @ k_reduced @ q.T
    rhs = conserving_probe(2, 2, 42)
    res = resolvent_solve(torch.zeros(4), rhs, lambda v: k @ v, lambda v: v,
                         gmres_tol=1e-12)
    diagnostic = diagnose_linear_solve(
        res, rhs, lambda v: k @ v, lambda v: v, tolerance=1e-12,
        iteration_cap=200, n_tau=2, n_levels=2, dense_limit=2,
    )
    expected = torch.linalg.matrix_norm(torch.linalg.inv(torch.eye(2) - k_reduced), 2)
    assert diagnostic.tangent_inverse_norm == pytest.approx(float(expected), rel=1e-13)
    assert diagnostic.directional_gain <= diagnostic.tangent_inverse_norm
    assert diagnostic.true_relative_defect < 1e-12
    assert diagnostic.gmres_relative_solution_error < 1e-12
    # A wrong returned solution must fail even with an optimistic Krylov history.
    res.delta_x_star = torch.zeros(4, dtype=torch.float64)
    bad = diagnose_linear_solve(res, rhs, lambda v: k @ v, lambda v: v,
                               tolerance=1e-12, iteration_cap=200,
                               n_tau=2, n_levels=2)
    assert not bad.converged and bad.true_relative_defect == pytest.approx(1.0)


@pytest.mark.unit
def test_amplification_aggregates_same_case_bounds_and_defers_unmeasured_norm():
    # Different magnitudes: cross-comparing max gain to min norm would FAIL.
    result = gate2._summarize_cases("2level", [("a", _case()),
                                   ("b", _case(gain=20.0, norm=30.0))])
    assert result.status == "PASS"
    assert gate2._summarize_cases("2level", [("a", _case(norm=None))]).status == "DEFERRED"
    assert gate2._summarize_cases("2level", [("a", _case(gain=4.0))]).status == "FAIL"
    assert gate2._summarize_cases("2level", [("a", _case(defect=1e-3))]).status == "FAIL"
    invalid_basepoint = _case()
    invalid_basepoint.basepoint_valid = False
    assert gate2._summarize_cases("2level", [("a", invalid_basepoint)]).status == "FAIL"


@pytest.mark.unit
def test_scaling_cannot_pass_capped_or_inaccurate_solves():
    cases = {100.0: _case(), 1000.0: _case(), 10000.0: _case()}
    assert gate3._summarize_cases("2level", 60, cases).status == "PASS"
    cases[10000.0] = _case(iterations=300)
    # A benign count ratio does not excuse a cap hit.
    cases[100.0] = _case(iterations=200)
    assert gate3._summarize_cases("2level", 60, cases).status == "FAIL"
    cases[10000.0] = _case(defect=1e-3)
    assert gate3._summarize_cases("2level", 60, cases).status == "FAIL"
    cases[10000.0] = _case()
    cases[10000.0].basepoint_valid = False
    assert gate3._summarize_cases("2level", 60, cases).status == "FAIL"


@pytest.mark.unit
def test_successful_homogeneity_defers_nonlinear_validity():
    # This linear equation has no nonlinear fixed point/remainder experiment.
    x = torch.zeros(2, dtype=torch.float64)
    b = torch.tensor([1e-3, -1e-3], dtype=torch.float64)
    def identity(v):
        return v

    def se(v):
        return 0.5*v
    residuals, passes, diagnostics = gate5._homogeneity_diagnostics(x, b, se, identity)
    assert passes and max(residuals) < 1e-12
    assert all(d.true_relative_defect < 1e-8 for d in diagnostics)
    result = gate5.run()
    assert result.status == "DEFERRED"
    assert result.measured is None
    assert "cannot establish a quadratic remainder" in result.message
