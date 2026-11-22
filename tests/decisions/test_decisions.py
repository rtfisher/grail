"""Tests for the decisions/ Decision-Gate harness.

Test IDs
--------
test_decisions_imports       — smoke test (Tier U): package imports clean.
test_run_decision_gates_fast — end-to-end fast-mode run (Tier I / integration).
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_decisions_imports() -> None:
    """Smoke test: decisions package imports without error.

    Verifies that the public API surface (run_decision_gates, DecisionGateVerdict,
    GateResult) is accessible and the per-gate modules can be imported.
    """
    from decisions import run_decision_gates, DecisionGateVerdict, GateResult  # noqa: F401
    from decisions.runner import GateResult, DecisionGateVerdict  # noqa: F401

    # Per-gate modules.
    import decisions.gate2_amplification # noqa: F401
    import decisions.gate3_gmres_scaling  # noqa: F401
    import decisions.gate5_linearity     # noqa: F401

    # GateResult dataclass construction.
    g = GateResult(
        gate_id=1,
        name="test",
        status="PASS",
        measured=0.1,
        expected=0.25,
        message="ok",
    )
    assert g.gate_id == 1
    assert g.status == "PASS"


@pytest.mark.integration
def test_run_decision_gates_fast(tmp_path, monkeypatch) -> None:
    """End-to-end fast-mode run of the Decision-Gate harness.

    Asserts:
    - Verdict has the three retained verification contracts.
    - Gate 5 cannot PASS from linear homogeneity.
    - Bounded-diagnostic deferrals are retained explicitly.
    - all_pass is False iff any implemented gate is FAIL or ERROR.
    - summary is a non-empty string.
    """
    from decisions import run_decision_gates, GateResult

    monkeypatch.chdir(tmp_path)
    verdict = run_decision_gates(fast=True)

    # --- Structural assertions ---

    # Three retained physical contracts.
    assert len(verdict.gates) == 3, (
        f"Expected 3 GateResult entries, got {len(verdict.gates)}"
    )

    assert [g.gate_id for g in verdict.gates] == [2, 3, 5]
    deferred_ids = {g.gate_id for g in verdict.gates if g.status == "DEFERRED"}
    assert deferred_ids <= {2, 5}
    gate5 = verdict.gates[2]
    assert gate5.status == "DEFERRED"
    assert "Nonlinear validity unassessed" in gate5.message or gate5.status == "ERROR"
    implemented = [g for g in verdict.gates if g.status != "DEFERRED"]
    for g in implemented:
        assert g.status in {"PASS", "FAIL", "ERROR"}

    # all_pass consistency: True iff no implemented gate is FAIL or ERROR.
    expected_all_pass = bool(implemented) and all(g.status == "PASS" for g in implemented)
    assert verdict.all_pass == expected_all_pass, (
        f"verdict.all_pass={verdict.all_pass} but implemented gate statuses are "
        f"{[g.status for g in implemented]}"
    )

    # Summary is non-empty.
    assert len(verdict.summary) > 0, "summary should be a non-empty string"

    # All implemented gate messages are non-empty.
    for g in implemented:
        assert len(g.message) > 0, f"Gate {g.gate_id} has empty message"
