"""Run the amplification, GMRES and local-response verification contracts."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import traceback


@dataclass
class GateResult:
    """A measured contract result; deferred conditions remain explicit."""
    gate_id: int
    name: str
    status: str
    measured: Optional[float]
    expected: Optional[float]
    message: str

    def __post_init__(self):
        if self.status not in {'PASS','FAIL','DEFERRED','ERROR'}:
            raise ValueError('Unknown verification status: '+self.status)


@dataclass
class DecisionGateVerdict:
    """Results of the three implemented physical verification contracts."""
    gates: list[GateResult]
    all_pass: bool
    summary: str


def _run_gates_for_setup(setup, fast):
    from . import gate2_amplification, gate3_gmres_scaling, gate5_linearity
    modules=[(2,'Amplification factor',gate2_amplification),
             (3,'GMRES scaling',gate3_gmres_scaling),
             (5,'Local-response validity',gate5_linearity)]
    results=[]
    for gate_id,name,module in modules:
        name=f'{name} [{setup}]'
        try:
            result=module.run(fast=fast,setup=setup)
            result=GateResult(result.gate_id,name,result.status,result.measured,
                              result.expected,result.message)
        except Exception as exc:
            result=GateResult(gate_id,name,'ERROR',None,None,
                              f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
        results.append(result)
    return results


def run_decision_gates(*, fast=True, setup='2level'):
    """Run contracts 2, 3 and 5 on two-, three- or five-level host physics.

    ``both`` interleaves the two- and three-level results. ``all_pass``
    concerns measured contracts only; it does not certify deferred nonlinear
    validity or an unmeasured full-space inverse norm.
    """
    if setup not in {'2level','3level','multilevel','both'}:
        raise ValueError('Unknown host setup: '+setup)
    if setup=='both':
        a=_run_gates_for_setup('2level',fast);b=_run_gates_for_setup('3level',fast)
        gates=[g for pair in zip(a,b) for g in pair]
    else:gates=_run_gates_for_setup(setup,fast)
    measured=[g for g in gates if g.status!='DEFERRED']
    all_pass=bool(measured) and all(g.status=='PASS' for g in measured)
    summary='\n'.join(f'Contract {g.gate_id} [{g.status}]: {g.name}: {g.message}' for g in gates)
    return DecisionGateVerdict(gates,all_pass,summary)
