"""Tier A — A-scf-fast-conv.  SPEC_scf §6 P1.

RKS/B3LYP/STO-3G H₂ at R_eq = 0.74 Å converges in ≤ 10 iterates with the
default DIIS settings.  SPEC says ~5–10 iters in this well-conditioned
regime; we allow up to 10 to absorb run-to-run variation in DIIS history.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyscf")

from scf._make_state import make_h2_state
from scf.host import picard_iterate_scf


@pytest.mark.analytic
def test_A_scf_fast_conv_at_R_eq() -> None:
    """A-scf-fast-conv: well-conditioned RKS H₂ converges in ≤ 10 iterates."""
    state = make_h2_state(R=0.74, basis="sto-3g", method="rks", xc="B3LYP")
    result = picard_iterate_scf(
        state, accelerator="diis", tol=1e-9, max_iter=100, diis_space=8,
    )
    assert result.converged is True, (
        f"R_eq did not converge: iter_count={result.iter_count}, "
        f"last residual={result.residual_history[-1] if result.residual_history else None}."
    )
    assert result.iter_count <= 10, (
        f"R_eq convergence took {result.iter_count} iters; SPEC says ~5–10."
    )
