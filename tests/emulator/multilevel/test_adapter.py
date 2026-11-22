"""Unit tests for the 5-level cycle adapter. SPEC_nlte_scaleup §M.4.

Test IDs:
  U-apnn5-cycle-adapter : adapter shape/dtype contract + SE-solver compliance.
  U-adapter-log-clamp   : adapter handles J_bar=0 without NaN.

No __init__.py (rootdir-relative discovery, --import-mode=importlib).
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# U-apnn5-cycle-adapter
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_apnn5_cycle_adapter() -> None:
    """U-apnn5-cycle-adapter: make_cycle_adapter_5level satisfies SE-solver contract.

    Verifies:
    1. Adapter accepts (J_bar_lines: (N_tau, K=10), T: (N_tau,), n_e: (N_tau,)).
    2. Returns (N_tau, L=5) float64 tensor.
    3. Output rows sum to 1 (softmax gate).
    4. ValueError raised when model.n_in mismatches atomic_model.K + 2.
    5. ValueError raised when model.n_out mismatches atomic_model.n_levels.
    """
    from emulator.apnn import APNN
    from emulator.multilevel.jvp import make_cycle_adapter_5level
    from atmosphere.atomic_model import AtomicModel

    atomic_model = AtomicModel.balmer_5level(epsilon=1e-3)
    K = atomic_model.K        # 10
    L = atomic_model.n_levels # 5

    # Correct model.
    model = APNN(n_in=K + 2, n_out=L, hidden_dims=(256, 256, 256, 256))
    model.eval()

    adapter = make_cycle_adapter_5level(model, atomic_model)

    N_tau = 15
    torch.manual_seed(3)
    J_bar_lines = torch.rand(N_tau, K, dtype=torch.float64) * 1e-12
    T   = torch.full((N_tau,), 8000.0, dtype=torch.float64)
    n_e = torch.full((N_tau,), 1e12,   dtype=torch.float64)

    pops = adapter(J_bar_lines, T, n_e)

    assert pops.shape == (N_tau, L), (
        f"Expected output shape ({N_tau}, {L}), got {pops.shape}"
    )
    assert pops.dtype == torch.float64, (
        f"Expected float64 output, got {pops.dtype}"
    )

    # Population conservation.
    row_sums = pops.sum(dim=-1)
    max_dev = (row_sums - 1.0).abs().max().item()
    assert max_dev < 1e-6, (
        f"Population sum != 1: max deviation = {max_dev:.2e}"
    )

    # Non-negativity.
    assert (pops >= 0.0).all().item(), "Adapter returned negative populations"

    # -----------------------------------------------------------------------
    # Validation: wrong n_in raises ValueError.
    # -----------------------------------------------------------------------
    bad_nin = APNN(n_in=K + 1, n_out=L)
    with pytest.raises(ValueError, match="n_in"):
        make_cycle_adapter_5level(bad_nin, atomic_model)

    # -----------------------------------------------------------------------
    # Validation: wrong n_out raises ValueError.
    # -----------------------------------------------------------------------
    bad_nout = APNN(n_in=K + 2, n_out=L - 1)
    with pytest.raises(ValueError, match="n_out"):
        make_cycle_adapter_5level(bad_nout, atomic_model)


# ---------------------------------------------------------------------------
# U-adapter-log-clamp
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_U_adapter_log_clamp() -> None:
    """U-adapter-log-clamp: adapter handles J_bar_lines=0 without NaN.

    Passes an all-zeros J_bar_lines tensor; the adapter must clamp before
    log10 and return finite, non-NaN output.
    """
    from emulator.apnn import APNN
    from emulator.multilevel.jvp import make_cycle_adapter_5level
    from atmosphere.atomic_model import AtomicModel

    atomic_model = AtomicModel.balmer_5level(epsilon=1e-3)
    K = atomic_model.K
    L = atomic_model.n_levels

    model = APNN(n_in=K + 2, n_out=L, hidden_dims=(64, 64))
    model.eval()

    adapter = make_cycle_adapter_5level(model, atomic_model)

    N_tau = 5
    J_bar_zero = torch.zeros(N_tau, K, dtype=torch.float64)
    T   = torch.full((N_tau,), 8000.0, dtype=torch.float64)
    n_e = torch.full((N_tau,), 1e12,   dtype=torch.float64)

    pops = adapter(J_bar_zero, T, n_e)

    assert pops.shape == (N_tau, L), f"Expected ({N_tau}, {L}), got {pops.shape}"
    assert not torch.isnan(pops).any().item(), "NaN in adapter output when J_bar=0"
    assert not torch.isinf(pops).any().item(), "Inf in adapter output when J_bar=0"
