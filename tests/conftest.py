"""pytest configuration shared across all test packages.

Markers and discovery rules are declared in pyproject.toml.

Per-setup test layout (SPEC_threelevel §M3a)
--------------------------------------------
As of M3a the test tree is organized by physics setup, not just by module:

  tests/
    atmosphere/
      test_atmosphere.py           — grid tests (shared; setup-agnostic)
      test_atmosphere_class.py     — Atmosphere dataclass tests (shared)
      twolevel/                    — tests specific to the 2-level setup
        test_rate_equations.py
        test_formal_solver.py
        test_fixed_point.py
    emulator/
      twolevel/
        test_emulator.py           — APNN + training + validation (2-level)
    integration/
      twolevel/
        test_integration.py        — cross-package integration (2-level)
    decisions/
      test_decisions.py            — Decision-Gate evaluator (setup-agnostic)
    grail/
      test_grail.py                — resolvent solver (setup-agnostic)

Tests at ``tests/<pkg>/twolevel/`` exercise the 2-level NLTE prototype code
that lives in ``atmosphere/twolevel/`` and ``emulator/twolevel/``.  Future
3-level tests will go in ``tests/<pkg>/threelevel/`` (SPEC_threelevel §M3c).

OpenMP workaround
-----------------
On macOS conda environments that include both PyTorch and h5py, two
copies of libomp.dylib end up linked into the same process (PyTorch
ships one, hdf5 / mkl ship another).  When both are loaded together,
the OpenMP runtime detects the conflict and aborts with::

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized.

The OpenMP project's documented (unsupported) workaround is to set
``KMP_DUPLICATE_LIB_OK=TRUE`` so the second runtime initialization is
silently allowed.  We do that here so the test suite "just works"
without users having to set the env var manually.

For production work the right fix is a single-OpenMP environment
(e.g.\\ install both PyTorch and hdf5 from the same conda channel that
links against the same libomp).  That is out of scope for the prototype.
"""

from __future__ import annotations

import os

# Set BEFORE any pytest plugin or test imports torch / h5py.  conftest.py
# is loaded before test collection, so this line runs at the right time.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
