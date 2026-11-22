# GRAIL

**Green's-function Resolvent for AI-accelerated Iteration Libraries**

[![Tests](https://github.com/rtfisher/grail/actions/workflows/tests.yml/badge.svg)](https://github.com/rtfisher/grail/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

GRAIL estimates how an error in a neural surrogate changes the converged
state of a coupled simulation. You supply the surrogate residual and two
Jacobian-vector product callbacks. GRAIL propagates the residual through
the cycle's linear response using a matrix-free GMRES solve.

For a regular fixed point, the response is `(I - K)^(-1) delta_T`, where
`K` differentiates the complete coupled cycle. This includes repeated
linear feedback. Accuracy for a nonlinear problem requires a sufficiently
small, smooth perturbation and accurate derivative callbacks.

The solver supports PyTorch and an optional JAX backend. Reference
integrations cover radiative transfer, electronic structure, a climate
column. The radiative-transfer library also retains the three-level
host used by the analytic checks and the five-level hybrid model.

## Installation

Python 3.10 or newer is required. Install from this repository:

```sh
git clone https://github.com/rtfisher/grail.git
cd grail
python -m pip install -e .
```

Optional dependencies are grouped by use:

```sh
python -m pip install -e '.[test,examples]'   # tests and numerical examples
python -m pip install -e '.[scf]'             # PySCF electronic structure
python -m pip install -e '.[jax]'             # optional JAX solver backend
```

## A complete first example

```python
import torch
from grail import resolvent_solve

x = torch.zeros(2, dtype=torch.float64)
residual = torch.tensor([0.01, -0.02], dtype=torch.float64)
result = resolvent_solve(
    x,
    residual,
    emulator_jvp=lambda v: 0.75 * v,
    host_jvp=lambda v: v,
    gmres_tol=1e-12,
)
torch.testing.assert_close(result.delta_x_star, 4 * residual)
print(result.delta_x_star)  # tensor([0.0400, -0.0800], dtype=torch.float64)
```

Run the same example with `python examples/quickstart.py`. The callbacks
compose as `emulator_jvp(host_jvp(v))`; they must differentiate the same
cycle and use consistent coordinates and evaluation points.

## Technical documentation

- [User manual](docs/user_manual/grail_user_manual.pdf): installation,
  integration recipes, interpreting output, calibration, and debugging.
- [Reference guide](docs/reference_guide/grail_reference_guide.pdf): mathematical
  foundations, state conventions, derivative contracts, and reference physics.
- [Testing](docs/testing.md): test groups and optional example artifacts.
- [Examples](examples/README.md): runnable examples, training utilities, and data.
- [Lean proofs](formalization/grail/README.md): pinned proof sources and
  standalone kernel and axiom checks.

The technical guides include their LaTeX sources and logo covers. Build them
from the repository root with `python tools/build_docs.py`.

## Tests

```sh
python -m pytest tests/grail tests/atmosphere tests/emulator tests/climate_rce/unit tests/regression tests/decisions/test_numerical_contracts.py -m "not analytic and not slow" -ra
```

Install the relevant extras to exercise other integrations. Some extended integration tests need generated ensembles; pytest reports
these as skips when the named artifact is absent. The maintained structured
two-level checkpoint and its offline test data are included under
`examples/data/structured/` and are required by its regression tests.

On macOS, installations mixing different OpenMP runtimes can require a
consistent conda environment. The test configuration supplies the existing
OpenMP compatibility setting; use a single-runtime environment for production.

## Scope

The core library consumes opaque callbacks and does not import host physics.
The reference models are examples rather than production-physics guarantees.
Split-conformal bands require exchangeable calibration and test scores and
cover the chosen prediction target; they do not automatically include
physical-model or discretization uncertainty. The climate example keeps its
state-dependent physical heating law fixed as the numerical timestep varies.

Version: **0.1.0a0**. License: **Apache-2.0**. Repository:
[github.com/rtfisher/grail](https://github.com/rtfisher/grail).
