# Testing the retained software

Install the `test` extra, and `scf` or `jax` when exercising those backends.

```sh
python -m pytest tests/grail tests/atmosphere tests/emulator tests/climate_rce/unit tests/regression tests/decisions/test_numerical_contracts.py -m "not analytic and not slow" -ra
```

The structured two-level model and training data are supplied and required by its
regression checks. Analytic tests include the two- and three-level host anchors
and five-level physical contracts. Dense grids and training checks remain explicit
`analytic` or `slow` opt-in groups. Extended integration tests may require generated
ensembles; absent optional data is reported as a skip. Core tests and supplied
examples verify composed derivatives, correction sign, true linear equation defects,
fixed physical heating, and calibration order statistics.

`python -m decisions --setup 2level --fast` runs the three retained physical
verification contracts. Deferred nonlinear validity and unmeasured full-space
inverse norms remain explicit; a passed homogeneity check does not certify them.
