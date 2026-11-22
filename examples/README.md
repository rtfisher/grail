# Executable reference examples

Run from the repository root after installing the relevant extras.

- `python examples/quickstart.py`: a complete two-component response solve.
- `python examples/structured_response.py`: the trained structured two-level closure,
  actual composed derivatives, independent reference solution and true linear defect.
- `python examples/scf_response.py`: a controlled local electronic-structure check;
  requires the `scf` extra and supplied HDF5 checkpoint.
- `python examples/calibration.py`: split-conformal order statistics and the
  infinite-band case when the calibration sample cannot support the requested level.
- `python examples/climate_fixed_heating.py --help`: physical heating held fixed
  while the numerical relaxation timestep changes.
- `python examples/train_structured.py`: deterministic training from the supplied
  two-level data; writes under `examples/output/structured_training/`.

Additional utilities examine the retained host physics, preconditioning and SCF
model training. The structured checkpoint, offline training data and metadata are
under `examples/data/structured/`. Commands describe mechanism and verification
examples; they do not claim production performance or an independent deployment study.
