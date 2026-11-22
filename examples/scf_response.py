"""Controlled SCF response verification using the actual composed derivative."""
from pathlib import Path
import json
import numpy as np
import torch

from grail import resolvent_solve
from scf import make_h2_state
from scf.diagnostics import make_cycle_scf, pack_pair
from scf.emulator import load_xc_emulator
from scf.fock_build import build_fock_truth
from scf.host import picard_iterate_scf
from scf.jvp import make_cycle_jvps_scf, make_fock_builder_with_nn


def main():
    torch.set_num_threads(1)
    state = make_h2_state(1.25, method="uhf", basis="sto-3g")
    model = load_xc_emulator(Path(__file__).parent/"scf_xc_emulator_uhf_sto3g.h5",
                              dtype=torch.float64)
    neural_fock = make_fock_builder_with_nn(model, state)
    eta = .001

    def controlled_fock(state, density):
        reference = build_fock_truth(state, density)
        neural = neural_fock(state, density)
        return tuple(a+eta*(b-a) for a, b in zip(reference, neural))

    truth = picard_iterate_scf(state, tol=1e-12, max_iter=200)
    emulator = picard_iterate_scf(state, fock_builder=controlled_fock,
                                 tol=1e-12, max_iter=200)
    assert truth.converged and emulator.converged
    x_t, x_e = pack_pair(truth.P_star), pack_pair(emulator.P_star)
    reference_cycle = make_cycle_scf(state, build_fock_truth)
    surrogate_cycle = make_cycle_scf(state, controlled_fock)
    residual = surrogate_cycle(x_e)-reference_cycle(x_e)
    first, second = make_cycle_jvps_scf(state, emulator.P_star, controlled_fock, h=1e-5)
    result = resolvent_solve(torch.from_numpy(x_e), torch.from_numpy(residual),
                             emulator_jvp=second, host_jvp=first,
                             gmres_tol=1e-10, gmres_maxiter=100)
    prediction = result.delta_x_star.detach().numpy()
    actual = x_e-x_t
    error = float(np.linalg.norm(prediction-actual)/np.linalg.norm(actual))
    defect = prediction-second(first(torch.from_numpy(prediction))).numpy()-residual
    relative_defect = float(np.linalg.norm(defect)/np.linalg.norm(residual))
    assert error < .01 and relative_defect < 1e-7
    print(json.dumps({"bond_length_angstrom": 1.25, "controlled_perturbation_amplitude": eta,
        "relative_shift_prediction_error": error, "true_linear_relative_defect": relative_defect,
        "scope": "controlled local-response verification; reference-containing blend"}, indent=2))


if __name__ == "__main__":
    main()
