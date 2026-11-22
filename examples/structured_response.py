"""Current structured neural closure, actual callbacks, and a reference check."""
from pathlib import Path
import json
import numpy as np
import torch

from atmosphere.twolevel.diagnostics import SourceProblem, truth_source, validated_atmosphere
from emulator.twolevel.full_envelope.structured import load_structured_checkpoint, make_structured_adapter
from grail import resolvent_solve


def main():
    torch.set_num_threads(1)
    epsilon = 1e-4
    model, _ = load_structured_checkpoint(Path(__file__).parent/"data/structured/model.pt")
    atmosphere = validated_atmosphere(T0=8000., ne0=1e12, epsilon=epsilon, tau_max=1e3)
    problem = SourceProblem(atmosphere, make_structured_adapter(model, epsilon))
    solved = problem.solve()
    assert solved["converged"]
    q_e = torch.from_numpy(problem.populations(solved["source"]))
    transport = torch.from_numpy(problem.L)
    incident = torch.from_numpy(problem.b*problem.B0)

    def host(q):
        return transport@(problem.Bstar*q/(1-2*q))+incident

    def neural(J):
        return problem.adapter(J, atmosphere.T, atmosphere.n_e)[:, 1]

    J_e = host(q_e)
    host_jvp = lambda v: torch.func.jvp(host, (q_e,), (v,))[1]
    neural_jvp = lambda v: torch.func.jvp(neural, (J_e,), (v,))[1]
    residual = torch.from_numpy(problem.population_cycle(q_e.numpy())-
                                problem.population_cycle(q_e.numpy(), 0))
    result = resolvent_solve(q_e, residual, emulator_jvp=neural_jvp,
                             host_jvp=host_jvp, gmres_tol=1e-11, gmres_maxiter=300)
    prediction = result.delta_x_star.detach()
    corrected = q_e-prediction
    # The reference solution is used only as an independent verification target.
    q_t = torch.from_numpy(problem.populations(truth_source(atmosphere, problem.transfer)))
    actual_shift = q_e-q_t
    relative_error = float(torch.linalg.vector_norm(prediction-actual_shift)/
                           torch.linalg.vector_norm(actual_shift))
    defect = prediction-neural_jvp(host_jvp(prediction))-residual
    relative_defect = float(torch.linalg.vector_norm(defect)/torch.linalg.vector_norm(residual))
    assert relative_error < .01 and relative_defect < 1e-8
    assert torch.linalg.vector_norm(corrected-q_t) < torch.linalg.vector_norm(q_e-q_t)
    print(json.dumps({"relative_shift_prediction_error": relative_error,
        "true_linear_relative_defect": relative_defect, "GMRES_iterations": result.iter_count,
        "correction_sign": "q_reference_estimate = q_emulator - delta_x_star"}, indent=2))


if __name__ == "__main__":
    main()
