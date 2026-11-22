"""Reproduce the fixed-heating climate experiment in a new output directory.

The model output represents an increment over a fixed 200000 s physical
timescale. Numerical timesteps change integration, never this heating law.
Single-threaded; no training, selection, or modification of prior results.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p/"grail/api.py").is_file())
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.optimize import root
import torch

from climate_rce.column import ColumnSpec
from climate_rce.emulator._smooth_fallback import make_smooth_fallback
from climate_rce.host import equilibrate_to_rce, _T_split_cycle_np, _q_ref_for_T
from climate_rce.jvp import (make_back_residual, make_emulator_jvp_rce,
                             make_host_jvp_rce, make_subgrid_with_nn)
from grail.api import resolvent_solve

torch.set_num_threads(1)
OUTPUT_TIMESCALE = 2e5
TIMESTEPS = [2e5, 1e4, 2e3, 1e3, 500., 250.]
SPEC = ColumnSpec(n_z=10, n_v=2, var_names=("T", "q"),
                  pressure=np.linspace(5000., 1e5, 10),
                  var_scales=(300., .01), n_state=10)
IDENTITY = np.eye(10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zero(spec, profile):
    return np.zeros_like(profile)


def dense(function, state, step=.003):
    return np.column_stack([(function(state+step*v)-function(state-step*v))/(2*step)
                            for v in IDENTITY])


def refine(function, initial, dt):
    solution = root(lambda t: (function(t)-t)/dt, initial,
                    jac=lambda t: (dense(function, t)-IDENTITY)/dt,
                    options={"xtol": 1e-10, "maxfev": 500})
    defect = float(np.linalg.norm(function(solution.x)-solution.x))
    valid = bool(np.isfinite(solution.x).all() and defect < 1e-8
                 and np.all((solution.x > 100) & (solution.x < 400)))
    return solution.x, {"scipy_success": bool(solution.success),
        "message": str(solution.message), "defect_kelvin": defect,
        "valid": valid, "evaluations": int(solution.nfev)}


def callbacks(profile, heating, model, dt, co2):
    host = make_host_jvp_rce(SPEC, profile, heating, dt=dt, co2_ppm=co2)
    emulator = make_emulator_jvp_rce(SPEC, profile, heating, dt=dt, co2_ppm=co2, model=model)
    return lambda v: emulator(host(v))


def one(dt, co2, amplitude, seed, *, steps=False):
    start = time.time()
    model = make_smooth_fallback(SPEC, amplitude=amplitude, seed=seed).double()
    heating = make_subgrid_with_nn(model, output_timescale=OUTPUT_TIMESCALE)
    ct = lambda t: _T_split_cycle_np(SPEC, t, zero, dt=dt, co2_ppm=co2)
    ce = lambda t: _T_split_cycle_np(SPEC, t, heating, dt=dt, co2_ppm=co2)
    # Independent initializations; neither root is warm-started from the other.
    tr = equilibrate_to_rce(SPEC, zero, dt=dt, co2_ppm=co2,
                            max_iter=2000, compute_spectral_radius=False)
    er = equilibrate_to_rce(SPEC, heating, dt=dt, co2_ppm=co2,
                            max_iter=2000, compute_spectral_radius=False)
    truth, truth_status = refine(ct, tr.x_star[0], dt)
    emulator, emulator_status = refine(ce, er.x_star[0], dt)
    profile_t = np.array([truth, _q_ref_for_T(SPEC, truth)])
    profile_e = np.array([emulator, _q_ref_for_T(SPEC, emulator)])
    # Operational prediction: only the emulator equilibrium is used here.
    rhs = make_back_residual(SPEC, profile_e, heating, zero, dt=dt, co2_ppm=co2)
    callback = callbacks(profile_e, heating, model, dt, co2)
    matrix = np.column_stack([callback(torch.from_numpy(v)).detach().numpy() for v in IDENTITY])
    solve = resolvent_solve(torch.from_numpy(emulator/300.), rhs,
                            emulator_jvp=callback, host_jvp=lambda v: v,
                            gmres_tol=1e-10, gmres_maxiter=200)
    prediction = solve.delta_x_star.detach().numpy()
    actual = (emulator-truth)/300.
    independent = dense(ce, emulator)
    dense_prediction = np.linalg.solve(IDENTITY-matrix, rhs.numpy())
    reference_rhs = make_back_residual(SPEC, profile_t, heating, zero, dt=dt, co2_ppm=co2).numpy()
    kt = dense(ct, truth)
    norm = float(1/np.linalg.svd(IDENTITY-kt, compute_uv=False)[-1])
    rho = float(max(abs(np.linalg.eigvals(kt))))
    # Sample the same physical states for every dt to verify law invariance.
    probe = np.linspace(245., 290., 10)
    probe_profile = np.array([probe, _q_ref_for_T(SPEC, probe)])
    row = {"dt": dt, "co2_ppm": co2, "amplitude": amplitude, "seed": seed,
        "output_timescale_seconds": OUTPUT_TIMESCALE,
        "initial_solver_flags": [bool(tr.converged), bool(er.converged)],
        "truth_solver": truth_status, "emulator_solver": emulator_status,
        "valid": truth_status["valid"] and emulator_status["valid"],
        "truth_kelvin": truth.tolist(), "emulator_kelvin": emulator.tolist(),
        "delta_actual": actual.tolist(), "delta_prediction": prediction.tolist(),
        "rhs": rhs.numpy().tolist(), "rhs_at_reference": reference_rhs.tolist(),
        "matrix": matrix.tolist(), "reference_matrix": kt.tolist(),
        "independent_fd_matrix": independent.tolist(),
        "weight": model.network[0].weight.detach().numpy().tolist(),
        "heating_at_probe_kelvin_per_second": heating(SPEC, probe_profile)[0].tolist(),
        "prediction_error": (actual-prediction).tolist(),
        "A_emp": float(np.linalg.norm(actual)/np.linalg.norm(reference_rhs)),
        "E": float(np.linalg.norm(actual-prediction)/np.linalg.norm(actual)),
        "A_op_reference": norm, "rho_reference": rho,
        "norm_over_positive_mode_proxy": norm*(1-rho),
        "max_temperature_shift_kelvin": float(np.max(np.abs(emulator-truth))),
        "physical_heating_max_kelvin_per_day": float(np.max(np.abs(heating(SPEC, profile_e)[0]))*86400),
        "callback_relative_error": float(np.linalg.norm(matrix-independent)/np.linalg.norm(independent)),
        "restoring_operator_relative_error": float(np.linalg.norm(matrix-independent)/np.linalg.norm(IDENTITY-independent)),
        "rhs_cycle_difference_error": float(np.linalg.norm(rhs.numpy()-(ce(emulator)-ct(emulator))/300.)),
        "true_linear_relative_defect": float(np.linalg.norm((IDENTITY-matrix)@prediction-rhs.numpy())/np.linalg.norm(rhs.numpy())),
        "GRAIL_vs_dense_relative_error": float(np.linalg.norm(prediction-dense_prediction)/np.linalg.norm(dense_prediction)),
        "zero_feedback_control_E": float(np.linalg.norm(actual-rhs.numpy())/np.linalg.norm(actual)),
        "seconds": time.time()-start}
    if steps:
        row["fd_steps"] = []
        for step in [3., .3, .03, .003, .0003]:
            k = dense(ct, truth, step)
            row["fd_steps"].append({"step_kelvin": step,
                "A_op": float(1/np.linalg.svd(IDENTITY-k, compute_uv=False)[-1])})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error("output directory must not exist; previous results are never overwritten")
    out.mkdir(parents=True)
    protocol = {"physical_heating_law": "Q_theta(T) = 300 K * model(T/300 K) / 200000 s",
        "output_timescale_seconds": OUTPUT_TIMESCALE, "headline_timesteps_seconds": TIMESTEPS,
        "headline": {"CO2_ppm": 280., "amplitude": 1e-4, "seed": 42},
        "ensemble": {"timesteps": [2e5, 1e4, 500., 250.], "CO2_ppm": [280., 400., 560.],
            "amplitudes": np.geomspace(5e-5, 5e-4, 8).tolist(), "seeds": list(range(8))},
        "prediction_point": "emulator equilibrium", "model": "prescribed affine surrogate, no climate training",
        "state": "10 temperatures / 300 K, Euclidean metric", "root_defect_kelvin_max": 1e-8,
        "temperature_window_kelvin": [100., 400.], "selection": "all declared attempts retained",
        "physical_interpretation": "timestep refinement at a fixed physical heating law; per-cycle gain is numerical"}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    start = time.time()
    headline, ensemble = [], []
    for dt in TIMESTEPS:
        row = one(dt, 280., 1e-4, 42, steps=True)
        headline.append(row)
        print("headline", dt, row["valid"], row["E"], flush=True)
    amplitudes = np.geomspace(5e-5, 5e-4, 8)
    for dt, co2, seed in itertools.product([2e5, 1e4, 500., 250.], [280., 400., 560.], range(8)):
        try:
            row = one(dt, co2, float(amplitudes[seed]), seed)
        except Exception as exc:
            row = {"dt": dt, "co2_ppm": co2, "seed": seed, "valid": False, "exception": repr(exc)}
        ensemble.append(row)
        (out/"progress.json").write_text(json.dumps(ensemble, indent=2)+"\n")
        print("ensemble", len(ensemble), row["valid"], row.get("E"), flush=True)
    files = ["climate_rce/jvp.py", "climate_rce/host.py", "climate_rce/radiation.py",
             "climate_rce/convection.py", "climate_rce/emulator/network.py",
             "climate_rce/emulator/_smooth_fallback.py", "grail/api.py", "grail/jacobian.py", "grail/gmres.py"]
    record = {"protocol": protocol, "headline": headline, "ensemble": ensemble,
        "valid_headline_count": sum(r["valid"] for r in headline),
        "valid_count": sum(r["valid"] for r in ensemble), "attempt_count": len(ensemble),
        "inputs": {p: sha(ROOT/p) for p in files}, "driver_sha256": sha(__file__),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                        "torch": torch.__version__, "numerical_threads": 1},
        "seconds": time.time()-start}
    (out/"results.json").write_text(json.dumps(record, indent=2)+"\n")
    print("complete", record["valid_headline_count"], "/6 headline;", record["valid_count"], "/96 ensemble", flush=True)


if __name__ == "__main__":
    main()
