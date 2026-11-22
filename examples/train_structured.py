"""Fixed-budget rate-coordinate training; all teacher calls occur offline."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from atmosphere.twolevel.rate_equations import rate_equation_solve
from emulator.twolevel.full_envelope.structured import StructuredRateAPNN

OUT = Path(__file__).resolve().parent / "output/structured_training"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data_seed, split_seed, training_seed, physical_seed = 2026111801, 2026111802, 2026111803, 2026111804
    protocol = {
        "model": StructuredRateAPNN.version, "width": 32, "samples": 12000,
        "data_seed": data_seed, "split_seed": split_seed, "training_seed": training_seed,
        "physical_test_seed": physical_seed, "physical_test_count": 128,
        "physical_test_bounds": {"T": [6000, 12000], "log10_ne": [10, 14],
            "log10_epsilon": [-6, -2], "log10_tau_max": [2, 4]},
        "training_bounds": {"T": [3000, 13000], "log10_ne": [8, 16],
            "log10_epsilon": [-7, float(np.log10(.03))]},
        "known_physics": ["source is affine in radiation intensity",
            "positive convex mixture of scattering and thermal emission",
            "prescribed epsilon scales the learned destruction probability",
            "conserving noninverted source-to-population conversion"],
        "learned_quantities": ["thermal source", "log-odds correction to destruction probability"],
        "teacher_use": "Two offline population labels per parameter draw; no teacher calls in inference or NN JVPs.",
        "loss": "MSE of log thermal source/Bstar plus destruction log-odds correction; no test-state feedback",
        "optimizer": "training-label affine initialization, 500 Adam epochs, then LBFGS max_iter=300 max_eval=400",
        "acceptance": {"all_96_original_and_128_fresh_roots_required": True,
            "max_nonlinear_relative_defect": 1e-9,
            "max_physical_population_source_flux_relative_error": 1e-3,
            "max_resolved_relative_shift_prediction_error": 1e-2,
            "max_GRAIL_true_linear_defect": 1e-8,
            "roundoff_limited_shifts": "Report separately with raw relative and absolute errors; do not omit cases."},
    }
    (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    rng = np.random.default_rng(physical_seed)
    physical = rng.uniform([6000, 10, -6, 2], [12000, 14, -2, 4], (128, 4))
    np.savez_compressed(OUT/"physical_test_inputs.npz", physical=physical)
    bounds = np.array([[np.log10(3000), np.log10(13000)], [8, 16], [-7, np.log10(.03)]])
    x = bounds[:, 0]+qmc.LatinHypercube(3, seed=data_seed).random(12000)*(bounds[:, 1]-bounds[:, 0])
    parameters = torch.from_numpy(10.0**x)
    T, ne, eps = parameters.T
    nu_0 = 2.467381e15
    Bstar = 2*6.62607015e-34*nu_0**3/2.99792458e8**2
    # Identify affine source coefficients using two independent offline
    # teacher queries in a well-conditioned intensity range.
    q0 = rate_equation_solve(torch.zeros_like(T), T, ne, epsilon=eps, nu_0=nu_0)
    source0 = Bstar*q0[:, 1]/(q0[:, 0]-q0[:, 1])
    J1 = source0/eps
    q1 = rate_equation_solve(J1, T, ne, epsilon=eps, nu_0=nu_0)
    thermal_label = Bstar*q1[:, 1]/(q1[:, 0]-q1[:, 1])
    probability_label = source0/thermal_label
    labels = torch.stack((torch.log(thermal_label/Bstar),
        torch.logit(probability_label)-torch.logit(eps)), -1)
    indices = np.random.default_rng(split_seed).permutation(len(x))
    train, val, test = np.split(indices, [9600, 10800])
    torch.manual_seed(training_seed)
    model = StructuredRateAPNN()
    xx = torch.from_numpy(x)
    invT = ((1/T)-2.05e-4)/1.3e-4
    # Fit the skip connection using training labels only, not a hardcoded
    # Boltzmann coefficient. The nonlinear correction remains trainable.
    design = torch.stack((invT[train], torch.ones(len(train), dtype=torch.float64)), -1)
    affine = torch.linalg.lstsq(design, labels[train, 0]).solution
    with torch.no_grad():
        model.thermal_linear.weight[0, 0] = affine[0]
        model.thermal_linear.bias[0] = affine[1]
    torch.save({"version": model.version, "state_dict": model.state_dict()}, OUT/"initial_model.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    start = time.time()

    def loss_on(index):
        logB, correction = model.rate_coordinates(xx[index])
        pred = torch.stack((logB, correction), -1)
        return torch.mean((pred-labels[index])**2)

    tidx = torch.from_numpy(train)
    for epoch in range(500):
        batches = tidx[torch.randperm(len(tidx))].split(512)
        losses = []
        for batch in batches:
            optimizer.zero_grad()
            loss = loss_on(batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            validation = float(loss_on(val))
        history.append([float(np.mean(losses)), validation])
        if epoch % 50 == 0 or epoch == 499:
            print(json.dumps({"epoch": epoch+1, "train": history[-1][0], "validation": validation,
                "seconds": time.time()-start}), flush=True)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=300, max_eval=400,
        tolerance_grad=1e-13, tolerance_change=1e-17, line_search_fn="strong_wolfe")
    lbfgs = []

    def closure():
        optimizer.zero_grad()
        loss = loss_on(train)
        loss.backward()
        lbfgs.append(float(loss.detach()))
        return loss

    optimizer.step(closure)
    metrics = {}
    with torch.no_grad():
        p, B = model.coefficients(T, ne, eps)
        for name, index in [("train", train), ("validation", val), ("test", test)]:
            erB = torch.abs(B[index]/thermal_label[index]-1).numpy()
            erp = torch.abs(p[index]/probability_label[index]-1).numpy()
            metrics[name] = {"n": len(index),
                "max_thermal_relative_error": float(erB.max()), "mean_thermal_relative_error": float(erB.mean()),
                "max_probability_relative_error": float(erp.max()), "mean_probability_relative_error": float(erp.mean())}
    metadata = {"protocol": protocol, "metrics": metrics, "seconds": time.time()-start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "teacher_calls_at_inference": 0, "teacher_JVPs_at_inference": 0,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path("emulator/twolevel/full_envelope/structured.py")]}}
    torch.save({"version": model.version, "width": model.width, "nu_0": model.nu_0,
        "state_dict": model.state_dict(), "metadata": metadata}, OUT/"model.pt")
    np.savez_compressed(OUT/"training.npz", inputs_log10=x, teacher_q0=q0.numpy(),
        teacher_q1=q1.numpy(), teacher_J1=J1.numpy(), labels=labels.numpy(),
        train_indices=train, validation_indices=val, test_indices=test,
        history=history, lbfgs_losses=lbfgs, predicted_probability=p.numpy(), predicted_thermal=B.numpy())
    metadata["checkpoint_sha256"] = hashlib.sha256((OUT/"model.pt").read_bytes()).hexdigest()
    (OUT/"training.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
