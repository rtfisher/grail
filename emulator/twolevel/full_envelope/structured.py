"""Trainable two-level closure that preserves the affine scattering structure.

The network learns the thermal source and a correction to the prescribed
destruction probability. Inference uses neither a Planck function nor a
reference rate solve. Known physics supplies the convex source mixture and
the exact conversion between source function and conserving populations.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn


class StructuredRateAPNN(nn.Module):
    version = "twolevel-structured-rates-v1"

    def __init__(self, width=32, nu_0=2.467381e15):
        super().__init__()
        self.width = width
        self.nu_0 = nu_0
        self.Bstar = 2*6.62607015e-34*nu_0**3/2.99792458e8**2
        self.network = nn.Sequential(nn.Linear(3, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2)).double()
        self.thermal_linear = nn.Linear(1, 1).double()

    def rate_coordinates(self, log_parameters):
        T = 10.0**log_parameters[..., 0]
        invT = ((1/T)-2.05e-4)/1.3e-4
        features = torch.stack((invT, (log_parameters[..., 1]-12)/4,
                                 (log_parameters[..., 2]+4.25)/2.75), -1)
        correction = 0.01*self.network(features)
        log_thermal = self.thermal_linear(invT[..., None]).squeeze(-1)+correction[..., 0]
        return log_thermal, correction[..., 1]

    def coefficients(self, T, ne, epsilon):
        parameters = torch.stack((torch.log10(T), torch.log10(ne),
                                   torch.log10(epsilon)), -1)
        log_thermal, destruction_correction = self.rate_coordinates(parameters)
        probability = torch.sigmoid(torch.logit(epsilon)+destruction_correction)
        thermal = self.Bstar*torch.exp(log_thermal)
        return probability, thermal

    def forward(self, log_inputs):
        J, T, ne, eps = (10.0**log_inputs).unbind(-1)
        probability, thermal = self.coefficients(T, ne, eps)
        source = (1-probability)*J+probability*thermal
        n2 = source/(self.Bstar+2*source)
        return torch.stack((1-n2, n2), -1)


def make_structured_adapter(model, epsilon):
    if not 0 < epsilon < 1:
        raise ValueError("destruction probability must lie in (0, 1)")
    evaluator = deepcopy(model).double().eval()

    def adapter(J, T, ne):
        inputs = torch.stack((torch.log10(J), torch.log10(T),
                               torch.log10(ne), torch.full_like(J, epsilon).log10()), -1)
        return evaluator(inputs)

    adapter.model = evaluator
    adapter.epsilon = epsilon
    return adapter


def save_structured_checkpoint(path, model, **metadata):
    """Save trained parameters and the model's declared physical structure."""
    torch.save({"version": model.version, "width": model.width, "nu_0": model.nu_0,
        "state_dict": model.state_dict(), "metadata": metadata}, Path(path))


def load_structured_checkpoint(path):
    """Load only a compatible structured-rate checkpoint, on the CPU."""
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("version") != StructuredRateAPNN.version:
        raise ValueError("not a structured two-level rate checkpoint")
    model = StructuredRateAPNN(payload["width"], payload["nu_0"])
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload["metadata"]
