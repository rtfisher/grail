"""Version 2 five-level SE surrogate with a positive-opacity population head.

Legacy APNN classes and checkpoint formats are unchanged. This model evaluates
in float64 and learns adjacent excitation-gap corrections about a physical rate
ratio. Its radiation encoder is shared by training and runtime: epsilon and T
enter through S_eff=(1-epsilon)J+epsilon B, and electron density is absent because
the master-epsilon reference solver does not use it. Strictly positive ordered
populations follow from exp(log gaps), without clipping populations or sources.
"""
from __future__ import annotations

import torch
from torch import nn

from atmosphere.atomic_model import AtomicModel
from atmosphere.multilevel.fixed_point import _source_from_populations_nlevel

_H, _KB, _C = 6.62607015e-34, 1.380649e-23, 2.99792458e8
SCHEMA = "positive-opacity-se-v2"


def radiation_features(J, T, atomic_model):
    """Natural log of the epsilon-conditioned line photon occupations."""
    J, T = J.double(), T.double()
    if not torch.isfinite(J).all() or (J < 0).any():
        raise ValueError("mean intensities must be finite and nonnegative")
    if not torch.isfinite(T).all() or (T <= 0).any():
        raise ValueError("temperature must be finite and positive")
    nu = torch.tensor([tr.nu_0 for tr in atomic_model.transitions], dtype=J.dtype, device=J.device)
    eps = torch.tensor([tr.epsilon for tr in atomic_model.transitions], dtype=J.dtype, device=J.device)
    amplitude = 2 * _H * nu**3 / _C**2
    thermal_occupation = 1 / torch.expm1(_H * nu[None, :] / (_KB * T[:, None]))
    occupation = (1 - eps) * J / amplitude + eps * thermal_occupation
    if (occupation <= 0).any():
        raise ValueError("zero effective line excitation is outside the positive-population domain")
    return torch.log(occupation)


class PositivePopulationAPNN(nn.Module):
    """Smooth float64 network with ordered n_i/g_i and exact conservation.

    Levels must be ordered in increasing energy; the five-level Balmer model
    meets that contract. The inverse adjacent excitation temperatures are
    positive by construction. Arbitrary inverted/masing media require a
    different opacity and formal-transfer model, and are outside this API.
    """
    schema = SCHEMA

    def __init__(self, atomic_model, hidden_dims=(96, 96, 96)):
        super().__init__()
        if atomic_model.n_levels != 5 or atomic_model.K != 10:
            raise ValueError("v2 currently supports the full five-level Balmer atom")
        self.n_in, self.n_out = atomic_model.K, atomic_model.n_levels
        self.hidden_dims = tuple(hidden_dims)
        self.adjacent = tuple(next(k for k, tr in enumerate(atomic_model.transitions)
                                   if tr.lower == j and tr.upper == j+1)
                              for j in range(self.n_out-1))
        self.register_buffer("log_g", torch.log(torch.tensor(atomic_model.statistical_weights, dtype=torch.float64)))
        self.register_buffer("center", torch.zeros(self.n_in, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(self.n_in, dtype=torch.float64))
        layers = []
        width = self.n_in
        for h in hidden_dims:
            layers += [nn.Linear(width, h, dtype=torch.float64), nn.Tanh()]
            width = h
        layers += [nn.Linear(width, self.n_out-1, dtype=torch.float64)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def log_gaps(self, features):
        features = features.double()
        # log(1+1/occupation), evaluated without exponential overflow.
        baseline = torch.nn.functional.softplus(-features[:, self.adjacent])
        return torch.log(baseline) + self.net((features-self.center) / self.scale)

    def forward(self, features):
        gaps = torch.exp(self.log_gaps(features))
        excitation = torch.cat([torch.zeros_like(gaps[:, :1]), -torch.cumsum(gaps, 1)], 1)
        out = torch.softmax(excitation + self.log_g, 1)
        if not torch.isfinite(out).all() or (out <= 0).any():
            raise ValueError("v2 excitation gaps exceed float64 representability")
        return out


def population_log_gaps(populations, atomic_model):
    _source_from_populations_nlevel(populations, atomic_model)
    g = torch.tensor(atomic_model.statistical_weights, dtype=populations.dtype, device=populations.device)
    excitation = torch.log(populations / g)
    gaps = excitation[:, :-1] - excitation[:, 1:]
    if (gaps <= 0).any():
        raise ValueError("training labels violate the ordered positive-opacity domain")
    return torch.log(gaps)


def make_positive_cycle_adapter(model, atomic_model):
    """The same epsilon-aware encoder used to construct training rows.

    Autograd is preserved, so coupled solves and JVPs differentiate the actual
    trained network. No reference-SE solve or output floor occurs at runtime.
    """
    def adapter(J, T, n_e):
        return model(radiation_features(J, T, atomic_model))
    return adapter


def save_positive_checkpoint(path, model, metadata):
    torch.save({"schema": SCHEMA, "hidden_dims": model.hidden_dims,
                "model_state_dict": model.state_dict(), "metadata": metadata}, path)


def load_positive_checkpoint(path, atomic_model):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("schema") != SCHEMA:
        raise ValueError("checkpoint is not a version-2 positive-opacity model")
    model = PositivePopulationAPNN(atomic_model, raw["hidden_dims"])
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model, raw["metadata"]


class RateRelaxedAPNN(PositivePopulationAPNN):
    """Trained neural initializer plus fixed positive rate-relaxation layers.

    The excited-state subsystem with n_0=1 is an M-matrix. Jacobi iterations
    use nonnegative incoming rates divided by strictly positive loss rates.
    Normalization is applied once after these homogeneous balance equations.
    This is a hybrid surrogate, not a standalone black-box replacement. It
    calls no reference solver; runtime and derivative include every layer.
    """
    schema = "rate-relaxed-positive-se-v3"

    def __init__(self, atomic_model, hidden_dims=(96,96,96), rate_sweeps=8):
        super().__init__(atomic_model, hidden_dims)
        self.rate_sweeps = int(rate_sweeps)
        self.pairs = tuple((tr.lower,tr.upper) for tr in atomic_model.transitions)
        self.register_buffer("amplitudes",torch.tensor([2*_H*tr.nu_0**3/_C**2 for tr in atomic_model.transitions],dtype=torch.float64))
        self.register_buffer("weights",torch.tensor(atomic_model.statistical_weights,dtype=torch.float64))

    def rate_system(self, features):
        occupation=torch.exp(features)
        N=len(features);incoming=features.new_zeros((N,self.n_out,self.n_out))
        for k,(lo,hi) in enumerate(self.pairs):
            effective=self.amplitudes[k]*occupation[:,k]
            incoming[:,hi,lo]=effective*(self.weights[hi]/self.weights[lo])
            incoming[:,lo,hi]=effective+self.amplitudes[k]
        losses=incoming.sum(1)
        return incoming,losses

    def forward(self, features):
        initial=super().forward(features)
        incoming,losses=self.rate_system(features)
        ratios=initial[:,1:]/initial[:,:1]
        for _ in range(self.rate_sweeps):
            ratios=(incoming[:,1:,0]+torch.einsum('nij,nj->ni',incoming[:,1:,1:],ratios))/losses[:,1:]
        out=torch.cat([torch.ones_like(ratios[:,:1]),ratios],1)
        out=out/out.sum(1,keepdim=True)
        if not torch.isfinite(out).all() or (out <= 0).any():
            raise ValueError("rate relaxation exceeds the positive float64 domain")
        for lo,hi in self.pairs:
            if (out[:,lo]*(self.weights[hi]/self.weights[lo])-out[:,hi] <= 0).any():
                raise ValueError("rate-relaxed populations have nonpositive line opacity")
        return out


def save_rate_relaxed_checkpoint(path, model, metadata):
    torch.save({"schema":model.schema,"hidden_dims":model.hidden_dims,
                "rate_sweeps":model.rate_sweeps,"model_state_dict":model.state_dict(),
                "metadata":metadata},path)


def load_rate_relaxed_checkpoint(path, atomic_model):
    raw=torch.load(path,map_location='cpu',weights_only=False)
    if raw.get('schema')!='rate-relaxed-positive-se-v3':
        raise ValueError('checkpoint is not a rate-relaxed v3 model')
    model=RateRelaxedAPNN(atomic_model,raw['hidden_dims'],raw['rate_sweeps'])
    for key in ("amplitudes", "weights", "log_g"):
        if not torch.equal(model.state_dict()[key], raw['model_state_dict'][key]):
            raise ValueError(f"checkpoint atomic-model {key} mismatch")
    model.load_state_dict(raw['model_state_dict']);model.eval()
    return model,raw['metadata']
