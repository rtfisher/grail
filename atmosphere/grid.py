"""Depth, frequency, and angle quadratures. the technical reference §3.1.

All grids carry float64 tensors; the Avrett-Hummer thermalization tests at
small epsilon are not honest in single precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DepthGrid:
    """Logarithmic depth grid in line optical depth tau. the technical reference §3.1.

    Stored as a 1-D Tensor `tau` of shape (n_tau,), strictly increasing,
    with tau[0] = tau_min and tau[-1] = tau_max.
    """

    tau: torch.Tensor

    @classmethod
    def log_spaced(cls, tau_min: float, tau_max: float, n_tau: int) -> "DepthGrid":
        """Logarithmically spaced grid from tau_min to tau_max, inclusive of both endpoints.

        the technical reference §3.1 defaults: tau_min ~ 1e-4, tau_max in {1e2, 1e4, 1e6}, n_tau in [40, 200].
        """
        if not (tau_min > 0 and tau_max > tau_min):
            raise ValueError(
                f"require 0 < tau_min < tau_max; got tau_min={tau_min}, tau_max={tau_max}"
            )
        if n_tau < 2:
            raise ValueError(f"require n_tau >= 2; got n_tau={n_tau}")
        tau = torch.logspace(
            np.log10(tau_min), np.log10(tau_max), n_tau, dtype=torch.float64
        )
        return cls(tau=tau)

    @property
    def n_tau(self) -> int:
        return self.tau.shape[0]


@dataclass
class FrequencyGrid:
    """Frequency grid across a Doppler line profile. the technical reference §3.1.

    Stored in dimensionless frequency offset x = (nu - nu_0) / Delta_nu_D, where
    Delta_nu_D is the Doppler width.  Field name `nu` aligns with SPEC notation;
    units are nominal line-profile widths.

    The line-profile-weighted mean intensity is
        J_bar(tau) = int phi(nu) J(tau, nu) d nu
                  ~= sum_i weights[i] * profile[i] * J(tau, nu[i])
    `profile` is phi(x) = exp(-x^2)/sqrt(pi). The trapezoidal integration
    weights are divided by their discrete profile mass. Thus the profile
    average is a positive convex combination even on coarse grids. The
    physical opacity profile is unchanged. Normalization alone does not
    guarantee frequency accuracy: frequency refinement remains necessary.
    """

    nu: torch.Tensor
    weights: torch.Tensor
    profile: torch.Tensor

    @classmethod
    def doppler(cls, n_nu: int = 21, half_width: float = 5.0) -> "FrequencyGrid":
        """Trapezoidal grid on x in [-half_width, +half_width] with Doppler profile.

        Trapezoidal rule on a Gaussian is exponentially convergent in both the
        truncation tail and the discretization error.
        """
        if n_nu < 3:
            raise ValueError(f"require n_nu >= 3; got n_nu={n_nu}")
        if half_width <= 0:
            raise ValueError(f"require half_width > 0; got half_width={half_width}")
        nu = torch.linspace(-half_width, half_width, n_nu, dtype=torch.float64)
        dx = (2.0 * half_width) / (n_nu - 1)
        weights = torch.full((n_nu,), dx, dtype=torch.float64)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        profile = torch.exp(-nu * nu) / np.sqrt(np.pi)
        # A mass error of only 1e-4 can overwhelm destruction epsilon=1e-6
        # and turn a passive scattering discretization into an amplifying one.
        # Normalize the *integration measure*, never the opacity profile.
        weights = weights / torch.sum(weights * profile)
        return cls(nu=nu, weights=weights, profile=profile)

    @property
    def n_nu(self) -> int:
        return self.nu.shape[0]


@dataclass
class AngleQuadrature:
    """Gauss-Legendre quadrature on mu in [0, 1]. the technical reference §3.1.

    The plane-parallel formal solver decomposes radiation into upward (mu > 0)
    and downward (mu < 0) streams.  We store positive |mu| only and use the
    same nodes/weights for both directions.  With sum_i weights[i] = 1 the
    mean intensity is
        J(tau) = (1/2) sum_i weights[i] (I_plus(tau, mu[i]) + I_minus(tau, mu[i]))
    by symmetry around mu = 0.

    n_mu-point Gauss-Legendre is exact for polynomials of degree <= 2*n_mu - 1.
    """

    mu: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def gauss_legendre(cls, n_mu: int = 5) -> "AngleQuadrature":
        """n_mu-point Gauss-Legendre on [0, 1] (transformed from [-1, 1])."""
        if n_mu < 1:
            raise ValueError(f"require n_mu >= 1; got n_mu={n_mu}")
        nodes_m11, weights_m11 = np.polynomial.legendre.leggauss(n_mu)
        mu_np = 0.5 * (nodes_m11 + 1.0)
        weights_np = 0.5 * weights_m11
        return cls(
            mu=torch.tensor(mu_np, dtype=torch.float64),
            weights=torch.tensor(weights_np, dtype=torch.float64),
        )

    @property
    def n_mu(self) -> int:
        return self.mu.shape[0]
