"""Independent affine transfer and physical source-coordinate diagnostics.

All arrays use independent source coordinates S/B0, one per depth. With
equal level weights, n2=S/(Bstar+2S), so this chart enforces conservation
without taking a finite difference larger than the excited fraction.
The independent NumPy recurrence implements positive linear SC only.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares

from atmosphere.atmosphere import Atmosphere
from atmosphere.grid import AngleQuadrature, DepthGrid, FrequencyGrid


def validated_atmosphere(T0=6000.0, ne0=1e10, epsilon=1e-4,
                         tau_max=1e4, n_tau=30, n_nu=41, n_mu=3):
    """Build an isothermal positive-linear-SC diagnostic configuration."""
    return Atmosphere.isothermal(T0=T0, n_e0=ne0, epsilon=epsilon,
        nu_0=2.467381e15,
        depth=DepthGrid.log_spaced(1e-4, tau_max, n_tau),
        frequency=FrequencyGrid.doppler(n_nu=n_nu),
        angle=AngleQuadrature.gauss_legendre(n_mu))


def planck_array(atmosphere):
    pref = 2 * 6.62607015e-34 * atmosphere.nu_0**3 / 2.99792458e8**2
    return pref / np.expm1(6.62607015e-34 * atmosphere.nu_0 /
                           (1.380649e-23 * atmosphere.T.numpy()))


def independent_transfer(atmosphere):
    """Return Lambda, b/B0, surface flux row and boundary flux/B0.

    J/B0 = Lambda @ (S/B0) + boundary. Coefficients are integrated
    independently using expm1 plus an explicit small-u expansion. The lower
    diffusion boundary is included; for these isothermal runs it equals B0.
    Surface flux is 2*pi*integral_0^1 mu I+ dmu, profile averaged.
    """
    tau = atmosphere.depth.tau.numpy()
    x = atmosphere.frequency.nu.numpy()
    weight = (atmosphere.frequency.weights * atmosphere.frequency.profile).numpy()
    n, f = len(tau), len(x)
    matrix, boundary = np.zeros((n, n)), np.zeros(n)
    fluxrow, fluxboundary = np.zeros(n), 0.0
    B = planck_array(atmosphere)
    B0 = B[0]
    for mu, wm in zip(atmosphere.angle.mu.numpy(), atmosphere.angle.weights.numpy()):
        u = np.diff(tau)[:, None] * np.exp(-x*x)[None, :] / mu
        attenuation = np.exp(-u)
        removed = -np.expm1(-u)
        alpha = (removed - u * attenuation) / u
        small = u < 1e-3
        alpha[small] = (u/2 - u*u/3 + u**3/8 - u**4/30 + u**5/144)[small]
        beta = removed-alpha
        # Stream carries n independent source basis responses and one
        # nonzero-boundary response; no author transfer function is imported.
        stream = np.zeros((f, n+1))
        dB = (B[-1]-B[-2])/(tau[-1]-tau[-2])
        stream[:, -1] = (B[-1]+mu*dB)/B0
        boundary[-1] += 0.5*wm*(weight @ stream[:, -1])
        for i in range(n-2, -1, -1):
            stream *= attenuation[i, :, None]
            stream[:, i+1] += alpha[i]
            stream[:, i] += beta[i]
            mean = weight @ stream
            matrix[i] += 0.5*wm*mean[:-1]
            boundary[i] += 0.5*wm*mean[-1]
        mean = weight @ stream
        fluxrow += 2*np.pi*wm*mu*mean[:-1]
        fluxboundary += 2*np.pi*wm*mu*mean[-1]
        stream = np.zeros((f, n))
        for i in range(n-1):
            stream *= attenuation[i, :, None]
            stream[:, i] += alpha[i]
            stream[:, i+1] += beta[i]
            matrix[i+1] += 0.5*wm*(weight @ stream)
    return matrix, boundary, fluxrow, fluxboundary


def truth_source(atmosphere, transfer=None):
    """Independent finite-slab source root including boundary irradiation."""
    transfer = independent_transfer(atmosphere) if transfer is None else transfer
    L, b = transfer[:2]
    eps = atmosphere.epsilon
    B = planck_array(atmosphere)
    return np.linalg.solve(np.eye(len(b))-(1-eps)*L,
                           (1-eps)*b + eps*B/B[0])


def source_from_populations(populations, Bstar):
    """Stable inverse in the positive, noninverted equal-weight domain."""
    return Bstar * populations[..., 1] / (populations[..., 0]-populations[..., 1])


class SourceProblem:
    """Reusable actual neural source cycle in independent normalized coordinates.

    ``cycle`` and ``jacobian`` evaluate the actual network and its derivative
    at the actual radiative intermediate. ``solve`` starts without truth
    information by default; an explicit initial state is recorded by callers.
    """
    def __init__(self, atmosphere, adapter, transfer=None):
        self.atmosphere = atmosphere
        self.transfer = independent_transfer(atmosphere) if transfer is None else transfer
        self.L, self.b = self.transfer[:2]
        self.B = planck_array(atmosphere)
        self.B0 = self.B[0]
        self.Bstar = 2*6.62607015e-34*atmosphere.nu_0**3/2.99792458e8**2
        self.adapter = adapter
        self._L = torch.from_numpy(self.L)
        self._b = torch.from_numpy(self.b)

    def torch_cycle(self, source, eta=1.0):
        J = (self._L @ source + self._b) * self.B0
        populations = self.adapter(J, self.atmosphere.T, self.atmosphere.n_e)
        neural = source_from_populations(populations, self.Bstar)/self.B0
        if eta == 1.0:
            return neural
        truth = (1-self.atmosphere.epsilon)*J/self.B0 + self.atmosphere.epsilon*torch.from_numpy(self.B)/self.B0
        return truth + eta*(neural-truth)

    def cycle(self, source, eta=1.0):
        with torch.no_grad():
            return self.torch_cycle(torch.as_tensor(source, dtype=torch.float64), eta).numpy()

    def jacobian(self, source, eta=1.0):
        source = torch.as_tensor(source, dtype=torch.float64)
        return torch.func.jacrev(lambda x: self.torch_cycle(x, eta))(source).detach().numpy()

    def populations(self, source):
        """Independent excited fractions, with ground fractions 1-q."""
        physical = np.asarray(source)*self.B0
        return physical/(self.Bstar+2*physical)

    def torch_population_cycle(self, q, eta=1.0):
        """Actual cycle in the conserving chart q=n2, n1=1-q."""
        source = self.Bstar*q/(1-2*q)/self.B0
        updated = self.torch_cycle(source, eta)*self.B0
        return updated/(self.Bstar+2*updated)

    def population_cycle(self, q, eta=1.0):
        with torch.no_grad():
            return self.torch_population_cycle(torch.as_tensor(q, dtype=torch.float64), eta).numpy()

    def population_jacobian(self, q, eta=1.0):
        q = torch.as_tensor(q, dtype=torch.float64)
        return torch.func.jacrev(lambda x: self.torch_population_cycle(x, eta))(q).detach().numpy()

    def solve(self, eta=1.0, initial=None, max_nfev=500, tolerance=1e-9):
        # Positive log coordinates avoid negative trial source/populations.
        initial = np.ones(len(self.b)) if initial is None else np.asarray(initial)
        def fun(logsource):
            source = np.exp(logsource)
            return np.log(self.cycle(source, eta))-logsource
        def jac(logsource):
            source = np.exp(logsource)
            out = self.cycle(source, eta)
            return self.jacobian(source, eta)*source[None,:]/out[:,None]-np.eye(len(source))
        result = least_squares(fun, np.log(initial), jac=jac, max_nfev=max_nfev,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        source = np.exp(result.x)
        defect = self.cycle(source, eta)-source
        relative = float(np.max(np.abs(defect)/source))
        return {"source": source, "defect": defect, "relative_defect": relative,
                "converged": bool(result.success and relative<tolerance),
                "nfev": result.nfev, "status": result.status, "message": result.message,
                "initial": initial, "eta": eta}
