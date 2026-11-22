"""Check public callbacks against the actual split program at nonbaseline CO2."""
import numpy as np
import pytest
import torch

from climate_rce.column import ColumnSpec
from climate_rce.emulator._smooth_fallback import make_smooth_fallback
from climate_rce.host import _T_split_cycle_np, _q_ref_for_T
from climate_rce.jvp import (make_back_residual, make_emulator_jvp_rce,
                             make_host_jvp_rce, make_subgrid_with_nn)


@pytest.mark.parametrize("co2_ppm", [280.0, 560.0, 840.0])
def test_callbacks_match_actual_split_cycle(co2_ppm):
    spec = ColumnSpec(n_z=10, n_v=2, var_names=("T", "q"),
                      pressure=np.linspace(5000., 1e5, 10),
                      var_scales=(300., .01), n_state=10)
    dt = 1e4
    temperature = np.linspace(230., 285., 10)
    profile = np.array([temperature, _q_ref_for_T(spec, temperature)])
    model = make_smooth_fallback(spec, amplitude=1e-3, seed=42).double()
    neural = make_subgrid_with_nn(model, dt=dt)

    def zero(spec, profile):
        return np.zeros_like(profile)

    def cycle(temperature, subgrid):
        return _T_split_cycle_np(spec, temperature, subgrid, dt=dt, co2_ppm=co2_ppm)

    rhs = make_back_residual(spec, profile, neural, zero, dt=dt, co2_ppm=co2_ppm)
    expected = (cycle(temperature, neural) - cycle(temperature, zero)) / 300.
    np.testing.assert_allclose(rhs.numpy(), expected, rtol=1e-10, atol=1e-15)
    host = make_host_jvp_rce(spec, profile, neural, dt=dt, co2_ppm=co2_ppm)
    emulator = make_emulator_jvp_rce(spec, profile, neural, dt=dt,
                                    co2_ppm=co2_ppm, model=model)
    direction = np.random.default_rng(7).normal(size=10)
    direction /= np.linalg.norm(direction)
    # Independent central differences of the host's NumPy physical program.
    step_kelvin = .003
    expected_jvp = (cycle(temperature + step_kelvin * direction, neural)
                    - cycle(temperature - step_kelvin * direction, neural)) / (2 * step_kelvin)
    actual_jvp = emulator(host(torch.from_numpy(direction))).detach().numpy()
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2e-7, atol=2e-9)
