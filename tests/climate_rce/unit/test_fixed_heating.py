"""The same physical heating law must survive changes in numerical timestep."""
import numpy as np
import pytest
import torch

from climate_rce.column import ColumnSpec
from climate_rce.emulator._smooth_fallback import make_smooth_fallback
from climate_rce.host import _T_split_cycle_np, _q_ref_for_T
from climate_rce.jvp import (make_back_residual, make_emulator_jvp_rce,
                             make_host_jvp_rce, make_subgrid_with_nn)


def setup():
    spec = ColumnSpec(n_z=10, n_v=2, var_names=("T", "q"),
                      pressure=np.linspace(5000., 1e5, 10),
                      var_scales=(300., .01), n_state=10)
    model = make_smooth_fallback(spec, amplitude=1e-4, seed=42).double()
    temperature = np.linspace(245., 290., 10)
    profile = np.array([temperature, _q_ref_for_T(spec, temperature)])
    return spec, model, temperature, profile


def zero(spec, profile):
    return np.zeros_like(profile)


def test_physical_heating_is_independent_of_cycle_timestep():
    spec, model, _, profile = setup()
    expected = model.predict(profile[0]/300.) * 300./2e5
    for dt in [250., 1e3, 1e4, 2e5]:
        heating = make_subgrid_with_nn(model, dt=dt, output_timescale=2e5)
        np.testing.assert_array_equal(heating(spec, profile)[0], expected)
    # The former convention changes the physical law; this control must differ.
    legacy = make_subgrid_with_nn(model, dt=250.)
    assert np.linalg.norm(legacy(spec, profile)[0])/np.linalg.norm(expected) == pytest.approx(800.)


@pytest.mark.parametrize("dt", [250., 1e4, 2e5])
@pytest.mark.parametrize("co2_ppm", [280., 560.])
def test_fixed_heating_callbacks_match_independent_numpy_cycle(dt, co2_ppm):
    spec, model, temperature, profile = setup()
    heating = make_subgrid_with_nn(model, dt=dt, output_timescale=2e5)

    def cycle(t, subgrid):
        return _T_split_cycle_np(spec, t, subgrid, dt=dt, co2_ppm=co2_ppm)

    rhs = make_back_residual(spec, profile, heating, zero, dt=dt, co2_ppm=co2_ppm)
    np.testing.assert_allclose(rhs.numpy(), (cycle(temperature, heating)-cycle(temperature, zero))/300.,
                               rtol=1e-7, atol=2e-16)
    host = make_host_jvp_rce(spec, profile, heating, dt=dt, co2_ppm=co2_ppm)
    emulator = make_emulator_jvp_rce(spec, profile, heating, dt=dt,
                                    co2_ppm=co2_ppm, model=model)
    # All independent directions catch a missing dt/output_timescale factor.
    step = .003
    for v in np.eye(10):
        fd = (-cycle(temperature+2*step*v, heating)+8*cycle(temperature+step*v, heating)
              -8*cycle(temperature-step*v, heating)+cycle(temperature-2*step*v, heating))/(12*step)
        actual = emulator(host(torch.from_numpy(v))).detach().numpy()
        np.testing.assert_allclose(actual, fd, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("timescale", [0., -1., float("nan"), float("inf")])
def test_invalid_physical_output_timescale_is_rejected(timescale):
    _, model, _, _ = setup()
    with pytest.raises(ValueError, match="finite and positive"):
        make_subgrid_with_nn(model, output_timescale=timescale)
