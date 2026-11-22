"""Physical passivity, independent recurrence, and conserving AD contracts."""
import numpy as np
import pytest
import torch
from scipy.integrate import quad

from atmosphere.grid import FrequencyGrid
from atmosphere.formal_solver import formal_solve, _parabolic_coefficients
from atmosphere.twolevel.fixed_point import line_profile_average
from atmosphere.twolevel.diagnostics import validated_atmosphere, independent_transfer, truth_source
from atmosphere.twolevel.fixed_point import _source_from_populations


@pytest.mark.parametrize("n_nu", [3, 11, 21, 41])
def test_profile_average_is_convex(n_nu):
    f = FrequencyGrid.doppler(n_nu)
    assert torch.all(f.weights*f.profile>0)
    assert abs(float(torch.sum(f.weights*f.profile))-1)<4e-16


def test_positive_linear_sc_matches_independent_recurrence():
    atm = validated_atmosphere(n_tau=30,n_nu=11)
    L,b,*_ = independent_transfer(atm)
    assert L.min()>=0
    assert b.min()>=0
    assert np.max(L.sum(1)+b)<=1+2e-14
    rng = np.random.default_rng(20261116)
    for source in [np.ones(30),rng.uniform(0,1,30)]:
        # Normalize the author's physical boundary to the independent B0.
        from atmosphere.twolevel.diagnostics import planck_array
        B0=planck_array(atm)[0]
        actual=line_profile_average(formal_solve(torch.from_numpy(source*B0)[:,None].expand(-1,11),atm),atm)/B0
        np.testing.assert_allclose(actual.numpy(),L@source+b,rtol=3e-13,atol=3e-15)
    s=truth_source(atm,(L,b))
    assert 0<s.min()<=s.max()<=1+2e-11


def test_invalid_population_states_are_rejected_without_a_source_floor():
    atm=validated_atmosphere(n_tau=2)
    for populations in [[[0.,0.],[1.,0.]],[[.4,.6],[1.,0.]],[[1.,-1e-12],[1.,0.]]]:
        with pytest.raises(ValueError,match="source requires"):
            _source_from_populations(torch.tensor(populations,dtype=torch.float64),atm)
    zero_upper=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
    torch.testing.assert_close(_source_from_populations(zero_upper,atm),torch.zeros(2,dtype=torch.float64))


@pytest.mark.parametrize("u,v",[(1e-5,2e-5),(5e-4,2e-4),(.1,.2),(3.,8.)])
def test_parabolic_moments_match_direct_integrals(u,v):
    actual=[float(c) for c in _parabolic_coefficients(torch.tensor(u,dtype=torch.float64),torch.tensor(v,dtype=torch.float64))]
    basis=[lambda t:t*(t-u)/(v*(u+v)),
           lambda t:-(t+v)*(t-u)/(u*v),
           lambda t:(t+v)*t/(u*(u+v))]
    reference=[quad(lambda t:fn(t)*np.exp(-(u-t)),0,u,epsabs=1e-14)[0] for fn in basis]
    np.testing.assert_allclose(actual,reference,rtol=2e-8,atol=1e-14)
