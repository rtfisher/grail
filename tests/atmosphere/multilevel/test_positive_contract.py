"""Physical regressions for the 2026-11-16 five-level repairs."""
import numpy as np
import pytest
import torch
from atmosphere.atomic_model import AtomicModel
from atmosphere.atmosphere import Atmosphere
from atmosphere.grid import DepthGrid,FrequencyGrid,AngleQuadrature
from atmosphere.multilevel.fixed_point import (_source_from_populations_nlevel,
    _anderson_step_nlevel,solve_equilibrium_nlevel,radiation_from_populations_nlevel,
    cycle_truth_nlevel,ali_loop_nlevel,_planck_nu)
from atmosphere.multilevel.rate_equations import rate_equation_solve_nlevel


def make_atm(eps=1e-6):
    return Atmosphere.balmer_isothermal(T0=6000,n_e0=1e12,epsilon=eps,
        depth=DepthGrid.log_spaced(tau_min=1e-4,tau_max=1e4,n_tau=32),
        frequency=FrequencyGrid.doppler(n_nu=11,half_width=5),
        angle=AngleQuadrature.gauss_legendre(n_mu=3))


def test_source_rejects_saturated_and_inverted_populations():
    atom=AtomicModel.balmer_5level()
    with pytest.raises(ValueError,match='strictly positive'):
        _source_from_populations_nlevel(torch.tensor([[1.,0,0,0,0]]),atom)
    with pytest.raises(ValueError,match='nonpositive line opacity'):
        _source_from_populations_nlevel(torch.full((1,5),0.2,dtype=torch.float64)*torch.tensor([1,100,1,1,1]),atom)


def test_source_resolves_small_excited_populations_without_floors():
    atom=AtomicModel.balmer_5level()
    p=torch.tensor([[1.,1e-30,1e-60,1e-90,1e-120]],dtype=torch.float64)
    s=_source_from_populations_nlevel(p,atom)
    torch.testing.assert_close(s,_source_from_populations_nlevel(p*1e-50,atom),rtol=2e-15,atol=0)
    assert torch.isfinite(s).all() and (s>0).all()
    tr=atom.transitions[0];g=atom.statistical_weights
    exact=2*6.62607015e-34*tr.nu_0**3/2.99792458e8**2*p[:,tr.upper]/(p[:,tr.lower]*g[tr.upper]/g[tr.lower]-p[:,tr.upper])
    torch.testing.assert_close(s[:,0],exact,rtol=2e-15,atol=0)


def test_anderson_secant_solves_affine_map():
    # T(x)=.8x+.2 has fixed point 1; two distinct residuals determine it.
    xs=[np.array([0.]),np.array([.2])]
    fs=[np.array([.2]),np.array([.16])]
    np.testing.assert_allclose(_anderson_step_nlevel(xs,fs,5),[1.],rtol=1e-14,atol=1e-14)


def test_cold_lte_excited_populations_keep_relative_precision():
    atom=AtomicModel.balmer_5level(epsilon=1.)
    T=torch.full((3,),2500.,dtype=torch.float64);J=torch.zeros((3,10),dtype=torch.float64)
    p=rate_equation_solve_nlevel(J,T,T,atomic_model=atom)
    assert (p>0).all()
    for tr in atom.transitions:
        ratio=(atom.statistical_weights[tr.upper]/atom.statistical_weights[tr.lower])*torch.exp(-6.62607015e-34*tr.nu_0/(1.380649e-23*T))
        torch.testing.assert_close(p[:,tr.upper]/p[:,tr.lower],ratio,rtol=2e-4,atol=0)


def test_closed_loop_truth_is_an_actual_positive_source_fixed_point():
    a=make_atm();r=solve_equilibrium_nlevel(a,tol=2e-9)
    assert r.converged,r.failure_reason
    s,j=radiation_from_populations_nlevel(r.populations,a)
    y=cycle_truth_nlevel(r.populations,a)
    sy=_source_from_populations_nlevel(y,a.atomic_model)
    assert float(((y-r.populations)/r.populations).abs().max())<2e-9
    assert float(((sy-s)/s).abs().max())<2e-9
    result=ali_loop_nlevel(a,tol=2e-9)
    assert result.converged,result.failure_reason
    torch.testing.assert_close(result.S_lines,_source_from_populations_nlevel(result.populations,a.atomic_model),rtol=0,atol=0)
    # A closed transition graph does not impose independent line balance.
    B=torch.stack([_planck_nu(tr.nu_0,a.T) for tr in a.atomic_model.transitions],1)
    assert float(((s-((1-1e-6)*j+1e-6*B))/s).abs().max())>1e-3
