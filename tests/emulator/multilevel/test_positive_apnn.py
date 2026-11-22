"""Versioned encoder, physical head and derivative contracts."""
import pytest
import torch
from atmosphere.atomic_model import AtomicModel
from atmosphere.multilevel.fixed_point import _source_from_populations_nlevel
from emulator.multilevel.positive_apnn import (PositivePopulationAPNN,radiation_features,
    make_positive_cycle_adapter,save_positive_checkpoint,load_positive_checkpoint)
from emulator.multilevel.jvp import make_cycle_adapter_5level


def test_epsilon_conditioning_and_positive_head(tmp_path):
    atom=AtomicModel.balmer_5level(epsilon=1e-6);net=PositivePopulationAPNN(atom)
    J=torch.zeros((2,10),dtype=torch.float64);T=torch.full((2,),8000.,dtype=torch.float64)
    features=radiation_features(J,T,atom)
    changed=radiation_features(J,T,AtomicModel.balmer_5level(epsilon=1e-2))
    assert float((changed-features).abs().min())>9
    adapter=make_cycle_adapter_5level(net,atom);p=adapter(J,T,T)
    assert (p>0).all() and torch.isfinite(_source_from_populations_nlevel(p,atom)).all()
    torch.testing.assert_close(p.sum(1),torch.ones(2,dtype=torch.float64),rtol=0,atol=3e-16)
    torch.testing.assert_close(p,net(features),rtol=0,atol=0)
    path=tmp_path/'v2.pt';save_positive_checkpoint(path,net,{'version':2})
    loaded,meta=load_positive_checkpoint(path,atom)
    torch.testing.assert_close(loaded(features),p,rtol=0,atol=0)
    assert meta['version']==2


def test_adapter_preserves_its_actual_derivative():
    atom=AtomicModel.balmer_5level(epsilon=1e-3);net=PositivePopulationAPNN(atom)
    # A nonzero learned correction ensures this probes the neural path.
    with torch.no_grad():net.net[-1].weight.fill_(0.001)
    f=torch.full((3,10),-15.,dtype=torch.float64)
    J=torch.exp(f).requires_grad_(True);T=torch.full((3,),8000.,dtype=torch.float64)
    adapter=make_positive_cycle_adapter(net,atom);direction=J.detach()*0.01
    _,actual=torch.autograd.functional.jvp(lambda j:adapter(j,T,T),J,direction)
    fd=(adapter(J+1e-3*direction,T,T)-adapter(J-1e-3*direction,T,T))/(2e-3)
    torch.testing.assert_close(actual,fd,rtol=2e-5,atol=1e-14)
    assert actual.abs().max()>0


def test_rate_relaxation_uses_actual_balance_without_reference_runtime(monkeypatch):
    from emulator.multilevel.positive_apnn import RateRelaxedAPNN
    from atmosphere.multilevel.rate_equations import rate_equation_solve_nlevel
    from atmosphere.multilevel.fixed_point import _planck_nu
    atom=AtomicModel.balmer_5level(epsilon=1e-4)
    T=torch.full((4,),8000.,dtype=torch.float64)
    B=torch.stack([_planck_nu(tr.nu_0,T) for tr in atom.transitions],1)
    J=B*torch.linspace(.1,.3,10,dtype=torch.float64)
    truth=rate_equation_solve_nlevel(J,T,T,atomic_model=atom)
    net=RateRelaxedAPNN(atom,rate_sweeps=4)
    with torch.no_grad():net.net[-1].bias.fill_(.05)
    def forbidden(*args,**kwargs):raise AssertionError('reference solve used at neural runtime')
    monkeypatch.setattr('atmosphere.multilevel.rate_equations.rate_equation_solve_nlevel',forbidden)
    adapter=make_cycle_adapter_5level(net,atom)
    pred=adapter(J,T,T)
    assert float(((pred-truth)/truth).abs().max())<1e-7
    assert (_source_from_populations_nlevel(pred,atom)>0).all()
    direction=J*.01
    _,ad=torch.autograd.functional.jvp(lambda j:adapter(j,T,T),J,direction)
    fd=(adapter(J+1e-3*direction,T,T)-adapter(J-1e-3*direction,T,T))/(2e-3)
    torch.testing.assert_close(ad,fd,rtol=2e-5,atol=1e-14)


def test_v3_checkpoint_rejects_atomic_mismatch(tmp_path):
    from dataclasses import replace
    from emulator.multilevel.positive_apnn import (RateRelaxedAPNN,
        save_rate_relaxed_checkpoint,load_rate_relaxed_checkpoint)
    atom=AtomicModel.balmer_5level(epsilon=1e-3)
    model=RateRelaxedAPNN(atom,rate_sweeps=4);path=tmp_path/'v3.pt'
    save_rate_relaxed_checkpoint(path,model,{'version':3})
    loaded,meta=load_rate_relaxed_checkpoint(path,AtomicModel.balmer_5level(epsilon=1e-6))
    assert loaded.rate_sweeps==4 and meta['version']==3
    changed=replace(atom,statistical_weights=(2,8,18,32,52))
    with pytest.raises(ValueError,match='atomic-model'):
        load_rate_relaxed_checkpoint(path,changed)
