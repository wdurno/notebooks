import json
from dataclasses import replace

import pytest
import torch

from mnist_experiment.fisher_memory_stm_capacity import core
from mnist_experiment.fisher_memory_stm_capacity.model import build_model, memory_indices, flat_prediction
from mnist_experiment.fisher_memory_grid_registration.runner import settings
from mnist_experiment.fisher_memory_grid_registration.learner import LIKELIHOOD
from mnist_experiment.combined_memory.replicated_combination import curvature,learner
from mnist_experiment.combined_memory.replicated_combination.config import Policy,configuration,HISTORY


@pytest.fixture
def model_case():
    torch.set_num_threads(2)
    cfg=configuration(json.loads((HISTORY/'resolved_config.json').read_text()),device='cpu',smoke=True)
    model,layout=build_model(77,stm_config='head_r1')
    theta=layout.flatten_module(model,detach=True)
    generator=torch.Generator().manual_seed(12)
    x=torch.randn(2,2,84,84,generator=generator)
    y=torch.randn(2,54,generator=generator)
    return cfg,model,layout,theta,x,y


def test_same_objective_and_gradient_as_historical_diagonal_matrix(model_case):
    cfg,m,l,t,x,y=model_case
    idx,_=memory_indices(l)
    d=torch.linspace(.001,.01,len(idx),dtype=torch.float64)
    recipe=Policy('B').recipe(cfg)
    fit=settings(cfg,recipe)
    ages=torch.tensor([.2,.8],dtype=torch.float64)
    new,meta=learner.prepare_objective(m,l,t,x,y,x,y,ages,d,t[idx],recipe.method,fit,19)
    old,_=core.prepare_objective(m,l,t,x,y,x,y,ages,torch.diag(d),t[idx],recipe.method,fit,19)
    candidate=(t+torch.linspace(-1e-3,1e-3,len(t))).requires_grad_()
    a,parts=new(candidate); b,original=old(candidate)
    assert torch.allclose(a,b,rtol=1e-7,atol=1e-7)
    for k in parts:
        assert torch.allclose(parts[k],original[k],rtol=1e-7,atol=1e-7),k
    ga=torch.autograd.grad(a,candidate)[0]
    gb=torch.autograd.grad(b,candidate)[0]
    assert torch.allclose(ga,gb,rtol=2e-6,atol=2e-6)


def test_merged_anchor_scores_and_no_training_mutation(model_case):
    _,m,l,t,x,y=model_case
    merged,ml,anchor,mapping=curvature.merge(m,l,t)
    assert len(anchor)==3268 and len(mapping)==len(ml.specs)
    assert torch.equal(flat_prediction(m,l,t,x),flat_prediction(merged,ml,anchor,x))
    scores=curvature.observed_scores(merged,ml,anchor,x,y,chunk_size=1)
    exact=torch.func.grad(lambda q: -curvature.gaussian_nll(
        flat_prediction(merged,ml,q,x[:1].double()),y[:1].double()).sum())(anchor.double())
    assert torch.allclose(scores[0],exact,rtol=1e-10,atol=1e-10)
    assert torch.equal(l.flatten_module(m,detach=True),t)


def test_fisher_change_zero_and_chunk_invariant(model_case):
    _,m,l,t,x,y=model_case
    idx,_=memory_indices(l)
    change,initial=curvature.paired_change(m,l,t,t,x,idx,chunk_size=1,output_chunk=9)
    assert torch.count_nonzero(change)==0
    after=t.clone(); after[:4]+=1e-4
    delta,fresh=curvature.paired_change(m,l,t,after,x,idx,chunk_size=2,output_chunk=6)
    assert torch.allclose(fresh,curvature.diagonal(m,l,after,x,idx,output_chunk=18),rtol=1e-10,atol=1e-10)
    assert torch.allclose(delta,fresh-initial,rtol=1e-8,atol=1e-10)


def test_diagonal_precision_against_original_jacobian_pass(model_case):
    _,m,l,t,x,y=model_case
    idx,_=memory_indices(l)
    original=torch.zeros(len(idx),dtype=torch.float64)
    for rows in core.output_jacobians(m,l,t,x,indices=idx,chunk_size=1):
        original+=rows.square().sum((0,1))/(4*len(x))
    direct=curvature.diagonal(m,l,t,x,idx,chunk_size=1)
    print('float32/float64 Jacobian diagonal relative error',float((direct-original).norm()/direct.norm()))
    assert torch.allclose(direct,original,rtol=1e-4,atol=1e-7)


def test_head_weight_move_preserves_head_fisher_not_encoder(model_case):
    _,m,l,t,x,y=model_case
    idx,_=memory_indices(l)
    after=t.clone()
    spec=next(s for s in l.specs if s.name=='classifier.weight')
    after[spec.start:spec.stop]+=1e-2
    delta,_=curvature.paired_change(m,l,t,after,x[:1],idx)
    head=(idx>=spec.start)
    assert torch.count_nonzero(delta[head])==0
    assert delta[~head].norm()>0


def state():
    return dict(schema_version=1,likelihood=LIKELIHOOD,nominal_arrival_size=4,
        theta=torch.zeros(3),ltm_indices=torch.tensor([0,1]),anchor_L=torch.zeros(2),
        fisher_L=dict(kind='diagonal',values=torch.ones(2,dtype=torch.float64)),capacity=0,
        replay=[],consolidated_ids=[],cumulative_count=0,accepted_arrivals=0,archive_updates=0)


def test_archive_split_retirement_weights_and_unique_evidence():
    s=state(); z=torch.tensor([2.,3.],dtype=torch.float64); q=1/3000
    once,_,_=learner.commit(s,s['theta'],[0,1,2,3],z,torch.zeros(2),pi=.025,q_archive=q)
    first,q1,_=learner.commit(s,s['theta'],[0],z,torch.zeros(2),pi=.025,q_archive=q)
    second,_,_=learner.commit(first,s['theta'],[1,2,3],z,torch.zeros(2),pi=.025,q_archive=q1)
    assert torch.allclose(once['fisher_L']['values'],second['fisher_L']['values'],rtol=1e-14,atol=1e-14)
    with pytest.raises(ValueError,match='already'):
        learner.commit(first,s['theta'],[0],z,torch.zeros(2),pi=.025,q_archive=q1)


def test_unprojected_signed_change_before_psd_clip():
    s=state()
    out,_,diagnostic=learner.commit(s,s['theta'],[0],torch.zeros(2),torch.tensor([-2.,1.]),pi=.025,q_archive=1/3000)
    assert diagnostic['negative_mass']>0 and diagnostic['projection_norm']>0
    assert out['fisher_L']['values'][0]==0


def test_fixed_references_reject_ridge_change():
    for name in ['R','B']:
        with pytest.raises(ValueError,match='fixed'):
            Policy(name,ridge=3.)
