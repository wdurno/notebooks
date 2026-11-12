"""Float64 diagonals and observed scores for the original 84px grid model."""

import copy
import torch
from torch.func import grad, vmap
from src.parameters import ParameterLayout
from mnist_experiment.combined_memory.curvature import _jacobian
from mnist_experiment.fisher_memory_stm_capacity.model import flat_prediction, memory_indices
from mnist_experiment.fisher_memory_grid_registration.curvature import gaussian_nll


class MergedHead(torch.nn.Linear):
    def forward(self, inputs, stm_scale=1.):
        return super().forward(inputs)


def merge(model, layout, theta):
    """Make an independent 3268-coordinate anchor without mutating training."""
    merged = copy.deepcopy(model).to(device=theta.device, dtype=theta.dtype)
    values = layout.unflatten_named(theta.detach())
    head = MergedHead(54, 54, device="meta", dtype=theta.dtype)
    gamma = model.classifier.lora_alpha / model.classifier.rank
    head.weight = torch.nn.Parameter((values["classifier.weight"] + gamma *
        values["classifier.lora_B"] @ values["classifier.lora_A"]).clone())
    head.bias = torch.nn.Parameter(values["classifier.bias"].clone())
    merged.classifier = head
    out = ParameterLayout.from_module(merged)
    anchor = out.flatten_module(merged, detach=True)
    if anchor.numel() != 3268:
        raise ValueError("incorrect merged anchor chart")
    source = {s.name:s for s in layout.specs}
    mapping = [dict(name=s.name, source_start=source[s.name].start, source_stop=source[s.name].stop,
                    merged_start=s.start, merged_stop=s.stop) for s in out.specs]
    return merged, out, anchor, mapping


def _validate(theta, x, selected):
    if theta.ndim != 1 or x.ndim != 4 or tuple(x.shape[1:]) != (2,84,84) or not len(x):
        raise ValueError("84px grids and a flat parameter vector required")
    if selected.ndim != 1 or selected.unique().numel() != len(selected) or not len(selected):
        raise ValueError("unique nonempty coordinate indices required")
    if (selected < 0).any() or (selected >= len(theta)).any() or not torch.isfinite(theta).all():
        raise ValueError("invalid selected coordinates or parameters")


def diagonal(model, layout, theta, x, indices=None, *, chunk_size=2, output_chunk=6, check=lambda:None):
    return paired_change(model, layout, theta, theta, x, indices,
                         chunk_size=chunk_size, output_chunk=output_chunk, check=check)[1]


def paired_change(model, layout, before, after, x, indices=None, *, chunk_size=2,
                  output_chunk=6, check=lambda:None):
    selected = (torch.arange(len(before),device=before.device) if indices is None else indices.to(before.device))
    _validate(before,x,selected)
    _validate(after,x,selected)
    if before.shape != after.shape or chunk_size < 1 or not 1 <= output_chunk <= 54:
        raise ValueError("incompatible finite-change inputs")
    minus, plus = before.detach().double(), after.detach().double()
    delta = torch.zeros(len(selected), dtype=torch.float64, device=before.device)
    fresh = torch.zeros_like(delta)
    zero = torch.equal(before,after)
    for images in x.split(chunk_size):
        images = images.double()
        for start in range(0,54,output_chunk):
            check()
            width = min(output_chunk,54-start)
            jp = _jacobian(model,layout,plus,images,selected,start,width)
            jm = jp if zero else _jacobian(model,layout,minus,images,selected,start,width)
            delta.add_(((jp-jm)*(jp+jm)).sum((0,1)))
            fresh.add_(jp.square().sum((0,1)))
    delta /= 4*len(x)
    fresh /= 4*len(x)
    if not torch.isfinite(delta).all() or not torch.isfinite(fresh).all():
        raise FloatingPointError("nonfinite Gaussian Fisher change")
    return delta, fresh


def observed_scores(model, layout, theta, x, y, *, chunk_size=2, check=lambda:None):
    """Positive per-grid score s = -grad(NLL), integrating no observed labels."""
    _validate(theta,x,torch.arange(len(theta),device=theta.device))
    if y.shape != (len(x),54):
        raise ValueError("one 54-coordinate target per grid required")
    fixed = theta.detach().double()
    def likelihood(value,image,target):
        return -gaussian_nll(flat_prediction(model,layout,value,image.unsqueeze(0)),target.unsqueeze(0)).sum()
    rows=[]
    for offset in range(0,len(x),chunk_size):
        check()
        with torch.no_grad():
            rows.append(vmap(grad(likelihood),in_dims=(None,0,0))(
                fixed,x[offset:offset+chunk_size].double(),y[offset:offset+chunk_size].double()).detach())
    result=torch.cat(rows)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite observed scores")
    return result
