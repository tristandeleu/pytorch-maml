import torch

from collections import OrderedDict
from maml.modules import MetaModule

def update_parameters(model, loss, step_size=0.5, first_order=False, out=None):
    if not isinstance(model, MetaModule):
        raise ValueError()
    grads = torch.autograd.grad(loss, model.meta_parameters(),
        create_graph=not first_order, retain_graph=True)
    if out is None:
        out = OrderedDict()
    if not isinstance(step_size, dict):
        step_size = OrderedDict((name, step_size)
            for (name, _) in model.meta_named_parameters())
    for (name, param), grad in zip(model.meta_named_parameters(), grads):
        out[name] = param - step_size[name] * grad

    return out
