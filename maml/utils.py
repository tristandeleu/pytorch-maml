import torch

from collections import OrderedDict
from maml.modules import MetaModule

def update_parameters(model, loss, params=None, step_size=0.5, first_order=False):
    if not isinstance(model, MetaModule):
        raise ValueError()
    grads = torch.autograd.grad(loss, model.meta_parameters(),
        create_graph=not first_order)

    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    out = OrderedDict()

    if not isinstance(step_size, dict):
        step_size = OrderedDict((name, step_size)
            for (name, _) in model.meta_named_parameters())

    for (name, param), grad in zip(params.items(), grads):
        out[name] = param - step_size[name] * grad

    return out

def tensors_to_device(tensors, device=torch.device('cpu')):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()
