import torch

from collections import OrderedDict
from maml.modules import MetaModule

def update_parameters(model, loss, params=None, step_size=0.5, first_order=False):
    if not isinstance(model, MetaModule):
        raise ValueError()

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss, params.values(),
        create_graph=not first_order)

    out = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            out[name] = param - step_size[name] * grad
    else:
        for (name, param), grad in zip(params.items(), grads):
            out[name] = param - step_size * grad

    return out

def compute_accuracy(logits, targets):
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

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
