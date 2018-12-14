import torch
import torch.nn as nn

from collections import OrderedDict

class MetaModule(nn.Module):
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def update_params(self, loss, step_size=0.5, first_order=False, out=None):
        grads = torch.autograd.grad(loss, self.meta_parameters(),
            create_graph=not first_order)
        if out is None:
            out = OrderedDict()
        if not isinstance(step_size, dict):
            step_size = OrderedDict((name, step_size)
                for (name, _) in self.meta_named_parameters())
        for (name, param), grad in zip(self.meta_named_parameters(), grads):
            out[name] = param - step_size[name] * grad

        return out
