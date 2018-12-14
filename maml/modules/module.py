import torch
import torch.nn as nn

from collections import OrderedDict

class MetaModule(nn.Module):
    def update_params(self, loss, step_size=0.5, first_order=False, out=None):
        grads = torch.autograd.grad(loss, self.parameters(),
            create_graph=not first_order)
        if out is None:
            out = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            out[name] = param - step_size * grad

        return out
