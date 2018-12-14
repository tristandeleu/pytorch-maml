import torch.nn as nn
from collections import OrderedDict

from maml.modules.utils import get_subdict

class MetaSequential(nn.Sequential):
    def forward(self, input, params=None):
        for name, module in self._modules.items():
            input = module(input, params=get_subdict(params, name))
        return input
