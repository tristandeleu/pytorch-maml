import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch._jit_internal import weak_module, weak_script_method
from maml.modules.module import MetaModule

@weak_module
class MetaConv1d(nn.Conv1d, MetaModule):
    @weak_script_method
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.conv1d(input, params['weight'], bias,
            self.stride, self.padding, self.dilation, self.groups)

@weak_module
class MetaConv2d(nn.Conv2d, MetaModule):
    @weak_script_method
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.conv2d(input, params['weight'], bias,
            self.stride, self.padding, self.dilation, self.groups)

@weak_module
class MetaConv3d(nn.Conv3d, MetaModule):
    @weak_script_method
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.conv3d(input, params['weight'], bias,
            self.stride, self.padding, self.dilation, self.groups)
