import torch.nn as nn

from collections import OrderedDict
from maml.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
    MetaSequential, MetaLinear)
from maml.modules.utils import get_subdict

def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.LeakyReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))

class MetaVGGNetwork(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(MetaVGGNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        
        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))

        self.classifier = MetaLinear(hidden_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs,
            params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features,
            params=get_subdict(params, 'classifier'))
        return logits
