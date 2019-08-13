import torch
import torch.nn.functional as F
import numpy as np

from maml.metalearners.maml import ModelAgnosticMetaLearning

__all__ = ['MetaSGD']


class MetaSGD(ModelAgnosticMetaLearning):
    def __init__(self, model, optimizer=None, init_step_size=0.1,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(MetaSGD, self).__init__(model, optimizer=optimizer,
            step_size=init_step_size, learn_step_size=True,
            per_param_step_size=True, num_adaptation_steps=num_adaptation_steps,
            scheduler=scheduler, loss_function=loss_function, device=device)
