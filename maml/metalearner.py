import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from collections import OrderedDict
from maml.utils import update_parameters

class ModelAgnosticMetaLearning(object):
    def __init__(self, model, optimizer, step_size=0.1, first_order=False,
                 learn_step_size=True, per_param_step_size=True,
                 scheduler=None, loss_function=F.cross_entropy, num_workers=4):
        self.model = model
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.loss_function = loss_function

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=param.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            # TODO: model.device doesn't exist, find an alternative
            # QKFIX: device=None instead of device=model.device
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=None, requires_grad=learn_step_size)

        if learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])

    def get_inner_loss(self, inputs, targets):
        logits = self.model(inputs.view(-1, *inputs.shape[2:]))
        inner_loss = self.loss_function(logits, targets.view(-1), reduction='none')
        inner_loss = torch.mean(inner_loss.view_as(targets), dim=1)
        return inner_loss

    def get_outer_loss(self, batch):
        if batch['test'] is None:
            raise RuntimeError('The batch does not contain any test dataset.')

        inner_loss = self.get_inner_loss(*batch['train'])
        outer_loss = torch.tensor(0.)
        for task_id, (test_inputs, test_targets) in enumerate(zip(*batch['test'])):
            params = update_parameters(self.model, inner_loss[task_id],
                step_size=self.step_size, first_order=self.first_order)
            test_logits = self.model(test_inputs, params=params)
            outer_loss += self.loss_function(test_logits, test_targets)
        outer_loss.div_(test_targets.size(0))
        return outer_loss

    def train(self, dataloader, max_batches=500):
        for loss in self.train_iter(dataloader, max_batches=max_batches):
            pass

    def train_iter(self, dataloader, max_batches=500):
        num_batches = 0
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.model.train()
                self.optimizer.zero_grad()

                outer_loss = self.get_outer_loss(batch)
                yield outer_loss

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1
