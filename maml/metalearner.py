import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from maml.utils import update_parameters, tensors_to_device, compute_accuracy

class ModelAgnosticMetaLearning(object):
    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])

    def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            params = None
            for step in range(self.num_adaptation_steps):
                train_logits = self.model(train_inputs, params=params)
                inner_loss = self.loss_function(train_logits, train_targets)
                results['inner_losses'][step, task_id] = inner_loss.item()

                if (step == 0) and is_classification_task:
                    results['accuracies_before'][task_id] = compute_accuracy(
                        train_logits, train_targets)

                self.model.zero_grad()
                params = update_parameters(self.model, inner_loss,
                    step_size=self.step_size, params=params,
                    first_order=(not self.model.training) or self.first_order)

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                    test_logits, test_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results

    def train(self, dataloader, max_batches=500, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies_after']))
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=500):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)
                yield results

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        results = {
            'mean_outer_loss': mean_outer_loss,
            'accuracies_after': mean_accuracy
        }

        return results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch)
                yield results

                num_batches += 1
