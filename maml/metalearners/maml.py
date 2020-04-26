import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters
from maml.utils import tensors_to_device, compute_accuracy

__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'FOMAML']


class ModelAgnosticMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
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
            params, adaptation_results = self.adapt(train_inputs, train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

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

    def adapt(self, inputs, targets, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
        params = None

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits = self.model(inputs, params=params)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if (step == 0) and is_classification_task:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                step_size=step_size, params=params,
                first_order=(not self.model.training) or first_order)

        return params, results

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

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results

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

MAML = ModelAgnosticMetaLearning

class FOMAML(ModelAgnosticMetaLearning):
    def __init__(self, model, optimizer=None, step_size=0.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(FOMAML, self).__init__(model, optimizer=optimizer, first_order=True,
            step_size=step_size, learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps, scheduler=scheduler,
            loss_function=loss_function, device=device)
