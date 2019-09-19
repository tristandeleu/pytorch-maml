import torch
import torch.nn.functional as F
import math
import os
import time
import json

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical
from torchvision.transforms import ToTensor, Resize, Compose

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from maml.metalearners import ModelAgnosticMetaLearning

def main(args):
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

        folder = os.path.join(args.output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)

        args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shots,
                                      num_test_per_class=args.num_shots_test)
    if args.dataset == 'sinusoid':
        transform = ToTensor()

        meta_train_dataset = Sinusoid(args.num_shots + args.num_shots_test,
            num_tasks=1000000, transform=transform, target_transform=transform,
            dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(args.num_shots + args.num_shots_test,
            num_tasks=1000000, transform=transform, target_transform=transform,
            dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
    elif args.dataset == 'omniglot':
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(args.folder, transform=transform,
            target_transform=Categorical(args.num_ways),
            num_classes_per_task=args.num_ways, meta_train=True,
            dataset_transform=dataset_transform, download=True)
        meta_val_dataset = Omniglot(args.folder, transform=transform,
            target_transform=Categorical(args.num_ways),
            num_classes_per_task=args.num_ways, meta_val=True,
            dataset_transform=dataset_transform)

        model = ModelConvOmniglot(args.num_ways, hidden_size=args.hidden_size)
    elif args.dataset == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(args.folder, transform=transform,
            target_transform=Categorical(args.num_ways),
            num_classes_per_task=args.num_ways, meta_train=True,
            dataset_transform=dataset_transform, download=True)
        meta_val_dataset = MiniImagenet(args.folder, transform=transform,
            target_transform=Categorical(args.num_ways),
            num_classes_per_task=args.num_ways, meta_val=True,
            dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(args.num_ways, hidden_size=args.hidden_size)
    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(args.dataset))

    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    metalearner = ModelAgnosticMetaLearning(model, meta_optimizer,
        first_order=args.first_order, num_adaptation_steps=args.num_steps,
        step_size=args.step_size, loss_function=F.cross_entropy, device=device)

    best_val_accuracy = None

    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    for epoch in range(args.num_epochs):
        metalearner.train(meta_train_dataloader, max_batches=args.num_batches,
                          verbose=args.verbose, desc='Training', leave=False)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))

        if (best_val_accuracy is None) \
                or (best_val_accuracy < results['accuracies_after']):
            best_val_accuracy = results['accuracies_after']
            if args.output_folder is not None:
                with open(args.model_path, 'wb') as f:
                    torch.save(model.state_dict(), f)

    if hasattr(meta_train_dataset, 'close'):
        meta_train_dataset.close()
        meta_val_dataset.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet'], default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', default=None,
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)
