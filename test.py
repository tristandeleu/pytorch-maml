import torch
import torch.nn.functional as F
import os
import json

from torchmeta.utils.data import BatchMetaDataLoader

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.folder is not None:
        config['folder'] = args.folder
    if args.num_steps > 0:
        config['num_steps'] = args.num_steps
    if args.num_batches > 0:
        config['num_batches'] = args.num_batches
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    benchmark = get_benchmark_by_name(config['dataset'],
                                      config['folder'],
                                      config['num_ways'],
                                      config['num_shots'],
                                      config['num_shots_test'],
                                      hidden_size=config['hidden_size'])

    with open(config['model_path'], 'rb') as f:
        benchmark.model.load_state_dict(torch.load(f, map_location=device))

    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            first_order=config['first_order'],
                                            num_adaptation_steps=config['num_steps'],
                                            step_size=config['step_size'],
                                            loss_function=benchmark.loss_function,
                                            device=device)

    results = metalearner.evaluate(meta_test_dataloader,
                                   max_batches=config['num_batches'],
                                   verbose=args.verbose,
                                   desc='Test')

    # Save results
    dirname = os.path.dirname(config['model_path'])
    with open(os.path.join(dirname, 'results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')
    parser.add_argument('config', type=str,
        help='Path to the configuration file returned by `train.py`.')
    parser.add_argument('--folder', type=int, default=None,
        help='Path to the folder the data is downloaded to. '
        '(default: path defined in configuration file).')

    # Optimization
    parser.add_argument('--num-steps', type=int, default=-1,
        help='Number of fast adaptation steps, ie. gradient descent updates '
        '(default: number of steps in configuration file).')
    parser.add_argument('--num-batches', type=int, default=-1,
        help='Number of batch of tasks per epoch '
        '(default: number of batches in configuration file).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()
    main(args)
