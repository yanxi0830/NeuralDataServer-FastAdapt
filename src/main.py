from __future__ import print_function
import argparse
import os
import torch
import numpy as np
import imp
import algorithms as alg
import torch.backends.cudnn
import pickle
from dataloader import DataLoader, GenericDataset
from algorithms.ClassificationModel import ClassificationModel


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    set_random_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='GenericFastAdapt',
                        help='config file with parameters of the experiment')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--imagedir', type=str, default='', help='path to image directory containing client images')
    parser.add_argument('--experts_dir', type=str, default='', help='path to directory containing experts')
    args_opt = parser.parse_args()

    exp_config_file = os.path.join('.', 'config', args_opt.exp + '.py')
    exp_directory = os.path.join('.', 'experiments', args_opt.exp + '-' + args_opt.experts_dir.replace('/', '-'))

    # Load the configuration params of the experiment
    print('Launching experiment: %s' % exp_config_file)
    config = imp.load_source("", exp_config_file).config
    config['exp_dir'] = exp_directory  # the place where logs, models, and other stuff will be stored

    config['image_directory'] = args_opt.imagedir

    data_test_opt = config['data_test_opt']
    dataset_test = GenericDataset(
        dataset_name=data_test_opt['dataset_name'],
        split=data_test_opt['split'],
        config=config)

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=data_test_opt['batch_size'],
        num_workers=args_opt.num_workers,
        shuffle=False)

    z = {}
    algorithm = ClassificationModel(config)

    for expert_fname in os.listdir(args_opt.experts_dir):
        expert_path = os.path.join(args_opt.experts_dir, expert_fname)
        if not os.path.isfile(expert_path):
            continue
        algorithm.init_network()
        algorithm.load_pretrained(expert_path)
        if args_opt.cuda:
            algorithm.load_to_gpu()
        eval_stats = algorithm.evaluate(dloader_test)
        z[expert_fname] = eval_stats['prec1']

    print(z)
    with open(os.path.join(exp_directory, 'z.pickle'), 'wb') as f:
        pickle.dump(z, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
