"""Define a generic class for training and testing learning algorithms."""
from __future__ import print_function
import os
import os.path
import imp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim

import utils
import datetime
import logging
from collections import OrderedDict

from pdb import set_trace as breakpoint


class Algorithm():
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir'])
        self.set_log_file_handler()

        self.logger.info('Algorithm options %s' % opt)
        self.opt = opt
        self.init_network()
        self.init_all_criterions()
        self.allocate_tensors()
        self.curr_epoch = 0
        self.optimizers = {}

        self.keep_best_model_metric_name = opt['best_metric'] if ('best_metric' in opt) else None
        self.cuda = False

    def set_experiment_dir(self, directory_path):
        self.exp_dir = directory_path
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.exp_dir, 'logs')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        now_str = datetime.datetime.now().__str__().replace(' ', '_')

        self.log_file = os.path.join(log_dir, 'LOG_INFO_' + now_str + '.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)

    def init_network(self):
        networks_defs = self.opt['networks']['model']
        def_file = networks_defs['def_file']
        net_opt = networks_defs['opt']
        self.logger.info('==> Initiliaze network from file %s with opts: %s' % (def_file, net_opt))
        self.network = imp.load_source("", def_file).create_model(net_opt)

    def load_pretrained(self, pretrained_path):
        self.logger.info('==> Load pretrained parameters from file %s:' % (pretrained_path))

        assert (os.path.isfile(pretrained_path))
        pretrained_model = torch.load(pretrained_path)
        pretrained_model = self.remove_module_prefix(pretrained_model)
        if pretrained_model['network'].keys() == self.network.state_dict().keys():
            self.network.load_state_dict(pretrained_model['network'])
        else:
            self.logger.info(
                '==> WARNING: network parameters in pre-trained file %s do not strictly match' % (pretrained_path))
            for pname, param in self.network.named_parameters():
                if pname in pretrained_model['network']:
                    self.logger.info('==> Copying parameter %s from file %s' % (pname, pretrained_path))
                    param.data.copy_(pretrained_model['network'][pname])

    def remove_module_prefix(self, checkpoint):
        self.logger.info('==> Clean up checkpoint state_dict for loading')
        state_dict = checkpoint['network']
        pretrained_key_name = list(state_dict.items())[0][0]
        if 'module' in pretrained_key_name:
            self.logger.info("==> Removing module prefix from checkpoint")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint['network'] = new_state_dict
            return checkpoint
        else:
            return checkpoint

    def init_all_optimizers(self):
        self.optimizers = {}

        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams != None:
                self.optimizers[key] = self.init_optimizer(
                    self.networks[key], oparams, key)

    # def init_optimizer(self, net, optim_opts, key):
    #     optim_type = optim_opts['optim_type']
    #     learning_rate = optim_opts['lr']
    #     optimizer = None
    #     parameters = filter(lambda p: p.requires_grad, net.parameters())
    #     self.logger.info('Initialize optimizer: %s with params: %s for netwotk: %s'
    #                      % (optim_type, optim_opts, key))
    #     if optim_type == 'adam':
    #         optimizer = torch.optim.Adam(parameters, lr=learning_rate,
    #                                      betas=optim_opts['beta'])
    #     elif optim_type == 'sgd':
    #         optimizer = torch.optim.SGD(parameters, lr=learning_rate,
    #                                     momentum=optim_opts['momentum'],
    #                                     nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,
    #                                     weight_decay=optim_opts['weight_decay'])
    #     else:
    #         raise ValueError('Not supported or recognized optim_type', optim_type)
    #
    #     return optimizer

    def init_all_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val['opt'] if ('opt' in val) else None
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            self.criterions[key] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        return getattr(nn, ctype)(copt)

    def load_to_gpu(self):
        self.network = torch.nn.DataParallel(self.network).cuda()
        self.cuda = True

        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.cuda()

    def evaluate(self, dloader):
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))

        self.dloader = dloader
        self.dataset_eval = dloader.dataset
        self.logger.info('==> Dataset: %s [%d images]' % (dloader.dataset.name, len(dloader)))
        self.network.eval()

        eval_stats = utils.DAverageMeter()
        self.bnumber = len(dloader())
        for idx, batch in enumerate(tqdm(dloader())):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch)
            eval_stats.update(eval_stats_this)

        self.logger.info('==> Results: %s' % eval_stats.average())
        print('==> Results: %s' % eval_stats.average())
        return eval_stats.average()

    # FROM HERE ON ARE ABSTRACT FUNCTIONS THAT MUST BE IMPLEMENTED BY THE CLASS
    # THAT INHERITS THE Algorithms CLASS
    def train_step(self, batch):
        """Implements a training step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es)
            * Backward propagation through the networks
            * Apply optimization step(s)
            * Return a dictionary with the computed losses and any other desired
                stats. The key names on the dictionary can be arbitrary.
        """
        pass

    def evaluation_step(self, batch):
        """Implements an evaluation step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es) or any other evaluation metrics.
            * Return a dictionary with the computed losses the evaluation
                metrics for that batch. The key names on the dictionary can be
                arbitrary.
        """
        pass

    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        self.tensors = {}
