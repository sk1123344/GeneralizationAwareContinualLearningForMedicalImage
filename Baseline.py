'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Baseline.py
@Author: sk
@Time: 2023/5/16-14:29
@e-mail: sk1123344@163.com
'''
import copy

import numpy as np
import shutil
import random
import time
from datetime import timedelta
import math
import os
import os.path as osp
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm

import torch
import torch.nn as nn
# from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import Factory
from Model import ModelEma
from Base import IncrementalLearner
from utils import check_and_mkdir, make_logger, find_usable_ports, get_time_ymdhm, dict_save, dict_append
# from metrics import F1Meter, ForgettingMeter
from Metrics import F1Meter, CLMeter
from Datasets import FundusDILDataset


# Constants
EPSILON = 1e-8


class BaselineIncModel(IncrementalLearner):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # self._device = torch.device(cfg['device'])
        self._device = torch.device(0)

        # Optimizer paras
        self._opt_name = cfg["optimizer"]
        self._lr = cfg["lr"]
        self._weight_decay = cfg["weight_decay"]
        self._n_epochs = cfg["epochs"]
        self._lr_decay = cfg["lr_decay"]
        self.disease_list = cfg['diseases']
        self._num_cls = len(self.disease_list)
        self.tasks = self._cfg['tasks']
        self._network = Factory.get_network(cfg['architecture'], self._num_cls, self._cfg)
        self._old_model = None
        self._save_model = None
        # self._ema_network = ModelEma(self.create_new_network_and_load(self._network.state_dict(), 0), self._cfg['ema_decay'])
        # Data
        self._inc_dataset = self._get_inc_dataset()
        base_name = self._cfg['base_name'] if not self._cfg['debug'] else 'experiments_debug'
        # base_name = 'experiments' if not self._cfg['debug'] else 'experiments_debug'
        self.save_path = osp.join(os.getcwd(), base_name, self._get_exp_name())
        if osp.exists(self.save_path) and not self._cfg['load_ckpt'] and not self._cfg['evaluation'] and not self._cfg['load_base_ckpt']:
            # a = input('overlap exist exp? [Y/N]:')
            # if a.lower() != 'y':
            #     exit(0)
            # else:
            print('find existence dir, rm')
            shutil.rmtree(self.save_path, ignore_errors=True)
            check_and_mkdir(self.save_path)
        self.val_save_path = check_and_mkdir(osp.join(self.save_path, 'val_tmp'))
        self.ckpt_save_path = os.path.join(self.save_path, "ckpts")
        if self._cfg["save_ckpt"]:
            check_and_mkdir(self.ckpt_save_path)

        # Logging
        self._logger = make_logger(self._get_exp_name(), self.save_path)
        self._logger.setLevel('DEBUG') if self._cfg['debug'] else self._logger.setLevel('INFO')
        self._tensorboard = None
        self.forgetting = None

        self.all_metric = CLMeter(cfg)
        self.weight_list = []
        self.bias_list = []

        self._logger.info(f'Initialization complete')

    def create_new_network_and_load(self, old_model_state_dict, task_i):
        network = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
        network.to(self._device)
        with torch.no_grad():
            network.load_state_dict(old_model_state_dict)
        return network

    def process_task(self, task_i):
        self.before_task(task_i)
        ckpt_p = "{}/{}step{}.ckpt".format(self.ckpt_save_path, 'ema_' if self._cfg['ema_model_use'] else '', task_i)
        ckpt_base = "{}/{}step{}.ckpt".format(osp.join(osp.split(self.save_path)[0], self._get_base_exp_name(), 'ckpts'), 'ema_' if self._cfg['ema_model_use'] else '', task_i)
        ckpt_sub_exp = "{}/{}step{}.ckpt".format(osp.join(osp.split(self.save_path)[0], self._get_sub_exp_name(self._cfg['sub_exp_tasks']), 'ckpts'), 'ema_' if self._cfg['ema_model_use'] else '', task_i)
        self._logger.info(f'base model path: {ckpt_base}')
        if (self._cfg['load_ckpt'] or self._cfg['evaluation']) and osp.exists(ckpt_p):
            self._logger.info(f'load step {task_i} ckpt model from {ckpt_p}')
            # self._logger.info(f'{torch.load(ckpt_base, map_location=self._device)}')
            self._network = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
            self._network.to(self._device)
            self._network.load_state_dict(torch.load(ckpt_p, map_location=self._device))
        elif task_i == 0 and self._cfg['load_base_ckpt'] and osp.exists(ckpt_base):
            self._logger.info(f'load base ckpt model from {ckpt_base}')
            self._network = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
            self._network.to(self._device)
            self._network.load_state_dict(torch.load(ckpt_base, map_location=self._device))
            # self._ema_network = ModelEma(self.create_new_network_and_load(self._network.state_dict(), task_i), self._cfg['ema_decay'])
        elif self._cfg['load_sub_exp_ckpt'] and osp.exists(ckpt_sub_exp):
            self._logger.info(f'load sub exp {task_i} ckpt model from {ckpt_sub_exp}')
            self._network = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
            self._network.to(self._device)
            self._network.load_state_dict(torch.load(ckpt_sub_exp, map_location=self._device))
        else:
            self.train_task(task_i, self.train_loader, self.val_loader)
        self.after_task(task_i)

    def _get_exp_name(self):
        return 'model-{}_backbone-{}_tasks-{}_data-{}_keypoint-{}'.format(
            self._cfg['model'],
            self._cfg['architecture'],
            self._cfg['tasks'],
            self._cfg['dataset'],
            self._cfg['exp_extra_name']
        )

    def _get_sub_exp_name(self, sub_tasks_):
        return 'model-{}_backbone-{}_tasks-{}_data-{}_keypoint-{}'.format(
            self._cfg['model'],
            self._cfg['architecture'],
            sub_tasks_,
            self._cfg['dataset'],
            self._cfg['exp_extra_name']
        )

    def _get_base_exp_name(self):
        # TODO: FT model path, first phase seems no need to modify (only when adding new module in the network)
        method_name = self._cfg['model']
        if 'ldam' in method_name or 'cos' in method_name or 'decoder' in method_name:
            baseline_name = '_'.join(['FT'] + method_name.split('_')[1:])
        else:
            baseline_name = 'FT'
        return 'model-{}_backbone-{}_tasks-{}_data-{}_keypoint-{}'.format(
            baseline_name,
            self._cfg['architecture'],
            self._cfg['tasks'],
            self._cfg['dataset'],
            self._cfg['base_extra_name']
        )

    def get_loader(self, s_name, task_i, dataset_=None):
        # TODO: USE DILDataset
        if dataset_ is None:
            dataset_ = self._inc_dataset.get_dataset(s_name, task_i)
            if 'JT' in self._cfg['model'] and s_name != 'test':
                for i in range(task_i):
                    old_dataset = self._inc_dataset.get_dataset(s_name, i)
                    dataset_.data_add(old_dataset.image, old_dataset.label)
        bs = self._cfg['batch_size']
        workers = self._cfg['workers']
        return DataLoader(dataset_, batch_size=bs, num_workers=workers, shuffle=True if s_name == 'train' else False, drop_last=True if s_name == 'train' else False)

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    def _before_task(self, task_i):
        # TODO: set loader/optimizer/replay loader
        self._logger.info(f"Begin step {task_i}")
        self._logger.debug(self._network)
        if task_i > 0:
            last_task = task_i - 1
            ckpt_p = "{}/{}step{}.ckpt".format(self.ckpt_save_path, 'ema_' if self._cfg['ema_model_use'] else '', last_task)
            if osp.exists(ckpt_p):
                self._logger.info(f'reload step {last_task} model from {ckpt_p} Consider remove')
                # self._network = torch.load(ckpt_p)
                self._network = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
                self._network.to(self._device)
                self._network.load_state_dict(torch.load(ckpt_p, map_location=self._device))

                # self._ema_network = ModelEma(self.create_new_network_and_load(self._network.state_dict(), task_i), self._cfg['ema_decay'])
                self._old_model = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
                self._old_model.to(self._device)
                self._old_model.load_state_dict(torch.load(ckpt_p, map_location=self._device))
                self._old_model.freeze_all()
        self.set_dataloader(task_i)
        self.set_optimizer(task_i, self._lr)

    def set_dataloader(self, task_i):
        self.train_loader, self.val_loader, self.test_loader = self.get_loader('train', task_i), self.get_loader('val', task_i), self.get_loader('test', task_i)

    def set_optimizer(self, task_i, lr):
        # TODO: scheduler use patience type
        weight_decay = self._weight_decay
        self._optimizer = Factory.get_optimizer(self._network, self._opt_name, lr, weight_decay)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max', factor=self._lr_decay, patience=self._cfg['patience'], min_lr=self._cfg['min_lr'])
        self.criterion = nn.CrossEntropyLoss()

    def display_weight_norm(self, task_i):
        # TODO: change to save the weight/bias and draw on a single figure
        color_list = ['r', 'g', 'b', 'c', 'y']
        save_p = osp.join(self.save_path, 'cls_embedding_norm_plt')
        check_and_mkdir(save_p)
        network = deepcopy(self._network)
        with torch.no_grad():
            if hasattr(network.fc, 'adaptive_pool'):
                weight = network.fc.adaptive_pool.weight
                weight_norm_ = (weight.data ** 2).sum(dim=1).sqrt()
                self.weight_list.append(weight_norm_)
                self._logger.info(f'weight size {weight.data.size()}, weight_norm size {weight_norm_.size()}')
                self._logger.info(f'norm {weight_norm_}')
                if task_i == self.tasks - 1:
                    plt.figure()
                    for i, weight_norm_ in enumerate(self.weight_list):
                        plt.scatter(range(1, len(weight_norm_) + 1), weight_norm_.cpu().numpy(), s=10, c=[color_list[i]] * self._num_cls)
                    plt.savefig(os.path.join(save_p, 'weight_norm_phase.png'))
                    plt.close()

            if hasattr(network.fc, 'weight'):
                weight_norm_ = (network.fc.weight.data ** 2).sum(dim=1).sqrt()
                self.weight_list.append(weight_norm_)
                self._logger.info(f'weight size {network.fc.weight.data.size()}, weight_norm size {weight_norm_.size()}')
                self._logger.info(f'norm {weight_norm_}')
                bias_norm_ = network.fc.bias.data if network.fc.bias is not None else None
                self.bias_list.append(bias_norm_)
                if task_i == self.tasks - 1:
                    plt.figure()
                    for i, weight_norm_ in enumerate(self.weight_list):
                        plt.scatter(range(1, len(weight_norm_) + 1), weight_norm_.cpu().numpy(), s=10, c=[color_list[i]] * self._num_cls)
                    plt.savefig(os.path.join(save_p, 'weight_norm_phase.png'))
                    plt.close()
                    if bias_norm_ is not None:
                        plt.figure()
                        for i, bias_norm_ in enumerate(self.bias_list):
                            plt.scatter(range(1, len(bias_norm_) + 1), bias_norm_.cpu().numpy(), s=10, c=[color_list[i]] * self._num_cls)
                        plt.savefig(os.path.join(save_p, 'bias_norm_phase.png'))
                        plt.close()

    def _train_task(self, task_i, train_loader, val_loader):
        # TODO: train and val
        self._logger.info(f"nb {len(train_loader.dataset)}")
        self._network.to(self._device)
        train_F1_meter = F1Meter(self._cfg)

        self._optimizer.zero_grad()

        scaler = torch.cuda.amp.GradScaler(init_scale=8192.0, enabled=self._cfg['amp_use'])
        best_mF1 = 0
        best_epoch = -1
        nan_flag = False
        for epoch in range(self._n_epochs):
            _loss = 0
            _total = 0
            train_F1_meter.reset()
            tqdm_train_loader = tqdm(train_loader, desc='training', ncols=75)
            self.train()
            for i, (inputs, targets) in enumerate(tqdm_train_loader, start=1):
                if self._cfg['debug'] and i > self._cfg['debug_iter']:
                    break
                _total += targets.size(0)
                self.train()
                self._optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self._cfg['amp_use']):
                    loss, outputs = self._forward_loss(inputs, targets, train_F1_meter)
                if torch.any(torch.isnan(loss)) and not self._cfg['amp_use']:
                    nan_flag = True
                    self._logger.info(f'epoch {epoch + 1} iter {i} produce NAN, break!')
                    break
                # loss = loss_ce
                if self._cfg['amp_use']:
                    scaler.scale(loss).backward()
                    scaler.step(self._optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self._optimizer.step()
                _loss += loss.clone().detach().cpu().numpy() * targets.size(0)
                train_F1_meter.add(torch.argmax(outputs[0].detach(), dim=1), targets)
            train_F1_meter.evaluation()
            self._logger.info(
                "[Task {}/{}, Epoch {}/{}]: Clf loss: {}, Train m_F1:{}".
                format(
                    task_i + 1,
                    self._cfg['tasks'],
                    epoch + 1,
                    self._n_epochs,
                    _loss / _total,
                    train_F1_meter.metric_dict['macro_F1'],
                ))
            val_meter = self.validate(task_i, val_loader, ema=self._cfg['ema_model_use'])
            self._scheduler.step(val_meter.metric_dict['macro_F1'])
            if val_meter.metric_dict['macro_F1'] > best_mF1:
                best_mF1 = val_meter.metric_dict['macro_F1']
                best_epoch = epoch
                self._save_model = deepcopy(self._network).to('cpu')
            self._network.to(self._device)
            if nan_flag:
                break
            if self._optimizer.param_groups[0]['lr'] <= self._cfg['min_lr'] or epoch == self._n_epochs - 1:
                self._logger.info(f'lr decrease to minimum, end training at epoch {epoch + 1}, best epoch = {best_epoch + 1}')
                break

    def _forward_loss(self, inputs, targets, train_acc_meter):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
        outputs = self._network(inputs)
        return self._compute_loss(inputs, targets, outputs)

    def _compute_loss(self, inputs, targets, outputs):
        # TODO: ADD a replay loader sampling
        criterion = self.criterion
        ce_loss = criterion(outputs[0], targets)
        return ce_loss, outputs

    def _metric_to_dict(self, meter_, ema_=False):
        # TODO: CHANGE TO Avg macro_F1/BWT/FWT/
        current_dict = self._ema_task_metric_dict if ema_ else self._task_metric_dict
        for metric_name in ['macro_F1', 'BWT', 'FWT']:
            dict_append(current_dict, metric_name, meter_.metric_dict[metric_name])
        dict_save(current_dict, '{}task_metric'.format('ema_' if ema_ else ''), self.save_path)

    def _after_task(self, task_i):
        # TODO: evaluation on test set
        self.display_weight_norm(task_i)
        self._network.eval()
        # self._ema_network.module.eval()
        self._logger.info("save model")
        if self._cfg["save_ckpt"]:
            torch.save(self._network.state_dict(), "{}/step{}.ckpt".format(self.ckpt_save_path, task_i))

        self.evaluation_all(task_i, self.tasks, ema=False)

    def _eval_task(self, data_loader, ema, cur_s_name='val'):
        ypred, ytrue = self._compute_accuracy_by_netout(data_loader, ema, cur_s_name)
        return ypred, ytrue

    def _compute_accuracy_by_netout(self, data_loader, ema, cur_s_name='val'):
        preds, ytrue = [], []
        if ema:
            network = self._ema_network.module
        else:
            network = self._network
        network.eval()
        network.to(self._device)
        with torch.no_grad():
            data_loader = tqdm(data_loader, ncols=75, desc='{}{}'.format(cur_s_name, '_ema' if ema else ''))
            for i, (inputs, lbls) in enumerate(data_loader):
                if self._cfg['debug'] and i > self._cfg['debug_iter']:
                    break
                inputs = inputs.to(self._device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self._cfg['amp_use']):
                    _preds = network(inputs)
                    if isinstance(_preds, (tuple, list)):
                        _preds = _preds[0]
                _preds = torch.argmax(_preds, dim=1)
                preds.append(_preds.detach().cpu().numpy())
                ytrue.append(lbls.long().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        ytrue = np.concatenate(ytrue, axis=0)
        return preds, ytrue

    def validate(self, task_i, data_loader, ema=False, cur_s_name='val'):
        val_acc_meter = None
        if data_loader is not None:
            ypred, ytrue = self._eval_task(data_loader, ema, cur_s_name)
            val_acc_meter = F1Meter(self._cfg)
            val_acc_meter.add(ypred, ytrue)
            val_acc_meter.evaluation()
            prefix = 'ema' if ema else ''
            self._logger.info(f"{prefix} {cur_s_name} mF1:{val_acc_meter.metric_dict['macro_F1']}")
        return val_acc_meter

    # TODO: add a evaluation for all test set, for loop, all task
    def evaluation_all(self, task_i, tasks_, ema=False):
        for i in range(tasks_):
            cur_dataloader = self.get_loader('test', i)
            meter_ = self.validate(i, cur_dataloader, ema, 'test')
            if ema:
                self.all_metric_ema.add(task_i, i, meter_.metric_dict['macro_F1'])
            else:
                self.all_metric.add(task_i, i, meter_.metric_dict['macro_F1'])


