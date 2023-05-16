'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Base.py
@Author: sk
@Time: 2023/5/16-14:22
@e-mail: sk1123344@163.com
'''
import abc
import logging
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import os.path as osp
from Metrics import F1Meter, CLMeter
import numpy as np
import torchvision.transforms as transforms
from Datasets import FundusDILDataset
from utils import dict_save, dict_append, acc_matrix_to_dict, dict_reform
import torch.distributed as dist

LOGGER = logging.Logger("IncLearn", level="INFO")


class IncrementalLearner(abc.ABC):
    """Base incremental learner.

    Methods are called in this order (& repeated for each new task):

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task
    """
    def __init__(self, *args, **kwargs):
        self._task_metric_dict = {}
        self._ema_task_metric_dict = {}

    def set_task_info(self):
        pass

    def before_task(self, task_i):
        LOGGER.info("Before task")
        self.eval()
        self._before_task(task_i)

    def train_task(self, task_i, train_loader, val_loader):
        LOGGER.info("train task")
        self.train()
        self._train_task(task_i, train_loader, val_loader)

    def after_task(self, task_i):
        LOGGER.info("after task")
        self.eval()
        self._after_task(task_i)

    def eval_task(self, data_loader):
        LOGGER.info("eval task")
        self.eval()
        return self._eval_task(data_loader)

    def get_memory(self):
        raise NotImplementedError

    def forgetting_writer(self):
        raise  NotImplementedError

    def close(self):
        if self._tensorboard is not None:
            self._tensorboard.close()
        if self._logger is not None:
            handlers = self._logger.handlers[:]
            for handler in handlers:
                self._logger.removeHandler(handler)
                handler.close()

    def metric_writer(self):
        self.all_metric.evaluation()
        # self.all_metric_ema.evaluation()
        dict_save(dict_reform(self.all_metric.metric_dict), 'cl_metric', self.save_path)
        # dict_save(dict_reform(self.all_metric_ema.metric_dict), 'cl_metric_ema', self.save_path)
        dict_save(acc_matrix_to_dict(self.all_metric.macro_f1_matrix), 'macro_f1', self.save_path)
        # dict_save(acc_matrix_to_dict(self.all_metric_ema.macro_f1_matrix), 'macro_f1_ema', self.save_path)

    def process_all(self, total_task):
        for i in range(total_task):
            self.process_task(i)
        # self.forgetting_writer()
        # dist.barrier()
        self.metric_writer()
        self.close()

    def process_task(self, task_i):
        raise NotImplementedError
        # self.before_task(task_i)
        # # torch.save(network.module.cpu(), "{}/step{}.ckpt".format(self.ckpt_save_path, self._task))
        # ckpd_p = "{}/step{}.ckpt".format(self.ckpt_save_path, self._task)
        # if (self._cfg['load_ckpt'] or self._cfg['evaluation']) and osp.exists(ckpd_p):
        #     self._logger.info(f'load step {task_i} model')
        #     self._network = torch.load(ckpd_p)
        #     # self._parallel_network = DataParallel(self._network, device_ids=self._cfg['device'])
        #     self._parallel_network = DataParallel(self._network, device_ids=[x for x in range(len(self._cfg['device']))])
        #     self._parallel_network.to(self._device)
        # else:
        #     self.train_task(task_i, self.get_loader('train', task_i), self.get_loader('test', task_i))
        # self.after_task(task_i)

    def _get_inc_dataset(self):
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomResizedCrop(self._cfg['scale_size'], scale=(0.8, 1), ratio=(1, 1)),
             transforms.ColorJitter(brightness=0.2), transforms.RandomHorizontalFlip(p=0.5)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((self._cfg['scale_size'], self._cfg['scale_size']))])
        if self._cfg['dataset'] == 'fundus':
            return FundusDILDataset(self._cfg['base_dir'], self._cfg['diseases'], transforms.Compose([transforms.Resize((self._cfg['scale_size'], self._cfg['scale_size']))]), train_transform_=train_transform, test_transform_=test_transform, load_to_ram_=self._cfg['load_to_ram'], random_seed=self._cfg['seed'])

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def _before_task(self, task_i):
        raise NotImplementedError

    def _train_task(self, task_i, train_loader, val_loader):
        raise NotImplementedError

    def _after_task(self, task_i):
        raise NotImplementedError

    def _eval_task(self, data_loader):
        raise NotImplementedError

    def validate(self, task_i, data_loader, ema=False, cur_s_name='val'):
        raise NotImplementedError
        # ypred, ytrue = self._eval_task(data_loader)
        # val_acc_meter = F1Meter()
        # val_acc_meter.add(ypred, ytrue)
        # val_acc_meter.evaluation()
        # self._logger.info(f"test mAP:{val_acc_meter.metric_dict['mAP']}, CR:{val_acc_meter.metric_dict['CR']}, CP:{val_acc_meter.metric_dict['CP']}, CF1:{val_acc_meter.metric_dict['CF1']}, OR:{val_acc_meter.metric_dict['OR']}, OP:{val_acc_meter.metric_dict['OP']}, OF1:{val_acc_meter.metric_dict['OF1']}")
        # return val_acc_meter
