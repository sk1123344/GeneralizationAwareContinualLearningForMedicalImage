'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Replay.py
@Author: sk
@Time: 2023/5/16-14:28
@e-mail: sk1123344@163.com
'''
import numpy as np
import shutil
import os
import os.path as osp
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
import Factory
from Model import ModelEma
from Metrics import F1Meter, CLMeter
from Baseline import BaselineIncModel
from Datasets import DILDataset, InfiniteBatchSampler


# Constants
EPSILON = 1e-8


class ReplayIncModel(BaselineIncModel):
    def __init__(self, cfg):
        super(ReplayIncModel, self).__init__(cfg)
        self.replay_buffer = {'image': [], 'label': None}
        self.replay_loader = None

    def sample_selection(self, task_i, data_loader=None, method='random', model_=None):
        replay_image_list = []
        replay_label_list = []
        random.seed(self._cfg['seed'])
        if model_ is not None:
            model_.eval()
        if method == 'random':
            dataset_ = self._inc_dataset.get_dataset('train', task_i=task_i)
            image_list = dataset_.image
            label_tensor = dataset_.label
            sample_num_per_cls = int(label_tensor.size(0) * self._cfg['sample_ratio'] / self._num_cls)
            for cls in range(self._num_cls):
                cls_label_index = np.atleast_1d(torch.nonzero(label_tensor == cls).squeeze().numpy())
                selected_index = random.sample(list(cls_label_index), min(sample_num_per_cls, cls_label_index.shape[0]))
                replay_image_list.extend([image_list[x] for x in selected_index])
                replay_label_list.append(label_tensor[selected_index])
            return replay_image_list, torch.cat(replay_label_list)

    def get_replay_loader(self, task_i, model_):
        # TODO: DO THIS IN BEFORE_TASK
        self._logger.info(f'replay data sampling')
        replay_image, replay_label = self.sample_selection(task_i - 1, method=self._cfg['sample_selection_method'])
        if self.replay_buffer['label'] is None:
            self.replay_buffer['label'] = replay_label
        else:
            self.replay_buffer['label'] = torch.cat([self.replay_buffer['label'], replay_label], dim=0)
        self.replay_buffer['image'].extend(replay_image)
        self._logger.info('current buffer size = {}'.format(self.replay_buffer['label'].size(0)))

        sampler_ = InfiniteBatchSampler(self.replay_buffer['label'].size(0), self._inc_dataset.data_dict['train'][task_i]['y'].size(0), self._cfg['batch_size'], drop_last=True)
        self._logger.debug('task {} sample num = {}, buffer sample = {}, label = {}'.format(task_i, self._inc_dataset.data_dict['train'][task_i]['y'].size(0), len(self.replay_buffer['image']), self.replay_buffer['label'].size(0)))
        self.replay_loader = DataLoader(DILDataset(self._inc_dataset.transform_dict['train'], self.replay_buffer['image'], self.replay_buffer['label'], self._inc_dataset.load_to_ram),
                                        batch_sampler=sampler_, num_workers=self._cfg['workers'])

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

                self._ema_network = ModelEma(self.create_new_network_and_load(self._network.state_dict(), task_i), self._cfg['ema_decay'])
                self._old_model = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
                self._old_model.to(self._device)
                self._old_model.load_state_dict(torch.load(ckpt_p, map_location=self._device))
                self._old_model.freeze_all()
            self.get_replay_loader(task_i, self._old_model)
        self.set_dataloader(task_i)
        self.set_optimizer(task_i, self._lr)

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
            if epoch == self._cfg['ema_start_epoch']:
                self._ema_network = ModelEma(self.create_new_network_and_load(self._network.state_dict(), task_i), self._cfg['ema_decay'])
                torch.cuda.empty_cache()
            self.train()
            for i, (inputs, targets) in enumerate(tqdm_train_loader, start=1):
                if self._cfg['debug'] and i > self._cfg['debug_iter']:
                    break
                if task_i > 0:
                    replay_inputs, replay_targets = self.replay_loader.__iter__().__next__()
                    inputs, targets = torch.cat([inputs, replay_inputs], dim=0), torch.cat([targets, replay_targets], dim=0)
                    self._logger.debug(f'{targets.size()}')
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
                if epoch >= self._cfg['ema_start_epoch']:
                    self._ema_network.update(self._network)
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
            self._ema_network.module.to(self._device)
            if nan_flag:
                break
            if self._optimizer.param_groups[0]['lr'] <= self._cfg['min_lr']:
                self._logger.info(f'lr decrease to minimum, end training at epoch {epoch}, best epoch = {best_epoch}')
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


