'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: BaselineCCAS.py
@Author: sk
@Time: 2023/5/16-14:27
@e-mail: sk1123344@163.com
'''
import numpy as np
import shutil
import os
import os.path as osp
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

import torch

from Model import ModelEma
from Metrics import F1Meter, CLMeter
from Replay import ReplayIncModel
from Loss import CSA


# Constants
EPSILON = 1e-8


class DomainAlignmentIncModel(ReplayIncModel):
    def __init__(self, cfg):
        super(DomainAlignmentIncModel, self).__init__(cfg)
        self.replay_buffer = {'image': [], 'label': None}
        self.replay_loader = None

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
            _loss_ce = 0
            _loss_csa = 0
            _total = 0
            train_F1_meter.reset()
            tqdm_train_loader = tqdm(train_loader, desc='training', ncols=75)
            # if epoch == self._cfg['ema_start_epoch']:
            #     self._ema_network = ModelEma(self.create_new_network_and_load(self._network.state_dict(), task_i), self._cfg['ema_decay'])
            #     torch.cuda.empty_cache()
            self.train()
            for i, (inputs, targets) in enumerate(tqdm_train_loader, start=1):
                if self._cfg['debug'] and i > self._cfg['debug_iter']:
                    break
                if task_i > 0:
                    replay_inputs, replay_targets = self.replay_loader.__iter__().__next__()
                    inputs, targets = torch.cat([inputs, replay_inputs], dim=0), torch.cat([targets, replay_targets], dim=0)
                    self._logger.debug(f'{targets.size()}')
                if self._cfg['ccas_train_data'] == 'new' and task_i != 0:
                    _total += int(targets.size(0) / 2)
                else:
                    _total += targets.size(0)
                self.train()
                if self._old_model is not None:
                    self._old_model.eval()
                self._optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self._cfg['amp_use']):
                    loss_ce, loss_csa, outputs = self._forward_loss(inputs, targets, train_F1_meter)
                    loss = loss_ce + loss_csa
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
                _loss_ce += loss_ce.clone().detach().cpu().numpy() * targets.size(0)
                _loss_csa += loss_csa.clone().detach().cpu().numpy() * targets.size(0)
                if self._cfg['ccas_train_data'] == 'new' and task_i != 0:
                    train_F1_meter.add(torch.argmax(outputs[0].detach(), dim=1), targets[:int(targets.size(0) / 2)])
                else:
                    train_F1_meter.add(torch.argmax(outputs[0].detach(), dim=1), targets)
                # if epoch >= self._cfg['ema_start_epoch']:
                #     self._ema_network.update(self._network)
            train_F1_meter.evaluation()
            self._logger.info(
                "[Task {}/{}, Epoch {}/{}]: loss: {}, ce_loss: {}, csa_loss: {}, Train m_F1:{}".
                format(
                    task_i + 1,
                    self._cfg['tasks'],
                    epoch + 1,
                    self._n_epochs,
                    _loss / _total,
                    _loss_ce / _total,
                    _loss_csa / _total,
                    train_F1_meter.metric_dict['macro_F1'],
                ))
            val_meter = self.validate(task_i, val_loader, ema=self._cfg['ema_model_use'])
            self._scheduler.step(val_meter.metric_dict['macro_F1'])
            if val_meter.metric_dict['macro_F1'] > best_mF1:
                best_mF1 = val_meter.metric_dict['macro_F1']
                best_epoch = epoch
                self._save_model = deepcopy(self._network).to('cpu')
            self._network.to(self._device)
            # self._ema_network.module.to(self._device)
            if nan_flag:
                break
            if self._optimizer.param_groups[0]['lr'] <= self._cfg['min_lr']:
                self._logger.info(f'lr decrease to minimum, end training at epoch {epoch}, best epoch = {best_epoch}')
                break

    def _forward_loss(self, inputs, targets, train_acc_meter):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
        if self._old_model is not None:
            if self._cfg['ccas_train_data'] == 'new':
                outputs = self._network(inputs[:int(targets.size(0) / 2)])
            else:
                outputs = self._network(inputs)
        else:
            outputs = self._network(inputs)
        if self._old_model is not None:
            with torch.no_grad():
                outputs_old = self._old_model(inputs[int(targets.size(0) / 2):])
            out = [outputs, outputs_old]
        else:
            out = [outputs]
        return self._compute_loss(inputs, targets, out)

    def _compute_loss(self, inputs, targets, outputs):
        criterion = self.criterion
        csa = CSA(self._cfg['ccas_margin'])
        if self._old_model is not None:
            if self._cfg['ccas_train_data'] == 'new':
                ce_loss = criterion(outputs[0][0], targets[:int(targets.size(0) / 2)])
            else:
                ce_loss = criterion(outputs[0][0], targets)
        else:
            ce_loss = criterion(outputs[0][0], targets)
        if self._old_model is not None:
            csa_loss = csa(outputs[0][1][:int(targets.size(0) / 2)], outputs[1][1], (targets[:int(targets.size(0) / 2)] == targets[int(targets.size(0) / 2):]).float())
        else:
            csa_loss = torch.tensor([0], dtype=torch.float).to(self._device)
        return ce_loss, csa_loss * self._cfg['ccas_lambda'], outputs[0]

