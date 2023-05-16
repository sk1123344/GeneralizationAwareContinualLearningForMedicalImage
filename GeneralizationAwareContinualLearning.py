'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: GeneralizationAwareContinualLearning.py
@Author: sk
@Time: 2023/5/16-14:24
@e-mail: sk1123344@163.com
'''
"""
Add Domain alignment
"""

import numpy as np
import shutil
import os
import os.path as osp
from copy import deepcopy
import matplotlib

matplotlib.use('Agg')
from tqdm import tqdm
import random
import torch
from Model import ModelEma
from torch.utils.data import DataLoader
from Metrics import F1Meter, CLMeter
from BaselineCCAS import DomainAlignmentIncModel
from Loss import CosineCSA
import Factory
from utils import distance_matrix_obtain
from torch.distributions.multivariate_normal import MultivariateNormal
from Loss import CosineLabelAwareMarginLoss, cosine_sim


# Constants
EPSILON = 1e-4


# TODO: add prototype/covariance save
class CosDistributionDomainAlignmentCosLabelDecoderIncModel(DomainAlignmentIncModel):
    def __init__(self, cfg):
        super(CosDistributionDomainAlignmentCosLabelDecoderIncModel, self).__init__(cfg)
        self.distribution_buffer = {'prototype': [], 'covariance': [], 'label': [], 'distance': []}
        self.contrastive_difference = {'positive': 0, 'negative': 0}

    def get_prototype(self, task_i):
        if task_i < 0:
            return
        if task_i == 0:
            dataset_ = self._inc_dataset.get_dataset('train', task_i, 'train')
            dataset_.duplicate_data(512)
            data_loader = DataLoader(dataset_,
                                     batch_size=self._cfg['batch_size'], drop_last=False,
                                     num_workers=self._cfg['workers'], shuffle=False)
        else:
            if self._cfg['dcas_only_base']:
                self._logger.info('Use Base Phase Distribution, Skip')
                return
            dataset_ = self._inc_dataset.get_replay_mix_dataset('train', task_i, self.replay_buffer['image'], self.replay_buffer['label'], 'train')
            dataset_.duplicate_data(512)
            data_loader = DataLoader(
                dataset_,
                batch_size=self._cfg['batch_size'], drop_last=False, num_workers=self._cfg['workers'], shuffle=False)
        self._logger.info(f'Current data num = {len(dataset_)} \nGet all features/labels')
        with torch.no_grad():
            self._network.eval()
            features = []
            label_all = []
            tqdm_loader = tqdm(data_loader, desc='Feature obtaining', ncols=75)
            for i, (inputs, targets) in enumerate(tqdm_loader, start=1):
                inputs = inputs.to(self._device)
                features.append(self._network(inputs)[1].to('cpu'))
                label_all.append(targets)
            features = torch.cat(features, dim=0)
            label_all = torch.cat(label_all, dim=0)
            self.distribution_buffer['prototype'] = []
            self.distribution_buffer['covariance'] = []
            self.distribution_buffer['label'] = []
            for label_ in torch.unique(label_all):
                selection = label_all == label_
                # selection = list(np.atleast_1d(torch.nonzero(label_all == label_).squeeze().numpy()))
                self.distribution_buffer['prototype'].append(torch.mean(features[selection, :], dim=0, keepdim=True))
                self.distribution_buffer['covariance'].append(torch.cov(features[selection, :].permute(1, 0)))
                self.distribution_buffer['label'].append(label_)
            self._logger.info('Prototype Num = {}'.format(len(self.distribution_buffer['prototype'])))
            prototype_tensor = torch.cat(self.distribution_buffer['prototype'], dim=0)
            dis_matrix = 1 - cosine_sim(prototype_tensor, prototype_tensor)
            self._logger.info(f'Task {task_i} Class distance matrix: {dis_matrix}')
            row, col = torch.triu_indices(torch.unique(label_all).size(0), torch.unique(label_all).size(0), 1)
            self.distribution_buffer['distance'].append(torch.min(dis_matrix[row, col]))
            self._logger.info('Task {} current distance set to {}'.format(task_i, self.distribution_buffer['distance'][-1]))

    def set_optimizer(self, task_i, lr):
        # TODO: scheduler use patience type
        weight_decay = self._weight_decay
        # self._logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))
        self._optimizer = Factory.get_optimizer(self._network, self._opt_name, lr, weight_decay)
        # self._scheduler = torch.optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=lr, steps_per_epoch=len(self.train_loader), epochs=self._cfg['epochs'], pct_start=0.2)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max', factor=self._lr_decay, patience=self._cfg['patience'], min_lr=self._cfg['min_lr'])
        self.criterion = CosineLabelAwareMarginLoss(self.train_loader.dataset.get_cls_sample_num(), s=self._cfg['loss_scale'], m=self._cfg['loss_margin'])
        # self.train_loader, self.val_loader = train_loader, val_loader

    def _before_task(self, task_i):
        self._logger.info(f"Begin step {task_i}")
        self._logger.debug(self._network)
        if task_i > 0:
            last_task = task_i - 1
            ckpt_p = "{}/{}step{}.ckpt".format(self.ckpt_save_path, 'ema_' if self._cfg['ema_model_use'] else '',
                                               last_task)
            if osp.exists(ckpt_p):
                self._logger.info(f'reload step {last_task} model from {ckpt_p} Consider remove')
                # self._network = torch.load(ckpt_p)
                self._network = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
                self._network.to(self._device)
                self._network.load_state_dict(torch.load(ckpt_p, map_location=self._device))

                # self._ema_network = ModelEma(self.create_new_network_and_load(self._network.state_dict(), task_i),
                #                              self._cfg['ema_decay'])
                self._old_model = Factory.get_network(self._cfg['architecture'], self._num_cls, self._cfg)
                self._old_model.to(self._device)
                self._old_model.load_state_dict(torch.load(ckpt_p, map_location=self._device))
                self._old_model.freeze_all()
            self.get_prototype(task_i - 1)
            self.get_replay_loader(task_i, self._old_model)
        self.set_dataloader(task_i)
        self.set_optimizer(task_i, self._lr)

    def _reset_sample_pairs_accumulation(self):
        self.contrastive_difference = {'positive': 0, 'negative': 0}

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
            #     self._ema_network = ModelEma(self.create_new_network_and_load(self._network.state_dict(), task_i),
            #                                  self._cfg['ema_decay'])
            #     torch.cuda.empty_cache()
            self.train()
            self._reset_sample_pairs_accumulation()
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
        # if self._old_model is not None:
        #     if self._cfg['ccas_train_data'] == 'new':
        #         ce_loss = criterion(outputs[0][1], targets[:int(targets.size(0) / 2)], self._network.fc.adaptive_pool.weight)
        #     else:
        #         ce_loss = criterion(outputs[0][1], targets, self._network.fc.adaptive_pool.weight)
        # else:
        #     ce_loss = criterion(outputs[0][1], targets, self._network.fc.adaptive_pool.weight)
        if self._old_model is not None:
            if self._cfg['dcas_margin']:
                csa = CosineCSA(self.distribution_buffer['distance'][-1])
            else:
                csa = CosineCSA(self._cfg['ccas_margin'])
            if self._cfg['ccas_train_data'] == 'new':
                ce_loss = criterion(outputs[0][1], targets[:int(targets.size(0) / 2)], self._network.fc.adaptive_pool.weight)
            else:
                ce_loss = criterion(outputs[0][1], targets, self._network.fc.adaptive_pool.weight)
            # TODO: sample data can be changed here according to contrastive_pairs, maybe only affect dcas_sampling
            if self._cfg['dcas_sampling']:
                if self._cfg['dcas_sample_adjust'] and self.contrastive_difference['positive'] + int(targets.size(0) / 2) <= self.contrastive_difference['negative']:
                    sample_ = targets[:int(targets.size(0) / 2)].clone().detach()
                else:
                    sample_ = targets[int(targets.size(0) / 2):]
                self.contrastive_difference['positive'] += torch.sum(sample_ == targets[:int(targets.size(0) / 2)]).cpu().numpy()
                self.contrastive_difference['negative'] += torch.sum(sample_ != targets[:int(targets.size(0) / 2)]).cpu().numpy()
                # self._logger.info('Current iter positive pairs = {}, negative pairs = {}'.format(self.contrastive_difference['positive'], self.contrastive_difference['negative']))
                proto_size = sample_.size(0)
                feature_d = self.distribution_buffer['prototype'][0].size(-1)
                proto_aug = MultivariateNormal(torch.cat(self.distribution_buffer['prototype'], dim=0)[sample_],
                                               torch.stack(self.distribution_buffer['covariance'], 0)[sample_] + EPSILON * torch.eye(feature_d, feature_d).repeat(
                                                   proto_size, 1).view(proto_size, feature_d, feature_d)).sample((1,)).squeeze(0).to(self._device)
                csa_loss = csa(outputs[0][1][:int(targets.size(0) / 2)], proto_aug, (targets[:int(targets.size(0) / 2)] == sample_).float())
                if self._cfg['dcas_replay_sample_alignment']:
                    csa_loss += csa(outputs[0][1][int(targets.size(0) / 2):], proto_aug, (targets[int(targets.size(0) / 2):] == sample_).float())
                if self._cfg['dcas_aug']:
                    ce_loss += criterion(proto_aug, sample_, self._network.fc.adaptive_pool.weight)
            else:
                csa_loss = csa(outputs[0][1][:int(targets.size(0) / 2)], outputs[1][1], (targets[:int(targets.size(0) / 2)] == targets[int(targets.size(0) / 2):]).float())
                if self._cfg['dcas_replay_sample_alignment']:
                    csa_loss += csa(outputs[0][1][int(targets.size(0) / 2):], outputs[1][1], (targets[int(targets.size(0) / 2):] == targets[int(targets.size(0) / 2):]).float())
        else:
            ce_loss = criterion(outputs[0][1], targets, self._network.fc.adaptive_pool.weight)
            csa_loss = torch.tensor([0], dtype=torch.float).to(self._device)
        return ce_loss, csa_loss * self._cfg['ccas_lambda'], outputs[0]
