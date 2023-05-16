'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Metrics.py
@Author: sk
@Time: 2023/5/16-14:22
@e-mail: sk1123344@163.com
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp
from sklearn.metrics import classification_report


def _to_array(data_):
    if isinstance(data_, torch.Tensor):
        return data_.clone().detach().cpu().numpy()
    if isinstance(data_, np.ndarray):
        return data_
    raise TypeError(type(data_))


def _to_tensor(data_):
    if isinstance(data_, torch.Tensor):
        return data_.clone().detach().cpu()
    if isinstance(data_, np.ndarray):
        return torch.tensor(data_, dtype=torch.float)
    raise TypeError(type(data_))


class CLMeter:
    def __init__(self, cfg):
        self.metric_dict = {
            'AVG': 0,
            'BWT': 0,
            'FWT': 0,
            'Plasticity': 0
        }
        self.cfg = cfg
        self.macro_f1_matrix = np.zeros((self.cfg['tasks'], self.cfg['tasks']))

    def reset(self):
        self.metric_dict = {
            'AVG': 0,
            'BWT': 0,
            'FWT': 0,
            'Plasticity': 0
        }
        self.macro_f1_matrix = np.zeros((self.cfg['tasks'], self.cfg['tasks']))

    def add(self, i, j, macro_f1):
        self.macro_f1_matrix[i, j] = macro_f1

    def _bwt(self):
        self.metric_dict['BWT'] = np.mean(self.macro_f1_matrix[-1, :-1] - np.diag(self.macro_f1_matrix)[:-1])

    def _fwt(self):
        self.metric_dict['FWT'] = np.mean(np.diag(self.macro_f1_matrix)[1:] - self.macro_f1_matrix[0, 1:])

    def _avg(self):
        self.metric_dict['AVG'] = np.mean(self.macro_f1_matrix[-1, :])

    def _plasticity(self):
        self.metric_dict['Plasticity'] = np.mean(np.diag(self.macro_f1_matrix))

    def evaluation(self):
        self._fwt()
        self._bwt()
        self._avg()
        self._plasticity()


class F1Meter:
    def __init__(self, cfg):
        self.pred = None
        self.ytrue = None
        self.cfg = cfg
        self.metric_dict = {
            'macro_F1': 0
        }
        self.CONSTANT = 1E-8
        self.reset()

    def reset(self):
        self.pred = None
        self.ytrue = None
        self.metric_dict = {
            'macro_F1': 0
            # 'AP': [],
            # 'mAP': 0,
            # 'OP': 0,
            # 'OR': 0,
            # 'OF1': 0,
            # 'CP': 0,
            # 'CR': 0,
            # 'CF1': 0,
            # 'CR_dict': {},
            # 'CP_dict': {},
            # 'CF1_dict': {}
        }

    def add(self, pred, ytrue):
        if isinstance(pred, torch.Tensor):
            pred = _to_array(pred)
            ytrue = _to_array(ytrue)
        if self.pred is None:
            self.pred = pred
            self.ytrue = ytrue
        else:
            self.pred = np.concatenate([self.pred, pred], axis=0)
            self.ytrue = np.concatenate([self.ytrue, ytrue], axis=0)

    # def _mAP(self):
    #     pred = _to_tensor(self.pred)
    #     ytrue = _to_tensor(self.ytrue)
    #     n_classes = pred.shape[1]
    #     for k in range(n_classes):
    #         sorted_, indices = torch.sort(pred[:, k], dim=0, descending=True)
    #
    #         label_ = ytrue[indices, k].squeeze()
    #         ones = torch.ones(label_.size())
    #         if sum(label_ == 1) == 0:
    #             precision_at_k = 0
    #         else:
    #             precision_at_k = (torch.cumsum(label_, dim=0) / torch.cumsum(ones, dim=0))[label_ == 1].mean()
    #         self.metric_dict['AP'].append(precision_at_k)
    #     self.metric_dict['mAP'] = np.mean(self.metric_dict['AP'])

    def _f1(self):
        phase_metric = classification_report(y_true=self.ytrue, y_pred=self.pred, target_names=self.cfg['diseases'], zero_division=0, output_dict=True)
        macro_f1 = phase_metric['macro avg']['f1-score']
        self.metric_dict['macro_F1'] = macro_f1

        # n, n_class = self.pred.shape
        # Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        # for k in range(n_class):
        #     k_str = str(k)
        #     score_ = self.pred[:, k]
        #     target_ = self.ytrue[:, k]
        #     Ng[k] = np.sum(target_ == 1)
        #     Np[k] = np.sum(score_ >= 0)
        #     Nc[k] = np.sum(target_ * (score_ >= 0))
        #     self.metric_dict['CR_dict'][k_str] = Nc[k] / (Ng[k] + self.CONSTANT)
        #     self.metric_dict['CP_dict'][k_str] = Nc[k] / (Np[k] + self.CONSTANT)
        #     self.metric_dict['CF1_dict'][k_str] = (2 * self.metric_dict['CR_dict'][k_str] * self.metric_dict['CP_dict'][k_str]) / (self.metric_dict['CR_dict'][k_str] + self.metric_dict['CP_dict'][k_str] + self.CONSTANT)
        # Np[Np == 0] = 1
        # Ng[Ng == 0] = 1e-8
        # OP = np.sum(Nc) / (np.sum(Np) + self.CONSTANT)
        # OR = np.sum(Nc) / (np.sum(Ng) + self.CONSTANT)
        # OF1 = (2 * OP * OR) / (OP + OR + self.CONSTANT)
        #
        # CP = np.sum(Nc / Np) / n_class
        # CR = np.sum(Nc / Ng) / n_class
        # CF1 = (2 * CP * CR) / (CP + CR + self.CONSTANT)
        # self.metric_dict['OP'] = OP
        # self.metric_dict['OR'] = OR
        # self.metric_dict['OF1'] = OF1
        # self.metric_dict['CP'] = CP
        # self.metric_dict['CR'] = CR
        # self.metric_dict['CF1'] = CF1

    def evaluation(self):
        self._f1()