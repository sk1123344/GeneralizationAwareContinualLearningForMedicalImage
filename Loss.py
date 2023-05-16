'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Loss.py
@Author: sk
@Time: 2023/5/16-14:31
@e-mail: sk1123344@163.com
'''
import torch
import torch.nn as nn
import math
import numpy as np


class CSA(nn.Module):
    def __init__(self, margin=1.0):
        super(CSA, self).__init__()
        self.margin = torch.tensor(margin, dtype=torch.float)

    def forward(self, x, y, class_eq):
        dist = nn.PairwiseDistance(p=2)(x, y)
        loss = class_eq * dist.pow(2)
        loss += (1 - class_eq) * (self.margin - dist).clamp(min=0).pow(2)
        return loss.mean()


class CosineCSA(nn.Module):
    def __init__(self, margin=0.4):
        super(CosineCSA, self).__init__()
        self.margin = torch.tensor(margin, dtype=torch.float)

    def forward(self, x, y, class_eq):
        cosine = torch.diag(cosine_sim(x, y))
        dist = 1 - cosine
        loss = class_eq * dist.pow(2)
        loss += (1 - class_eq) * (self.margin - dist).clamp(min=0).pow(2)
        return loss.mean()


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.linalg.vector_norm(x1, 2, dim)
    w2 = torch.linalg.vector_norm(x2, 2, dim)
    return ip / torch.outer(w1, w2).clamp(min=eps)


class CosineMarginLoss(nn.Module):
    def __init__(self, s=30, m=0.4, weight=None):
        super(CosineMarginLoss, self).__init__()
        self.s = s
        self.m = m
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets, fc_weights):
        cosine = cosine_sim(inputs, fc_weights)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        outputs = self.s * (cosine - one_hot * self.m)
        return self.ce_loss(outputs, targets)


class CosineLabelAwareMarginLoss(nn.Module):
    def __init__(self, cls_sample_num_list, s=30, m=0.4, weight=None):
        super(CosineLabelAwareMarginLoss, self).__init__()
        self.s = s
        m_list = 1.0 / np.sqrt(np.sqrt(cls_sample_num_list))
        m_list = m_list * (m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets, fc_weights):
        cosine = cosine_sim(inputs, fc_weights)
        one_hot = torch.zeros_like(cosine).to(targets.device)
        self.m_list = self.m_list.to(targets.device)
        # one_hot.scatter_(1, targets.view(-1, 1), self.m_list[targets].view(-1, 1).to(targets.device))
        one_hot.scatter_(1, targets.view(-1, 1), self.m_list[targets].view(-1, 1))
        outputs = self.s * (cosine - one_hot)
        return self.ce_loss(outputs, targets)


class LabelDistributionAwareLoss(nn.Module):
    def __init__(self, cls_sample_num_list, c=10, s=1, weight=None):
        super(LabelDistributionAwareLoss, self).__init__()
        self.c = c
        self.s = s
        m_list = cls_sample_num_list
        m_list = 1.0 / np.sqrt(np.sqrt(m_list)) * self.c
        self.m_list = torch.tensor(m_list, dtype=torch.float)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, targets):
        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, targets.view(-1, 1), self.m_list[targets].view(-1, 1).to(targets.device))
        outputs = self.s * (outputs - one_hot)
        return self.ce_loss(outputs, targets)


class ArcMarginLoss(nn.Module):
    def __init__(self, s=30, m=0.5, weight=None, easy_margin=False):
        super(ArcMarginLoss, self).__init__()
        self.s = s
        self.m = m
        # self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, targets, fc_weights):
        cosine = cosine_sim(inputs, fc_weights)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        outputs = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        outputs *= self.s
        return self.ce_loss(outputs, targets)
