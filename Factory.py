'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Factory.py
@Author: sk
@Time: 2023/5/16-14:31
@e-mail: sk1123344@163.com
'''
import torch
from torch import nn
from torch import optim

from Model import ResNet18Fundus, ResNet18DecoderFundus, ResNet18FundusHF, ResNet18DecoderFundus1


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_optimizer(model, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        params = add_weight_decay(model, weight_decay)
        return optim.Adam(params, lr=lr, weight_decay=0)
    elif optimizer == 'adamw':
        params = [
            {'params': [p for n, p in model.backbone.named_parameters() if p.requires_grad]},
            {'params': [p for n, p in model.fc.named_parameters() if p.requires_grad], 'weight_decay': 0}
        ]
        return optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    elif optimizer == "sgd":
        params = [
            {'params': [p for n, p in model.backbone.named_parameters() if p.requires_grad]},
            {'params': [p for n, p in model.fc.named_parameters() if p.requires_grad], 'weight_decay': 0}
        ]
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0)
    else:
        raise NotImplementedError


def get_network(architecture, num_cls, cfg):
    model_name = cfg['model']
    if model_name in ('FT', 'JT', 'Replay', 'CCAS', 'DCAS', 'UDCAS', 'PL', 'EWC', 'MAS', 'MixReplay', 'GPM'):
        return ResNet18Fundus(num_cls, use_pretrained_backbone=cfg['pretrain_model'])
    if model_name in ('FT_cos', 'JT_cos', 'FT_arc', 'JT_arc', 'FT_ldam', 'FT_coslabel'):
        return ResNet18Fundus(num_cls, bias=False, use_pretrained_backbone=cfg['pretrain_model'])
    if model_name in ('FT_decoder', 'JT_decoder', 'FT_cos_decoder', 'JT_cos_decoder', 'FT_ldam_decoder', 'DCAS_ldam_decoder', 'DCAS_cos_decoder', 'CDCAS_cos_decoder', 'FT_coslabel_decoder', 'CDCAS_coslabel_decoder'):
        return ResNet18DecoderFundus(num_cls, bias=False, use_pretrained_backbone=cfg['pretrain_model'])
    if model_name in ('FT_HF',):
        return ResNet18FundusHF(num_cls, bias=False, use_pretrained_backbone=cfg['pretrain_model'])
    else:
        raise NotImplementedError("Unknown architecture {}.".format(architecture))
