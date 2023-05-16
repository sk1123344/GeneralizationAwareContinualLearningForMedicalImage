'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Method.py
@Author: sk
@Time: 2023/5/16-14:33
@e-mail: sk1123344@163.com
'''
from Baseline import BaselineIncModel
from Replay import ReplayIncModel
from BaselineCCAS import DomainAlignmentIncModel
from GeneralizationAwareContinualLearning import CosDistributionDomainAlignmentCosLabelDecoderIncModel


def model_selection(model_name):
    if model_name in ('FT', 'FT_HF', 'JT'):
        return BaselineIncModel
    if model_name in ('Replay',):
        return ReplayIncModel
    if model_name in ('CCAS',):
        return DomainAlignmentIncModel
    if model_name in ('CDCAS_coslabel_decoder',):
        return CosDistributionDomainAlignmentCosLabelDecoderIncModel
    return NotImplementedError(model_name)
