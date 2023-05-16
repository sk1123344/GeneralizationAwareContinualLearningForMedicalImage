'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: main.py
@Author: sk
@Time: 2023/5/16-14:35
@e-mail: sk1123344@163.com
'''
import time

import yaml
import os
import os.path as osp
import torch
from Method import model_selection


dataset_path_list = [
        r'/amax/home/shukuang/data/DIL712_preprocessed/ODIR_ALL_512',
        r'/amax/home/shukuang/data/DIL712_preprocessed/RIADD_REFUGE_512',
        r'/amax/home/shukuang/data/DIL712_preprocessed/ISEE_ALL_512',
    ]


def _cfg_auto_adjust(_cfg, epochs_, tasks_, load_sub_exp_ckpt_, sample_ratio_, base_name_, dataset_name_, dataset_order_):
    if dataset_name_ == 'fundus':
        base_dir_ = [dataset_path_list[x] for x in dataset_order_]
        base_name_ += '_'
        base_name_ += '-'.join([osp.basename(x).split('_')[0] for x in base_dir_])
        _cfg['base_dir'] = base_dir_[:tasks_]
    if dataset_name_ == 'cataract':
        _cfg['base_dir'] = [
            r'/amax/home/shukuang/data/DIL712_preprocessed/cataract/0',
            r'/amax/home/shukuang/data/DIL712_preprocessed/cataract/2'
        ]
        _cfg['tasks'] = 2
        tasks_ = 2
        _cfg['diseases'] = ['0', '1', '2', '3', '4']
    if _cfg['dataset'] in ('fundus', 'cataract'):
        _cfg['tasks'] = tasks_
        _cfg['load_sub_exp_ckpt'] = load_sub_exp_ckpt_
        # _cfg['load_base_ckpt'] = True
        _cfg['load_ckpt'] = False
        if tasks_ == 3:
            _cfg['load_sub_exp_ckpt'] = load_sub_exp_ckpt_
            _cfg['sub_exp_tasks'] = tasks_ - 1
            _cfg['load_base_ckpt'] = not load_sub_exp_ckpt_
            _cfg['load_ckpt'] = False
        if _cfg['model'] in ('Replay', 'CCAS', 'DCAS', 'CDCAS_coslabel_decoder', 'MixReplay'):
            _cfg['sample_ratio'] = sample_ratio_
            _cfg['exp_extra_name'] += 'sample_ratio-{}'.format(_cfg['sample_ratio'])
        if _cfg['model'] in ('DCAS', 'DCAS_ldam_decoder', 'DCAS_cos_decoder', 'CDCAS_cos_decoder'):
            _cfg['exp_extra_name'] += 'AutoMargin-{}_DistributionSampling-{}_CSA-{}_SamplingAdjust-{}_ProtoAug-{}_ReplayAlign-{}_BaseDistribution-{}'.format(
                '1' if _cfg['dcas_margin'] else '0', '1' if _cfg['dcas_sampling'] else '0', _cfg['ccas_lambda'], '1' if _cfg['dcas_sample_adjust'] else '0',
                '1' if _cfg['dcas_aug'] else '0', '1' if _cfg['dcas_replay_sample_alignment'] else '0', '1' if _cfg['dcas_only_base'] else '0')
        if _cfg['model'] in ('UDCAS',):
            _cfg['exp_extra_name'] += 'PseudoLabel-Type-{}-Threshold-{}_DCSA-{}'.format(
                _cfg['pseudo_label_type'], _cfg['pseudo_label_determine_threshold'], _cfg['ccas_lambda'])
        if _cfg['model'] in ('PL',):
            _cfg['exp_extra_name'] += 'PseudoLabel-Type-{}'.format(_cfg['pseudo_label_type'])
        _cfg['epochs'] = epochs_
        _cfg['base_name'] = base_name_
    return _cfg


if __name__ == '__main__':

    dataset_order = [0, 1, 2]
    debug = False
    tasks = 3
    load_sub_exp_ckpt = False
    sample_ratio = 0.1
    epochs = 120
    device = 1
    # experiment fundus 3 tasks or cataract
    dataset_name = 'fundus'
    base_name = 'experiment{}'.format('-cataract' if dataset_name == 'cataract' else 'fundus')
    run_list = ['cfg_cdcas_coslabel_decoder.yaml']
    cfg_list = [yaml.load(open(osp.join(os.getcwd(), x), 'r'), Loader=yaml.FullLoader) for x in run_list]
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(device)
    torch.backends.cudnn.benchmark = True
    for cfg in cfg_list:
        cfg['debug'] = debug
        cfg = _cfg_auto_adjust(cfg, epochs, tasks, load_sub_exp_ckpt, sample_ratio, base_name, dataset_name, dataset_order)
        print(cfg)
        if cfg['debug']:
            cfg['epochs'] = 1
            cfg['early_stop_epochs'] = 1
        model = model_selection(cfg['model'])(cfg)
        total_task = cfg['tasks']
        model.process_all(total_task)



