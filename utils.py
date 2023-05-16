'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: utils.py
@Author: sk
@Time: 2023/5/16-14:23
@e-mail: sk1123344@163.com
'''
import functools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import os.path as osp
import platform
import time
import collections
import cv2


def check_and_mkdir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_, exist_ok=True)
    return dir_


def port_use_windows(port_):
    if os.popen('netstat -an | findstr :' + str(port_)).readlines():
        port_is_use = True
    else:
        port_is_use = False
    return port_is_use


def port_use_linux(port_):
    if os.popen('netstat -na | grep :' + str(port_)).readlines():
        port_is_use = True
    else:
        port_is_use = False
    return port_is_use


def select_platform():
    machine = platform.platform().lower()
    if 'windows-' in machine:
        return port_use_windows
    elif 'linux-' in machine:
        return port_use_linux
    else:
        raise NotImplementedError


def find_usable_ports(start_port_):
    port_use_func = select_platform()
    while True:
        if port_use_func(start_port_):
            start_port_ += 1
            continue
        else:
            return start_port_


def get_time_ymdhm():
    return time.strftime("%Y-%m-%d %H.%M", time.localtime())


# @functools.lru_cache()
def make_logger(log_name, save_dir):
    """Set up the logger for saving log file on the disk
    Args:
        cfg: configuration dict

    Return:
        logger: a logger for record essential information
        :param save_dir:
        :param log_name:
    """
    import logging
    import os
    from logging.config import dictConfig
    import time

    # set up logger
    log_file = '{}.log'.format(log_name)
    # if folder not exist,create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file_path = os.path.join(save_dir, log_file)

    logging_config = dict(
        version=1,
        formatters={'f_t': {
            'format': '\n %(asctime)s | %(levelname)s | %(name)s \t %(message)s'
        }},
        handlers={
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'formatter': 'f_t',
                'level': logging.DEBUG
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'f_t',
                'level': logging.DEBUG,
                'filename': log_file_path,
            }
        },
        root={
            'handlers': ['stream_handler', 'file_handler'],
            'level': logging.DEBUG,
            'propagate': False
        },
    )

    # print(log_file_path)
    logging_config['handlers']['file_handler']['filename'] = log_file_path

    # open(log_file_path, 'a+').close()  # Clear the content of logfile
    # get logger from dictConfig
    dictConfig(logging_config)

    logger = logging.getLogger()

    return logger


def dict_save(dict_, name_, path_=None):
    pd.DataFrame(dict_).to_csv(os.path.join(os.getcwd() if path_ is None else path_, '{}.csv'.format(name_)), index=False)


def dict_append(dict_, column_name_, data_):
    if column_name_ not in dict_.keys():
        dict_[column_name_] = [data_] if not isinstance(data_, list) else data_
    else:
        dict_[column_name_].append(data_) if not isinstance(data_, list) else dict_[column_name_].extend(data_)
    return dict_


def dict_reform(dict_):
    for k_ in dict_.keys():
        if not isinstance(dict_[k_], (list, tuple)):
            dict_[k_] = [dict_[k_]]
    return dict_


def acc_matrix_to_dict(acc_matrix_):
    """

    :param acc_matrix_: square
    :return:
    """
    dict_ = collections.OrderedDict()
    for i_ in range(acc_matrix_.shape[1]):
        dict_append(dict_, 'task{}'.format(str(i_)), list(acc_matrix_[:, i_]))
    return dict_


def distance_matrix_obtain(x1, x2):
    """
    assume tensor and same size
    :param x1:
    :param x2:
    :return:
    """
    n_size = x1.size(0)
    d_size = x1.size(1)
    return nn.PairwiseDistance(p=2, eps=1e-10)(x1.repeat(n_size, 1), x2.repeat(1, n_size).view(n_size ** 2, d_size)).view(n_size, n_size)


def distance_matrix_obtain_arbitrary(x1, x2):
    x1_size = x1.size(0)
    x2_size = x2.size(0)
    d_size = x1.size(1)
    return nn.PairwiseDistance(p=2, eps=1e-10)(x1.repeat(x2_size, 1), x2.repeat(1, x1_size).view(x1_size * x2_size, d_size)).view(x2_size, x1_size).permute(1, 0)


def to_onehot(target_, num_cls_):
    return torch.zeros((target_.size(0), num_cls_)).scatter_(1, target_.unsqueeze(1).cpu(), 1)


def get_kernel(kernel_len_=27, sigma_=20):
    return cv2.getGaussianKernel(kernel_len_, sigma_) * cv2.getGaussianKernel(kernel_len_, sigma_).T


def normalize_zero_one(x_):
    return (x_ - torch.min(x_)) / (torch.max(x_) - torch.min(x_))


class MatrixMinimumSelectionCalculation:
# def matrix_minimum_selection_calculation(x_):
    """
    assume col >= row, the number can't in the same col, calculate a minimum selection scheme
    here assume square matrix
    :param x_:
    :return:
    """
    def __init__(self, x_):
        self.x_ = x_
        self.minimum = torch.sum(torch.diag(x_))
        self.d_size = x_.size(0)
        self.selection_mark = torch.zeros((self.d_size, self.d_size), dtype=torch.long)
        self.select_num = torch.tensor([-1] * self.d_size, dtype=torch.long)
        self.select_num_list = []

    def get_results(self):
        """

        :return: CenterLabel -> ClassLabel
        """
        self.recursion(0, 0)
        return self.select_num_list[-1], self.minimum

    def recursion(self, cur_row_, cur_sum_):
        for i in range(self.d_size):
            if self.selection_mark[cur_row_, i] == 1:
                continue
            temp_sum_ = cur_sum_ + self.x_[cur_row_, i]
            if temp_sum_ > self.minimum:
                continue
            else:
                if cur_row_ == self.d_size - 1:
                    self.minimum = temp_sum_
                    self.select_num[cur_row_] = i
                    self.select_num_list.append(self.select_num.clone())
                else:
                    self.selection_mark[:, i] = 1
                    self.select_num[cur_row_] = i
                    self.recursion(cur_row_ + 1, cur_sum_ + self.x_[cur_row_, i])
                    self.selection_mark[:, i] = 0


def _bwt(macro_f1_matrix):
    return np.mean(macro_f1_matrix[-1, :-1] - np.diag(macro_f1_matrix)[:-1])


def _fwt(macro_f1_matrix):
    return np.mean(np.diag(macro_f1_matrix)[1:] - macro_f1_matrix[0, 1:])


def _avg(macro_f1_matrix):
    return np.mean(macro_f1_matrix[-1, :])


def _plasticity(macro_f1_matrix):
    return np.mean(np.diag(macro_f1_matrix))

