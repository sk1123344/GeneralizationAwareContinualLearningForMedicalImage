'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Datasets.py
@Author: sk
@Time: 2023/5/16-14:23
@e-mail: sk1123344@163.com
'''
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
from sklearn.utils import shuffle
import copy
import os
from PIL import Image
import random
import skimage.io
import urllib.request
import pickle
from tqdm import tqdm
import torch.nn as nn


class SessionSampler(Sampler):
    """
    use np.random.seed to reproduce the sampling results of GPM
    """
    def __init__(self, current_data_length_, random_seed_, shuffle_=True):
        super(SessionSampler, self).__init__(None)
        self.random_seed_ = random_seed_
        self.shuffle = shuffle_
        self.r = np.arange(current_data_length_)
        if self.shuffle:
            np.random.seed(random_seed_)
            np.random.shuffle(self.r)
        self.sequence_list = list(self.r)

    def sample_list_update(self):
        if self.shuffle:
            np.random.seed(self.random_seed_)
            np.random.shuffle(self.r)
            self.sequence_list = list(self.r)

    def __iter__(self):
        return iter(self.sequence_list)

    def __len__(self):
        return 1


def image_load_preprocess(image_path_, transform_):
    try:
        image = Image.fromarray(skimage.io.imread(image_path_)).convert('RGB')
    except Exception as e:
        print('wrong path: {}'.format(image_path_))
        image = np.ones((84, 84, 3), dtype=float)
    return transform_(image) if transform_ is not None else image


def l2_distance_calculation(tensor_a_, tensor_b_):
    """
    calculate for each row
    :param tensor_a_: [n, dim]
    :param tensor_b_:
    :return: n
    """
    assert tensor_b_.size() == tensor_a_.size(), print('different size for tensor_a {}, tensor_b {}'.format(tensor_a_.size(), tensor_b_.size()))
    return torch.sqrt(torch.sum((tensor_a_ - tensor_b_) ** 2, dim=1))


def _herding_process(features_, sample_num_, herding_method):
    """

    :param features_:
    :return:
    """
    prototype = torch.mean(features_, dim=0)
    if herding_method == 'l2':
        distance_ = l2_distance_calculation(features_, prototype.repeat((features_.size(0), 1)))
    elif herding_method == 'cos':
        distance_ = nn.CosineSimilarity(dim=1)(features_, prototype.repeat((features_.size(0), 1)))
    else:
        raise ValueError('herding method {}'.format(herding_method))
    return torch.sort(distance_, dim=0, descending=False)[1].cpu().numpy()[: sample_num_]


class InfiniteBatchSampler(Sampler):
    """
    used for replay, which can sample from a small buffer according to current phase's sample num and batch size and need not to reinitialize
    the dataloader every iteration
    """
    def __init__(self, buffer_size, epoch_sample_size, batch_size, drop_last=False):
        super(InfiniteBatchSampler, self).__init__(None)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('batch size = {}'.format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError('drop last = {}'.format(drop_last))
        self.buffer_size = buffer_size
        self.epoch_sample_size = epoch_sample_size
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        total_iter = self.__len__()
        for idx in range(total_iter):
            if idx == total_iter - 1:
                batch_size = self.epoch_sample_size % self.batch_size
            else:
                batch_size = self.batch_size
            # selected = list(np.random.choice(range(self.buffer_size), size=batch_size, replace=False if batch_size <= self.buffer_size else True))
            # print(selected)
            # yield selected
            yield list(np.random.choice(range(self.buffer_size), size=batch_size, replace=False if batch_size <= self.buffer_size else True))

    def __len__(self):
        if self.drop_last:
            return self.epoch_sample_size // self.batch_size
        else:
            return (self.epoch_sample_size + self.batch_size - 1) // self.batch_size


class DILDataset(Dataset):
    def __init__(self, transform_, image_data_, label_data_, load_to_ram=False, return_origin_data=False):
        self.image = image_data_
        self.label = label_data_
        self.transform_ = transform_
        self.load_to_ram = load_to_ram
        self.return_origin_data = return_origin_data

    def get_cls_sample_num(self):
        label_num_list = []
        for label_ in torch.unique(self.label):
            label_num_list.append(torch.sum(self.label == label_))
        return label_num_list

    def duplicate_data(self, minimum_num=512):
        image_list = []
        label_list = []
        for label_ in torch.unique(self.label):
            selection = list(np.atleast_1d(torch.nonzero(self.label == label_).squeeze().numpy()))
            if len(selection) > minimum_num:
                image_list.extend([self.image[x] for x in selection])
                label_list.append(self.label[selection])
                continue
            cur_len = len(selection)
            cur_multiplier = 1
            while cur_len <= minimum_num:
                cur_multiplier *= 2
                cur_len *= 2
            image_list.extend([self.image[x] for x in selection] * cur_multiplier)
            label_list.append(self.label[selection].repeat(cur_multiplier))
        self.image = image_list
        self.label = torch.cat(label_list, dim=0)

    def data_replace(self, image_, label_):
        self.image = image_
        self.label = label_

    def data_add(self, image_, label_):
        self.image.extend(image_)
        self.label = torch.cat([self.label, label_], dim=0)

    def __getitem__(self, item):
        if self.return_origin_data:
            return self.transform_(self.image[item]) if self.load_to_ram else image_load_preprocess(self.image[item], self.transform_),  self.label[item], self.image[item]
        else:
            return self.transform_(self.image[item]) if self.load_to_ram else image_load_preprocess(self.image[item], self.transform_), self.label[item]

    def __len__(self):
        return self.label.size(0)


class BaseDataset:
    def __init__(self, train_transform_, test_transform_, load_to_ram_=False, validation_percentage=0, random_seed=4):
        """
        data_dict{'train': {'phase_num': {'x': np.array, 'y': torch.tensor}, ...}, ...}
        up_to_now/joint_data{'train': {'x': np.array, 'y': torch.tensor}, ...}
        :param train_transform_:
        :param test_transform_:
        :param load_to_ram_:
        :param validation_percentage:
        :param random_seed:
        """
        self.current_phase = 0
        self.use_validation = True if 0 < validation_percentage < 1 else False
        assert 0 <= validation_percentage < 1, print('invalid validation percentage {}'.format(validation_percentage))
        self.validation_percentage_ = validation_percentage
        self.random_seed = random_seed
        self._set_random_seed()
        self.transform_dict = {
            'train': train_transform_,
            'val': test_transform_,
            'test': test_transform_
        }
        self.task_num = None
        self.total_nc = None
        self.data_dict = {}
        self.load_to_ram = load_to_ram_
        self.memory_dict = {'x': [], 'y': [], 'total_capacity': 0, 'current_use': 0, 'current_use_for_each_class': {}}

    def total_phase_obtain(self):
        return self.task_num

    def _set_random_seed(self):
        seed_ = self.random_seed
        random.seed(seed_)
        np.random.seed(seed_)
        torch.manual_seed(seed_)
        torch.cuda.manual_seed(seed_)
        torch.cuda.manual_seed_all(seed_)

    def data_dict_obtain_(self):
        raise NotImplementedError('data dict not implemented')

    def memory_update(self):
        raise NotImplementedError('memory update not implemented')

    def check_if_use_validation(self):
        return self.use_validation

    def get_dataset(self, s_name_, task_i):
        """

        :param s_name_:
        :param task_i:
        :return:
        """
        raise NotImplementedError

    def reset_random_seed(self, seed_):
        self.random_seed = seed_


class FundusDILDataset(BaseDataset):
    def __init__(self, dir_list_, disease_list_, resize_transform_, **kwargs):
        super(FundusDILDataset, self).__init__(**kwargs)
        self.dir_list = dir_list_
        self.disease_list = disease_list_
        self.task_num = len(dir_list_)
        self.total_nc = len(disease_list_)
        self.resize_transform = resize_transform_
        self.disease_to_label_dict, self.label_to_disease_dict = self._get_disease_to_label_mapping_dict()
        self.data_dict = self.data_dict_obtain_()

    def _get_disease_to_label_mapping_dict(self):
        disease_to_label_dict = {}
        label_to_disease_dict = {}
        for idx, disease in enumerate(self.disease_list):
            disease_to_label_dict[disease] = idx
            label_to_disease_dict[idx] = disease
        return disease_to_label_dict, label_to_disease_dict

    def get_dataset(self, s_name_, task_i, transform_name_='train'):
        if s_name_ == 'train':
            return DILDataset(self.transform_dict[transform_name_], self.data_dict['train'][task_i]['x'], self.data_dict['train'][task_i]['y'], self.load_to_ram)
        if s_name_ in ('val', 'test'):
            return DILDataset(self.transform_dict['test'], self.data_dict[s_name_][task_i]['x'], self.data_dict[s_name_][task_i]['y'], self.load_to_ram)

    def get_replay_mix_dataset(self, s_name, task_i, replay_image_, replay_label_, transform_name_='train'):
        if s_name == 'train':
            return DILDataset(self.transform_dict[transform_name_], replay_image_ + self.data_dict[s_name][task_i]['x'], torch.cat([replay_label_, self.data_dict[s_name][task_i]['y']], dim=0), self.load_to_ram)

    def data_dict_obtain_(self):
        data_dict = {'train': {}, 'test': {}, 'val': {}}
        use_validation_flag = 1
        for phase, set_dir in enumerate(self.dir_list):
            for set_name in data_dict.keys():
                if phase not in data_dict[set_name].keys():
                    data_dict[set_name][phase] = {'x': [], 'y': []}
                for label_name in self.disease_list:
                    if set_name in os.listdir(set_dir):
                        data_path = os.path.join(set_dir, set_name, label_name)
                        label_id = self.disease_to_label_dict[label_name]
                        for image_name in sorted(os.listdir(data_path)):
                            image_path = os.path.join(data_path, image_name)
                            data_dict[set_name][phase]['y'].append(torch.tensor([label_id], dtype=torch.long))
                            if self.load_to_ram:
                                if len(data_dict[set_name][phase]['x']) == 0:
                                    data_dict[set_name][phase]['x'] = np.expand_dims(image_load_preprocess(image_path, self.resize_transform), axis=0)
                                else:
                                    data_dict[set_name][phase]['x'] = np.concatenate([data_dict[set_name][phase]['x'], np.expand_dims(image_load_preprocess(image_path, self.resize_transform), axis=0)], axis=0)
                            else:
                                data_dict[set_name][phase]['x'].append(image_path)
                    elif set_name in ('train', 'test'):
                        raise AssertionError('no {} under {}'.format(set_name, set_dir))
                    else:
                        use_validation_flag = 0
                data_dict[set_name][phase]['y'] = torch.cat(data_dict[set_name][phase]['y'], dim=0)
        self.use_validation = True if use_validation_flag == 1 else False
        return data_dict
