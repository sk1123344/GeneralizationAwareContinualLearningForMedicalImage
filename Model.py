'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Model.py
@Author: sk
@Time: 2023/5/16-14:25
@e-mail: sk1123344@163.com
'''
from torchvision.models import resnet18
import torch
import torch.nn as nn
from torch.nn import functional as F
from Transformer import TransformerDecoderFundus, TransformerDecoderFundus1
from utils import get_kernel, normalize_zero_one


class HighFrequency(nn.Module):
    def __init__(self, kernel_len=7, sigma=10):
        super(HighFrequency, self).__init__()
        kernel = get_kernel(kernel_len, sigma)
        kernel = torch.tensor(kernel, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding = nn.ReplicationPad2d(int(kernel_len / 2))
        # print(kernel.size())

    def forward(self, x):
        x_pad = self.padding(x)
        low_frequency_x = F.conv2d(x_pad, self.weight, groups=x.shape[1])
        return normalize_zero_one(x - low_frequency_x)


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        with torch.no_grad():
            # self.module = deepcopy(model)
            self.module = model
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class IncrementalBaseModel(nn.Module):
    def __init__(self):
        super(IncrementalBaseModel, self).__init__()
        self.backbone = None
        self.fc = None

    # def incremental_fc(self, incremental_num_classes):
    #     weight = self.fc.weight.data
    #     bias = self.fc.bias.data if hasattr(self.fc, 'bias') else None
    #     in_feature, out_feature = self.fc.in_features, self.fc.out_features
    #
    #     self.fc = nn.Linear(in_feature, incremental_num_classes + out_feature, bias=True if bias is not None else False)
    #     self.fc.weight.data[:out_feature] = weight[:out_feature]
    #     if bias is not None:
    #         self.fc.bias.data[:out_feature] = bias[:out_feature]

    def freeze_all(self):
        for param_ in self.backbone.parameters():
            param_.requires_grad = False
        for param_ in self.fc.parameters():
            param_.requires_grad = False
        self.freeze_bn()
        self.backbone.eval()
        self.fc.eval()

    def freeze_bn(self):
        for n, p in self.backbone.named_modules():
            if isinstance(p, nn.BatchNorm2d):
                p.eval()
                p.weight.requires_grad = False
                p.bias.requires_grad = False

    def feature_extract(self, input_):
        return self.backbone(input_)

    def get_feature_length(self):
        return self.fc.in_features


class ResNet18Fundus(IncrementalBaseModel):
    def __init__(self, num_classes=50, bias=True, use_pretrained_backbone=False):
        super(ResNet18Fundus, self).__init__()
        self.backbone = resnet18(num_classes=1000, pretrained=use_pretrained_backbone)
        self.backbone.fc = nn.Identity()
        # self.backbone.avgpool = nn.Identity()
        # self.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes, bias=bias)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x), x


class ResNet18FundusHF(IncrementalBaseModel):
    def __init__(self, num_classes=50, bias=True, use_pretrained_backbone=False):
        super(ResNet18FundusHF, self).__init__()
        self.backbone = resnet18(num_classes=1000, pretrained=use_pretrained_backbone)
        conv_weight = self.backbone.conv1.weight
        hf_conv_weight = torch.zeros_like(conv_weight)
        nn.init.kaiming_normal_(hf_conv_weight.data)
        self.backbone.conv1.weight = nn.Parameter(torch.cat([conv_weight, hf_conv_weight], dim=1), requires_grad=True)
        self.high_frequency = HighFrequency()
        self.backbone.fc = nn.Identity()
        # self.backbone.avgpool = nn.Identity()
        # self.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes, bias=bias)

    def forward(self, x):
        x = torch.cat([x, self.high_frequency(x).detach()], dim=1)
        x = self.backbone(x)
        return self.fc(x), x


class ResNet18DecoderFundus(IncrementalBaseModel):
    def __init__(self, num_classes=50, bias=False, use_pretrained_backbone=False):
        super(ResNet18DecoderFundus, self).__init__()
        self.backbone = resnet18(num_classes=1000, pretrained=use_pretrained_backbone)
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.fc = TransformerDecoderFundus(2, 512, 4, num_classes)
        # self.fc = nn.Linear(512, num_classes, bias=bias)

    def forward(self, x):
        x = self.backbone(x).view(-1, 512, 16, 16)
        return self.fc(x)


class ResNet18DecoderFundus1(IncrementalBaseModel):
    def __init__(self, num_classes=50, bias=False, use_pretrained_backbone=False):
        super(ResNet18DecoderFundus1, self).__init__()
        self.backbone = resnet18(num_classes=1000, pretrained=use_pretrained_backbone)
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.fc = TransformerDecoderFundus(4, 512, 4, num_classes)
        # self.fc = nn.Linear(512, num_classes, bias=bias)

    def forward(self, x):
        x = self.backbone(x).view(-1, 512, 16, 16)
        return self.fc(x)
