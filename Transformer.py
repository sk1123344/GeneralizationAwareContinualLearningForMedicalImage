'''
#！/usr/bin/env python
# -*- coding:utf-8 -*-
@File: Transformer.py
@Author: sk
@Time: 2023/5/16-14:26
@e-mail: sk1123344@163.com
'''
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from typing import Optional
from copy import deepcopy


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", use_self_attn=False):
        super().__init__()
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, query_embedding, spatial_feature):
        """

        :param query_embedding:
        :param spatial_feature: assume been viewed and permuted
        :return:
        """
        # norm + MSA
        norm_query = self.norm1(query_embedding)
        if self.use_self_attn:
            norm_query = self.self_attn(norm_query, norm_query, norm_query)[0]

        # cross attention + residual
        query_embedding1 = self.cross_attn(norm_query, spatial_feature, spatial_feature)[0]
        query_embedding = query_embedding + self.dropout1(query_embedding1)

        # norm + MLP + residual
        query_embedding = self.norm2(query_embedding)
        query_embedding2 = self.linear2(self.dropout(self.activation(self.linear1(query_embedding))))
        query_embedding = query_embedding + self.dropout2(query_embedding2)

        return query_embedding


class TransformerDecoderLayer1(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", use_self_attn=False):
        """
        norm(x+dropout(x)) when in and norm when out
        :param d_model:
        :param nhead:
        :param dim_feedforward:
        :param dropout:
        :param activation:
        :param use_self_attn:
        """
        super().__init__()
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_in = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, query_embedding, spatial_feature):
        """

        :param query_embedding:
        :param spatial_feature: assume been viewed and permuted
        :return:
        """
        # norm + MSA
        norm_query = self.norm1(query_embedding + self.dropout_in(query_embedding))
        if self.use_self_attn:
            norm_query = self.self_attn(norm_query, norm_query, norm_query)[0]

        # cross attention + residual
        query_embedding1 = self.cross_attn(norm_query, spatial_feature, spatial_feature)[0]
        query_embedding = query_embedding + self.dropout1(query_embedding1)

        # norm + MLP + residual
        query_embedding = self.norm2(query_embedding)
        query_embedding2 = self.linear2(self.dropout(self.activation(self.linear1(query_embedding))))
        query_embedding = self.norm_out(query_embedding + self.dropout2(query_embedding2))

        return query_embedding


class TransformerDecoder(nn.Module):
    def __init__(self, num_blocks, d_model, nhead, num_classes, token='task', dim_backbone=512, dim_feedforward=1024, dropout=0.1, activation="relu", use_self_attn=False, block=TransformerDecoderLayer1):
        """

        :param num_blocks:
        :param d_model:
        :param nhead:
        :param token: cls/task
        :param dim_feedforward:
        :param dropout:
        :param activation:
        :param use_self_attn:
        """
        super().__init__()
        self.token_type = token
        self.d_model = d_model
        self.num_classes = num_classes
        if token == 'cls':
            self.token = nn.Parameter(torch.zeros(1, num_classes, d_model))
            self.adaptive_pool = nn.Linear(d_model, 1)
            # self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        elif token == 'task':
            self.token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.adaptive_pool = nn.Linear(d_model, num_classes)
        else:
            raise NotImplementedError(token)
        self.feature_embedding_layer = nn.Linear(dim_backbone, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 16 * 16, d_model))
        self.decoder = nn.ModuleList([block(d_model, nhead, dim_feedforward, dropout, activation, use_self_attn) for i in range(num_blocks)])

    def get_cls_token(self, spatial_feature, cls_index):
        bs, channel, width, height = spatial_feature.size()
        spatial_feature = spatial_feature.view(bs, channel, width * height).permute(0, 2, 1)
        embedded_feature = self.feature_embedding_layer(spatial_feature)
        query = self.token.expand(bs, -1, -1)
        for decoder_layer in self.decoder:
            query = decoder_layer(query, embedded_feature)
        if self.token_type == 'cls':
            if cls_index is None:
                return query
            return query[:, cls_index, :].squeeze(1)
        elif self.token_type == 'task':
            return query[:, 0, :].squeeze(1)

    def forward(self, spatial_feature):
        """

        :param spatial_feature: b c w h
        :return:
        """
        bs, channel, width, height = spatial_feature.size()
        spatial_feature = spatial_feature.view(bs, channel, width * height).permute(0, 2, 1)
        embedded_feature = self.feature_embedding_layer(spatial_feature)
        embedded_feature += self.pos_embedding
        query = self.token.expand(bs, -1, -1)
        for decoder_layer in self.decoder:
            query = decoder_layer(query, embedded_feature)
        out = self.adaptive_pool(query)
        if self.token_type == 'cls':
            out = out.squeeze(-1)
        elif self.token_type == 'task':
            out = out.squeeze(1)
        else:
            raise NotImplementedError
        return out


class TransformerDecoderFundus(nn.Module):
    def __init__(self, num_blocks, d_model, nhead, num_classes, token='task', dim_backbone=512, dim_feedforward=1024, dropout=0.1, activation="relu", use_self_attn=False, block=TransformerDecoderLayer1):
        """

        :param num_blocks:
        :param d_model:
        :param nhead:
        :param token: cls/task
        :param dim_feedforward:
        :param dropout:
        :param activation:
        :param use_self_attn:
        """
        super().__init__()
        self.token_type = token
        self.d_model = d_model
        self.num_classes = num_classes
        # if token == 'cls':
        #     self.token = nn.Parameter(torch.zeros(1, num_classes, d_model))
        #     self.adaptive_pool = nn.Linear(d_model, 1)
        # elif token == 'task':
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.adaptive_pool = nn.Linear(d_model, num_classes, bias=False)
        # else:
        #     raise NotImplementedError(token)
        self.feature_embedding_layer = nn.Linear(dim_backbone, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 16 * 16, d_model))
        self.decoder = nn.ModuleList([block(d_model, nhead, dim_feedforward, dropout, activation, use_self_attn) for i in range(num_blocks)])

    def forward(self, spatial_feature):
        """

        :param spatial_feature: b c w h
        :return:
        """
        bs, channel, width, height = spatial_feature.size()
        spatial_feature = spatial_feature.view(bs, channel, width * height).permute(0, 2, 1)
        embedded_feature = self.feature_embedding_layer(spatial_feature)
        embedded_feature += self.pos_embedding
        query = self.token.expand(bs, -1, -1)
        for decoder_layer in self.decoder:
            query = decoder_layer(query, embedded_feature)
        out = self.adaptive_pool(query)
        out = out.squeeze(1)
        return out, query.squeeze(1)


class TransformerDecoderFundus1(nn.Module):
    def __init__(self, num_blocks, d_model, nhead, num_classes, token='task', dim_backbone=512, dim_feedforward=1024, dropout=0.1, activation="relu", use_self_attn=False, block=TransformerDecoderLayer1):
        """
        use cat[feature_embedding, token] as query
        :param num_blocks:
        :param d_model:
        :param nhead:
        :param token: cls/task
        :param dim_feedforward:
        :param dropout:
        :param activation:
        :param use_self_attn:
        """
        super().__init__()
        self.token_type = token
        self.d_model = d_model
        self.num_classes = num_classes
        # if token == 'cls':
        #     self.token = nn.Parameter(torch.zeros(1, num_classes, d_model))
        #     self.adaptive_pool = nn.Linear(d_model, 1)
        # elif token == 'task':
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.adaptive_pool = nn.Linear(d_model, num_classes, bias=False)
        # else:
        #     raise NotImplementedError(token)
        self.feature_embedding_layer = nn.Linear(dim_backbone, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 16 * 16, d_model))
        self.decoder = nn.ModuleList([block(d_model, nhead, dim_feedforward, dropout, activation, use_self_attn) for i in range(num_blocks)])

    def forward(self, spatial_feature):
        """

        :param spatial_feature: b c w h
        :return:
        """
        bs, channel, width, height = spatial_feature.size()
        spatial_feature = spatial_feature.view(bs, channel, width * height).permute(0, 2, 1)
        embedded_feature = self.feature_embedding_layer(spatial_feature)
        embedded_feature += self.pos_embedding
        token = self.token.expand(bs, -1, -1)
        query = torch.cat([embedded_feature, token], dim=1)
        for decoder_layer in self.decoder:
            query = decoder_layer(query, embedded_feature)
        out = self.adaptive_pool(query[:, 0, :])
        out = out.squeeze(1)
        return out, query.squeeze(1)
