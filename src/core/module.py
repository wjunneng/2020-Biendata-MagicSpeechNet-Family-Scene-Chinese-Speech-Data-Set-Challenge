# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import math
import torch

import numpy as np
from torch import nn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0

        # 假定d_k == d_v
        self.d_k = n_feat // n_head
        self.n_head = n_head

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_o = nn.Linear(n_feat, n_feat)

        self.atten = None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):
        """
        Compute 'Scaled Dot Product Attention'
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)

        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.n_head, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.n_head, self.d_k)
        v = self.linear_v(key).view(n_batch, -1, self.n_head, self.d_k)

        # 维度转换 (batch, head, time1, d_k)
        q = q.transpose(1, 2)
        # 维度转换 (batch, head, time2, d_k)
        k = k.transpose(1, 2)
        # 维度转换 (batch, head, time2, d_k)
        v = v.transpose(1, 2)

        # (batch, head, time1, time2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # (batch, 1, time1, time2)
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            # (batch, head, time1, time2)
            self.atten = torch.softmax(scores, dim=-1).masked_fill(mask=mask, value=0.0)
        else:
            # # (batch, head, time1, time2)
            self.atten = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.atten)
        # (batch, head, time1, d_k)
        x = torch.matmul(p_attn, v)
        # (batch, time1, d_model)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_feat)

        # (batch, time1, d_model)
        return self.linear_o(x)
