# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import torch
import torch.nn as nn
import logging

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)


class LabelSmoothingLoss(nn.Module):
    """
    Provides Label-Smoothing loss.

    Args:
        vocab_size (int): the number of classfication
        ignore_index (int): Indexes that are ignored when calculating loss
        smoothing (float): ratio of smoothing (confidence = 1.0 - smoothing)
        dim (int): dimention of calculation loss
        logit (torch.Tensor): probability distribution value from model and it has a logarithm shape
        target (torch.Tensor): ground-thruth encoded to integers which directly point a word in label

    Returns: label_smoothed
        - **label_smoothed** (float): sum of loss

    Reference:
        https://github.com/pytorch/pytorch/issues/7455
    """

    def __init__(self, vocab_size, ignore_index, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, logit, target):
        with torch.no_grad():
            label_smoothed = torch.zeros_like(logit)
            label_smoothed.fill_(self.smoothing / (self.vocab_size - 1))
            label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
            label_smoothed[target == self.ignore_index, :] = 0
        return torch.sum(-label_smoothed * logit)
