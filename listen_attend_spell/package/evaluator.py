# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import logging
import torch
from listen_attend_spell.package.utils import get_distance

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)


def evaluate(model, queue, criterion, device):
    r"""
    Args:
        model (torch.nn.Module): Model to be evaluated
        queue (queue): queue for threading
        criterion (torch.nn): one of PyTorch’s loss function. Refer to http://pytorch.org/docs/master/nn.html#loss-functions for a list of them.
        device (torch.cuda): device used ('cuda' or 'cpu')

    Returns: loss, cer
        - **loss** (float): loss of evalution
        - **cer** (float): character error rate
    """
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_distance = 0
    total_length = 0
    total_sentence_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)
            target = scripts[:, 1:]

            model.module.flatten_parameters()
            y_hat, logit = model(feats, scripts, teacher_forcing_ratio=0.0, use_beam_search=False)
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            distance, length = get_distance(target, y_hat)
            total_distance += distance
            total_length += length
            total_sentence_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_distance / total_length
