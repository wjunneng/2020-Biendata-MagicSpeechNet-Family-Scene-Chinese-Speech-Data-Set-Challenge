# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import pandas as pd
import pickle
import Levenshtein as Lev

from listen_attend_spell.package import args
import logging

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)


def char_distance(target, y_hat):
    """
    Calculating charater distance between target & y_hat

    Args:
        target: sequence of target
        y_hat: sequence of y_Hat

    Returns: distance, length
        - **distance**: distance between target & y_hat
        - **length**: length of target sequence
    """
    target = target.replace(' ', '')
    y_hat = y_hat.replace(' ', '')
    distance = Lev.distance(y_hat, target)
    length = len(target.replace(' ', ''))

    return distance, length


def get_distance(targets, y_hats, id2char, eos_id):
    """
    Provides total character distance between targets & y_hats

    Args:
        targets (torch.Tensor): set of ground truth
        y_hats (torch.Tensor): predicted y values (y_hat) by the model
        id2char (dict): id2char[id] = ch
        eos_id (int): identification of <end of sequence>

    Returns: total_distance, total_length
        - **total_distance**: total distance between targets & y_hats
        - **total_length**: total length of targets sequence
    """
    total_distance = 0
    total_length = 0

    for i in range(len(targets)):
        target = label_to_string(targets[i], id2char, eos_id)
        y_hat = label_to_string(y_hats[i], id2char, eos_id)
        distance, length = char_distance(target, y_hat)
        total_distance += distance
        total_length += length
    return total_distance, total_length


def get_label(tokens, sos_id, eos_id):
    """
    Provides specific file`s label to list format.

    Args:
        tokens (str): specific path of label file
        bos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>

    Returns: label
        - **label** (list): list of bos + sequence of label + eos
    """
    label = list()
    label.append(int(sos_id))
    for token in tokens:
        label.append(int(token))
    label.append(int(eos_id))
    return label


def label_to_string(labels, id2char, eos_id):
    """
    Converts label to string (number => Hangeul)

    Args:
        labels (list): number label
        id2char (dict): id2char[id] = ch
        eos_id (int): identification of <end of sequence>

    Returns: sentence
        - **sentence** (str or list): Hangeul representation of labels
    """
    if len(labels.shape) == 1:
        sentence = str()
        for label in labels:
            if label.item() == eos_id:
                break
            sentence += id2char[label.item()]
        return sentence

    elif len(labels.shape) == 2:
        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == eos_id:
                    break
                sentence += id2char[label.item()]
            sentences.append(sentence)
        return sentences


def save_epoch_result(train_result, valid_result):
    """ save result of training (unit : epoch) """
    train_dict, train_loss = train_result
    valid_dict, valid_loss = valid_result
    train_dict["loss"].append(train_loss)
    valid_dict["loss"].append(valid_loss)

    train_df = pd.DataFrame(train_dict)
    valid_df = pd.DataFrame(valid_dict)
    train_df.to_csv(args.data_train_result_path, encoding="utf-8", index=False)
    valid_df.to_csv(args.data_dev_result_path, encoding="utf-8", index=False)


def save_pickle(save_var, savepath, message=""):
    """ save pickle file """
    with open(savepath, "wb") as f:
        pickle.dump(save_var, f)
    logger.info(message)
