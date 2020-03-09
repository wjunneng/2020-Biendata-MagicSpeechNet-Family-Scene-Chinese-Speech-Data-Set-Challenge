import pandas as pd
import pickle
import Levenshtein as Lev
from listen_attend_spell.package.definition import *


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


def get_label(filepath, sos_id, eos_id, target_dict=None):
    """
    Provides specific file`s label to list format.

    Args:
        filepath (str): specific path of label file
        bos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        target_dict (dict): dictionary of filename and labels

    Returns: label
        - **label** (list): list of bos + sequence of label + eos
    """
    assert target_dict is not None, "target_dict is None"
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')

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
    train_dict, train_loss, train_cer = train_result
    valid_dict, valid_loss, valid_cer = valid_result
    train_dict["loss"].append(train_loss)
    train_dict["cer"].append(train_cer)
    valid_dict["loss"].append(valid_loss)
    valid_dict["cer"].append(valid_cer)

    train_df = pd.DataFrame(train_dict)
    valid_df = pd.DataFrame(valid_dict)
    train_df.to_csv(TRAIN_RESULT_PATH, encoding="cp949", index=False)
    valid_df.to_csv(VALID_RESULT_PATH, encoding="cp949", index=False)


def save_step_result(train_step_result, loss, cer):
    """ save result of training (unit : K time step) """
    train_step_result["loss"].append(loss)
    train_step_result["cer"].append(cer)
    train_step_df = pd.DataFrame(train_step_result)
    train_step_df.to_csv(TRAIN_STEP_RESULT_PATH, encoding="cp949", index=False)


def save_pickle(save_var, savepath, message=""):
    """ save pickle file """
    with open(savepath, "wb") as f:
        pickle.dump(save_var, f)
    logger.info(message)
