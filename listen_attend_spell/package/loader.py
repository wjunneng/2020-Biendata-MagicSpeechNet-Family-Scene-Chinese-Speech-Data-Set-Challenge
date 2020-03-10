# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import csv
import pickle
import torch
import threading
import math
import random
import pandas as pd
from tqdm import trange

from listen_attend_spell.package import args

import logging

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)


class MultiLoader(object):
    """
    Multi Data Loader using Threads.

    Args:
        dataset (package.dataset.BaseDataset): object of BaseDataset
        queue (Queue.queue): queue for threading
        batch_size (int): size of batch
        worker_num (int): the number of cpu cores used
    """

    def __init__(self, dataset_list, queue, batch_size, worker_num):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.loader = list()

        for idx in range(self.worker_num):
            self.loader.append(BaseDataLoader(self.dataset_list[idx], self.queue, self.batch_size, idx))

    def start(self):
        for idx in range(self.worker_num):
            self.loader[idx].start()

    def join(self):
        for idx in range(self.worker_num):
            self.loader[idx].join()


class BaseDataLoader(threading.Thread):
    """
    Base Data Loader

    Args:
        dataset (package.dataset.BaseDataset): object of BaseDataset
        queue (Queue.queue): queue for threading
        batch_size (int): size of batch
        thread_id (int): identification of thread
    """

    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()
            for _ in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break
                feat, label = self.dataset.get_item(self.index)
                if feat is not None:
                    items.append((feat, label))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % self.thread_id)


def _collate_fn(batch):
    """ functions that pad to the maximum sequence length """

    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(args.PAD_TOKEN)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    return seqs, targets, seq_lengths, target_lengths


def load_data_list(wav_path, text_path, vocab_dict):
    """
    Provides set of audio path & label path

    Args:
        wav_path (str): csv file with training or test data list
        text_path (None | str): txt file with training or test data list

    Returns: audio_paths, label_paths
        - **audio_paths** (list): set of audio path
        - **label_paths** (list): set of label path
    """
    uttid_audio = pd.read_csv(wav_path, "r", delimiter=" ", encoding="utf-8", header=None)
    uttid_audio = dict(zip(uttid_audio[0], uttid_audio[1]))

    utt_ids, audio_paths, label_texts = [], [], []
    if text_path is not None:
        with open(text_path, encoding='utf-8', mode='r') as file:
            for line in file.readlines():
                line = line.strip('\n').split(' ')
                id = line[0]
                utt_ids.append(id)
                label_texts.append([vocab_dict[i] if i in vocab_dict else vocab_dict['<UNK>'] for i in line[1:]])
                audio_paths.append(uttid_audio[id])

        return utt_ids, audio_paths, label_texts
    else:
        return uttid_audio.keys(), uttid_audio.values()


def load_label(label_path, encoding='utf-8'):
    """
    Provides char2id, id2char

    Args:
        label_path (str): csv file with character labels

    Returns: char2id, id2char
        - **char2id** (dict): char2id[ch] = id
        - **id2char** (dict): id2char[id] = ch
    """
    char2id = dict()
    id2char = dict()
    with open(label_path, mode='r', encoding=encoding) as f:
        labels = csv.reader(f, delimiter=' ')

        for row in labels:
            char2id[row[0]] = int(row[1])
            id2char[int(row[1])] = row[0]

    return char2id, id2char


def load_pickle(filepath, message=""):
    """
    load pickle file

    Args:
        filepath (str): Path to pickle file to load

    Returns: load_result
        -**load_result** : load result of pickle
    """
    with open(filepath, "rb") as f:
        load_result = pickle.load(f)
        logger.info(message)
        return load_result
