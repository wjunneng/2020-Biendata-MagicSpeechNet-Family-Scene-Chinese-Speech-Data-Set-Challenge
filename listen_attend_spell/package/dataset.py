# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import random
from torch.utils.data import Dataset
from listen_attend_spell.package.feature import spec_augment, get_librosa_melspectrogram
from listen_attend_spell.package.utils import get_label
from listen_attend_spell.package import args

import logging

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)


class BaseDataset(Dataset):
    """
    Dataset for audio & label matching

    Args:
        audio_paths (list): set of audio path
        label_paths (list): set of label paths
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        target_dict (dict): dictionary of filename and labels
        input_reverse (bool): flag indication whether to reverse input feature or not (default: True)
        use_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        augment_ratio (float): ratio of spec-augmentation applied data (default: 1.0)
        pack_by_length (bool): pack by similar sequence length
        batch_size (int): mini batch size
    """

    def __init__(self, audio_paths, label_paths, input_reverse=True, use_augment=True, batch_size=None,
                 augment_ratio=0.3, pack_by_length=True):
        self.audio_paths = list(audio_paths)
        self.label_paths = list(label_paths)
        self.sos_id = args.SOS_TOKEN
        self.eos_id = args.EOS_TOKEN
        self.batch_size = batch_size
        self.input_reverse = input_reverse
        self.augment_ratio = augment_ratio
        self.augment_flags = [False] * len(self.audio_paths)
        self.pack_by_length = pack_by_length
        if use_augment:
            self.augmentation()
        if pack_by_length:
            self.sort_by_length()
            self.audio_paths, self.label_paths, self.augment_flags = self.batch_shuffle(remain_drop=False)
        else:
            bundle = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
            random.shuffle(bundle)
            self.audio_paths, self.label_paths, self.augment_flags = zip(*bundle)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

    def get_item(self, idx):
        label = get_label(self.label_paths[idx], sos_id=self.sos_id, eos_id=self.eos_id)
        feat = get_librosa_melspectrogram(self.audio_paths[idx], n_mels=args.input_dim, mel_type='log_mel',
                                          input_reverse=self.input_reverse)
        # exception handling
        if feat is None:
            return None, None
        if self.augment_flags[idx]:
            feat = spec_augment(feat, T=8, F=20, time_mask_num=2, freq_mask_num=2)
        return feat, label

    def augmentation(self):
        """ Apply Spec-Augmentation """
        augment_end_idx = int(0 + ((len(self.audio_paths) - 0) * self.augment_ratio))
        logger.info("Applying Augmentation...")

        for idx in range(augment_end_idx):
            self.augment_flags.append(True)
            self.audio_paths.append(self.audio_paths[idx])
            self.label_paths.append(self.label_paths[idx])

    def shuffle(self):
        """ Shuffle Dataset """
        if self.pack_by_length:
            self.audio_paths, self.label_paths, self.augment_flags = self.batch_shuffle(remain_drop=False)
        else:
            bundle = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
            random.shuffle(bundle)
            self.audio_paths, self.label_paths, self.augment_flags = zip(*bundle)

    def sort_by_length(self):
        """ descending sort by sequence length """
        target_lengths = list()
        for idx, label_path in enumerate(self.label_paths):
            key = label_path.split('/')[-1].split('.')[0]
            target_lengths.append(len(self.target_dict[key].split()))

        bundle = list(zip(target_lengths, self.audio_paths, self.label_paths, self.augment_flags))
        _, self.audio_paths, self.label_paths, self.augment_flags = zip(*sorted(bundle, reverse=True))

    def batch_shuffle(self, remain_drop=False):
        """ batch shuffle """
        audio_batches, label_batches, flag_batches = [], [], []
        tmp_audio_paths, tmp_label_paths, tmp_augment_flags = [], [], []
        index = 0

        while True:
            if index == len(self.audio_paths):
                if len(tmp_audio_paths) != 0:
                    audio_batches.append(tmp_audio_paths)
                    label_batches.append(tmp_label_paths)
                    flag_batches.append(tmp_augment_flags)
                break
            if len(tmp_audio_paths) == self.batch_size:
                audio_batches.append(tmp_audio_paths)
                label_batches.append(tmp_label_paths)
                flag_batches.append(tmp_augment_flags)
                tmp_audio_paths, tmp_label_paths, tmp_augment_flags = [], [], []
            tmp_audio_paths.append(self.audio_paths[index])
            tmp_label_paths.append(self.label_paths[index])
            tmp_augment_flags.append(self.augment_flags[index])
            index += 1

        remain_audio, remain_label, remain_flag = audio_batches[-1], label_batches[-1], flag_batches[-1]
        audio_batches, label_batches, flag_batches = audio_batches[:-1], label_batches[:-1], flag_batches[:-1]

        bundle = list(zip(audio_batches, label_batches, flag_batches))
        random.shuffle(bundle)
        audio_batches, label_batches, flag_batches = zip(*bundle)

        audio_paths, label_paths, augment_flags = [], [], []

        for (audio_batch, label_batch, augment_flag) in zip(audio_batches, label_batches, flag_batches):
            audio_paths.extend(audio_batch)
            label_paths.extend(label_batch)
            augment_flags.extend(augment_flag)

        audio_paths = list(audio_paths)
        label_paths = list(label_paths)
        augment_flags = list(augment_flags)

        if not remain_drop:
            audio_paths.extend(remain_audio)
            label_paths.extend(remain_label)
            augment_flags.extend(remain_flag)

        return audio_paths, label_paths, augment_flags
