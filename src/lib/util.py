# -*- coding=utf-8 -*-
import os
import sys
import json
import copy
import string
import torch
import numpy as np
import torchaudio as ta

from zhon import hanzi
from torch.utils.data import Dataset
from src.conf import args

os.chdir(sys.path[0])


class Util(object):
    def __init__(self):
        pass

    @staticmethod
    def segment_wav(src_wav, tgt_wav, start_time, end_time):
        """
        音频切分

        :param src_wav:
        :param tgt_wav:
        :param start_time:
        :param end_time:
        :return:
        """
        span = end_time - start_time
        cmds = 'sox %s %s trim %f %f' % (src_wav, tgt_wav, start_time, span)
        os.system(cmds)

    @staticmethod
    def time2sed(t):
        """
        时间转秒
        :param t:
        :return:
        """
        h, m, s = t.strip().split(':')

        return float(h) * 3600 + float(m) * 60 + float(s)

    @staticmethod
    def load_json(json_file):
        """
        读取 json
        :param self:
        :return:
        """
        with open(file=json_file, mode='r') as file:
            json_lines = file.readlines()
            json_str = ''.join(json_lines).replace('\n', '').replace(' ', '').replace(',}', '}')

            return json.loads(json_str)

    @staticmethod
    def deal_train_dev():
        """
        处理训练集和验证集
        :return:
        """
        for name in ['train', 'dev']:
            save_dir = os.path.join(args.project_dir, 'data', name, 'wav')
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)

            seg_wav_list = []
            sub_audio_dir = os.path.join(args.audio_dir, name)
            for wav in os.listdir(sub_audio_dir):
                if wav[0] == '.':
                    continue  # 跳过隐藏文件

                suffix = None
                if name == 'dev':
                    parts = wav.split('_')
                    jf = '_'.join(parts[:-1]) + '.json'
                    suffix = parts[-1]
                else:
                    jf = wav[:-4] + '.json'

                utt_list = Util.load_json(os.path.join(args.transcription_dir, name, jf))
                for i in range(len(utt_list)):
                    utt_info = utt_list[i]
                    session_id = utt_info['session_id']

                    if name == 'dev':
                        tgt_id = session_id + '_' + str(i) + '_' + suffix
                    else:
                        tgt_id = session_id + '_' + str(i) + '.wav'

                    # 句子切分
                    start_time = Util.time2sed(utt_info['start_time']['original'])
                    end_time = Util.time2sed(utt_info['end_time']['original'])

                    src_wav = os.path.join(sub_audio_dir, wav)
                    tgt_wav = os.path.join(save_dir, tgt_id)
                    Util.segment_wav(src_wav=src_wav, tgt_wav=tgt_wav, start_time=start_time, end_time=end_time)
                    seg_wav_list.append((tgt_id, tgt_wav, utt_info['words']))

            with open(os.path.join(args.project_dir, 'data', name, 'wav.scp'), 'w') as ww:
                with open(os.path.join(args.project_dir, 'data', name, 'transcrpts.txt'), 'w', encoding='utf-8') as tw:
                    for uttid, wavdir, text in seg_wav_list:
                        ww.write(uttid + ' ' + wavdir + '\n')
                        tw.write(uttid + ' ' + text + '\n')

            print('prepare %s dataset done!' % name)

    @staticmethod
    def deal_test():
        """
        处理测试集
        :return:
        """
        if os.path.exists(args.data_test_wav_dir) is False:
            os.makedirs(args.data_test_wav_dir)

        seg_wav_list = []
        sub_audio_dir = os.path.join(args.audio_dir, 'test')
        for wav in os.listdir(sub_audio_dir):
            # 跳过隐藏文文件和IOS的音频文件
            if wav[0] == '.' or 'IOS' not in wav:
                continue

            jf = '_'.join(wav.split('_')[:-1]) + '.json'
            utt_list = Util.load_json(os.path.join(args.transcription_dir, 'test_no_ref_noise', jf))
            for i in range(len(utt_list)):
                utt_info = utt_list[i]
                session_id = utt_info['session_id']
                uttid = utt_info['uttid']

                # 如果句子已经标注，则跳过
                if 'word' in utt_info:
                    continue

                # 句子切分
                start_time = Util.time2sed(utt_info['start_time'])
                end_time = Util.time2sed(utt_info['end_time'])

                tgt_id = uttid + '.wav'
                src_wav = os.path.join(sub_audio_dir, wav)
                tgt_wav = os.path.join(args.data_test_wav_dir, tgt_id)
                Util.segment_wav(src_wav=src_wav, tgt_wav=tgt_wav, start_time=start_time, end_time=end_time)
                seg_wav_list.append((uttid, tgt_wav))

        with open(args.data_test_wav_path, 'w') as ww:
            for uttid, wavdir in seg_wav_list:
                ww.write(uttid + ' ' + wavdir + '\n')

        print('prepare test dataset done!')

    @staticmethod
    def text_normlization(seq):
        """
        文本归一化
        :param seq:
        :return:
        """
        new_seq = []
        for c in seq:
            if c == '+':
                # 文档中有加号，单独处理，避免删除
                new_seq.append(c)
            elif c in string.punctuation or c in hanzi.punctuation:
                # 删除全部的半角符号和全角符号
                continue
            else:
                if str(c).isalpha():
                    new_seq.append(c.lower())

        return ' '.join(new_seq)

    @staticmethod
    def normal_train_dev():
        """
        对训练和验证集进行标准化
        :return:
        """
        for name in ['train', 'dev']:
            with open(os.path.join(args.data_dir, name, 'transcrpts.txt'), 'r') as tr:
                with open(os.path.join(args.data_dir, name, 'text.txt'), 'w') as tw:
                    for line in tr:
                        parts = line.split()
                        uttid = parts[0]
                        seqs = ''.join(parts[1:])

                        # 直接跳过包含特殊标记的句子
                        if '[' in seqs:
                            continue

                        seqs = Util.text_normlization(seqs)
                        tw.write(uttid + ' ' + seqs + '\n')

            print('Normlize %s TEXT!' % name)

    @staticmethod
    def get_seq_mask(targets):
        """
        遮掉未来的文本信息
        :param targets:
        :return:
        """
        batch_size, steps = targets.size()
        seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
        seq_mask = torch.tril(seq_mask).bool()

        return seq_mask


class DataUtil(object):
    def __init__(self):
        pass

    # 词表生成
    def generate_vocab_table(self):
        """
        根据训练集文本生成词表，并加入起始标记<BOS>,结束标记<EOS>,填充标记<PAD>,以及未识别词标记<UNK>

        :return: 返回模型词表大小
        """
        vocab_dict = {}
        for name in ['train', 'dev']:
            with open(os.path.join(args.data_dir, name, 'text.txt'), mode='r', encoding='utf-8') as file:
                for line in file.readlines():
                    chars = line.strip().split()[1:]
                    for char in chars:
                        if char not in vocab_dict:
                            vocab_dict[char] = 1
                        else:
                            vocab_dict[char] += 1

        vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        vocab = copy.deepcopy(args.vocab)
        for index, item in enumerate(vocab_list):
            vocab[item[0]] = index + 4

        print('There are {} units in Vocabulary!'.format(len(vocab)))

        with open(args.vocab_path, mode='w', encoding='utf-8') as file:
            for key, value in vocab.items():
                file.write(key + ' ' + str(value) + '\n')

        return len(vocab)

    @staticmethod
    def collate_fn(batch):
        """
        收集函数，将同一批内的特征填充到相同的长度，并在文本中加上起始和结束标记
        :param batch:
        :return:
        """
        uttids = [data[0] for data in batch]
        features_length = [data[1].shape[0] for data in batch]
        max_feat_length = max(features_length)
        padded_features = []

        if len(batch[0]) == 3:
            targets_length = [len(data[2]) for data in batch]
            max_text_length = max(targets_length)
            padded_targets = []

        for parts in batch:
            feat = parts[1]
            feat_len = feat.shape[0]
            padded_features.append(np.pad(feat, ((
                                                     0, max_feat_length - feat_len), (0, 0)), mode='constant',
                                          constant_values=0.0))

            if len(batch[0]) == 3:
                target = parts[2]
                text_len = len(target)
                padded_targets.append(
                    [args.vocab['<BOS>']] + target + [args.vocab['<EOS>']] + [args.vocab['<PAD>']] * (
                            max_text_length - text_len))

        if len(batch[0]) == 3:
            return uttids, torch.FloatTensor(padded_features), torch.LongTensor(padded_targets)
        else:
            return uttids, torch.FloatTensor(padded_features)


class AudioDataset(Dataset):
    def __init__(self, wav_list, text_list=None, unit2idx=None):

        self.unit2idx = unit2idx

        self.file_list = []
        for wavscpfile in wav_list:
            with open(wavscpfile, 'r', encoding='utf-8') as wr:
                for line in wr:
                    uttid, path = line.strip().split()
                    self.file_list.append([uttid, path])

        if text_list is not None:
            self.targets_dict = {}
            for textfile in text_list:
                with open(textfile, 'r', encoding='utf-8') as tr:
                    for line in tr:
                        parts = line.strip().split()
                        uttid = parts[0]
                        label = []
                        for c in parts[1:]:
                            label.append(self.unit2idx[c] if c in self.unit2idx else self.unit2idx['<UNK>'])
                        self.targets_dict[uttid] = label
            self.file_list = self.filter(self.file_list)  # 过滤掉没有标注的句子
            assert len(self.file_list) == len(self.targets_dict)
        else:
            self.targets_dict = None

        self.lengths = len(self.file_list)

    def __getitem__(self, index):
        uttid, path = self.file_list[index]
        wavform, _ = ta.load_wav(path)  # 加载wav文件
        feature = ta.compliance.kaldi.fbank(wavform, num_mel_bins=40)  # 计算fbank特征
        # 特征归一化
        mean = torch.mean(feature)
        std = torch.std(feature)
        feature = (feature - mean) / std

        if self.targets_dict is not None:
            targets = self.targets_dict[uttid]
            return uttid, feature, targets
        else:
            return uttid, feature

    def filter(self, feat_list):
        new_list = []
        for (uttid, path) in feat_list:
            if uttid not in self.targets_dict: continue
            new_list.append([uttid, path])
        return new_list

    def __len__(self):
        return self.lengths

    @property
    def idx2char(self):
        return {i: c for (c, i) in self.unit2idx.items()}


if __name__ == '__main__':
    pass
    # 处理训练集和验证集
    # Util.deal_train_dev()

    # 处理测试集
    # Util.deal_test()

    # 标准化训练和验证集
    # Util.normal_train_dev()

    # 词表生成
    # DataUtil().generate_vocab_table()
