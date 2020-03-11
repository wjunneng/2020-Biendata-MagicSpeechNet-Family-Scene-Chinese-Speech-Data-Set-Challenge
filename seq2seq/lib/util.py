# -*- coding=utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import json
import copy
import string
import torch
import numpy as np
import librosa
import pandas as pd
import torchaudio as ta

from torch import nn
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from zhon import hanzi
from torch.utils.data import Dataset

from seq2seq.conf import args

from seq2seq.lib import spec_augment_pytorch, melscale_pytorch
from seq2seq.lib import wavio


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
    def time2sec(t):
        """
        时间转秒
        :param t:
        :return:
        """
        h, m, s = t.strip().split(":")

        return float(h) * 3600 + float(m) * 60 + float(s)

    @staticmethod
    def load_json(json_file):
        """
        读取 json
        :param self:
        :return:
        """
        with open(json_file, 'r') as f:
            lines = f.readlines()
            json_str = ''.join(lines).replace('\n', '').replace(' ', '').replace(',}', '}')

            return json.loads(json_str)

    @staticmethod
    def deal_train_dev():
        """
        处理训练集和验证集
        :return:
        """
        for name in ['train', 'dev']:
            save_dir = os.path.join(args.data_dir, name, 'wav')
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)

            seg_wav_list = []
            sub_audio_dir = os.path.join(args.audio_dir, name)
            for wav in os.listdir(sub_audio_dir):
                if wav[0] == '.':
                    # 跳过隐藏文件
                    continue

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
                    start_time = Util.time2sec(utt_info['start_time']['original'])
                    end_time = Util.time2sec(utt_info['end_time']['original'])

                    src_wav = os.path.join(sub_audio_dir, wav)
                    tgt_wav = os.path.join(save_dir, tgt_id)
                    Util.segment_wav(src_wav=src_wav, tgt_wav=tgt_wav, start_time=start_time, end_time=end_time)
                    seg_wav_list.append((tgt_id, tgt_wav, utt_info['words']))

            with open(os.path.join(args.data_dir, name, 'wav.scp'), 'w') as ww:
                with open(os.path.join(args.data_dir, name, 'transcrpts.txt'), 'w', encoding='utf-8') as tw:
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
                continue  # 跳过隐藏文件和非IOS的音频文件

            jf = '_'.join(wav.split('_')[:-1]) + '.json'
            utt_list = Util.load_json(os.path.join(args.transcription_dir, 'test_no_ref_noise', jf))
            for i in range(len(utt_list)):
                utt_info = utt_list[i]
                session_id = utt_info['session_id']
                uttid = utt_info['uttid']
                # 如果句子已经标注，则跳过
                if 'words' in utt_info:
                    continue
                # 句子切分
                start_time = Util.time2sec(utt_info['start_time'])
                end_time = Util.time2sec(utt_info['end_time'])

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
        import jieba

        new_seq = []
        for c in list(jieba.cut(seq, cut_all=False)):
            if c == '+':
                # 文档中有加号，所以单独处理，避免删除
                new_seq.append(c)
            elif c in string.punctuation or c in hanzi.punctuation:
                # 删除全部的半角标点和全角标点
                continue
            else:
                if c.encode('UTF-8').isalpha():
                    # 大写字母转小写
                    c = c.lower()
                new_seq.append(c)
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
                        if '[' in seqs:
                            continue  # 直接跳过包含特殊标记的句子
                        seqs = Util.text_normlization(seqs)
                        tw.write(uttid + ' ' + seqs + '\n')

            print('Normlize %s TEXT!' % name)

    @staticmethod
    # 遮掉未来的文本信息
    def get_seq_mask(targets):
        batch_size, steps = targets.size()
        seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
        seq_mask = torch.tril(seq_mask).bool()
        return seq_mask

    @staticmethod
    # 定义优化器以及学习率更新函数
    def get_learning_rate(step):
        return args.lr_factor * args.model_size ** (-0.5) * min(step ** (-0.5), step * args.warmup_steps ** (-1.5))

    @staticmethod
    def draw(nframes, framerate, data):
        """
        :param nframes: 采样点个数
        :param framerate: 采样频率
        :return:
        """
        from matplotlib import pyplot as plt

        # 采样点的时间间隔
        sample_time = 1 / framerate
        # 20毫秒左右
        print('一帧持续的时间{}'.format(sample_time))
        # 声音信号的长度
        time = nframes / framerate

        x_seq = np.arange(0, time, sample_time)
        x_seq = x_seq[:nframes, ]
        plt.plot(x_seq, data.reshape([-1, 1]), 'blue')
        plt.xlabel("time (s)")
        plt.show()

    @staticmethod
    def mfcc(data, sampling_rate, n_mfcc):
        """
        Compute mel-scaled feature using librosa
        :param data:
        :param sampling_rate:
        :param n_mfcc:
        :return:
        """
        data = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
        # data = np.expand_dims(data, axis=-1)
        return data

    @staticmethod
    def padding(data, input_length):
        """
            Padding of samples to make them of same length
        """
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        return data

    @staticmethod
    def padding_numpy(data, input_len):
        """
            二维数组 padding
        """
        if data.shape[1] > input_len:
            return data[:, :input_len]
        else:
            return np.pad(data, ((0, 0), (0, input_len - data.shape[1])), 'constant')

    @staticmethod
    def apply_per_channel_energy_norm(data, sampling_rate):
        """
        Compute Per-Channel Energy Normalization (PCEN)
        :param data:
        :param sampling_rate:
        :return:
        """
        # Compute mel-scaled spectrogram
        S = librosa.feature.melspectrogram(data, sr=sampling_rate, power=1)
        # Convert an amplitude spectrogram to dB-scaled spectrogram
        # log_S = librosa.amplitude_to_db(S, ref=np.max)
        pcen_S = librosa.core.pcen(S)

        return pcen_S

    @staticmethod
    def wavelet_denoising(data):
        """
        Wavelet Denoising using scikit-image
        NOTE: Wavelet denoising is an effective method for SNR improvement in environments with
                  wide range of noise types competing for the same subspace.
        """
        sigma_est = estimate_sigma(data, multichannel=True, average_sigmas=True)
        im_bayes = denoise_wavelet(data, multichannel=True, convert2ycbcr=False, method='BayesShrink',
                                   mode='soft')
        # im_visushrink = denoise_wavelet(data, multichannel=False, convert2ycbcr=True, method='VisuShrink',
        #                                 mode='soft')
        #
        # # VisuShrink is designed to eliminate noise with high probability, but this
        # # results in a visually over-smooth appearance. Here, we specify a reduction
        # # in the threshold by factors of 2 and 4.
        # im_visushrink2 = denoise_wavelet(data, multichannel=True, convert2ycbcr=False, method='VisuShrink',
        #                                  mode='soft', sigma=sigma_est / 2)
        # im_visushrink4 = denoise_wavelet(data, multichannel=False, convert2ycbcr=True, method='VisuShrink',
        #                                  mode='soft', sigma=sigma_est / 4)
        return im_bayes

    @staticmethod
    def generate_result():
        data_result_path = args.data_reault_path
        sample_submission_path = args.sample_submission_path
        data_submission_path = args.data_submission_path

        result = pd.read_csv(data_result_path)
        sample_submission = pd.read_csv(sample_submission_path)
        result = dict(zip(result['id'], result['words']))
        for index in range(sample_submission.shape[0]):
            value = result[sample_submission.iloc[index, 0]]
            if str(value) == 'nan':
                sample_submission.iloc[index, 1] = '嗯'
            else:
                sample_submission.iloc[index, 1] = str(value)

        sample_submission.to_csv(data_submission_path, index=None)
        print(sample_submission)

    @staticmethod
    def preprocessing_result():
        data_submission_path = args.data_submission_path
        sample_submission = pd.read_csv(data_submission_path)

        words = []
        for word in sample_submission['words']:
            str = word
            if len(word) > 2:
                str = word[:2]
                for index in range(2, len(word)):
                    if len(set(word[index - 2:index])) == 1:
                        continue
                    else:
                        str += word[index]
            words.append(str)
        sample_submission['words'] = words
        sample_submission.to_csv(args.data_pre_submission_path, index=None)

    @staticmethod
    def trim(data, cfg_trim):
        threshold_attack = cfg_trim["threshold_attack"]
        threshold_release = cfg_trim["threshold_release"]
        attack_margin = cfg_trim["attack_margin"]
        release_margin = cfg_trim["release_margin"]

        data_size = len(data)
        cut_head = 0
        cut_tail = data_size

        # Square
        w = np.power(np.divide(data, np.max(data)), 2)

        # Gaussian kernel
        sig = 20000
        time = np.linspace(-40000, 40000)
        kernel = np.exp(-np.square(time) / 2 / sig / sig)

        # Smooth and normalize
        w = np.convolve(w, kernel, mode='same')
        w = np.divide(w, np.max(w))

        # Detect crop sites
        for sample in range(data_size):
            sample_num = sample
            sample_amp = w[sample_num]
            if sample_amp > threshold_attack:
                cut_head = np.max([sample_num - attack_margin, 0])
                break

        for sample in range(data_size):
            sample_num = data_size - sample - 1
            sample_amp = w[sample_num]
            if sample_amp > threshold_release:
                cut_tail = np.min([sample_num + release_margin, data_size])
                break

        # print("trimmed audio length = ", cut_tail-cut_head+1)

        data_copy = data[cut_head:cut_tail]
        del w, time, kernel, data

        return data_copy


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
                for line in file:
                    chars = line.strip().split()[1:]
                    for c in chars:
                        if c in vocab_dict:
                            vocab_dict[c] += 1
                        else:
                            vocab_dict[c] = 1

        vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        vocab = copy.deepcopy(args.vocab)
        for i in range(len(vocab_list)):
            c = vocab_list[i][0]
            vocab[c] = i + 4

        print('There are {} units in Vocabulary!'.format(len(vocab)))
        with open(args.vocab_path, mode='w', encoding='utf-8') as file:
            for c, id in vocab.items():
                file.write(c + ' ' + str(id) + '\n')

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
            padded_features.append(
                np.pad(feat, ((0, max_feat_length - feat_len), (0, 0)), mode='constant', constant_values=0.0))

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


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class AudioDataset(Dataset):
    def __init__(self, wav_list, text_list=None, unit2idx=None, enhance_type='fbank'):
        self.enhance_type = enhance_type
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

        feature = None

        # fbank
        if self.enhance_type == 'fbank':
            # 加载wav文件
            wavform, _ = ta.load_wav(path)
            # 计算fbank特征
            feature = ta.compliance.kaldi.fbank(wavform, num_mel_bins=args.input_size)

        # time mask + frequency mask
        elif self.enhance_type == 'freq_mask_time_mask':
            (rate, width, sig) = wavio.readwav(path)
            sig = sig.ravel()
            sig = Util.trim(sig,
                            {"use": True, "threshold_attack": 0.01, "threshold_release": 0.01, "attack_margin": 5000,
                             "release_margin": 5000})
            stft = torch.stft(torch.FloatTensor(sig),
                              args.N_FFT,
                              hop_length=int(0.01 * rate),
                              win_length=int(0.030 * rate),
                              window=torch.hamming_window(int(0.030 * rate)),
                              center=False,
                              normalized=False,
                              onesided=True)
            stft = (stft[:, :, 0].pow(2) + stft[:, :, 1].pow(2)).pow(0.5)

            if args.use_mel_scale:
                amag = stft.clone().detach()
                # reshape spectrogram shape to [batch_size, time, frequency]
                amag = amag.view(-1, amag.shape[0], amag.shape[1])
                # melspec with same shape
                # mel = melscale_pytorch.mel_scale(amag, sample_rate=rate, n_mels=args.N_FFT // 2 + 1)
                mel = melscale_pytorch.mel_scale(amag, sample_rate=rate, n_mels=args.input_size)
                if args.use_specaug:
                    specaug_prob = 1  # augment probability
                    if np.random.uniform(0, 1) < specaug_prob:
                        # apply augment
                        mel = spec_augment_pytorch.spec_augment(mel, time_warping_para=80, frequency_masking_para=54,
                                                                time_masking_para=40, frequency_mask_num=1,
                                                                time_mask_num=1)
                # squeeze back to [frequency, time]
                feat = mel.view(mel.shape[1], mel.shape[2])
                feature = feat.transpose(0, 1).clone().detach()
                del sig, stft, amag, mel
            else:
                # use baseline feature
                amag = stft.numpy()
                feat = torch.FloatTensor(amag)
                feature = torch.FloatTensor(feat).transpose(0, 1)
                del sig, stft, amag

        # ################# librosa
        # if args.using_mfcc:
        #     y_16k, sr_16k = librosa.core.load(path, sr=16000, res_type='kaiser_fast')
        #     # mfcc
        #     y_16k = Util.mfcc(y_16k, sampling_rate=sr_16k, n_mfcc=args.input_size)
        #     # 标准化
        #     y_16k = Util.apply_per_channel_energy_norm(data=y_16k.flatten(), sampling_rate=sr_16k)
        #     # 去静音
        #     # y_16k = Util.wavelet_denoising(data=y_16k)
        #     # 填充
        #     y_16k = Util.padding_numpy(y_16k, args.input_size)
        #     # Util.draw(nframes=y_16k.size, framerate=16000, data=y_16k)
        #
        #     feature = torch.from_numpy(y_16k)
        #     feature = (feature - torch.mean(feature)) / torch.std(feature)

        feature = (feature - torch.mean(feature)) / torch.std(feature)
        if self.targets_dict is not None:
            targets = self.targets_dict[uttid]
            return uttid, feature, targets
        else:
            return uttid, feature

    def filter(self, feat_list):
        new_list = []
        for (uttid, path) in feat_list:
            if uttid not in self.targets_dict:
                continue
            new_list.append([uttid, path])
        return new_list

    def __len__(self):
        return self.lengths

    @property
    def idx2char(self):
        return {i: c for (c, i) in self.unit2idx.items()}


if __name__ == '__main__':
    pass
    # # 处理训练集和验证集
    # Util.deal_train_dev()

    # # 处理测试集
    # Util.deal_test()

    # # 标准化训练和验证集
    # Util.normal_train_dev()

    # # 词表生成
    # DataUtil().generate_vocab_table()

    # 后处理
    # Util.preprocessing_result()
