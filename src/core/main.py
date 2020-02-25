#!/usr/bin/env python
# coding: utf-8

# 将音频切分保存到wav文件夹中，并生成wav.scp和transcripts文件
# wav.scp文件格式:
# id1 path1
# id2 path2
# transcripts文件格式：
# id1 你好
# id2 今天天气怎么样

import os
import json

data_rootdir = './Magicdata'  # 指定解压后数据的根目录

audiodir = os.path.join(data_rootdir, 'audio')
trans_dir = os.path.join(data_rootdir, 'transcription')


# 音频切分
def segment_wav(src_wav, tgt_wav, start_time, end_time):
    span = end_time - start_time
    cmds = 'sox %s %s trim %f %f' % (src_wav, tgt_wav, start_time, span)
    os.system(cmds)


# 将时间格式转化为秒为单位
def time2sec(t):
    h, m, s = t.strip().split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


# 读取json文件内容
def load_json(json_file):
    with open(json_file, 'r') as f:
        lines = f.readlines()
        json_str = ''.join(lines).replace('\n', '').replace(' ', '').replace(',}', '}')
        return json.loads(json_str)


# 训练集和开发集数据处理
for name in ['train', 'dev']:
    save_dir = os.path.join('./data', name, 'wav')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seg_wav_list = []
    sub_audio_dir = os.path.join(audiodir, name)
    for wav in os.listdir(sub_audio_dir):
        if wav[0] == '.':
            continue  # 跳过隐藏文件

        if name == 'dev':
            parts = wav.split('_')
            jf = '_'.join(parts[:-1]) + '.json'
            suffix = parts[-1]
        else:
            jf = wav[:-4] + '.json'

        utt_list = load_json(os.path.join(trans_dir, name, jf))
        for i in range(len(utt_list)):
            utt_info = utt_list[i]
            session_id = utt_info['session_id']

            if name == 'dev':
                tgt_id = session_id + '_' + str(i) + '_' + suffix
            else:
                tgt_id = session_id + '_' + str(i) + '.wav'

            # 句子切分
            start_time = time2sec(utt_info['start_time']['original'])
            end_time = time2sec(utt_info['end_time']['original'])

            src_wav = os.path.join(sub_audio_dir, wav)
            tgt_wav = os.path.join(save_dir, tgt_id)
            segment_wav(src_wav, tgt_wav, start_time, end_time)
            seg_wav_list.append((tgt_id, tgt_wav, utt_info['words']))

    with open(os.path.join('./data', name, 'wav.scp'), 'w') as ww:
        with open(os.path.join('./data', name, 'transcrpts.txt'), 'w', encoding='utf-8') as tw:
            for uttid, wavdir, text in seg_wav_list:
                ww.write(uttid + ' ' + wavdir + '\n')
                tw.write(uttid + ' ' + text + '\n')

    print('prepare %s dataset done!' % name)

# 测试集数据处理
save_dir = os.path.join('./data', 'test', 'wav')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

seg_wav_list = []
sub_audio_dir = os.path.join(audiodir, 'test')
for wav in os.listdir(sub_audio_dir):

    if wav[0] == '.' or 'IOS' not in wav:
        continue  # 跳过隐藏文件和非IOS的音频文件

    jf = '_'.join(wav.split('_')[:-1]) + '.json'

    utt_list = load_json(os.path.join(trans_dir, 'test_no_ref_noise', jf))
    for i in range(len(utt_list)):
        utt_info = utt_list[i]
        session_id = utt_info['session_id']
        uttid = utt_info['uttid']
        if 'words' in utt_info: continue  # 如果句子已经标注，则跳过
        # 句子切分
        start_time = time2sec(utt_info['start_time'])
        end_time = time2sec(utt_info['end_time'])

        tgt_id = uttid + '.wav'
        src_wav = os.path.join(sub_audio_dir, wav)
        tgt_wav = os.path.join(save_dir, tgt_id)
        segment_wav(src_wav, tgt_wav, start_time, end_time)
        seg_wav_list.append((uttid, tgt_wav))

with open(os.path.join('./data', 'test', 'wav.scp'), 'w') as ww:
    for uttid, wavdir in seg_wav_list:
        ww.write(uttid + ' ' + wavdir + '\n')

print('prepare test dataset done!')

# ### 2.2.2 文本归一化处理
# 这里要对文本数据进行归一化处理，其中包括大写字母都转化为小写字母，过滤掉标点符号和无意义的句子。

# In[ ]:


# 过滤掉各种标点符号
# 过滤掉包含[*]、[LAUGH]、[SONANT]、[ENs]、[MUSIC]的句子
# 大写字母转小写
import os
import string
import zhon.hanzi


def text_normlization(seq):
    new_seq = []
    for c in seq:
        if c == '+':
            new_seq.append(c)  # 文档中有加号，所以单独处理，避免删除
        elif c in string.punctuation or c in zhon.hanzi.punctuation:
            continue  # 删除全部的半角标点和全角标点
        else:
            if c.encode('UTF-8').isalpha(): c = c.lower()  # 大写字母转小写
            new_seq.append(c)
    return ' '.join(new_seq)


for name in ['train', 'dev']:
    with open(os.path.join('./data', name, 'transcrpts.txt'), 'r') as tr:
        with open(os.path.join('./data', name, 'text'), 'w') as tw:
            for line in tr:
                parts = line.split()
                uttid = parts[0]
                seqs = ''.join(parts[1:])
                if '[' in seqs: continue  # 直接跳过包含特殊标记的句子
                seqs = text_normlization(seqs)
                tw.write(uttid + ' ' + seqs + '\n')

    print('Normlize %s TEXT!' % name)

# # 3. 思路分析
# ## 3.1 常见的语音识别方法
# 目前的语音识别方法可以大致分为混合模型和端到端模型结构。混合模型以Chain模型为代表（基于Kaldi工具包构建），是目前工业上主流的模型结构，其训练过程步骤相较繁琐。端到端模型是最近几年比较火的模型结构，以CTC模型、基于注意力机制的序列到序列模型，以及Transducer模型为代表，其中基于注意力机制的序列到序列模型效果最好。中文的端到端模型一般使用汉字作为建模单元，不需要发音词典等先验信息，直接将音频文本对输入模型就能直接进行训练，省去了繁琐的准备过程。并且在很多数据集例如AISHELL上面，基于注意力机制的端到端模型的效果已经远超Chain模型，详情见[ESPnet](https://github.com/espnet/espnet)。
# 
# ## 3.2 基线系统方法
# 本文中决定使用目前最好的端到端模型（Speech-Transformer）构建基线系统。
# 
# 基于注意力机制的序列到序列模型结构包含**编码器**和**解码器**两部分：
# - **编码器** 编码器用来将输入特征编码为高层的特征表示
# - **解码器** 解码器从起始标记“<BOS>”开始，利用注意力机制不断从编码特征中抽取有用信息并解码出一个汉字，不断重复这一过程，直到解码出结束标记“<EOS>”，由此完成整个解码过程。
# 
# Speech-Transformer丢弃了传统序列到序列模型的循环神经网络结构，使用自注意力机制进行替代，不仅仅提高了模型的训练速度，也使得解码精度大大提高。详情可以参考论文[Speech-Transformer:A No-Recurrence Sequence-to-Sequence Model for Speech Recognition](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005884.pdf)和[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
# 
# 模型结构图参考[Speech-Transformer结构图](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005884.pdf)
# 

# # 4. 基线系统的构建
# ## 4.1 实验环境
# 实验在Linux系统上进行，要求具备以下软件和硬件环境。
# - 至少具备一个GPU
# - python >= 3.6
# - pytorch >= 1.2.0
# - torchaudio >= 0.3.0

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# ## 4.2 数据处理与加载
# ### 4.2.1 词表生成
# 根据训练集文本生成词表，并加入起始标记<BOS\>,结束标记<EOS\>,填充标记<PAD\>,以及未识别词标记<UNK\>

# In[ ]:


# 词表生成
import os

vocab_dict = {}

for name in ['train', 'dev']:
    with open(os.path.join('./data', name, 'text'), 'r', encoding='utf-8') as fr:
        for line in fr:
            chars = line.strip().split()[1:]
            for c in chars:
                if c in vocab_dict:
                    vocab_dict[c] += 1
                else:
                    vocab_dict[c] = 1

vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
for i in range(len(vocab_list)):
    c = vocab_list[i][0]
    vocab[c] = i + 4

print('There are %d units in Vocabulary!' % len(vocab))
with open(os.path.join('./data', 'vocab'), 'w', encoding='utf-8') as fw:
    for c, id in vocab.items():
        fw.write(c + ' ' + str(id) + '\n')

# ### 4.2.2 构建特征提取与加载模块

# In[ ]:


import os
import torch
import numpy as np
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader

PAD = 0
BOS = 1
EOS = 2
UNK = 3


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


# 收集函数，将同一个批内的特征填充到同样的长度，并在文本中加上起始标记和结束标记
def collate_fn(batch):
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
                [BOS] + target + [EOS] + [PAD] * (max_text_length - text_len))

    if len(batch[0]) == 3:
        return uttids, torch.FloatTensor(padded_features), torch.LongTensor(padded_targets)
    else:
        return uttids, torch.FloatTensor(padded_features)


# ## 4.3 模型构建
# 模型的主题结构包含**编码器**和**解码器**。
# ### 4.3.1 基本模块构建
# 主要是模型的一些子模块，包括多头注意力机制，前馈模块以及位置编码模块。  
# #### 多头注意力机制模块

# In[ ]:


# 多头注意力机制
class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


# #### 前馈模块

# In[ ]:


# 前馈模块
class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units * 2)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.w_1(x)
        x = F.glu(x)
        return self.w_2(self.dropout(x))


# #### 位置编码模块

# In[ ]:


# 位置编码
class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout_rate=0.0, max_len=5000):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ### 4.3.3 解码器

# In[ ]:


# 编码器
# 对输入数据进行维度变换，时间维度降采样，以及进行位置编码，模块输出长度是输入长度的四分之一
class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate=0.0):
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


# 定义Transformer的每层的网络结构
class TransformerEncoderLayer(nn.Module):
    def __init__(self, attention_heads, d_model, linear_units, residual_dropout_rate):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)

    def forward(self, x, mask):
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        x = residual + self.dropout1(self.self_attn(x, x, x, mask))
        x = self.norm1(x)

        residual = x
        x = residual + self.dropout2(self.feed_forward(x))
        x = self.norm2(x)

        return x, mask


class TransformerEncoder(nn.Module):

    def __init__(self, input_size, d_model=256, attention_heads=4, linear_units=2048, num_blocks=6,
                 repeat_times=1, pos_dropout_rate=0.0, slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0,
                 residual_dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.embed = Conv2dSubsampling(input_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(attention_heads, d_model, linear_units, residual_dropout_rate) for _ in
            range(num_blocks)
        ])

    def forward(self, inputs):
        enc_mask = torch.sum(inputs, dim=-1).ne(0).unsqueeze(-2)
        enc_output, enc_mask = self.embed(inputs, enc_mask)

        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        for _, block in enumerate(self.blocks):
            enc_output, enc_mask = block(enc_output, enc_mask)

        return enc_output, enc_mask


# ### 4.3.3 解码器

# In[ ]:


# 定义解码层
class TransformerDecoderLayer(nn.Module):

    def __init__(self, attention_heads, d_model, linear_units, residual_dropout_rate):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model)
        self.src_attn = MultiHeadedAttention(attention_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features

        :param torch.Tensor tgt: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        :param torch.Tensor memory: encoded source features (batch, max_time_in, size)
        :param torch.Tensor memory_mask: mask for memory (batch, max_time_in)
        """
        residual = tgt
        x = residual + self.dropout1(self.self_attn(tgt, tgt, tgt, tgt_mask))
        x = self.norm1(x)

        residual = x
        x = residual + self.dropout2(self.src_attn(x, memory, memory, memory_mask))
        x = self.norm2(x)

        residual = x
        x = residual + self.dropout3(self.feed_forward(x))
        x = self.norm3(x)

        return x, tgt_mask


# 遮掉未来的文本信息
def get_seq_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask


# 定义解码器
class TransformerDecoder(nn.Module):
    def __init__(self, output_size, d_model=256, attention_heads=4, linear_units=2048, num_blocks=6,
                 residual_dropout_rate=0.1, share_embedding=False):
        super(TransformerDecoder, self).__init__()

        self.embedding = torch.nn.Embedding(output_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.blocks = nn.ModuleList([
            TransformerDecoderLayer(attention_heads, d_model, linear_units,
                                    residual_dropout_rate) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(d_model, output_size)

        if share_embedding:
            assert self.embedding.weight.size() == self.output_layer.weight.size()
            self.output_layer.weight = self.embedding.weight

    def forward(self, targets, memory, memory_mask):

        dec_output = self.embedding(targets)
        dec_output = self.pos_encoding(dec_output)

        dec_mask = get_seq_mask(targets)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        logits = self.output_layer(dec_output)

        return logits

    def recognize(self, preds, memory, memory_mask, last=True):

        dec_output = self.embedding(preds)
        dec_mask = get_seq_mask(preds)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        logits = self.output_layer(dec_output)

        log_probs = F.log_softmax(logits[:, -1] if last else logits, dim=-1)

        return log_probs


# ### 4.3.4 整体结构

# In[ ]:


# 定义整体模型结构
class Transformer(nn.Module):
    def __init__(self, input_size, vocab_size, d_model=320, n_heads=4, d_ff=1280, num_enc_blocks=6, num_dec_blocks=6,
                 residual_dropout_rate=0.1, share_embedding=True):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.encoder = TransformerEncoder(input_size=input_size, d_model=d_model,
                                          attention_heads=n_heads,
                                          linear_units=d_ff,
                                          num_blocks=num_enc_blocks,
                                          residual_dropout_rate=residual_dropout_rate)

        self.decoder = TransformerDecoder(output_size=vocab_size,
                                          d_model=d_model,
                                          attention_heads=n_heads,
                                          linear_units=d_ff,
                                          num_blocks=num_dec_blocks,
                                          residual_dropout_rate=residual_dropout_rate,
                                          share_embedding=share_embedding)

        self.crit = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # 1. forward encoder
        enc_states, enc_mask = self.encoder(inputs)

        # 2. forward decoder
        target_in = targets[:, :-1].clone()
        logits = self.decoder(target_in, enc_states, enc_mask)

        # 3. compute attention loss
        target_out = targets[:, 1:].clone()
        loss = self.crit(logits.reshape(-1, self.vocab_size), target_out.view(-1))

        return loss


# ## 4.4 训练过程与模型保存

# In[ ]:


import math

total_epochs = 60  # 模型迭代次数
model_size = 320  # 模型维度
n_heads = 4  # 注意力机制头数
num_enc_blocks = 6  # 编码器层数
num_dec_blocks = 6  # 解码器层数
residual_dropout_rate = 0.1  # 残差连接丢弃率
share_embedding = True  # 是否共享编码器词嵌入的权重

batch_size = 16  # 指定批大小
warmup_steps = 12000  # 热身步数
lr_factor = 1.0  # 学习率因子
accu_grads_steps = 8  # 梯度累计步数

input_size = 40  # 输入特征维度

# 加载词表
unit2idx = {}
with open('./data/vocab', 'r', encoding='utf-8') as fr:
    for line in fr:
        unit, idx = line.strip().split()
        unit2idx[unit] = int(idx)
vocab_size = len(unit2idx)  # 输出词表大小

# 模型定义
model = Transformer(input_size=input_size,
                    vocab_size=vocab_size,
                    d_model=model_size,
                    n_heads=n_heads,
                    d_ff=model_size * 4,
                    num_enc_blocks=num_enc_blocks,
                    num_dec_blocks=num_dec_blocks,
                    residual_dropout_rate=residual_dropout_rate,
                    share_embedding=share_embedding)

if torch.cuda.is_available():
    model.cuda()  # 将模型加载到GPU中

train_wav_list = ['./data/train/wav.scp', './data/dev/wav.scp']
train_text_list = ['./data/train/text', './data/dev/text']
dataset = AudioDataset(train_wav_list, train_text_list, unit2idx=unit2idx)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2, pin_memory=False,
                                         collate_fn=collate_fn)


# 定义优化器以及学习率更新函数
def get_learning_rate(step):
    return lr_factor * model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


lr = get_learning_rate(step=1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

if not os.path.exists('./model'): os.makedirs('./model')

global_step = 1
step_loss = 0
print('Begin to Train...')
for epoch in range(total_epochs):
    print('*****  epoch: %d *****' % epoch)
    for step, (_, inputs, targets) in enumerate(dataloader):
        # 将输入加载到GPU中
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        loss = model(inputs, targets)
        loss.backward()
        step_loss += loss.item()

        if (step + 1) % accu_grads_steps == 0:
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            if global_step % 10 == 0:
                print('-Training-Epoch-%d, Global Step:%d, lr:%.8f, Loss:%.5f' % \
                      (epoch, global_step, lr, step_loss / accu_grads_steps))
            global_step += 1
            step_loss = 0

            # 学习率更新
            lr = get_learning_rate(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # 模型保存
    checkpoint = model.state_dict()
    torch.save(checkpoint, os.path.join('./model', 'model.epoch.%d.pt' % epoch))
print('Done!')
