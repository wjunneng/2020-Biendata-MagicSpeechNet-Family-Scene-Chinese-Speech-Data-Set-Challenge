# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import math
import torch

import numpy as np
from torch import nn
import torch.nn.functional as F
from seq2seq.lib.util import Util
from seq2seq.conf import args
from seq2seq.lib.util import LabelSmoothingLoss


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


# 编码器『未』
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


# 定义Transformer的每层的网络结构『未』
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


# 定义Transformer编码器『未』
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


# 定义Transformer解码层
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

        dec_mask = Util.get_seq_mask(targets)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        logits = self.output_layer(dec_output)

        return logits

    def recognize(self, preds, memory, memory_mask, last=True):

        dec_output = self.embedding(preds)
        dec_mask = Util.get_seq_mask(preds)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        logits = self.output_layer(dec_output)

        log_probs = F.log_softmax(logits[:, -1] if last else logits, dim=-1)

        return log_probs


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

        # 交叉熵
        self.crit = nn.CrossEntropyLoss()
        # 标签平滑
        # self.crit = LabelSmoothingLoss(classes=args.vocab_size, smoothing=0.2)

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


# 定义解码识别模块
class Recognizer():
    def __init__(self, model, unit2char=None, beam_width=5, max_len=100):

        self.model = model
        self.model.eval()
        self.unit2char = unit2char
        self.beam_width = beam_width
        self.max_len = max_len

    def recognize(self, inputs):

        enc_states, enc_masks = self.model.encoder(inputs)

        # 将编码状态重复beam_width次
        beam_enc_states = enc_states.repeat([self.beam_width, 1, 1])
        beam_enc_mask = enc_masks.repeat([self.beam_width, 1, 1])

        # 设置初始预测标记 <BOS>, 维度为[beam_width, 1]
        preds = torch.ones([self.beam_width, 1], dtype=torch.long, device=enc_states.device) * args.vocab['<BOS>']

        # 定义每个分支的分数, 维度为[beam_width, 1]
        global_scores = torch.FloatTensor([0.0] + [-float('inf')] * (self.beam_width - 1))
        global_scores = global_scores.to(enc_states.device).unsqueeze(1)

        # 定义结束标记，任意分支出现停止标记1则解码结束， 维度为 [beam_width, 1]
        stop_or_not = torch.zeros_like(global_scores, dtype=torch.bool)

        def decode_step(pred_hist, scores, flag):
            """ decode an utterance in a stepwise way"""
            batch_log_probs = self.model.decoder.recognize(pred_hist, beam_enc_states, beam_enc_mask).detach()
            last_k_scores, last_k_preds = batch_log_probs.topk(self.beam_width)  # 计算每个分支最大的beam_width个标记
            # 分数更新
            scores = scores + last_k_scores
            scores = scores.view(self.beam_width * self.beam_width)
            # 保存所有路径中的前k个路径
            scores, best_k_indices = torch.topk(scores, k=self.beam_width)
            scores = scores.view(-1, 1)
            # 更新预测
            pred = pred_hist.repeat([self.beam_width, 1])
            pred = torch.cat((pred, last_k_preds.view(-1, 1)), dim=1)
            best_k_preds = torch.index_select(pred, dim=0, index=best_k_indices)
            # 判断最后一个是不是结束标记
            flag = torch.eq(best_k_preds[:, -1], args.vocab['<EOS>']).view(-1, 1)
            return best_k_preds, scores, flag

        with torch.no_grad():

            for _ in range(1, self.max_len + 1):
                preds, global_scores, stop_or_not = decode_step(preds, global_scores, stop_or_not)
                # 判断是否停止，任意分支解码到结束标记或者达到最大解码步数则解码停止
                if stop_or_not.sum() > 0: break

            max_indices = torch.argmax(global_scores, dim=-1).long()
            preds = preds.view(self.beam_width, -1)
            best_preds = torch.index_select(preds, dim=0, index=max_indices)

            # 删除起始标记 BOS
            best_preds = best_preds[0, 1:]
            results = []
            for i in best_preds:
                if int(i) == args.vocab['<EOS>']:
                    break
                results.append(self.unit2char[int(i)])
        return ''.join(results)
