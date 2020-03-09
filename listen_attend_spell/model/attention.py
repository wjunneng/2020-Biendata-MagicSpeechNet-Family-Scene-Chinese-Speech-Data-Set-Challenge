# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an self attention mechanism on the output features from the decoder.
    「A Structured Self-Attentive Sentence Embedding」 Paper
    https://arxiv.org/abs/1703.03130

    .. math::
        \begin{array}{ll}
        x = encoder_outputs*decoder_output \\
        attn_score = exp(x_i) / sum_j exp(x_j) \\
        output = \tanh(w * (attn_score * encoder_outputs) + b * output)
        \end{array}

    Args:
        decoder_hidden_size (int): The number of expected features in the output

    Inputs: decoder_output, encoder_outputs
        - **decoder_output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **encoder_outputs** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Examples::

        > attention = Attention(hidden_size)
        > output = attention(decoder_output, encoder_outputs)
    """

    def __init__(self, decoder_hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(decoder_hidden_size * 2, decoder_hidden_size)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)
        input_size = encoder_outputs.size(1)

        # get attention score
        attn_score = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        # get attention distribution
        attn_distribution = F.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # get context vector
        context = torch.bmm(attn_distribution, encoder_outputs)
        # concatenate attn_val & decoder_output
        combined = torch.cat((context, decoder_output), dim=2)
        output = torch.tanh(self.w(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output
