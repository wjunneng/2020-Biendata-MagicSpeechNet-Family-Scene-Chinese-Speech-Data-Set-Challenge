# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from collections import namedtuple
import torchaudio as ta
from torchaudio import transforms
import matplotlib.pyplot as plt
import random
import torch

from seq2seq.lib.sparse_image_warp import sparse_image_warp


class SpecAugment(object):
    @staticmethod
    def tfm_spectro(ad, to_db_scale=False, n_fft=1024, ws=None, hop=None, f_min=0.0, f_max=-80, pad=0, n_mels=128):
        # We must reshape signal for torchaudio to generate the spectrogram.
        mel = transforms.MelSpectrogram(sample_rate=ad.sr, n_mels=n_mels, n_fft=n_fft, win_length=ws, hop_length=hop,
                                        f_min=f_min, f_max=f_max, pad=pad, )(ad.sig.reshape(1, -1))
        # swap dimension, mostly to look sane to a human.
        mel = mel.permute(0, 2, 1)

        if to_db_scale:
            mel = transforms.AmplitudeToDB(stype='magnitude', top_db=f_max).forward(mel)

        return mel

    @staticmethod
    def time_warp(spec, W=5):
        num_rows = spec.shape[1]
        spec_len = spec.shape[2]
        device = spec.device

        y = num_rows // 2
        horizontal_line_at_ctr = spec[0][y]
        assert len(horizontal_line_at_ctr) == spec_len

        point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
        assert isinstance(point_to_warp, torch.Tensor)

        # Uniform distribution from (0,W) with chance to be up to W negative
        dist_to_warp = random.randrange(-W, W)
        src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                             torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
        warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)

        return warped_spectro.squeeze(3)

    @staticmethod
    def tensor_to_img(spectrogram):
        plt.figure(figsize=(14, 1))  # arbitrary, looks good on my screen.
        plt.imshow(spectrogram[0])
        plt.show()
        print(spectrogram.shape)


# if __name__ == '__main__':
#     path = '/home/wjunneng/Ubuntu/2020-Biendata-MagicSpeechNet-Family-Scene-Chinese-Speech-Data-Set-Challenge/data/test/wav/MDT_Conversation_003-001.wav'
#     # path = '/home/wjunneng/Ubuntu/2020-Biendata-MagicSpeechNet-Family-Scene-Chinese-Speech-Data-Set-Challenge/data/party-crowd.wav'
#     SpecAugment(wav_path=path).main()
