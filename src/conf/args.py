# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# project_dir: .../2020-Biendata-MagicSpeechNet-Family-Scene-Chinese-Speech-Data-Set-Challenge
project_dir = '/'.join(os.path.abspath('..').split('/')[:-1])
# src dir: .../2020-Biendata-MagicSpeechNet-Family-Scene-Chinese-Speech-Data-Set-Challenge/src
src_dir = os.path.join(project_dir, 'src')
# 当前文件路径: .../2020-Biendata-MagicSpeechNet-Family-Scene-Chinese-Speech-Data-Set-Challenge/src/conf
conf_dir = os.path.join(src_dir, 'conf')
# Magicdata dir
magicdata_dir = os.path.join(project_dir, 'Magicdata')
# audio dir
audio_dir = os.path.join(magicdata_dir, 'audio')
# transcription dir
transcription_dir = os.path.join(magicdata_dir, 'transcription')

# data dir
data_dir = os.path.join(project_dir, 'data')

# -* train *-
# data train dir
data_train_dir = os.path.join(data_dir, 'train')
# data train wav dir
data_train_wav_dir = os.path.join(data_train_dir, 'wav')
# data train transcrpts path
data_train_transcrpts_path = os.path.join(data_train_dir, 'trainscrpts.txt')
# data train wav path
data_train_wav_path = os.path.join(data_train_dir, 'wav.scp')

# -* dev *-
# data dev dir
data_dev_dir = os.path.join(data_dir, 'dev')
# data dev wav dir
data_dev_wav_dir = os.path.join(data_dev_dir, 'wav')
# data dev transcrpts path
data_dev_transcrpts_path = os.path.join(data_dev_dir, 'trainscrpts.txt')
# data dev wav path
data_dev_wav_path = os.path.join(data_dev_dir, 'wav.scp')

# -* test *-
# data test dir
data_test_dir = os.path.join(data_dir, 'test')
# data test wav dir
data_test_wav_dir = os.path.join(data_test_dir, 'wav')
# data test wav path
data_test_wav_path = os.path.join(data_test_dir, 'wav.scp')

vocab_path = os.path.join(data_dir, 'vocab.txt')
vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
vocab_size = 3863   # 需要时刻注意是否更新