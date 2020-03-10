# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
from listen_attend_spell.package.loader import load_label

# project dir
project_dir = '/home/wjunneng/Ubuntu/2020-Biendata-MagicSpeechNet-Family-Scene-Chinese-Speech-Data-Set-Challenge'

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
# data train text
data_train_text_path = os.path.join(data_train_dir, 'text.txt')

# -* dev *-
# data dev dir
data_dev_dir = os.path.join(data_dir, 'dev')
# data dev wav dir
data_dev_wav_dir = os.path.join(data_dev_dir, 'wav')
# data dev transcrpts path
data_dev_transcrpts_path = os.path.join(data_dev_dir, 'trainscrpts.txt')
# data dev wav path
data_dev_wav_path = os.path.join(data_dev_dir, 'wav.scp')
# data dev text
data_dev_text_path = os.path.join(data_dev_dir, 'text.txt')

# -* test *-
# data test dir
data_test_dir = os.path.join(data_dir, 'test')
# data test wav dir
data_test_wav_dir = os.path.join(data_test_dir, 'wav')
# data test wav path
data_test_wav_path = os.path.join(data_test_dir, 'wav.scp')

# -* model *-
# data model dir
data_model_dir = os.path.join(data_dir, 'model')

# -* result *-
# data result path
data_reault_path = os.path.join(data_dir, 'result.csv')
data_train_result_path = os.path.join(data_dir, 'train_result.csv')
data_dev_result_path = os.path.join(data_dir, 'dev_result.csv')
sample_submission_path = os.path.join(magicdata_dir, 'sample_submission.csv')
data_submission_path = os.path.join(data_dir, 'submission.csv')
vocab_path = os.path.join(data_dir, 'vocab.txt')

char2id, id2char = load_label(vocab_path, encoding='utf-8')  # 2,040
# if you want to use total character label
# change => char2id, id2char = load_label('./data/label/test_labels.csv', encoding='utf-8') # 2,337
SOS_TOKEN = int(char2id['<BOS>'])
EOS_TOKEN = int(char2id['<EOS>'])
PAD_TOKEN = int(char2id['<PAD>'])
train_dict = {'loss': [], 'cer': []}
valid_dict = {'loss': [], 'cer': []}

input_dim = 48
