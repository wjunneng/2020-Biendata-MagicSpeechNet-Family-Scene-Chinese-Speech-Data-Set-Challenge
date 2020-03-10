# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

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
sample_submission_path = os.path.join(magicdata_dir, 'sample_submission.csv')
data_submission_path = os.path.join(data_dir, 'submission.csv')

vocab_path = os.path.join(data_dir, 'vocab.txt')
vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
# 需要时刻注意是否更新
vocab_size = 3864

seed = 42

# 增强类型
# enhance = {'0': 'fbank',
#            '1': 'freq_mask_time_mask',
#            '2': 'fbank',
#            '3': 'freq_mask_time_mask',
#            '4': 'fbank',
#            '5': 'freq_mask_time_mask',
#            '6': 'fbank',
#            '7': 'freq_mask_time_mask',
#            '8': 'fbank',
#            '9': 'freq_mask_time_mask',
#            }

enhance = {'0': 'freq_mask_time_mask',
           '1': 'freq_mask_time_mask',
           '2': 'freq_mask_time_mask',
           '3': 'freq_mask_time_mask',
           '4': 'freq_mask_time_mask',
           '5': 'freq_mask_time_mask',
           '6': 'freq_mask_time_mask',
           '7': 'freq_mask_time_mask',
           '8': 'freq_mask_time_mask',
           '9': 'freq_mask_time_mask',
           }

# -* model arguments *-
# 模型迭代次数
total_epochs = 40
# 模型维度
model_size = 320
# 注意力机制头数
n_heads = 4
# 编码器层数
num_enc_blocks = 6
# 解码器层数
num_dec_blocks = 6
# 残差连接丢弃率
residual_dropout_rate = 0.1
# 是否共享编码器词嵌入的权重
share_embedding = True
# 热身步数
warmup_steps = 12000
# 学习率因子
lr_factor = 1.0
# 梯度累计步数
accu_grads_steps = 8

# 输入特征维度
input_size = 40
# 指定批大小 [batch:16->Global Step:1280]
batch_size = 16
N_FFT = 512
use_mel_scale = True
use_specaug = False
