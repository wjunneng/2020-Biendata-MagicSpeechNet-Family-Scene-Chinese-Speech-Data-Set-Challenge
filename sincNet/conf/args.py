import os

# -*- coding:utf-8 -*-
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
vocab_size = 3863
pt_file = None

# -* model arguments *-
fs = 16000
cw_len = 200
cw_shift = 10

cnn_N_filt = '80,60,60'
cnn_len_filt = '251,5,5'
cnn_max_pool_len = '3,3,3'
cnn_use_laynorm_inp = 'True'
cnn_use_batchnorm_inp = 'False'
cnn_use_laynorm = 'True,True,True'
cnn_use_batchnorm = 'False,False,False'
cnn_act = 'leaky_relu,leaky_relu,leaky_relu'
cnn_drop = '0.0,0.0,0.0'

fc_lay = '2048,2048,2048'
fc_drop = '0.0,0.0,0.0'
fc_use_laynorm_inp = 'True'
fc_use_batchnorm_inp = 'False'
fc_use_batchnorm = 'True,True,True'
fc_use_laynorm = 'False,False,False'
fc_act = 'leaky_relu,leaky_relu,leaky_relu'

class_lay = '462'
class_drop = '0.0'
class_use_laynorm_inp = 'False'
class_use_batchnorm_inp = 'False'
class_use_batchnorm = 'False'
class_use_laynorm = 'False'
class_act = 'softmax'

lr = 0.001
batch_size = 128
N_epochs = 1500
N_batches = 800
N_eval_epoch = 8
seed = 1234
