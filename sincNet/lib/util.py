# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import torch
import math
import shutil

import soundfile as sf
from torch.autograd import Variable
import configparser as ConfigParser
from optparse import OptionParser
from torch import nn
import numpy as np
import scipy

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Util(object):
    @staticmethod
    def read_list(file_path):
        with open(file_path, "r", encoding='ISO-8859-1') as file:
            list_sig = []
            for x in file.readlines():
                list_sig.append(x.rstrip())

        return list_sig

    @staticmethod
    def copy_folder(in_folder, out_folder):
        if not (os.path.isdir(out_folder)):
            shutil.copytree(in_folder, out_folder, ignore=Util.ig_f)

        return True

    @staticmethod
    def ig_f(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    @staticmethod
    def flip(x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.contiguous()
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:,
            getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]

        return x.view(xsize)

    @staticmethod
    def sinc(band, t_right):
        y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
        y_left = Util.flip(y_right, 0)

        tmp = Variable(torch.ones(1)).to(DEVICE)
        y = torch.cat([y_left, tmp, y_right])

        return y

    @staticmethod
    def act_fun(act_type):
        if act_type == "relu":
            return nn.ReLU()

        if act_type == "tanh":
            return nn.Tanh()

        if act_type == "sigmoid":
            return nn.Sigmoid()

        if act_type == "leaky_relu":
            return nn.LeakyReLU(0.2)

        if act_type == "elu":
            return nn.ELU()

        if act_type == "softmax":
            return nn.LogSoftmax(dim=1)

        if act_type == "linear":
            # initializzed like this, but not used in forward!
            return nn.LeakyReLU(1)

    @staticmethod
    def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp):
        # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
        sig_batch = np.zeros([batch_size, wlen])
        lab_batch = np.zeros(batch_size)

        snt_id_arr = np.random.randint(N_snt, size=batch_size)

        rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

        for i in range(batch_size):

            # select a random sentence from the list
            # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
            # signal=signal.astype(float)/32768

            [signal, fs] = sf.read(wav_lst[snt_id_arr[i]].split(' ')[-1])

            # accesing to a random chunk
            snt_len = signal.shape[0]
            # randint(0, snt_len-2*wlen-1)
            snt_beg = np.random.randint(snt_len - wlen - 1)
            snt_end = snt_beg + wlen

            channels = len(signal.shape)
            if channels == 2:
                print('WARNING: stereo to mono: ' + data_folder + wav_lst[snt_id_arr[i]])
                signal = signal[:, 0]

            sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
            print(lab_dict)
            lab_batch[i] = lab_dict[wav_lst[snt_id_arr[i]].split(' ')[-1]]

        inp = Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
        lab = Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())

        return inp, lab


class DataUtil(object):
    @staticmethod
    def str_to_bool(s):
        if s == 'True':
            return True
        elif s == 'False':
            return False
        else:
            raise ValueError

    @staticmethod
    def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp):
        # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
        sig_batch = np.zeros([batch_size, wlen])
        lab_batch = np.zeros(batch_size)

        snt_id_arr = np.random.randint(N_snt, size=batch_size)

        rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

        for i in range(batch_size):
            # select a random sentence from the list  (joint distribution)
            [fs, signal] = scipy.io.wavfile.read(data_folder + wav_lst[snt_id_arr[i]])
            signal = signal.astype(float) / 32768

            # accesing to a random chunk
            snt_len = signal.shape[0]
            snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
            snt_end = snt_beg + wlen

            sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
            lab_batch[i] = lab_dict[wav_lst[snt_id_arr[i]]]

        inp = torch.from_numpy(sig_batch).float().cuda().contiguous()  # Current Frame
        lab = torch.from_numpy(lab_batch).float().cuda().contiguous()

        return inp, lab

    @staticmethod
    def read_conf_inp(cfg_file):
        parser = OptionParser()
        (options, args) = parser.parse_args()

        Config = ConfigParser.ConfigParser()
        Config.read(cfg_file)

        # [data]
        options.tr_lst = Config.get('data', 'tr_lst')
        options.te_lst = Config.get('data', 'te_lst')
        options.lab_dict = Config.get('data', 'lab_dict')
        options.data_folder = Config.get('data', 'data_folder')
        options.output_folder = Config.get('data', 'output_folder')
        options.pt_file = Config.get('data', 'pt_file')

        # [windowing]
        options.fs = Config.get('windowing', 'fs')
        options.cw_len = Config.get('windowing', 'cw_len')
        options.cw_shift = Config.get('windowing', 'cw_shift')

        # [cnn]
        options.cnn_N_filt = Config.get('cnn', 'cnn_N_filt')
        options.cnn_len_filt = Config.get('cnn', 'cnn_len_filt')
        options.cnn_max_pool_len = Config.get('cnn', 'cnn_max_pool_len')
        options.cnn_use_laynorm_inp = Config.get('cnn', 'cnn_use_laynorm_inp')
        options.cnn_use_batchnorm_inp = Config.get('cnn', 'cnn_use_batchnorm_inp')
        options.cnn_use_laynorm = Config.get('cnn', 'cnn_use_laynorm')
        options.cnn_use_batchnorm = Config.get('cnn', 'cnn_use_batchnorm')
        options.cnn_act = Config.get('cnn', 'cnn_act')
        options.cnn_drop = Config.get('cnn', 'cnn_drop')

        # [dnn]
        options.fc_lay = Config.get('dnn', 'fc_lay')
        options.fc_drop = Config.get('dnn', 'fc_drop')
        options.fc_use_laynorm_inp = Config.get('dnn', 'fc_use_laynorm_inp')
        options.fc_use_batchnorm_inp = Config.get('dnn', 'fc_use_batchnorm_inp')
        options.fc_use_batchnorm = Config.get('dnn', 'fc_use_batchnorm')
        options.fc_use_laynorm = Config.get('dnn', 'fc_use_laynorm')
        options.fc_act = Config.get('dnn', 'fc_act')

        # [class]
        options.class_lay = Config.get('class', 'class_lay')
        options.class_drop = Config.get('class', 'class_drop')
        options.class_use_laynorm_inp = Config.get('class', 'class_use_laynorm_inp')
        options.class_use_batchnorm_inp = Config.get('class', 'class_use_batchnorm_inp')
        options.class_use_batchnorm = Config.get('class', 'class_use_batchnorm')
        options.class_use_laynorm = Config.get('class', 'class_use_laynorm')
        options.class_act = Config.get('class', 'class_act')

        # [optimization]
        options.lr = Config.get('optimization', 'lr')
        options.batch_size = Config.get('optimization', 'batch_size')
        options.N_epochs = Config.get('optimization', 'N_epochs')
        options.N_batches = Config.get('optimization', 'N_batches')
        options.N_eval_epoch = Config.get('optimization', 'N_eval_epoch')
        options.seed = Config.get('optimization', 'seed')

        return options
