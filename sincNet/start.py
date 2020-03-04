# -*- coding:utf-8 -*-

import os
import sys

os.chdir(sys.path[0])
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from sincNet.core.module import MLP
from sincNet.core.module import SincNet as CNN
from sincNet.lib.util import Util, DataUtil
from sincNet.conf import args

# [data]
tr_lst = args.data_train_wav_path
te_lst = args.data_dev_wav_path
pt_file = args.pt_file
class_dict_file = args.vocab_path
data_folder = args.data_dir + '/'
output_folder = args.data_model_dir

# [windowing]
fs = int(args.fs)
cw_len = int(args.cw_len)
cw_shift = int(args.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, args.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, args.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, args.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = DataUtil.str_to_bool(args.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = DataUtil.str_to_bool(args.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(DataUtil.str_to_bool, args.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(DataUtil.str_to_bool, args.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, args.cnn_act.split(',')))
cnn_drop = list(map(float, args.cnn_drop.split(',')))

# [dnn]
fc_lay = list(map(int, args.fc_lay.split(',')))
fc_drop = list(map(float, args.fc_drop.split(',')))
fc_use_laynorm_inp = DataUtil.str_to_bool(args.fc_use_laynorm_inp)
fc_use_batchnorm_inp = DataUtil.str_to_bool(args.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(DataUtil.str_to_bool, args.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(DataUtil.str_to_bool, args.fc_use_laynorm.split(',')))
fc_act = list(map(str, args.fc_act.split(',')))

# [class]
class_lay = list(map(int, args.class_lay.split(',')))
class_drop = list(map(float, args.class_drop.split(',')))
class_use_laynorm_inp = DataUtil.str_to_bool(args.class_use_laynorm_inp)
class_use_batchnorm_inp = DataUtil.str_to_bool(args.class_use_batchnorm_inp)
class_use_batchnorm = list(map(DataUtil.str_to_bool, args.class_use_batchnorm.split(',')))
class_use_laynorm = list(map(DataUtil.str_to_bool, args.class_use_laynorm.split(',')))
class_act = list(map(str, args.class_act.split(',')))

# [optimization]
lr = float(args.lr)
batch_size = int(args.batch_size)
N_epochs = int(args.N_epochs)
N_batches = int(args.N_batches)
N_eval_epoch = int(args.N_eval_epoch)
seed = int(args.seed)

# training list
wav_lst_tr = Util.read_list(tr_lst)
snt_tr = len(wav_lst_tr)

# test list
wav_lst_te = Util.read_list(te_lst)
snt_te = len(wav_lst_te)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = 128

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop,
            }

CNN_net = CNN(CNN_arch)
CNN_net.cuda()

# Loading label dictionary
lab_dict = {}
with open(class_dict_file, mode='r', encoding='utf-8') as file:
    for line in file.readlines():
        key = line.split(' ')[0]
        value = line.strip('\n').split(' ')[1]
        lab_dict[key] = int(value)

DNN1_arch = {'input_dim': CNN_net.out_dim,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act,
             }

DNN1_net = MLP(DNN1_arch)
DNN1_net.cuda()

DNN2_arch = {'input_dim': fc_lay[-1],
             'fc_lay': class_lay,
             'fc_drop': class_drop,
             'fc_use_batchnorm': class_use_batchnorm,
             'fc_use_laynorm': class_use_laynorm,
             'fc_use_laynorm_inp': class_use_laynorm_inp,
             'fc_use_batchnorm_inp': class_use_batchnorm_inp,
             'fc_act': class_act,
             }

DNN2_net = MLP(DNN2_arch)
DNN2_net.cuda()

if pt_file is not None:
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
    DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

for epoch in range(N_epochs):

    test_flag = 0
    CNN_net.train()
    DNN1_net.train()
    DNN2_net.train()

    loss_sum = 0
    err_sum = 0

    for i in range(N_batches):
        [inp, lab] = Util.create_batches_rnd(batch_size, data_folder, wav_lst_tr, snt_tr, wlen, lab_dict, 0.2)
        pout = DNN2_net(DNN1_net(CNN_net(inp)))

        pred = torch.max(pout, dim=1)[1]
        loss = cost(pout, lab.long())
        err = torch.mean((pred != lab.long()).float())

        optimizer_CNN.zero_grad()
        optimizer_DNN1.zero_grad()
        optimizer_DNN2.zero_grad()

        loss.backward()
        optimizer_CNN.step()
        optimizer_DNN1.step()
        optimizer_DNN2.step()

        loss_sum = loss_sum + loss.detach()
        err_sum = err_sum + err.detach()

    loss_tot = loss_sum / N_batches
    err_tot = err_sum / N_batches

    # Full Validation  new
    if epoch % N_eval_epoch == 0:

        CNN_net.eval()
        DNN1_net.eval()
        DNN2_net.eval()
        test_flag = 1
        loss_sum = 0
        err_sum = 0
        err_sum_snt = 0

        with torch.no_grad():
            for i in range(snt_te):

                # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
                # signal=signal.astype(float)/32768

                [signal, fs] = sf.read(data_folder + wav_lst_te[i].upper())

                signal = torch.from_numpy(signal).float().cuda().contiguous()
                lab_batch = lab_dict[wav_lst_te[i]]

                # split signals into chunks
                beg_samp = 0
                end_samp = wlen

                N_fr = int((signal.shape[0] - wlen) / (wshift))

                sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
                lab = Variable((torch.zeros(N_fr + 1) + lab_batch).cuda().contiguous().long())
                pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().cuda().contiguous())
                count_fr = 0
                count_fr_tot = 0
                while end_samp < signal.shape[0]:
                    sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                    beg_samp = beg_samp + wshift
                    end_samp = beg_samp + wlen
                    count_fr = count_fr + 1
                    count_fr_tot = count_fr_tot + 1
                    if count_fr == Batch_dev:
                        inp = Variable(sig_arr)
                        pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
                        count_fr = 0
                        sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()

                if count_fr > 0:
                    inp = Variable(sig_arr[0:count_fr])
                    pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))

                pred = torch.max(pout, dim=1)[1]
                loss = cost(pout, lab.long())
                err = torch.mean((pred != lab.long()).float())

                [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                err_sum_snt = err_sum_snt + (best_class != lab[0]).float()

                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()

            err_tot_dev_snt = err_sum_snt / snt_te
            loss_tot_dev = loss_sum / snt_te
            err_tot_dev = err_sum / snt_te

        print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
            epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        with open(output_folder + "/res.res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                      'DNN1_model_par': DNN1_net.state_dict(),
                      'DNN2_model_par': DNN2_net.state_dict(),
                      }
        torch.save(checkpoint, output_folder + '/model_raw.pkl')

    else:
        print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))
