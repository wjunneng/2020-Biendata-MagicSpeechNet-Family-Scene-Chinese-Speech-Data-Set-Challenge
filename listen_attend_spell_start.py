# -*- coding: utf-8 -*-
"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

- Korean Speech Recognition
Team: Kai.Lib
    ● Team Member
        ○ Kim-Soo-Hwan: KW University elcomm. senior
        ○ Bae-Se-Young: KW University elcomm. senior
        ○ Won-Cheol-Hwang: KW University elcomm. senior

Model Architecture:
    ● Listen, Attend and Spell (Seq2seq with Attention)

Data:
    ● A.I Hub Dataset

Score:
    ● CRR: Character Recognition Rate
    ● CER: Character Error Rate based on Edit Distance

GitHub repository : https://github.com/sh951011/Korean-ASR
Documentation : https://sh951011.github.io/Korean-Speech-Recognition/index.html
"""
# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import queue
import torch.nn as nn
import torch.optim as optim
import random
import torch
import time
import math

from listen_attend_spell.model.speller import Speller
from listen_attend_spell.model.listener import Listener
from listen_attend_spell.model.listenAttendSpell import ListenAttendSpell
from listen_attend_spell.package.dataset import BaseDataset
from listen_attend_spell.package import args
from listen_attend_spell.package.evaluator import evaluate
from listen_attend_spell.package.hparams import HyperParams
from listen_attend_spell.package.loader import load_data_list, MultiLoader, BaseDataLoader
from listen_attend_spell.package.loss import LabelSmoothingLoss
from listen_attend_spell.package.trainer import supervised_train
from listen_attend_spell.package.utils import save_epoch_result

import logging

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


def train():
    hparams = HyperParams()

    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    cuda = hparams.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    listener = Listener(
        feat_size=args.input_dim,
        hidden_size=hparams.hidden_size,
        dropout_p=hparams.dropout,
        layer_size=hparams.listener_layer_size,
        bidirectional=hparams.use_bidirectional,
        rnn_cell='gru',
        use_pyramidal=hparams.use_pyramidal
    )
    speller = Speller(
        vocab_size=len(args.char2id),
        max_len=hparams.max_len,
        k=8,
        hidden_size=hparams.hidden_size << (1 if hparams.use_bidirectional else 0),
        sos_id=args.SOS_TOKEN,
        eos_id=args.EOS_TOKEN,
        layer_size=hparams.speller_layer_size,
        rnn_cell='gru',
        dropout_p=hparams.dropout,
        use_attention=hparams.use_attention,
        device=device
    )
    model = ListenAttendSpell(listener, speller, use_pyramidal=hparams.use_pyramidal)
    model.flatten_parameters()
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=hparams.init_lr)
    if hparams.use_label_smooth:
        criterion = LabelSmoothingLoss(len(args.char2id), ignore_index=args.PAD_TOKEN, smoothing=0.1, dim=-1).to(device)
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=args.PAD_TOKEN).to(device)

    vocab_dict = {}
    with open(args.vocab_path, mode='r', encoding='utf-8') as file:
        for line in file:
            vocab_dict[line.split(' ')[0]] = int(line.split(' ')[1].strip('\n'))

    train_utt_ids, train_audio_paths, train_label_texts = load_data_list(wav_path=args.data_train_wav_path,
                                                                         text_path=args.data_train_text_path,
                                                                         vocab_dict=vocab_dict)

    dev_utt_ids, dev_audio_paths, dev_label_texts = load_data_list(wav_path=args.data_dev_wav_path,
                                                                   text_path=args.data_dev_text_path,
                                                                   vocab_dict=vocab_dict)
    # seperating the train dataset by the number of workers
    train_dataset_list = []
    for idx in range(hparams.worker_num):
        train_dataset_list.append(
            BaseDataset(audio_paths=train_audio_paths,
                        label_paths=train_label_texts,
                        input_reverse=hparams.input_reverse,
                        use_augment=hparams.use_augment,
                        batch_size=hparams.batch_size,
                        augment_ratio=hparams.augment_ratio,
                        pack_by_length=False)
        )

    valid_dataset = BaseDataset(
        audio_paths=dev_audio_paths,
        label_paths=dev_label_texts,
        batch_size=hparams.batch_size,
        input_reverse=hparams.input_reverse,
        use_augment=False,
        pack_by_length=False
    )

    logger.info('start')
    train_begin = time.time()

    total_time_step = math.ceil((len(train_audio_paths)) / hparams.batch_size)
    for epoch in range(hparams.max_epochs):
        train_queue = queue.Queue(hparams.worker_num << 1)
        for train_dataset in train_dataset_list:
            train_dataset.shuffle()
        train_loader = MultiLoader(train_dataset_list, train_queue, hparams.batch_size, hparams.worker_num)
        train_loader.start()
        train_loss = supervised_train(
            model=model,
            total_time_step=total_time_step,
            hparams=hparams,
            queue=train_queue,
            criterion=criterion,
            epoch=epoch,
            optimizer=optimizer,
            device=device,
            train_begin=train_begin,
            worker_num=hparams.worker_num,
            print_time_step=100,
            teacher_forcing_ratio=hparams.teacher_forcing
        )
        torch.save(model, os.path.join(args.data_model_dir, "epoch%s.pt" % str(epoch)))
        logger.info('Epoch %d (Training) Loss %0.4f' % (epoch, train_loss))
        train_loader.join()
        valid_queue = queue.Queue(hparams.worker_num << 1)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, hparams.batch_size, 0)
        valid_loader.start()

        valid_loss, valid_cer = evaluate(model, valid_queue, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, valid_loss, valid_cer))

        valid_loader.join()

        save_epoch_result(train_result=[args.train_dict, train_loss],
                          valid_result=[args.valid_dict, valid_loss, valid_cer])
        logger.info('Epoch %d Training result saved as a csv file complete !!' % epoch)


# def predict():
#     """ Test for Model Performance """
#     hparams = HyperParams()
#     hparams.logger_hparams()
#     cuda = hparams.use_cuda and torch.cuda.is_available()
#     device = torch.device('cuda' if cuda else 'cpu')
#     model = torch.load("model_path")
#     model.set_beam_size(k=8)
#
#     audio_paths, label_paths = load_data_list(data_list_path=TEST_LIST_PATH, dataset_path=DATASET_PATH)
#
#     target_dict = load_targets(label_paths)
#     logger.info('start')
#
#     test_dataset = BaseDataset(
#         audio_paths=audio_paths[:],
#         label_paths=label_paths[:],
#         sos_id=SOS_TOKEN,
#         eos_id=EOS_TOKEN,
#         target_dict=target_dict,
#         input_reverse=hparams.input_reverse,
#         use_augment=False
#     )
#
#     test_queue = queue.Queue(hparams.worker_num << 1)
#     test_loader = BaseDataLoader(test_dataset, test_queue, hparams.batch_size, 0)
#     test_loader.start()
#
#     logger.info('evaluate() start')
#     total_distance = 0
#     total_length = 0
#     total_sent_num = 0
#
#     model.eval()
#
#     with torch.no_grad():
#         while True:
#             feats, targets, feat_lengths, script_lengths = queue.get()
#             if feats.shape[0] == 0:
#                 break
#
#             feats = feats.to(device)
#             targets = targets.to(device)
#             target = targets[:, 1:]
#
#             model.module.flatten_parameters()
#             y_hat, _ = model(
#                 feats=feats,
#                 targets=targets,
#                 teacher_forcing_ratio=0.0,
#                 use_beam_search=True
#             )
#             distance, length = get_distance(target, y_hat, id2char, EOS_TOKEN)
#             total_distance += distance
#             total_length += length
#             total_sent_num += target.size(0)
#
#     CER = total_distance / total_length
#     logger.info('test() completed')
#     return CER


if __name__ == '__main__':
    # if you use Multi-GPU, delete this line
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logger.info("device : %s" % torch.cuda.get_device_name(0))
    logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
    logger.info("CUDA version : %s" % torch.version.cuda)
    logger.info("PyTorch version : %s" % torch.__version__)

    train()
