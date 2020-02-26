# -*- coding:utf-8 -*-
import os
import sys
import time

os.chdir(sys.path[0])

import torch
from torch.utils.data import DataLoader
from src.lib.util import AudioDataset, DataUtil, Util
from src.conf import args
from src.core.module import Transformer, Recognizer


class Run(object):
    def __init__(self):
        self.args = args

    def train(self):
        # unit2idx
        unit2idx = {}
        with open(self.args.vocab_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                unit, idx = line.strip().split()
                unit2idx[unit] = int(idx)

        # 模型定义
        model = Transformer(input_size=self.args.input_size,
                                 vocab_size=self.args.vocab_size,
                                 d_model=self.args.model_size,
                                 n_heads=self.args.n_heads,
                                 d_ff=self.args.model_size * 4,
                                 num_enc_blocks=self.args.num_enc_blocks,
                                 num_dec_blocks=self.args.num_dec_blocks,
                                 residual_dropout_rate=self.args.residual_dropout_rate,
                                 share_embedding=self.args.share_embedding)
        if torch.cuda.is_available():
            model.cuda()  # 将模型加载到GPU中

        # 根据生成词表指定大小
        vocab_size = len(unit2idx)
        print('Set the size of vocab: %d' % vocab_size)

        # 将模型加载
        train_wav_list = [self.args.data_train_wav_path, self.args.data_dev_wav_path]
        train_text_list = [self.args.data_train_text_path, self.args.data_dev_wav_path]
        dataset = AudioDataset(train_wav_list, train_text_list, unit2idx=unit2idx)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2, pin_memory=False,
                                collate_fn=DataUtil.collate_fn)

        lr = Util.get_learning_rate(step=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

        if not os.path.exists(self.args.data_model_dir):
            os.makedirs(self.args.data_model_dir)

        global_step = 1
        step_loss = 0
        print('Begin to Train...')
        for epoch in range(self.args.total_epochs):
            print('***** epoch: %d *****' % epoch)
            for step, (_, inputs, targets) in enumerate(dataloader):
                # 将输入加载到GPU中
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                loss = model(inputs, targets)
                loss.backward()
                step_loss += loss.item()

                if (step + 1) % self.args.accu_grads_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                    optimizer.step()
                    optimizer.zero_grad()
                    if global_step % 10 == 0:
                        print('-Training-Epoch-%d, Global Step:%d, lr:%.8f, Loss:%.5f' % (
                            epoch, global_step, lr, step_loss / self.args.accu_grads_steps))
                    global_step += 1
                    step_loss = 0

                    # 学习率更新
                    lr = Util.get_learning_rate(global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

            # 模型保存
            checkpoint = model.state_dict()
            torch.save(checkpoint, os.path.join(self.args.data_model_dir, 'model.epoch.%d.pt' % epoch))
        print('Done!')

    def predict(self):
        # 定义评估模型
        eval_model = Transformer(input_size=self.args.input_size,
                                 vocab_size=self.args.vocab_size,
                                 d_model=self.args.model_size,
                                 n_heads=self.args.n_heads,
                                 d_ff=self.args.model_size * 4,
                                 num_enc_blocks=self.args.num_enc_blocks,
                                 num_dec_blocks=self.args.num_dec_blocks,
                                 residual_dropout_rate=0.0,
                                 share_embedding=self.args.share_embedding)

        if torch.cuda.is_available():
            eval_model.cuda()  # 将模型加载到GPU中

        # 将模型加载
        idx2unit = {}
        with open(self.args.vocab_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                unit, idx = line.strip().split()
                idx2unit[int(idx)] = unit

        wav_list = [self.args.data_test_wav_path]
        test_dataset = AudioDataset(wav_list)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                                      pin_memory=False, collate_fn=DataUtil.collate_fn)

        # checkpoints = torch.load('./model/model.pt', map_location=lambda storage, loc: storage)
        checkpoints = torch.load(os.path.join(self.args.data_model_dir, 'model.epoch.2.pt'))
        eval_model.load_state_dict(checkpoints)

        recognizer = Recognizer(eval_model, unit2char=idx2unit)

        csv_writer = open(self.args.data_reault_path, 'w', encoding='utf-8')
        csv_writer.write('id,words\n')
        print('Begin to decode test set!')
        total_num = len(test_dataloader)
        for step, (uttid, inputs) in enumerate(test_dataloader):
            # 将输入加载到GPU中
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            preds = recognizer.recognize(inputs)
            # 打印输出结果
            print('[%5d/%d] %s %s' % (step, total_num, uttid[0], preds))
            csv_writer.write(','.join([uttid[0], preds]) + '\n')
        csv_writer.close()
        print('Done!')


if __name__ == '__main__':
    start = time.clock()
    Run().train()
    current_time = time.clock()
    print('train using time: {}'.format(current_time - start))
    Run().predict()
    print('predict using time: {}'.format(time.clock() - current_time))
