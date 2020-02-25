# -*- coding=utf-8 -*-
import os
import sys
import json

from zhon import hanzi
import string
from src.conf import args

os.chdir(sys.path[0])


class Util(object):
    def __init__(self):
        pass

    @staticmethod
    def segment_wav(src_wav, tgt_wav, start_time, end_time):
        """
        音频切分

        :param src_wav:
        :param tgt_wav:
        :param start_time:
        :param end_time:
        :return:
        """
        span = end_time - start_time
        cmds = 'sox %s %s trim %f %f' % (src_wav, tgt_wav, start_time, span)
        os.system(cmds)

    @staticmethod
    def time2sed(t):
        """
        时间转秒
        :param t:
        :return:
        """
        h, m, s = t.strip().split(':')

        return float(h) * 3600 + float(m) * 60 + float(s)

    @staticmethod
    def load_json(json_file):
        """
        读取 json
        :param self:
        :return:
        """
        with open(file=json_file, mode='r') as file:
            json_lines = file.readlines()
            json_str = ''.join(json_lines).replace('\n', '').replace(' ', '').replace(',}', '}')

            return json.loads(json_str)

    @staticmethod
    def deal_train_dev():
        """
        处理训练集和验证集
        :return:
        """
        for name in ['train', 'dev']:
            save_dir = os.path.join(args.project_dir, 'data', name, 'wav')
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)

            seg_wav_list = []
            sub_audio_dir = os.path.join(args.audio_dir, name)
            for wav in os.listdir(sub_audio_dir):
                if wav[0] == '.':
                    continue  # 跳过隐藏文件

                suffix = None
                if name == 'dev':
                    parts = wav.split('_')
                    jf = '_'.join(parts[:-1]) + '.json'
                    suffix = parts[-1]
                else:
                    jf = wav[:-4] + '.json'

                utt_list = Util.load_json(os.path.join(args.transcription_dir, name, jf))
                for i in range(len(utt_list)):
                    utt_info = utt_list[i]
                    session_id = utt_info['session_id']

                    if name == 'dev':
                        tgt_id = session_id + '_' + str(i) + '_' + suffix
                    else:
                        tgt_id = session_id + '_' + str(i) + '.wav'

                    # 句子切分
                    start_time = Util.time2sed(utt_info['start_time']['original'])
                    end_time = Util.time2sed(utt_info['end_time']['original'])

                    src_wav = os.path.join(sub_audio_dir, wav)
                    tgt_wav = os.path.join(save_dir, tgt_id)
                    Util.segment_wav(src_wav=src_wav, tgt_wav=tgt_wav, start_time=start_time, end_time=end_time)
                    seg_wav_list.append((tgt_id, tgt_wav, utt_info['words']))

            with open(os.path.join(args.project_dir, 'data', name, 'wav.scp'), 'w') as ww:
                with open(os.path.join(args.project_dir, 'data', name, 'transcrpts.txt'), 'w', encoding='utf-8') as tw:
                    for uttid, wavdir, text in seg_wav_list:
                        ww.write(uttid + ' ' + wavdir + '\n')
                        tw.write(uttid + ' ' + text + '\n')

            print('prepare %s dataset done!' % name)

    @staticmethod
    def deal_test():
        """
        处理测试集
        :return:
        """
        if os.path.exists(args.data_test_wav_dir) is False:
            os.makedirs(args.data_test_wav_dir)

        seg_wav_list = []
        sub_audio_dir = os.path.join(args.audio_dir, 'test')
        for wav in os.listdir(sub_audio_dir):
            # 跳过隐藏文文件和IOS的音频文件
            if wav[0] == '.' or 'IOS' not in wav:
                continue

            jf = '_'.join(wav.split('_')[:-1]) + '.json'
            utt_list = Util.load_json(os.path.join(args.transcription_dir, 'test_no_ref_noise', jf))
            for i in range(len(utt_list)):
                utt_info = utt_list[i]
                session_id = utt_info['session_id']
                uttid = utt_info['uttid']

                # 如果句子已经标注，则跳过
                if 'word' in utt_info:
                    continue

                # 句子切分
                start_time = Util.time2sed(utt_info['start_time'])
                end_time = Util.time2sed(utt_info['end_time'])

                tgt_id = uttid + '.wav'
                src_wav = os.path.join(sub_audio_dir, wav)
                tgt_wav = os.path.join(args.data_test_wav_dir, tgt_id)
                Util.segment_wav(src_wav=src_wav, tgt_wav=tgt_wav, start_time=start_time, end_time=end_time)
                seg_wav_list.append((uttid, tgt_wav))

        with open(args.data_test_wav_path, 'w') as ww:
            for uttid, wavdir in seg_wav_list:
                ww.write(uttid + ' ' + wavdir + '\n')

        print('prepare test dataset done!')

    @staticmethod
    def text_normlization(seq):
        """
        文本归一化
        :param seq:
        :return:
        """
        new_seq = []
        for c in seq:
            if c == '+':
                # 文档中有加号，单独处理，避免删除
                new_seq.append(c)
            elif c in string.punctuation or c in hanzi.punctuation:
                # 删除全部的半角符号和全角符号
                continue
            else:
                if str(c).isalpha():
                    new_seq.append(c.lower())

        return ' '.join(new_seq)

    @staticmethod
    def normal_train_dev():
        """
        对训练和验证集进行标准化
        :return:
        """
        for name in ['train', 'dev']:
            with open(os.path.join(args.data_dir, name, 'transcrpts.txt'), 'r') as tr:
                with open(os.path.join(args.data_dir, name, 'text.txt'), 'w') as tw:
                    for line in tr:
                        parts = line.split()
                        uttid = parts[0]
                        seqs = ''.join(parts[1:])

                        # 直接跳过包含特殊标记的句子
                        if '[' in seqs:
                            continue

                        seqs = Util.text_normlization(seqs)
                        tw.write(uttid + ' ' + seqs + '\n')

            print('Normlize %s TEXT!' % name)


if __name__ == '__main__':
    # 处理训练集和验证集
    # Util.deal_train_dev()

    # 处理测试集
    # Util.deal_test()

    # 标准化训练和验证集
    Util.normal_train_dev()