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







