#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2020/08/11 Tue
# TIME: 15:35:32
# DESCRIPTION: 配置文件
import os
ROOT = os.path.dirname(__file__)


class Config():
    max_length = 64
    vocab_size = 10997
    emb_dim = 128

    train_file_path = os.path.join(ROOT, 'resource/conversations.corpus.json')
    vocab_file_path = os.path.join(ROOT, 'resource/vocab.json')
    # load_path = os.path.join(ROOT, 'ckpts/Seq2seq_0811_23h45m20s.pth')
    load_path = None

    max_epoch = 50
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-5
    lr_decay = 0.95


opt = Config()
