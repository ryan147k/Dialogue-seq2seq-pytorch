#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2020/08/11 Tue
# TIME: 15:15:37
# DESCRIPTION: 处理文件的一些操作
import json
import os
ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


def get_vocab_file(filepath):
    """生成词表文件"""
    fin = open(filepath, 'r', encoding='utf-8')
    conversations = json.loads(fin.read())['conversations']

    vocab = {
        '_unk': 0,
        '_pad': 1,
        '</s>': 2,  # 开始符
        '</e>': 3   # 结束符
    }
    num = 4
    for conv in conversations:
        for sentence in conv:
            for word in sentence:
                if word not in vocab.keys():
                    vocab[word] = num
                    num += 1
    fout = open(os.path.join(ROOT, 'resource/vocab.json'), 'w', encoding='utf-8')
    fout.write(json.dumps(vocab, ensure_ascii=False, indent=2))


# get_vocab_file(os.path.join(ROOT, 'resource/conversations.corpus.json'))
