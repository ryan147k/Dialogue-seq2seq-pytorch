#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2020/08/11 Tue
# TIME: 16:06:57
# DESCRIPTION: 工具函数
from config import opt
import json


def word2ix(word_list):
    """
    句子转成数字列表
    """
    vocab_dic = json.loads(
        open(opt.vocab_file_path, 'r', encoding='utf-8').read()
    )

    ix = []
    for word in word_list:
        if word not in vocab_dic.keys():
            word = '_unk'
        ix.append(vocab_dic[word])
    return ix


def ix2word(index_list):
    """
    数字列表转句子
    """
    vocab_dic = json.loads(
        open(opt.vocab_file_path, 'r', encoding='utf-8').read()
    )
    vocab_dic_reverse = {}
    for k, v in vocab_dic.items():
        vocab_dic_reverse[v] = k

    sentence = ''
    for ix in index_list:
        sentence += vocab_dic_reverse[ix]
    return sentence
