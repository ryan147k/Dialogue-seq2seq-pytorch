#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2020/08/11 Tue
# TIME: 15:12:45
# DESCRIPTION:
import torch as t
from torch.utils.data import Dataset
from config import opt
import json
from utils import word2ix


class Mydataset(Dataset):
    def __init__(self):
        super(Mydataset, self).__init__()
        fin = open(opt.train_file_path, 'r', encoding='utf-8')
        self.conversations = json.loads(fin.read())['conversations']
        self.vocab_dic = json.loads(
            open(opt.vocab_file_path, 'r', encoding='utf-8').read()
        )
        self.sample_num = [(len(item) - 1) for item in self.conversations]

    def __getitem__(self, index):
        cnt = 0
        # 找到样本(x, y)
        for i, num in enumerate(self.sample_num):
            cnt += num
            if index < cnt:
                index = num - (cnt - index)
                x = list(self.conversations[i][index])
                y = list(self.conversations[i][index+1])
                break

        # 构造encoder input
        encoder_input = self._reconsturct(x)
        # 构造decoder input
        decoder_input = y.copy()
        decoder_input.insert(0, '</s>')
        decoder_input = self._reconsturct(decoder_input)
        # 构造decoder output
        decoder_output = decoder_input.copy()
        decoder_output.pop(0)
        decoder_output.append('</e>')

        # 转成数字
        encoder_input = word2ix(encoder_input)
        decoder_input = word2ix(decoder_input)
        decoder_output = word2ix(decoder_output)

        return t.LongTensor(encoder_input), t.LongTensor(decoder_input), t.LongTensor(decoder_output)

    def __len__(self):
        return sum(self.sample_num)

    def _reconsturct(self, input: list):
        """对句子填充 or 截断"""
        assert(type(input) is list)
        if len(input) > opt.max_length:
            input = input[:opt.max_length]
        else:
            for _ in range(opt.max_length - len(input)):
                input.append('_pad')
        return input
