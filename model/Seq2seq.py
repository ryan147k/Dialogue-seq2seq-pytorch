#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2020/08/11 Tue
# TIME: 16:58:28
# DESCRIPTION: Seq2Seq 模型
import torch as t
import torch.nn as nn
from .BasicModule import BasicModule


class Seq2seq(BasicModule):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super(Seq2seq, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # encoder
        self.encoder = nn.LSTM(emb_dim, hidden_size,
                               batch_first=True, bidirectional=True)
        # decoder
        self.decoder = nn.LSTMCell(emb_dim, hidden_size)
        self.liner = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y):
        """
        x: encoder输入 (batch_size, time_step, input_size)
        y: decoder输入 (batch_size, time_step, input_size)
        """
        x = self.embedding(x)
        y = self.embedding(y)
        # 对x进行encode
        _, (h_n, c_n) = self.encoder(x)

        output = t.Tensor()
        # encoder的输出是decoder的输入, 因为是双向, 先所以求和
        h_d, c_d = t.sum(h_n, dim=0), t.sum(c_n, dim=0)
        # 将每个time_step送入decoder cell
        for i in range(y.shape[1]):
            input = y[:, i]     # 保持batch_size, 取每个batch的第i个字
            h_d, c_d = self.decoder(input, (h_d, c_d))
            prob = self.liner(h_d).unsqueeze(dim=1)     # 线性分类
            output = t.cat((output, prob), dim=1)   # 将每个time_step的结果拼接
        return output

    def predict(self, x, start_symbol, time_step):
        """
        x: encoder输入 (batch_size, time_step, input_size)
        start_emb: 开始符的embedding
        time_step: decoder的输出step长度
        """
        x = self.embedding(x)
        # 对x进行encode
        _, (h_n, c_n) = self.encoder(x)

        output = t.LongTensor()
        # encoder的输出是decoder的输入, 因为是双向, 先所以求和
        h_d, c_d = t.sum(h_n, dim=0), t.sum(c_n, dim=0)
        # 将每个time_step送入decoder cell
        input = self.embedding(start_symbol)
        for i in range(time_step):
            h_d, c_d = self.decoder(input, (h_d, c_d))
            prob = self.liner(h_d).squeeze()     # 线性分类
            _, index = t.max(prob, dim=0)   # 找到最大概率索引
            index = index.unsqueeze(dim=0)
            input = self.embedding(index)  # 更新input
            output = t.cat((output, index))
        return output
