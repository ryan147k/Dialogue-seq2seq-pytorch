#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2020/08/11 Tue
# TIME: 17:08:35
# DESCRIPTION:
import torch as t
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import Mydataset
from config import opt
from model import Seq2seq
from utils import word2ix, ix2word


def train():
    train_writer = SummaryWriter(log_dir='./log/train')

    # 模型
    seq2seq = Seq2seq(vocab_size=opt.vocab_size,
                      emb_dim=opt.emb_dim,
                      hidden_size=opt.vocab_size//2)
    if opt.load_path:
        seq2seq.load(opt.load_path)

    # 数据
    train_dataset = Mydataset()
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    # 损失函数和优化器
    loss_fn = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(seq2seq.parameters(), lr=lr, weight_decay=opt.weight_decay)

    previous_loss = 1e9
    iteration = 0
    for epoch in range(opt.max_epoch):
        for ii, (encoder_input, decoder_input, decoder_label) in enumerate(train_dataloader):
            # forward pass
            decoder_hat = seq2seq.forward(encoder_input, decoder_input)
            decoder_hat = decoder_hat.view(-1, decoder_hat.shape[-1])   # (batch_size, time_step, class_num) -> (N, class_num)
            decoder_label = decoder_label.flatten()     # (batch_size, time_step) -> (N)

            # 计算loss
            loss = loss_fn(decoder_hat, decoder_label)
            loss.backward()

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            if ii % 1000 == 0:
                print("epoch:{}  loss:{}  lr:{}".format(epoch, loss.item(), lr))
                train_writer.add_scalar('Loss', loss.item(), iteration)
                iteration += 1
                if loss > previous_loss:
                    lr = lr * opt.lr_decay
                else:
                    previous_loss = loss

        # 保存检查点
        seq2seq.save()


def predict(input):
    # 模型
    seq2seq = Seq2seq(vocab_size=opt.vocab_size,
                      emb_dim=opt.emb_dim,
                      hidden_size=opt.vocab_size*2)
    if opt.load_path:
        seq2seq.load(opt.load_path)

    input = t.LongTensor(word2ix(input)).unsqueeze(dim=0)
    start_symbol = t.LongTensor(word2ix(['</s>']))
    output = seq2seq.predict(input, start_symbol, opt.max_length)
    output = output.numpy().tolist()
    output = ix2word(output)
    print(output)


train()
# predict("你好吗?")
