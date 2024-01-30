# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class BERT_LSTM(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_LSTM, self).__init__()
        self.bert = bert
        
        self.lstm = DynamicLSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, num_layers=1, batch_first=True,bidirectional=True)
        self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        hidden_state = self.bert(text_bert_indices)[0]
        # print("x:",x)
        x_len = torch.sum(text_bert_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(hidden_state, x_len)
        out = self.dense(h_n[0])
        return out
