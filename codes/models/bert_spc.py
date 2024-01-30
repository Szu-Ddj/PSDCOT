# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt,tokenizer):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids,bert_mask_ids = inputs
        # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]
        pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids,attention_mask = bert_mask_ids)[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        # print(logits.size(), logits.device)
        return logits
        # return pooled_output
