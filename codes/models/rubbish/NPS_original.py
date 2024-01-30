# -*- coding: utf-8 -*-

from binascii import a2b_hqx
from cmath import tau
from filecmp import dircmp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
# from layers import ListModule
from torch.autograd import Variable
# from capsule import Capsule
# from attentionlayer import Attn
# from transformers import RobertaTokenizer
import re
from senticnet.senticnet import SenticNet
sn = SenticNet()

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(
            in_features, out_features))  # (hd,hd)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1  # degree of the i-th token
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:

            return output + self.bias
        else:

            return output

class MultiAtt(nn.Module):
    def __init__(self,hid_dim,heads=4,batch_first=True):
        super(MultiAtt, self).__init__()
        self.att = nn.MultiheadAttention(hid_dim,heads,batch_first=batch_first)
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

    def forward(self,q,k,v):
        return self.att(self.w_q(q),self.w_k(k),self.w_v(v))

class NPS(nn.Module):
    def __init__(self, bert, opt):
        super(NPS, self).__init__()
        self.opt = opt


        self.bert = bert
        self.hid_dim = self.bert.config.hidden_size
        self.lambdaa = opt.lambdaa
        self.hop = int(opt.hop)

        # self.gc1 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc2 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc3 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc4 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc5 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc6 = GraphConvolution(self.hid_dim, self.hid_dim)

        # self.conv1 = nn.Conv1d(self.hid_dim, self.hid_dim, 3, padding=1)
        # self.conv2 = nn.Conv1d(self.hid_dim, self.hid_dim, 3, padding=1)
        # self.conv3 = nn.Conv1d(self.hid_dim, self.hid_dim, 3, padding=0)

        # self.fc_out = nn.Linear(self.hid_dim, opt.polarities_dim)
        self.fc_out = nn.Linear(self.hid_dim,opt.polarities_dim)
        self.fc_target = nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_reasons = nn.Linear(opt.n_views+3,1)

        self.att = nn.MultiheadAttention(self.hid_dim,4,batch_first=True)
        self.w_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.w_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.w_v = nn.Linear(self.hid_dim, self.hid_dim)

        self.attr = nn.ModuleList([MultiAtt(self.hid_dim) for _ in range(opt.n_views + 2)])

        self.text_embed_dropout = nn.Dropout(opt.dropout)

        self.layer_norm = torch.nn.LayerNorm(self.hid_dim, eps=1e-12)


    def forward(self, inputs):
        bert_text_target_inputs, bert_text_target_type, bert_text_target_mask, bert_reason_inputs, bert_reason_type, bert_reason_mask = inputs
        #rule_indices batch_size*num_classi*max_len
        # text_len = torch.sum(text_indices != 1, dim=-1)
        # print(text_indices,reason_indices)
        bert_reason_inputs = bert_reason_inputs.reshape(-1,bert_reason_inputs.size(-1))
        bert_reason_type = bert_reason_type.reshape(-1,bert_reason_inputs.size(-1))
        bert_reason_mask = bert_reason_mask.reshape(-1,bert_reason_inputs.size(-1))

        text_out,text_pooler_out = self.bert(bert_text_target_inputs,token_type_ids=bert_text_target_type, attention_mask=bert_text_target_mask,return_dict=False)
        target_out = torch.where(bert_text_target_type.unsqueeze(-1) != 0,text_out,0)
        target_out = torch.sum(target_out,dim=1)

        reason_out,reason_pooler_out = self.bert(bert_reason_inputs,token_type_ids=bert_reason_type, attention_mask=bert_reason_mask,return_dict=False)

        text_out = torch.where(bert_text_target_type.unsqueeze(-1) == 0,text_out,0)
        batch_size = text_out.shape[0]
        # seq_len = text_out.shape[1]
        # hidden_size = text_out.shape[2]
        # print(reason_pooler_out.size())
        reason_pooler_out = reason_pooler_out.reshape(batch_size,-1,reason_pooler_out.size(-1))
        reason_out = reason_out.reshape(batch_size,-1,reason_out.size(1),reason_out.size(2))


        reason_pooler_out = reason_pooler_out.unsqueeze(2)
        _reason_pooler_out = reason_pooler_out.clone()
        for reason_index in range(reason_out.size(1)): 
            _reason_pooler_out[:,reason_index,:], _ = self.attr[reason_index](reason_pooler_out[:,reason_index,:],reason_out[:,reason_index,:],reason_out[:,reason_index,:])





        # target_out = target_out.unsqueeze(1)
        # for index in range(self.hop):
        #     for reason_index in range(reason_out.size(1)):
        #         reason_kl = _reason_pooler_out[:,reason_index,:].clone()
        #         if index != 0:
        #             reason_kl = reason_kl + self.lambdaa * self.layer_norm(out)
        #         reason_weight= F.softmax(torch.matmul(reason_kl,text_out.transpose(1,2)),dim=-1)
        #         reason_rule = torch.matmul(reason_weight,text_out)
        #         if reason_index == 0:
        #             reason_rules = reason_rule
        #         else:
        #             reason_rules = torch.cat((reason_rules,reason_rule),dim=1)
        
        #     if index != 0:
        #         target_out = target_out + self.lambdaa * self.layer_norm(target_rule)
        #     target_weight = F.softmax(torch.matmul(target_out,text_out.transpose(1,2)),dim=-1)
        #     target_rule = torch.matmul(target_weight,text_out)
        #     reason_rules = torch.cat((reason_rules,target_out),dim=1)
        

        #     reason_rules,_ = self.att(self.w_q(reason_rules),self.w_k(reason_rules),self.w_v(reason_rules))
        #     out = self.fc_reasons(reason_rules.transpose(1,2)).squeeze(2).unsqueeze(1)

        # out = self.fc_out(out.squeeze(1))
        # # print(out.size())
        # return out

        #begin 20230727
        for reason_index in range(reason_out.size(1)):
            reason_kl = _reason_pooler_out[:,reason_index,:].clone()
            for index in range(self.hop):
                reason_weight= F.softmax(torch.matmul(reason_kl,text_out.transpose(1,2)),dim=-1)
                reason_rule = torch.matmul(reason_weight,text_out)
                if index != self.hop - 1:
                    reason_kl = reason_kl + self.lambdaa * self.layer_norm(reason_rule)
            if reason_index == 0:
                reason_rules = reason_rule
            else:
                reason_rules = torch.cat((reason_rules,reason_rule),dim=1)
        
        target_out = target_out.unsqueeze(1)
        for index in range(self.hop):
            target_weight = F.softmax(torch.matmul(target_out,text_out.transpose(1,2)),dim=-1)
            target_rule = torch.matmul(target_weight,text_out)
            if index != self.hop - 1:
                target_out = target_out + self.lambdaa * self.layer_norm(target_rule)
        reason_rules = torch.cat((reason_rules,target_out),dim=1)
        
        # print(reason_rules.size(),)
        out,_ = self.att(self.fc_reasons(reason_rules.transpose(1,2)).transpose(1,2),self.w_k(reason_rules),self.w_v(reason_rules))
        # out = self.fc_reasons(reason_rules.transpose(1,2)).squeeze(2)
        out = self.fc_out(out.squeeze(1))
        # print(out.size())
        # print(out.size())
        return out
        #end==

        # for reason_index in range(reason_out.size(1)):
        #     reason_kl = reason_pooler_out[:,reason_index,:].unsqueeze(1).clone()
        #     for index in range(self.hop):
        #         if index != self.hop - 1:
        #             reason_weight= F.softmax(torch.matmul(reason_kl,text_out.transpose(1,2)),dim=-1)
        #             reason_rule = torch.matmul(reason_weight,text_out)
        #             reason_kl = reason_kl + self.lambdaa * self.layer_norm(reason_rule)
        #         else:
        #             reason_weight= torch.matmul(reason_kl,text_out.transpose(1,2))
        #             reason_rule = torch.matmul(reason_weight,text_out)

        #     if reason_index == 0:
        #         reason_rules = reason_rule
        #     else:
        #         reason_rules = torch.cat((reason_rules,reason_rule),dim=1)
        # target_out = target_out.unsqueeze(1)
        # for index in range(self.hop):
        #     if index != self.hop - 1:
        #         target_weight = F.softmax(torch.matmul(target_out,text_out.transpose(1,2)),dim=-1)
        #         target_rule = torch.matmul(target_weight,text_out)
        #         target_out = target_out + self.lambdaa * self.layer_norm(target_rule)
        #     else:
        #         target_weight = torch.matmul(target_out,text_out.transpose(1,2))
        #         target_rule = torch.matmul(target_weight,text_out)
        # reason_rules = torch.cat((reason_rules,target_out),dim=1)

        # out = self.fc_reasons(reason_rules.transpose(1,2)).squeeze(2)
        # out = self.fc_out(out)
        # return out
        





        # for reason_index in range(reason_out.size(1)):
        #     reason_kl = reason_pooler_out[:,reason_index,:].unsqueeze(1).clone()
        #     for index in range(self.hop):
        #         reason_weight= F.softmax(torch.matmul(reason_kl,text_out.transpose(1,2)),dim=-1)
        #         reason_rule = torch.matmul(reason_weight,text_out)
        #         if index != self.hop - 1:
        #             reason_kl = reason_kl + self.lambdaa * self.layer_norm(reason_rule)
        #     if reason_index == 0:
        #         reason_rules = reason_rule
        #     else:
        #         reason_rules = torch.cat((reason_rules,reason_rule),dim=1)
        # target_out = target_out.unsqueeze(1)
        # for index in range(self.hop):
        #     target_weight = F.softmax(torch.matmul(target_out,text_out.transpose(1,2)),dim=-1)
        #     target_rule = torch.matmul(target_weight,text_out)
        #     if index != self.hop - 1:
        #         target_out = target_out + self.lambdaa * self.layer_norm(target_rule)
        # reason_rules = torch.cat((reason_rules,target_out),dim=1)
        # # print(reason_rules.transpose(1,2).size())
        # reason_rules,_ = self.att(self.w_q(reason_rules),self.w_k(reason_rules),self.w_v(reason_rules))

        # out = self.fc_reasons(reason_rules.transpose(1,2)).squeeze(2)
        # out = self.fc_out(out)
        # # print(out.size())
        # return out