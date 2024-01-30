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
class MultiAttRule(nn.Module):
    def __init__(self,hid_dim,heads=4,batch_first=True):
        super(MultiAttRule, self).__init__()
        self.att = nn.MultiheadAttention(hid_dim,heads,batch_first=batch_first)
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        
    def forward(self,q,k,v):
        return self.att(self.w_q(q),self.w_k(k),self.w_v(v))
    
class MultiAttKnowledge(nn.Module):
    def __init__(self,hid_dim,n_view,heads=4,batch_first=True):
        super(MultiAttKnowledge, self).__init__()
        self.att = nn.MultiheadAttention(hid_dim,heads,batch_first=batch_first)
        # self.w_q = nn.Linear(n_view,1)
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

    def forward(self,q,k,v):
        return self.att(self.w_q(q),self.w_k(k),self.w_v(v))

class NPS_prompt(nn.Module):
    def __init__(self, bert,bert2, opt,tokenizer):
        super(NPS_prompt, self).__init__()
        self.opt = opt
        self.tokenizer = tokenizer

        self.bert = bert
        self.bert2 = bert2
        self.hid_dim = bert.config.hidden_size
        self.lambdaa_r = opt.lambdaa_r
        self.lambdaa_k = opt.lambdaa_k
        self.hop_r = int(opt.hop_r)
        self.hop_k = int(opt.hop_k)

        # self.gc1 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc2 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc3 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc4 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc5 = GraphConvolution(self.hid_dim, self.hid_dim)
        # self.gc6 = GraphConvolution(self.hid_dim, self.hid_dim)

        # self.conv1 = nn.Conv1d(self.hid_dim, self.hid_dim, 3, padding=1)
        # self.conv2 = nn.Conv1d(self.hid_dim, self.hid_dim, 3, padding=1)
        # self.conv3 = nn.Conv1d(self.hid_dim, self.hid_dim, 3, padding=0)
        # self.fc_out_r = nn.Linear(self.hid_dim, opt.polarities_dim)
        # self.fc_out_k = nn.Linear(self.hid_dim, opt.polarities_dim)
        self.fc_out = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.fc_out2 = nn.Linear(self.hid_dim, opt.polarities_dim)
        self.fc_target = nn.Linear(self.hid_dim,self.hid_dim)

        self.att_k = MultiAttKnowledge(self.hid_dim,opt.n_views+3)
        self.att_r = MultiAttRule(self.hid_dim)
        # self.att = nn.MultiheadAttention(self.hid_dim,4,batch_first=True)
        # self.w_q = nn.Linear(self.hid_dim, self.hid_dim)
        # self.w_k = nn.Linear(self.hid_dim, self.hid_dim)
        # self.w_v = nn.Linear(self.hid_dim, self.hid_dim)

        self.text_embed_dropout = nn.Dropout(opt.dropout)

        self.layer_norm = torch.nn.LayerNorm(self.hid_dim, eps=1e-12)


    def forward(self, inputs):
        bert_text_target_inputs, bert_text_target_type, bert_text_target_mask, bert_reason_inputs, bert_reason_type, bert_reason_mask,mlm_labels = inputs

        bert_reason_inputs = bert_reason_inputs.reshape(-1,bert_reason_inputs.size(-1))
        bert_reason_type = bert_reason_type.reshape(-1,bert_reason_inputs.size(-1))
        bert_reason_mask = bert_reason_mask.reshape(-1,bert_reason_inputs.size(-1))

        reason_out,reason_pooler_out = self.bert(bert_reason_inputs,token_type_ids=bert_reason_type,attention_mask=bert_reason_mask,return_dict=False)
        text_out,text_pooler_out = self.bert2(bert_text_target_inputs,token_type_ids=bert_text_target_type,attention_mask=bert_text_target_mask,return_dict=False)
        text_out = torch.where(bert_text_target_type.unsqueeze(-1) == 0,text_out,0)
        target_out = torch.where(bert_text_target_type.unsqueeze(-1) != 0,text_out,0)
        target_out = torch.sum(target_out,dim=1)
        batch_size = text_out.shape[0]
        text_out = self.text_embed_dropout(text_out)
        reason_out = self.text_embed_dropout(reason_out)
        reason_out = reason_out.reshape(batch_size,-1,reason_out.size(1),reason_out.size(2))
        reason_out = reason_out[mlm_labels >= 0].view(reason_out.size(0),reason_out.size(1),1,reason_out.size(-1)).squeeze(2)
        reason_pooler_out = reason_pooler_out.reshape(batch_size,-1,reason_pooler_out.size(-1))
        target_out = self.fc_target(target_out)

        #Select rule
        for index in range(self.hop_r):
            # for reason_index in range(reason_out.size(1)):
                # reason_weight = F.softmax(torch.matmul(text_pooler_out.unsqueeze(1), reason_out[:,reason_index,:].transpose(1,2)),dim=-1)
                # reason_rule = torch.matmul(reason_weight,reason_out[:,reason_index,:])
                # if reason_index == 0:
                #     reason_rules = reason_rule
                # else:
                #     reason_rules = torch.cat((reason_rules, reason_rule),dim = 1)

            rules_out = reason_out
            # standard_r  =torch.mean(rules_out,dim=1,keepdim=True)
            _tmp,_score = self.att_r(text_pooler_out.unsqueeze(1),rules_out,rules_out)

            _tmp = _tmp.squeeze(1)
            if index != self.hop_r - 1:
                score = F.gumbel_softmax(_score/1,dim=2,hard=True)
                # target_out = self.lambdaa * torch.matmul(score,rules_out).squeeze() + target_out
                text_pooler_out = self.lambdaa_r * self.layer_norm(torch.matmul(score,rules_out).squeeze()) + text_pooler_out
            else:
                out = _tmp
        # print(self.tokenizer.prompt_label_idx)
        # self.tokenizer.get_labels()
        # out = [
        #     torch.mm(
        #         out[:,:],
        #         self.bert.embeddings.word_embeddings.weight[i].transpose(1,0)
        #     ) for i in self.tokenizer.prompt_label_idx
        # ]
        # rout = self.fc_out_r(out)
        rout = out


        # #VBN
        for reason_index in range(reason_out.size(1)):
            reason_kl = reason_out[:,reason_index,:].clone().unsqueeze(1)
            for index in range(self.hop_k):
                reason_weight= F.gumbel_softmax(torch.matmul(reason_kl,text_out.transpose(1,2)),dim=-1,hard=True)
                reason_rule = torch.matmul(reason_weight,text_out)
                # print(reason_kl.size(),reason_weight.size(),reason_rule.size(),text_out.size())
                if index != self.hop_k - 1:
                    reason_kl = reason_kl + self.lambdaa_k * self.layer_norm(reason_rule)
            if reason_index == 0:
                reason_rules = reason_rule
            else:
                reason_rules = torch.cat((reason_rules,reason_rule),dim=1)
        
        target_out = target_out.unsqueeze(1)
        for index in range(self.hop_k):
            target_weight = F.gumbel_softmax(torch.matmul(target_out,text_out.transpose(1,2)),dim=-1,hard=True)
            target_rule = torch.matmul(target_weight,text_out)
            if index != self.hop_k - 1:
                target_out = target_out + self.lambdaa_k * self.layer_norm(target_rule)
        # print(reason_rules.size(),target_out.size())
        reason_rules = torch.cat((reason_rules,target_out),dim=1)
        standard_k = torch.mean(reason_rules,dim=1,keepdim=True)
        out,_ = self.att_k(standard_k,reason_rules,reason_rules)
        # out,_ = self.att_k(self.fc_reasons(reason_rules.transpose(1,2)).transpose(1,2),self.w_k(reason_rules),self.w_v(reason_rules))
        # kout = self.fc_out_k(out.squeeze(1))
        kout = out.squeeze(1)

        # print(type(kout),type(rout))
        out = self.fc_out(torch.cat((rout,kout),dim=-1))
        # print(type(out))
        out = self.fc_out2(out)
        # print(type(out))
        # self.tokenizer.get_labels()
        # out = [
        #     torch.mm(
        #         out[:,:],
        #         self.bert.embeddings.word_embeddings.weight[i].transpose(1,0)
        #     ) for i in self.tokenizer.prompt_label_idx
        # ]
        return out
