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
    
class MultiAttPreRea(nn.Module):
    def __init__(self,hid_dim,heads=4,batch_first=True):
        super(MultiAttPreRea, self).__init__()
        self.att = nn.MultiheadAttention(hid_dim,heads,batch_first=batch_first)
        # self.w_q = nn.Linear(n_view,1)
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

    def forward(self,q,k,v):
        return self.att(self.w_q(q),self.w_k(k),self.w_v(v))

class MPLN(nn.Module):
    def __init__(self, bert,bert2, opt,tokenizer):
        super(MPLN, self).__init__()
        self.opt = opt
        self.tokenizer = tokenizer
        self.bert = bert
        self.bert2 = bert2
        self.hid_dim = bert.config.hidden_size
        self.lambdaa = opt.lambdaa
        self.hop = int(opt.hop)
        self.fc = nn.Linear(self.hid_dim,3)
        self.text_embed_dropout = nn.Dropout(opt.dropout)
        self.layer_norm = torch.nn.LayerNorm(self.hid_dim, eps=1e-12)


    def forward(self, inputs):
        prompt_inputs,prompt_mask,bert_knowledge_inputs,bert_knowledge_mask,mlm_labels = inputs

        prompt_inputs = prompt_inputs.reshape(-1,prompt_inputs.size(-1))
        prompt_mask = prompt_mask.reshape(-1,prompt_mask.size(-1))
        prompt_out,koutt = self.bert(prompt_inputs,attention_mask=prompt_mask,return_dict=False)
        knowledge_out,koutt2 = self.bert2(bert_knowledge_inputs,attention_mask=bert_knowledge_mask,return_dict=False)
        batch_size = knowledge_out.shape[0]
        knowledge_out = self.text_embed_dropout(knowledge_out)
        prompt_out = self.text_embed_dropout(prompt_out)
        prompt_out = prompt_out.reshape(batch_size,-1,prompt_out.size(1),prompt_out.size(2))
        prompt_out = torch.where(mlm_labels.unsqueeze(-1) >= 0,prompt_out,torch.zeros_like(prompt_out))
        # prompt_out = prompt_out[mlm_labels >= 0].view(prompt_out.size(0),prompt_out.size(1),1,prompt_out.size(-1)).squeeze(2)


        for prompt_index in range(prompt_out.size(1)):
            knowledge_out_c = knowledge_out.clone()
            prompt_single = prompt_out[:,prompt_index]
            for index in range(self.hop):
                alpha_weight= torch.matmul(prompt_single,knowledge_out_c.transpose(1,2))
                if index == self.hop - 1:
                    alpha_weight = F.softmax(alpha_weight.sum(1, keepdim=True), dim=2)
                    knowledge_h = torch.matmul(alpha_weight,knowledge_out_c)
                else:
                    knowledge_h = torch.matmul(alpha_weight,knowledge_out_c)
                    knowledge_out_c = knowledge_out_c + self.lambdaa * self.layer_norm(torch.sigmoid(knowledge_h))

            if prompt_index == 0:
                knowledge_hs = knowledge_h
            else:
                knowledge_hs = torch.cat((knowledge_hs,knowledge_h),dim=1)



        out = torch.mean(knowledge_hs,1)
        self.tokenizer.get_labels()
        out = [
            torch.mm(
                out[:,:],
                self.bert.embeddings.word_embeddings.weight[i].transpose(1,0)
            ) for i in self.tokenizer.prompt_label_idx
        ]
        return out
