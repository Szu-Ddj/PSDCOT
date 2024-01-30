# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, \
                         AutoTokenizer
import csv
import random
from prompts import get_prompt_template
from senticnet.senticnet import SenticNet
SN = SenticNet()

def build_tokenizer(fnames, max_seq_len, dat_fname, add_num):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            if type(fname) == list:
                for sign_fname in fname:
                    if 'add' not in sign_fname:
                        _add_num = 1000000
                    else:
                        _add_num = add_num
                    with open(sign_fname,'r') as f:
                        l1s = csv.DictReader(f)
                        for l1,_ in zip(l1s,range(_add_num)):
                            text += l1['Tweet'] + ' ' + l1['Reason'] + ' ' + l1['Target']
            else:
                if 'add' not in fname:
                    _add_num = 1000000
                else:
                    _add_num = add_num
                with open(fname,'r') as f:
                    l1s = csv.DictReader(f)
                    for l1,_ in zip(l1s,range(_add_num)):
                        text += l1['Tweet'] + ' ' + l1['Reason'] + ' ' + l1['Target']

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else '/home/dingdaijun/data_list/dingdaijun/glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        words_num = {}
        for word in words:
            if words_num.get(word) == None:
                words_num[word] = 0
            words_num[word] += 1
        for word in words_num:
            if word not in self.word2idx and words_num[word] > 1:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
        print('tokenizer is over:', len(self.word2idx))

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len,pretrained_bert_name,lid=0, tid=1,model_name='MPLN'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len
        self.tokenizer.add_tokens([])
        self.temps = get_prompt_template()
        self.lid = lid

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', add = 0):

        if 'prompt' in self.model_name or self.model_name == 'MPLN':
            truc = False
        else:
            truc = True

        if len(text) == 1:
            sequence = self.tokenizer.encode_plus(text[0],max_length=self.max_seq_len + add,padding='max_length',truncation=truc,return_tensors='pt').values()
        else:
            sequence = self.tokenizer.encode_plus(text[0],text_pair=text[1],max_length=self.max_seq_len + add,padding='max_length',truncation=truc,return_tensors='pt').values()
        sequence = [item.squeeze(0) for item in sequence]

        if len(sequence[0]) > self.max_seq_len + add:
            # print("===========it's Prompt==========")
            for index in range(len(sequence)):
                sequence[index] = sequence[index][len(sequence[index]) - self.max_seq_len - add:]
        return sequence
        # sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        # if len(sequence) == 0:
        #     sequence = [0]
        # if reverse:
        #     sequence = sequence[::-1]
        # if len(sequence) - add > self.max_seq_len:
        #     print('***length out***', len(sequence) - add - self.max_seq_len, 'add: ',add)
        # return pad_and_truncate(sequence, self.max_seq_len + add, padding=padding, truncating=truncating)
        
    def get_senticnet_n_hop(self,all_seed_words, n, tokenizer):
        if n ==0:
            return all_seed_words
        elif n > 0:
            expanded_seed_words = all_seed_words.copy()

            for label, seed_words in enumerate(all_seed_words):
                for _h in range(0, n):
                    new_words = seed_words.copy()
                    for word in seed_words:
                        try:
                            new_words.extend(SN.semantics(word))
                        except:
                            pass

                    # 去重
                    seed_words = list(set(new_words))
                    res = []
                    for item in seed_words:
                        if len(tokenizer.encode(' '.join(['the', item]), add_special_tokens = False)[1:]) == 1:
                            res.append(item)
                expanded_seed_words[label] = res
        else:
            raise Exception("N>=0")
        return expanded_seed_words
    def get_wordnet_n_hop(self,all_seed_words, n, tokenizer):
        if n ==0:
            return all_seed_words
        elif n > 0:
            expanded_seed_words = all_seed_words.copy()

            for label, seed_words in enumerate(all_seed_words):
                for _h in range(0, n):
                    new_words = seed_words.copy()
                    for word in seed_words:
                        try:
                            for _word in wn.synsets(word):
                                new_words.extend(_word.lemma_names())
                            # new_words.extend([_word.lemma_names() for _word in wn.synsets(word)])
                        except:
                            pass
                    # 去重
                    seed_words = list(set(new_words))
                    res = []
                    for item in seed_words:
                        if len(tokenizer.encode(' '.join(['the', item]), add_special_tokens = False)[1:]) == 1:
                            res.append(item)
                expanded_seed_words[label] = res
        else:
            raise Exception("N>=0")
        return expanded_seed_words
    
    def get_label_words(self,tokenizer,lid = 0):
        if lid == 0:
            return [
                ['against'],
                ['favor'],
                ['neutral']
            ]
        if lid == 1:
            all_seed_words = [
                ['against'],
                ['favor'],
                ['neutral']
            ]
            return self.get_senticnet_n_hop(all_seed_words, 1, tokenizer)
        elif lid == 2:
            all_seed_words = [
                ['against'],
                ['favor'],
                ['neutral']
            ]
            return self.get_senticnet_n_hop(all_seed_words, 2, tokenizer)
        elif lid == 3:
            all_seed_words = [
                ['against'],
                ['favor'],
                ['neutral']
            ]
            return self.get_senticnet_n_hop(all_seed_words, 3, tokenizer)
        elif lid == 4:
            return [
                ['opposing'],
                ['support'],
                ['neutral']
            ]
        elif lid == 5:
            all_seed_words = [
                ['opposed','wrong'],
                ['in favor of'],
                ['neutral']
            ]

        else:
            raise Exception("label_id error, please choose correct id")
        
    def get_labels(self):
        self.verbalizer = self.get_label_words(self.lid)
        mask_ids = []
        mask_pos = 0
        temp = self.temps[0].format(text_a = 'a', text_b = 'a', mask = self.tokenizer.mask_token)
        temp = temp.split(' ')

        _temp = temp.copy()
        
        original = self.tokenizer.encode(' '.join(temp), add_special_tokens = False)
        
        for i in range(len(_temp)):
            if _temp[i] == self.tokenizer.mask_token:
                _mask_pos = i
        for i in range(len(original)):
            if original[i] == self.tokenizer.mask_token_id:
                mask_pos = i

        sign = True
        for index, name in enumerate(self.verbalizer):
            mask_id = []
            for item in name:
                _temp[_mask_pos] = item
                final = self.tokenizer.encode(' '.join(_temp), add_special_tokens = False)
                if len(final) != len(original):
                    sign = False
                mask_id.append(final[mask_pos])
            mask_ids.append(mask_id)
        assert sign
        self.prompt_label_idx = mask_ids
        return self.prompt_label_idx

import csv
import random
class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer,opt):
        all_data = []
        match = {'AGAINST':0,'FAVOR':1,'NONE':2,'0':0,'1':1,'2':2,'against':0,'favor':1,'none':2}
        # match = {'AGAINST':0,'FAVOR':1,'NONE':2,'-1':0,'1':1,'0':2}
        def deal_data(sign_fname):
            features = []
            sall_data = []
            with open(sign_fname,'r') as f:
                lines = csv.DictReader(f)
                for index, item in enumerate(lines):
                    text = item['Tweet']
                    polarity = match[item['Stance']]
                    Reason = item['Ans'].split('.')
                    Reason = '.'.join(Reason[:])
                    target = item['Target']
                    if 'prompt' in opt.model_name or opt.model_name == 'MPLN':
                        prompt_texts =  [template.format(text_a = f'{text}', text_b = target, mask = tokenizer.tokenizer.mask_token) for template in tokenizer.temps]

                    prompt_inputs = np.ones((len(prompt_texts),tokenizer.max_seq_len),dtype=int) * tokenizer.tokenizer.pad_token_id
                    prompt_mask = np.ones((len(prompt_texts),tokenizer.max_seq_len),dtype=int) * tokenizer.tokenizer.pad_token_id

                    if 'roberta' in opt.pretrained_bert_name or 'bart' in opt.pretrained_bert_name:
                        for indexx,prompt_text in enumerate(prompt_texts):
                            prompt_inputs[indexx],prompt_mask[indexx] = tokenizer.text_to_sequence([prompt_text])
                    else:
                        for indexx,prompt_text in enumerate(prompt_texts):
                            prompt_inputs[indexx],_,prompt_mask[indexx] = tokenizer.text_to_sequence([prompt_text])
                    
                    mlm_labels = np.where(prompt_inputs == tokenizer.tokenizer.mask_token_id,1,-1)
                    bert_knowledge = tokenizer.text_to_sequence([Reason])
                    bert_text = tokenizer.text_to_sequence([f"Sentence: {text}, Target: {target}",Reason])
                    if 'roberta' in opt.pretrained_bert_name or 'bart' in opt.pretrained_bert_name:
                        data = {
                            'bert_knowledge_inputs': bert_knowledge[0],
                            'bert_knowledge_mask': bert_knowledge[1],

                            'prompt_inputs': prompt_inputs,
                            'prompt_mask': prompt_mask,

                            'bert_text_inputs': bert_text[0],
                            'bert_text_mask': bert_text[1], 

                            'mlm_labels': mlm_labels,
                            'polarity': polarity,
                        }
                    else:
                        data = {
                            'bert_knowledge_inputs': bert_knowledge[0],
                            'bert_knowledge_mask': bert_knowledge[2],

                            'prompt_inputs': prompt_inputs,
                            'prompt_mask': prompt_mask,

                            'bert_text_inputs': bert_text[0],
                            'bert_text_mask': bert_text[2], 

                            'mlm_labels': mlm_labels,
                            'polarity': polarity,
                        }
                    sall_data.append(data)
            return sall_data


        if type(fname) == list:
            for sign_fname in fname:
                sall_data = deal_data(sign_fname)
                all_data.extend(sall_data)
        else:
            sall_data = deal_data(fname)
            all_data.extend(sall_data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
