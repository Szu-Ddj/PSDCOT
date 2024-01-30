# import argparse
import torch
import transformers
from senticnet.senticnet import SenticNet
# from nltk.corpus import wordnet as wn

# from transformers import BertConfig, BertTokenizer, BertModel, \
#                          RobertaConfig, RobertaTokenizer, RobertaModel, \
#                          AlbertTokenizer, AlbertConfig, AlbertModel
sn = SenticNet()

def get_prompt_template(tid = 1):
    if tid == 1:
        temp = ["{text_a}. According to the sentiment expression in the text, the stance polarity is {mask} .",
        "{text_a}. The stance polarity of this text is {mask} .",
        "{text_a}. The stance polarity toward the {text_b} is {mask} ."]
    elif tid == 2:
        temp = ["{text_a}. According to the sentiment expression in the text, the stance polarity is {mask} .",
        "{text_a}. The stance polarity of this text is {mask} .",
        "{text_a}. The attitude toward the {text_b} is {mask} ."]
    elif tid == 3:
        temp = ["{text_a}. According to the sentiment expression in the text, the stance polarity is {mask} .",
        "'{text_a} is {mask} of {text_b} .",
        "{text_a}. The stance polarity toward the {text_b} is {mask} ."]
        temp = '{text_a} is {mask} of {text_b} .'
    elif tid == 4:
        temp = '{text_a} . The {text_b} made me feel {mask} .'
    else:
        raise Exception("template_id error, please choose correct id")
    return temp

    
def get_senticnet_n_hop(all_seed_words, n, tokenizer):
    if n ==0:
        return all_seed_words
    elif n > 0:
        expanded_seed_words = all_seed_words.copy()

        for label, seed_words in enumerate(all_seed_words):
            for _h in range(0, n):
                new_words = seed_words.copy()
                for word in seed_words:
                    try:
                        new_words.extend(sn.semantics(word))
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
def get_wordnet_n_hop(all_seed_words, n, tokenizer):
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
def get_label_words(tokenizer,lid = 1):
    if lid == 1:
        return [
            ['against'],
            ['favor'],
            ['neutral']
        ]
    elif lid == 2:
        return [
            ['opposing'],
            ['support'],
            ['neutral']
        ]
    elif lid == 3:
        all_seed_words = [
            ['opposed'],
            ['in favor of'],
            ['neutral']
        ]
        return get_senticnet_n_hop(all_seed_words, 2, tokenizer)
    elif lid == 4:
        return [
                ["wrong", "bad", "stupid"],
                ["beautiful", "good", "great"],
                ["neutral", "unique", "cool"],
            ]

    else:
        raise Exception("label_id error, please choose correct id")