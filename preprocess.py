import sys, os
import argparse
from collections import OrderedDict
import random
import torch

EOU = 0
UNK = 1
VOCAB_SIZE = 5000


parser = argparse.ArgumentParser(description='Preprocess data for Character Language Model')
parser.add_argument('--raw', type=str, default='data/raw_data.pt',
                            help='path to raw data in specified format')
parser.add_argument('--processed', type=str, default='data/processed_data.pt',
                            help='path to save processed data')
args = parser.parse_args()


def build_dictionary(text, level='char'):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = {}
    print(f'num sentences : {len(text)}')
    for i,cc in enumerate(text):
        if level == 'word':
            words = cc.split(" ")
        elif level == 'char':
            words = list(cc)
        words = list(filter(None, words))
        for w in words:
            if w not in wordcount:
                wordcount[w]  = 0
            wordcount[w] += 1
    sorted_words = sorted(list(wordcount.keys()), key = lambda x:wordcount[x], reverse=True)
    worddict = OrderedDict()
    for idx, word in enumerate(sorted_words):
        worddict[word] = idx + 2
    return worddict

def convert_utterance_to_indices(utterance, word_dict, level='char'):
    if not utterance:
        return []
    utterance = utterance.split(" ") if level == 'word' else utterance
    indices = [word_dict.get(w) if word_dict.get(w, VOCAB_SIZE) < VOCAB_SIZE else UNK
                    for w in utterance]
    return indices

def process_field(text):
    raw_text = [d for d in text if isinstance(d, str)]
    dictionary = build_dictionary(raw_text, level='char')
    textind = [convert_utterance_to_indices(item, dictionary, level='char') for item in raw_text]
    return raw_text, dictionary, textind

if __name__ == "__main__":
    with open(args.raw, 'r', encoding='utf-8') as f:
        text = [line.lower() for line in f]
    random.shuffle(text)
    raw_text, dictionary, textind = process_field(text)
    torch.save({
    'text': raw_text,
    'dictionary': dictionary,
    'textind': textind}, args.processed)
