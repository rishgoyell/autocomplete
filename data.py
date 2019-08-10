import os
from io import open
import torch
import random

class Corpus(object):
    def __init__(self, path):
        D = torch.load(path)
        self.dictionary = D['dictionary']
        self.revmap = list(D['dictionary'].items())
        divind = [int(0.8 * len(D['text'])), int(0.9 * len(D['text']))]
        self.textind = {}
        self.textind['train'] = D['textind'][:divind[0]]
        self.textind['valid'] = D['textind'][divind[0]:divind[1]]
        self.textind['test'] = D['textind'][divind[1]:]

    def get_batch(self, bsz=64, randomize=True, start_ind=-1, set='train'):
        '''
        INPUT
        bsz: batch_size
        randomize: flag for generating random batch
        start_ind: if randomize is false, get batch with this starting index
        set: get batch from train/test/dev

        OUTPUT
        textind: tensor of token indices [seq_len, bsz]
        lengths: list of lengths of sequences, excluding [EOU] token
        '''
        if randomize:
            batch_inds = random.sample(range(len(self.textind[set])), bsz)
        else:
            batch_inds = [i for i in range(start_ind, min(start_ind + bsz, len(self.textind[set])))]
        textind = [self.textind[set][i] for i in batch_inds]
        lengths = [len(x)+1 for x in textind]
        if 0 in lengths:
            print("empty string found")
        maxlength = max(lengths) + 1
        textind = torch.stack([torch.tensor([1] + textind[i] + [0]*(maxlength-lengths[i])) for i in range(len(textind))])
        return textind.t(), torch.LongTensor(lengths)

    def idx2word(self, idx):
        #idx = idx.data[0]
        if idx == 0:
            return "EOU"
        elif idx == 1:
            return "UNK"
        search_idx = idx - 2
        if search_idx >= len(self.revmap):
            return "NA"
        word, idx_ = self.revmap[search_idx]
        assert idx_ == idx
        return word

    def word2idx(self, word, vocab_size=160):
        if word == "EOU":
            return 0
        idx = self.dictionary.get(word)
        if idx and idx <= vocab_size:
            return idx
        return 1

    def decode_indexlist(self, indices):
        '''
        INPUT : list of word indices
        OUTPUT : list of words corresponding to indices
        '''
        length = 0
        while length < len(indices) and indices[length] != 0:
            length += 1
        indices = indices[:length]
        word_list = [self.idx2word(idx) for idx in indices if idx != 0]
        return word_list

    def encode_wordlist(self, words):
        '''
        INPUT : list of sanitized word
        OUTPUT : list of indices corresponding to words
        '''
        return torch.tensor([self.word2idx(w) for w in words]).long()
