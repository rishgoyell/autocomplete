# coding: utf-8
import sys, os
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import random
import numpy as np

import data
import model

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Character Language Model')
parser.add_argument('--data', type=str, default='data/processed.pt',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=32,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=10.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=64,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='models/model.pt',
                    help='path to save the final model')

args = parser.parse_args()
print("PARAMETERS:")
for item in vars(args).items():
    print(item)
print("==============================================")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary) + 2
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss(reduction='none')   ##

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(mode='valid'):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    masksum = 0
    with torch.no_grad():
        for i in range(0, len(corpus.textind[mode]), args.batch_size):
            data, lengths = corpus.get_batch(args.batch_size, False, i, mode)
            data = data.to(device)
            lengths = lengths.to(device)
            hidden = model.init_hidden(data.shape[1])
            for seqind, j in enumerate(range(0, data.shape[0]-1, args.bptt)):
                ei = min(j + args.bptt, data.shape[0]-1)
                partoutput, hidden = model(data[j:ei], hidden)
                lossmat = criterion(partoutput.transpose(1,2), data[j+1:ei+1])
                if (lengths >= ei).sum() == lengths.shape[0]:
                    total_loss += lossmat.sum()
                    masksum += lossmat.shape[0] * lossmat.shape[1]
                else:
                    mask = (torch.arange(ei-j).to(device).expand(len(lengths), ei-j) < (lengths-j).unsqueeze(1)).t().float()
                    total_loss += (lossmat*mask).sum()
                    masksum += mask.sum()
    return total_loss / masksum

def train():
    # Turn on training mode which enables dropout.
    model.train()
    random.shuffle(corpus.textind['train'])
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, len(corpus.textind['train']), args.batch_size)):
        data, lengths = corpus.get_batch(args.batch_size, False, i, 'train')
        data = data.to(device)
        lengths = lengths.to(device)
        hidden = model.init_hidden(data.shape[1])
        loss = 0
        masksum = 0
        model.zero_grad()
        for seqind, j in enumerate(range(0, data.shape[0]-1, args.bptt)):
            # data.shape[0] - 1 to not let EOU pass as input
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            ei = min(j + args.bptt, data.shape[0]-1)
            hidden = repackage_hidden(hidden)
            partoutput, hidden = model(data[j:ei], hidden)
            lossmat = criterion(partoutput.transpose(1,2), data[j+1:ei+1])
            if (lengths >= ei).sum() == lengths.shape[0]:
                temploss = lossmat.sum()
                tempmasksum = lossmat.shape[0] * lossmat.shape[1]
            else:
                mask = (torch.arange(ei-j).to(device).expand(len(lengths), ei-j) < (lengths-j).unsqueeze(1)).t().float()
                temploss = (lossmat*mask).sum()
                tempmasksum = mask.sum()
            loss += temploss.data
            masksum += tempmasksum.data
            (temploss/tempmasksum).backward()
        loss /= masksum
        # loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss

        if batch % args.log_interval == 0 and batch != 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.3f} | ppl {:8.3f}'.format(
                epoch, batch, len(corpus.textind['train']) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate('valid')
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f} | '
                'valid ppl {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        sys.stdout.flush()
        # train()
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate('test')
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
