import torch
import numpy as np
import math
from difflib import SequenceMatcher

def beam_forward(input, model, args):
    device = torch.device("cuda" if input.is_cuda else "cpu")
    batch_size = input.shape[1]
    hidden_state = model.init_hidden(batch_size)
    h_state_mat = torch.stack([hidden_state[0].cpu() for i in range(args.beamsize)])  # (args.beamsize, num_directions, d_batch_size, word_size)
    c_state_mat = torch.stack([hidden_state[1].cpu() for i in range(args.beamsize)])
    seq_ind_mat = torch.zeros([args.beamsize, batch_size, args.maxlen]).long()
    seq_prob_mat = torch.zeros([args.beamsize, batch_size])
    final_ind_mat = torch.zeros([args.beamsize, batch_size, args.maxlen]).long()
    final_prob_mat = torch.ones([args.beamsize, batch_size])
    beam_occupied = torch.tensor([0] * batch_size).long()
    LogSoftmax = torch.nn.LogSoftmax(dim=1)

    for i in range(args.maxlen):
        predicted_word_probs_list = []
        predicted_word_ids_list = []
        h_state_list = []
        c_state_list = []

        for z in range(args.beamsize):
            if i == 0:
                seq_prob = torch.zeros([batch_size]).to(device)
            else:
                seq_prob = seq_prob_mat[z].to(device)
                input = seq_ind_mat[z, :, i - 1].to(device).unsqueeze(0)  # (1, d_batch_size)
            hidden_state = (h_state_mat[z].to(device), c_state_mat[z].to(device))
            predicted_word_embeddings, hidden_state = model(input, hidden_state)
            h_state_list.append(hidden_state[0].cpu())
            c_state_list.append(hidden_state[1].cpu())
            predicted_word_probs, predicted_word_ids = LogSoftmax(predicted_word_embeddings[-1]).topk(args.beamsize)    # ([d_batch_size, args.beamsize])
            predicted_word_probs_list.append((predicted_word_probs.t() + seq_prob).cpu())
            predicted_word_ids_list.append(predicted_word_ids.t().cpu())
            if i == 0 and z == 0:
                break

        predicted_word_probs_list = torch.cat(predicted_word_probs_list)    # (args.beamsize^2, d_batch_size)
        seq_prob_mat, mat_ind = predicted_word_probs_list.topk(args.beamsize, dim=0)    # (args.beamsize, d_batch_size)
        # seq_prob_mat = seq_prob_mat/seq_prob_mat.max()
        predicted_word_ids_list = torch.cat(predicted_word_ids_list)       # (args.beamsize^2, d_batch_size)
        mat_ind_ext = (mat_ind / args.beamsize).unsqueeze(2).repeat(1, 1, args.maxlen)
        seq_ind_mat = torch.gather(seq_ind_mat, 0, mat_ind_ext)
        seq_ind_mat[:, :, i] = torch.gather(predicted_word_ids_list, 0, mat_ind)     # (args.beamsize, d_batch_size)

        ################################################
        zero_ind = (seq_ind_mat[:, :, i] == 0).nonzero()    # (args.beamsize, d_batch_size)
        minval, minind = final_prob_mat.min(0)
        for j in range(zero_ind.shape[0]):
            if beam_occupied[zero_ind[j,1]] < args.beamsize:
                final_ind_mat[beam_occupied[zero_ind[j,1]],zero_ind[j,1],:] = seq_ind_mat[zero_ind[j,0], zero_ind[j,1],:]
                final_prob_mat[beam_occupied[zero_ind[j,1]],zero_ind[j,1]] = seq_prob_mat[zero_ind[j,0], zero_ind[j,1]]/i
                seq_prob_mat[zero_ind[j,0], zero_ind[j,1]] = -math.inf
                beam_occupied[zero_ind[j,1]] += 1
            elif minval[zero_ind[j,1]] < seq_prob_mat[zero_ind[j,0], zero_ind[j,1]]/i:
                final_ind_mat[minind[zero_ind[j,1]],zero_ind[j,1],:] = seq_ind_mat[zero_ind[j,0], zero_ind[j,1],:]
                final_prob_mat[minind[zero_ind[j,1]],zero_ind[j,1]] = seq_prob_mat[zero_ind[j,0], zero_ind[j,1]]/i
                seq_prob_mat[zero_ind[j,0], zero_ind[j,1]] = -math.inf

        if beam_occupied.sum() == args.beamsize * batch_size:
            break
        #################################################

        h_state_list = torch.stack(h_state_list)  # (args.beamsize, num_directions, batch_size, word_size)
        c_state_list = torch.stack(c_state_list)
        mat_ind = (mat_ind / args.beamsize).view(mat_ind.shape[0], 1, mat_ind.shape[1], 1).repeat(1, h_state_mat.shape[1], 1, h_state_mat.shape[3])
        h_state_mat = torch.gather(h_state_list, 0, mat_ind)
        c_state_mat = torch.gather(c_state_list, 0, mat_ind)

    #####################################################
    for i in range(batch_size):
        rem = args.beamsize - beam_occupied[i]
        final_ind_mat[beam_occupied[i]:,i,:] = seq_ind_mat[:rem, i, :]
        final_prob_mat[beam_occupied[i]:,i] = seq_prob_mat[:rem, i]

    final_prob_mat, mat_ind = final_prob_mat.topk(args.beamsize, dim=0)    # (args.beamsize, d_batch_size)
    mat_ind = (mat_ind).unsqueeze(2).repeat(1, 1, args.maxlen)
    final_ind_mat = torch.gather(final_ind_mat, 0, mat_ind)
    #####################################################
    return final_ind_mat # [args.beamsize, batch_size, maxlen]



def decode_suggestion(prompt, indxt, corpus):
    '''
    prompt : list of prompt characters
    indxt : torch LongTensor [seqlen]
    '''
    sep = ""
    outstring = corpus.decode_indexlist(indxt.tolist())
    return f'{sep.join(prompt)}{sep}{sep.join(outstring)}'


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def rerank4diversity(suggestions, topk=5):
    l = ["".join(wl) for wl in suggestions]
    s = [1. for i in l]
    for i in range(topk-1):
        s = [s[j] if j<i+1 else s[j] * (1-similar(l[i].lower(), o.lower())) for j,o in enumerate(l)]
        z = zip(s, l)
        s, l = zip(*sorted(z, key=lambda x:x[0], reverse=True))
    return l
