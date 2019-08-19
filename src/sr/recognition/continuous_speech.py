# -*- coding: utf-8 -*-
from .decode import decode_hmm_states
from .hmm import HMM
from .hmm_state import NES
from itertools import chain
import os
import pickle
import warnings
import copy
import numpy as np


def build_state_sequences(hmms, label_matrix):
    seq = []
    # count number of hmms and then number of states
    n_states = 1  # 1 non-emitting state at the very beginning
    for labels in label_matrix:
        for l in labels:
            n_states += len(hmms[l].gmm_states)
        n_states += 1  # 1 non-emitting state after each layer

    trans = np.full((n_states, n_states), np.inf)
    seq.append(NES())
    prev_nes_idx = 0
    trans_offset = 0
    for labels in label_matrix:
        seq.append(NES())  # NES after this layer, but added first for convenience
        trans_offset = len(seq)
        for l in labels:
            hmm = hmms[l]
            seq += hmm.gmm_states
            n_states = len(hmm.gmm_states)
            trans[trans_offset: trans_offset + n_states, trans_offset:trans_offset + n_states] = hmm.transitions
            for i in range(n_states):
                trans[trans_offset + i, prev_nes_idx] = 0  # transition from previous nes to a gmm state
                trans[trans_offset - 1, trans_offset + i] = 0  # transition from a gmm state to an nes
        prev_nes_idx = trans_offset - 1
    return seq, trans, trans_offset - 1


def continuous_train(data, models, label_seqs, use_gmm=True, n_gaussians=4, use_em=True, max_iteration=1000):
    # remember old models
    old_models = copy.deepcopy(models)

    for iter in range(max_iteration):
        converged = True
        print('=' * 25)
        print('Continuous training iteration:', iter)
        print('Rearranging data for hmm training...')
        segments = {w: [] for w in chain.from_iterable(label_seqs)}
        data_len = len(data)

        print('Building state sequences')
        sequences_and_transitions = [build_state_sequences(old_models, [[l] for l in labels]) for labels in label_seqs]

        print('Rearranging data, this may take a while...')
        for i in range(data_len):
            seq = sequences_and_transitions[i][0]
            trans = sequences_and_transitions[i][1]
            end_idx = sequences_and_transitions[i][2]
            x = data[i]

            _, path = decode_hmm_states(x, seq, trans, end_points=[[end_idx, -1]])
            # FIXME
            path = reversed(path[:, 0].tolist())

            print('Progress:', str(int(100 * i / data_len)) + "%", end='\r')

            # find segments for every digit, a segment is ended and started by 'NES'
            # each segment is a sequence of mfcc features, thus it's a 2d array-like
            i_nes = [i for i, x in enumerate(path) if x == "NES"]
            i_nes_len = len(i_nes)
            for index in range(i_nes_len - 1):
                start_i = i_nes[index] + 1
                segments[path[start_i]].append(x[start_i:i_nes[index + 1]])

        print('Complete data rearrangement')
        print("=" * 25)

        # remove templates that are too small to do dtw on
        for w in segments.keys():
            seg = segments[w]
            seg_len = len(seg)
            del_i = []
            for si in range(seg_len):
                if seg[si].shape[0] < 5:
                    warnings.warn('Removing digit templates that are too small', UserWarning)
                    del_i.append(si)

            del_i.sort(reverse=True)
            for si in del_i:
                del seg[si]

        # continuous training
        print('Doing HMM training...')
        for w in segments.keys():
            # train a new HMM model using the segments
            m = HMM(5)
            m.fit(segments[w], n_gaussians, use_gmm, use_em)
            converged = converged and m == old_models[w]
            old_models[w] = m
            # TODO: use command line argument for output model path
            file = open(os.path.join('models-continuous-4gaussians-em-realign', str(w) + '.pkl'), 'wb')
            pickle.dump(m, file)
            file.close()

        if converged:
            print('Continuous training converged')
            break
