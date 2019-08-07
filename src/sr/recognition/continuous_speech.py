# -*- coding: utf-8 -*-
from .model_graph import *
from .dtw import *
from .hmm import *
from itertools import chain
import os
import pickle


def continuous_train(data, model_graphs, word_indices, use_gmm=True, n_gaussians=5, use_em=False):
    print('rearranging data for continuous training...')
    segments = {w: [] for w in chain.from_iterable(word_indices)}

    data_len = len(data)
    for i in range(data_len):
        m = model_graphs[i]
        x = data[i]
        # w = word_indices[i]
        _, path = sentence_viterbi(x, m)
        # find segments for every digit, a segment is ended and started by 'NES'
        # each segment is a sequence of mfcc features, thus it's a 2d array-like
        i_nes = [i for i, x in enumerate(path) if x == "NES"]
        i_nes_len = len(i_nes)
        for i in range(i_nes_len - 1):
            start_i = i_nes[i] + 1
            segments[path[start_i]].append(np.array(x[start_i:i_nes[i + 1]]))

    # continuous training
    print('doing continuous training...')
    print('=' * 25)
    for w in segments.keys():
        # train a new HMM model using the segments
        m = HMM(n_gaussians)
        m.fit(segments[w], n_gaussians, use_gmm, use_em)
        file = open(os.path.join('models', str(w) + '.pkl'), 'wb')
        pickle.dump(m, file)
        file.close()
