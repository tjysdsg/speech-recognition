# -*- coding: utf-8 -*-
from .model_graph import *
from .dtw import *
from .hmm import *
from itertools import chain
import os
import pickle
import warnings
import copy


def continuous_train(data, models, lables, use_gmm=True, n_gaussians=4, use_em=True, max_iteration=1000):
    # remember old models
    old_models = copy.deepcopy(models)

    for iter in range(max_iteration):
        converged = True
        print('=' * 25)
        print('Continuous training iteration:', iter)
        print('Rearranging data for hmm training...')
        segments = {w: [] for w in chain.from_iterable(lables)}
        data_len = len(data)

        print('Building model graphs')
        # make model_graphs for current models
        model_graphs = [ContinuousGraph([]) for _ in lables]
        n_labels = len(lables)
        for i in range(n_labels):
            lb = lables[i]
            model_graphs[i].add_non_emitting_state()
            for digit in lb:
                model_graphs[i].add_model(old_models[digit], digit)
                model_graphs[i].add_non_emitting_state()

        for i in range(data_len):
            m = model_graphs[i]
            x = data[i]

            _, path = sentence_viterbi(x, m)
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
