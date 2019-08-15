# -*- coding: utf-8 -*-
from .model_graph import *
from .dtw import *
from .hmm import *
from itertools import chain
import os
import pickle
import gc
import multiprocessing


def continuous_train(data, models, lables, use_gmm=True, n_gaussians=4, use_em=True, max_iteration=1,
                     use_cache=False):
    # remember old models
    old_models = models
    for iter in range(max_iteration):
        converged = True
        print('=' * 25)
        print('continuous training iteration:', iter)
        print('building model graphs')
        # make model_graphs
        model_graphs = [ContinuousGraph([]) for _ in lables]
        n_labels = len(lables)
        for i in range(n_labels):
            lb = lables[i]
            model_graphs[i].add_non_emitting_state()
            for digit in lb:
                model_graphs[i].add_model(models[digit], digit)
                model_graphs[i].add_non_emitting_state()

        print('rearranging data for continuous training...')
        segments = {w: [] for w in chain.from_iterable(lables)}
        data_len = len(data)

        if use_cache:
            cache_file = open('cache/segments.pkl', 'rb')
            segments = pickle.load(cache_file)
            cache_file.close()
        else:
            for i in range(data_len):
                m = model_graphs[i]
                x = data[i]

                _, path = sentence_viterbi(x, m)
                print('progress:', i / data_len)

                # find segments for every digit, a segment is ended and started by 'NES'
                # each segment is a sequence of mfcc features, thus it's a 2d array-like
                i_nes = [i for i, x in enumerate(path) if x == "NES"]
                i_nes_len = len(i_nes)
                for index in range(i_nes_len - 1):
                    start_i = i_nes[index] + 1
                    segments[path[start_i]].append(x[start_i:i_nes[index + 1]])

            # remove templates that are too small to do dtw on
            for w in segments.keys():
                seg = segments[w]
                seg_len = len(seg)
                del_i = []
                for si in range(seg_len):
                    if seg[si].shape[0] < 5:
                        del_i.append(si)

                del_i.sort(reverse=True)
                for si in del_i:
                    del seg[si]

            cache_file = open('cache/segments.pkl', 'wb')
            pickle.dump(segments, cache_file)
            cache_file.close()

        # continuous training
        print('doing hmm training...')
        for w in segments.keys():
            # train a new HMM model using the segments
            m = HMM(5)
            m.fit(segments[w], n_gaussians, use_gmm, use_em)
            converged = converged and m == old_models[w]
            old_models[w] = m
            file = open(os.path.join('models', str(w) + '.pkl'), 'wb')
            pickle.dump(m, file)
            file.close()

        if converged:
            print('continuous training converged')
            break
