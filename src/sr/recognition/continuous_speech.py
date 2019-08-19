# -*- coding: utf-8 -*-
from .decode import decode_hmm_states
from .kmeans import kmeans
from .hmm_state import GMM, NES, mahalanobis
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


def continuous_train(data, models, label_seqs, n_gaussians=4, max_iteration=1000):
    # remember old models
    old_models = models
    new_models = copy.deepcopy(models)

    for iter in range(max_iteration):
        print('=' * 25)
        print('Continuous training iteration:', iter)
        print('Rearranging data for hmm training...')
        segments = {}
        data_len = len(data)

        print('Building state sequences')
        sequences_and_transitions = [build_state_sequences(new_models, [[l] for l in labels]) for labels in label_seqs]

        print('Rearranging data, this may take a while...')
        for i in range(data_len):
            print('Progress:', str(int(100 * i / data_len)) + "%", end='\r', flush=True)
            seq = sequences_and_transitions[i][0]
            trans = sequences_and_transitions[i][1]
            end_idx = sequences_and_transitions[i][2]
            x = data[i]

            _, path = decode_hmm_states(x, seq, trans, end_points=[[end_idx, -1]])
            path = list(reversed(path.tolist()))
            start_idx = None
            for pt in path:
                r = pt[0]
                c = pt[1]
                if start_idx is None:
                    start_idx = c
                if type(seq[r]) is NES and start_idx < c:
                    if segments.get(seq[r]) is None:
                        segments[seq[r]] = [x[start_idx: c]]
                    else:
                        segments[seq[r]].append(x[start_idx: c])
                        start_idx = None

        print('Complete data rearrangement')
        print("=" * 25)

        # remove templates that are too small to do dtw on
        # for w in segments.keys():
        #     seg = segments[w]
        #     seg_len = len(seg)
        #     del_i = []
        #     for si in range(seg_len):
        #         if seg[si].shape[0] < 5:
        #             warnings.warn('Removing digit templates that are too small', UserWarning)
        #             del_i.append(si)

        #     del_i.sort(reverse=True)
        #     for si in del_i:
        #         del seg[si]

        # continuous training
        print('Doing HMM training...')
        for state in segments.keys():
            seg = segments[state]

            # train gmm states
            n_splits = int(np.log(n_gaussians))
            assert n_splits > 0

            mu = np.mean(seg, axis=1)
            centroids = np.array([mu])
            weights = np.full(n_gaussians, 1 / data.shape[0])
            for i in range(n_splits):
                lcentroids = centroids * 0.9
                hcentroids = centroids * 1.1
                centroids = np.concatenate([lcentroids, hcentroids], axis=0)
                clusters, centroids, variance = kmeans(data, 2 ** (i + 1), centroids, dist_fun=mahalanobis)

                # calculate and update weights of distributions
                cs, c_counts = np.unique(clusters, return_counts=True)
                for c in cs:
                    weights[c] = c_counts[c] / data.shape[0]

                # update model parameters
                state.update_models(centroids, variance, weights[:2 ** (i + 1)])
                # Expectation-maximization
                state.em(data, 2 ** (i + 1))

        # save new models to files
        # TODO: use command line argument for output model path
        for i in range(len(new_models)):
            file = open(os.path.join('models-continuous-4gaussians-em-realign', str(i) + '.pkl'), 'wb')
            pickle.dump(new_models[i], file)
            file.close()
        # check if converged
        convergence = [new_m == old_m for new_m, old_m in zip(new_models, old_models)]
        converged = np.alltrue(convergence)
        if not converged:
            old_models = new_models
            new_models = copy.deepcopy(old_models)
        else:
            print('Continuous training converged')
            break
