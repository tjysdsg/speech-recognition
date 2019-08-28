# -*- coding: utf-8 -*-
from .decode import decode_hmm_states
from .kmeans import kmeans
from .hmm_state import GMM, NES, mahalanobis
from .hmm import HMM
import os
import pickle
from typing import List, AnyStr
import copy
import numpy as np


def build_state_sequences(hmms: List[HMM], label_matrix: List[List[int]]):
    seq = []
    # count the number of number of states in total
    n_states = 1  # one non-emitting state at the very beginning
    for labels in label_matrix:
        for l in labels:
            n_states += len(hmms[l].gmm_states)
        n_states += 1  # one non-emitting state after each layer

    trans = np.full((n_states, n_states), np.inf)
    seq.append(NES())
    start_state_indices = [[] for _ in label_matrix]
    end_state_indices = [[] for _ in label_matrix]
    nes_indices = [0]
    # NOTE: assume that all hmms have the same number of states
    n_states = len(hmms[0].gmm_states)

    for layer in range(len(label_matrix)):
        labels = label_matrix[layer]
        for l in labels:
            trans_offset = len(seq)
            start_state_indices[layer].append(trans_offset)
            end_state_indices[layer].append(trans_offset + n_states - 1)
            hmm = hmms[l]
            assert n_states == len(hmm.gmm_states)
            seq += hmm.gmm_states
            trans[trans_offset: trans_offset + n_states, trans_offset:trans_offset + n_states] = hmm.transitions
        seq.append(NES())
        nes_indices.append(len(seq) - 1)

    prev_nes_ii = 0
    next_nes_ii = 1
    for layer in range(len(label_matrix)):
        for s in start_state_indices[layer]:
            trans[s, nes_indices[prev_nes_ii]] = 0
        for e in end_state_indices[layer]:
            trans[nes_indices[next_nes_ii], e] = 0
        prev_nes_ii += 1
        next_nes_ii += 1

    return seq, trans, end_state_indices[-1]


def continuous_train(data: List[np.ndarray], models: List[HMM], label_seqs: List[List[int]], output_path: AnyStr,
                     n_gaussians: int = 4,
                     n_segments: int = 5,
                     max_iteration: int = 1000):
    # remember old models
    old_models = models
    new_models = copy.deepcopy(models)

    # create a hash table that maps the index of a model to a list of its states
    # it is intended for updating the transition costs of models
    modelidx_state_map = {}
    n_models = len(new_models)
    for i in range(n_models):
        modelidx_state_map[i] = []
        for s in new_models[i].gmm_states:
            modelidx_state_map[i].append(s)

    for iter in range(max_iteration):
        print('=' * 25)
        print('Continuous training iteration:', iter)
        gmm_data = {}
        data_len = len(data)

        print('Building state sequences')
        sequences_and_transitions = [build_state_sequences(new_models, [[l] for l in labels]) for labels in label_seqs]

        print('Rearranging data, this may take a while...')
        for i in range(data_len):
            seq = sequences_and_transitions[i][0]
            trans = sequences_and_transitions[i][1]
            end_pts = sequences_and_transitions[i][2]
            x = data[i]

            _, path = decode_hmm_states(x, seq, trans, end_points=list(map(lambda x: [x, -1], end_pts)))
            path = list(reversed(path.tolist()))
            start_idx = None
            current_state = None
            # TODO: use `array[1:] - array[:-1]` vectorization to optimized the process of finding gmm state segments
            for pt in path:
                r = pt[0]
                c = pt[1]
                if start_idx is None and type(seq[r]) is not NES:
                    start_idx = c
                    current_state = r
                if r != current_state and start_idx is not None and start_idx < c:
                    if gmm_data.get(seq[current_state]) is None:
                        gmm_data[seq[current_state]] = [x[start_idx: c]]
                    else:
                        gmm_data[seq[current_state]].append(x[start_idx: c])
                    start_idx = None
                    current_state = None
            print('Progress:', str(int(100 * i / data_len)) + "%", end='\r', flush=True)

        print('Complete data rearrangement')
        print("=" * 25)

        # continuous training
        print('Doing HMM training...')
        for state in gmm_data.keys():
            assert type(state) is not NES
            # enforce type checking
            state: GMM = state

            seg = np.vstack(gmm_data[state])

            # train gmm states
            n_splits = int(np.log(n_gaussians))
            assert n_splits > 0

            centroids = np.mean(seg, axis=0)
            centroids = centroids.reshape((1, centroids.shape[0]))
            weights = np.full(n_gaussians, 1 / n_segments)
            for i in range(n_splits):
                lcentroids = centroids * 0.9
                hcentroids = centroids * 1.1
                centroids = np.concatenate([lcentroids, hcentroids], axis=0)
                clusters, centroids, variance = kmeans(seg, 2 ** (i + 1), centroids, dist_fun=mahalanobis)

                # calculate and update weights of distributions
                cs, c_counts = np.unique(clusters, return_counts=True)
                for c in cs:
                    weights[c] = c_counts[c] / n_segments

                # update model parameters
                state.update_models(centroids, variance, weights[:2 ** (i + 1)])
                # Expectation-maximization
                state.em(seg, 2 ** (i + 1))

        print('Updating other model parameters...')
        # update transition costs for each hmm
        for mi in range(n_models):
            states = modelidx_state_map[mi]
            for si in range(len(states)):
                if gmm_data.get(states[si]) is None:
                    # FIXME
                    import warnings
                    warnings.warn("No MFCC data for state", UserWarning)
                    continue
                state_data = gmm_data[states[si]]
                seg_len = 0
                n_temps = 0
                for sd in state_data:
                    sd: np.ndarray = sd
                    n_temps += 1
                    seg_len += sd.shape[0]
                p_jump = n_temps / seg_len
                if si < len(states) - 1:
                    new_models[mi].transitions[si + 1, si] = -np.log(p_jump)
                new_models[mi].transitions[si, si] = -np.log(1 - p_jump)

        # save new models to files
        for i in range(len(new_models)):
            file = open(os.path.join(output_path, str(i) + '.pkl'), 'wb')
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
