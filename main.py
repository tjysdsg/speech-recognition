# -*- coding: utf-8 -*-
import numpy as np
import pickle
from typing import List
import os, re
from config import digit_names, data_path, filename_index_map
from sr import *

if __name__ == "__main__":
    models: List[HMM] = []
    # for digit in digit_names:
    for digit in range(11):
        # file = open('models-4gaussians-em/' + str(digit) + '.pkl', 'rb')
        file = open('models-continuous-4gaussians-em-realign/' + str(digit) + '.pkl', 'rb')
        models.append(pickle.load(file))
        file.close()

    gmm_id_modelidx_map = {}
    for model_idx in range(len(models)):
        for g in models[model_idx].gmm_states:
            g: GMM = g
            gmm_id_modelidx_map[g.id] = model_idx

    # get all test files using regular expressions
    print('loading data...')
    sequence_regex = re.compile('(?<=_)[OZ0-9]+(?=A)')
    test_filenames = [f for f in os.listdir('test') if os.path.isfile(os.path.join('test', f))]

    sequences = [re.search(sequence_regex, test_filename).group(0) for test_filename in test_filenames]

    labels = [list(map(lambda x: filename_index_map[x], s)) for s in sequences]
    data = [load_wav_as_mfcc('test/' + f) for f in test_filenames]

    print('Building state sequences...')
    seq, trans, end_pts = build_state_sequences(models, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 7)
    print('Recognizing...')


    def split_result(seq, cond):
        ret = None
        found = False
        for e in seq:
            if not cond(e):
                if not found:
                    ret = e
                    found = True
            elif ret is not None:
                yield ret
                ret = None
                found = False
        if ret is not None:
            yield ret


    n = len(test_filenames)
    correct = 0
    n_digits = 0
    digit_ndiff = 0
    for x, l in zip(data, labels):
        _, matched = decode_hmm_states(x, seq, trans, end_points=list(map(lambda x: [x, -1], end_pts)))
        matched = matched[:, 0][::-1]
        # remove consecutive duplicates
        matched = matched[np.insert(np.diff(matched).astype(np.bool), 0, True)]
        # split by NES
        matched = list(split_result(matched, lambda x: type(seq[x]) is NES))
        # find model indices
        matched = list(map(lambda idx: gmm_id_modelidx_map[seq[idx].id], matched))

        n_digits += len(l)
        if matched == l:
            correct += 1
            print('Correct:', matched)
        else:
            print('Incorrect:', matched, l)
            # find how many digits are different
            matched = np.asarray(matched)
            l = np.asarray(l)
            diff = matched - l
            n_diff = np.count_nonzero(diff)
            print('Diff:', n_diff)
            digit_ndiff += n_diff

    print('Sequence accuracy:', correct / n)
    print('Digit accuracy:', (n_digits - digit_ndiff) / n_digits)
