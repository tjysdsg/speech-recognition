# -*- coding: utf-8 -*-
from sr.audio_capture import *
from sr.feature import *
from sr.recognition import *
from scipy.io import wavfile
import numpy as np
import pickle
import re
import os
import python_speech_features as psf
from config import *


def delta_feature(feat):
    delta = np.zeros(feat.shape)
    for i in range(len(feat)):
        if i == 0:
            delta[i] = feat[i + 1] - feat[i]
        elif i == len(feat) - 1:
            delta[i] = feat[i] - feat[i - 1]
        else:
            delta[i] = feat[i + 1] - feat[i - 1]
    return delta


def load_wav_as_mfcc(path):
    fb, mfcc = mfcc_features(path)
    df = delta_feature(mfcc)
    ddf = delta_feature(df)
    features = np.concatenate([mfcc, df, ddf], axis=1)
    features = standardize(features)
    return features


def load_wav_as_mfcc1(path):
    """
    Another version of load_wav_as_mfcc using library python_speech_feature to calculate
    mfcc, to check if this implementation is correct.
    """
    sample_rate, signal = wavfile.read(path)
    mfcc = psf.mfcc(signal, sample_rate, nfilt=40, preemph=0.95, appendEnergy=False, winfunc=np.hamming)
    df = delta_feature(mfcc)
    ddf = delta_feature(df)
    features = np.concatenate([mfcc, df, ddf], axis=1)
    features = standardize(features)
    return features


def make_HMM(filenames, n_segs, use_gmm, use_em):
    print('Loading wav files to mfcc features')
    ys = [load_wav_as_mfcc(filename) for filename in filenames]
    m = HMM(n_segs)
    print('Fitting HMMs')
    model = m.fit(ys, n_gaussians=4, use_gmm=use_gmm, use_em=use_em)
    return model


def train(filenames, model_folder, model_name, n_segs, use_gmm, use_em):
    models = make_HMM(filenames, n_segs, use_gmm=use_gmm, use_em=use_em)
    file = open(os.path.join(model_folder, model_name + '.pkl'), 'wb')
    pickle.dump(models, file)


def test(models, folder, file_patterns):
    n_passed = 0
    n_tests = 0

    # get the best model for every digit
    for digit in range(len(digit_names)):
        # get all test files using regular expresssions
        filenames = [os.path.join(folder, file) for file in os.listdir(folder) if
                     re.match(file_patterns[digit], file)]
        # update the total number of tests
        n_tests += len(filenames)

        # do evaluation on all models, find the best one
        # NOTE: the evaluation is based on costs
        for f in filenames:
            input = load_wav_as_mfcc(f)
            best_model = 0
            c = np.inf
            # find the best model
            for i in range(len(models)):
                m = models[i]
                cost = m.evaluate(input)
                if cost < c:
                    c = cost
                    best_model = i
            # if the best model is correct, move on
            if best_model == digit:
                n_passed += 1
            # if the best model is wrong, log info
            else:
                print("Digit:", digit_names[digit], "is wrong")
    return n_passed / n_tests


if __name__ == "__main__":
    if True:
        models = []
        for digit in digit_names:
            file = open('models-4gaussians-em-realign/' + digit + '.pkl', 'rb')
            models.append(pickle.load(file))
            file.close()

        # get filenames
        f = open('/home/tjy/repos/aurora_digits/TRAIN.filelist')
        filenames = [line.rstrip('\n\r ') for line in f]
        f.close()
        # get transcripts
        import re

        ex = re.compile(r'^[A-Za-z\s]+')
        f = open('/home/tjy/repos/aurora_digits/TRAIN.transcripts')
        transcripts = [ex.match(line)[0].rstrip('\n\r ') for line in f]
        transcripts = [t.replace('sil', '') for t in transcripts]
        f.close()

        digit_name_idx_map = {'one': 0,
                              'two': 1,
                              'three': 2,
                              'four': 3,
                              'five': 4,
                              'six': 5,
                              'seven': 6,
                              'eight': 7,
                              'nine': 8,
                              'oh': 9,
                              'zero': 10}

        print('loading data')
        data = [load_wav_as_mfcc(os.path.join(data_path, 'train', f + '.wav')) for f in filenames]
        print('loading transcripts')

        labels = [list(map(lambda x: digit_name_idx_map[x], t.split())) for t in transcripts]
        print('-' * 25)
        print('labels:')
        print(labels)
        print('=' * 25)

        continuous_train(data, models, labels)
