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
    if False:
        # continuous speech recognition
        # get all the models from pickle files
        models = []
        for digit in digit_names:
            file = open('models/' + digit + '.pkl', 'rb')
            models.append(pickle.load(file))
            file.close()

        model_graph = LayeredHMMGraph()
        nes0 = model_graph.add_non_emitting_state()
        model_graph.add_layer_from_models(models[1:9])
        model_graph.add_non_emitting_state()
        model_graph.add_layer_from_models(models)
        model_graph.add_non_emitting_state()
        model_graph.add_layer_from_models(models)
        nes1 = model_graph.add_non_emitting_state()

        model_graph.add_edge(nes0, nes1)

        model_graph.add_layer_from_models(models)
        model_graph.add_non_emitting_state()
        model_graph.add_layer_from_models(models)
        model_graph.add_non_emitting_state()
        model_graph.add_layer_from_models(models)
        model_graph.add_non_emitting_state()
        model_graph.add_layer_from_models(models)
        model_graph.add_non_emitting_state()

        # record('test.wav')
        # get all test files using regular expressions
        f = 'test.wav'
        x = load_wav_as_mfcc(f)
        _, matched = sentence_viterbi(x, model_graph)
        print(matched)

    if True:
        models = []
        for digit in digit_names:
            file = open('models-4gaussians-em-realign/' + digit + '.pkl', 'rb')
            models.append(pickle.load(file))
            file.close()
        filenames = ['FAC_13A',
                     'FAC_1473533A',
                     'FAC_172A',
                     'FAC_1911446A',
                     'FAC_1A',
                     'FAC_1B',
                     'FAC_1O1A',
                     'FAC_24Z52A',
                     'FAC_26555A',
                     'FAC_282A',
                     'FAC_29A',
                     'FAC_29OA',
                     'FAC_2A',
                     'FAC_2B',
                     'FAC_32421A',
                     'FAC_33O31A',
                     'FAC_369O5A',
                     'FAC_37O1641A',
                     'FAC_39ZA',
                     'FAC_3A',
                     'FAC_3B',
                     'FAC_434A',
                     'FAC_43A',
                     'FAC_47A',
                     'FAC_4876A',
                     'FAC_4915A',
                     'FAC_4A',
                     'FAC_4B',
                     'FAC_54A',
                     'FAC_5821A',
                     'FAC_59Z2A',
                     'FAC_5A',
                     'FAC_5B',
                     'FAC_5Z26ZA',
                     'FAC_5Z31ZZ4A',
                     ]

        transcripts = ['one three',
                       'one four seven three five three three',
                       'one seven two',
                       'one nine one one four four six',
                       'one',
                       'one',
                       'one oh one',
                       'two four zero five two',
                       'two six five five five',
                       'two eight two',
                       'two nine',
                       'two nine oh',
                       'two',
                       'two',
                       'three two four two one',
                       'three three oh three one',
                       'three six nine oh five',
                       'three seven oh one six four one',
                       'three nine zero',
                       'three',
                       'three',
                       'four three four',
                       'four three',
                       'four seven',
                       'four eight seven six',
                       'four nine one five',
                       'four',
                       'four',
                       'five four',
                       'five eight two one',
                       'five nine zero two',
                       'five',
                       'five',
                       'five zero two six zero',
                       'five zero three one zero zero four']

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
        data = np.array([load_wav_as_mfcc(os.path.join(data_path, 'train', f + '.wav')) for f in filenames])
        print('loading transcripts')

        labels = [list(map(lambda x: digit_name_idx_map[x], t.split())) for t in transcripts]
        print('-' * 25)
        print('labels:')
        print(labels)
        print('=' * 25)

        # make model_graphs
        model_graphs = [ContinuousGraph([]) for _ in labels]
        n_labels = len(labels)
        for i in range(n_labels):
            lb = labels[i]
            model_graphs[i].add_non_emitting_state()
            for digit in lb:
                model_graphs[i].add_model(models[digit], digit)
                model_graphs[i].add_non_emitting_state()

        continuous_train(data, model_graphs, labels)
