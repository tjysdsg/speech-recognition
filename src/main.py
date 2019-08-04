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
from multiprocessing import Process


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


def make_HMM(filenames, n_segs):
    print('Loading wav files to mfcc features')
    ys = [load_wav_as_mfcc(filename) for filename in filenames]
    m = HMM(n_segs)
    print('Fitting HMMs')
    model = m.fit(ys, n_gaussians=4, use_em=True)
    return model


def train(filenames, model_folder, model_name):
    models = make_HMM(filenames, 5)
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


def cli():
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='CLI for speech recognition functionality.')
    parser.add_argument('action', metavar='ACTION', type=str, nargs=1, choices=['train', 'test', 'record'],
                        help='Action to perform. Can be one of the following:\n train \n test \n record')
    args = parser.parse_args()

    if args.action[0] == 'train':
        print('training...')
        # data folder location
        folder = os.path.join(data_path, 'train')
        processes = []
        for digit in digit_names:
            filenames = [os.path.join(folder, f) for f in os.listdir(folder) if
                         re.match('[A-Z]+_' + digit + '[AB].wav', f)]

            if use_multiprocessing:
                # create a new process for every digit
                p = Process(target=train, args=(filenames, 'models', digit))
                p.start()
                processes.append(p)
            else:
                train(filenames, 'models', digit)
        # wait for all processes to end (if use multiple processes)
        if use_multiprocessing:
            # wait until all processes finished their jobs
            for p in processes:
                p.join()

    if args.action[0] == 'test':
        print('testing...')
        # get all the models from pickle files
        models = []
        for digit in digit_names:
            file = open('models-4gaussians-em-realign/' + digit + '.pkl', 'rb')
            models.append(pickle.load(file))
            file.close()
        # get file patterns for each digit
        file_patterns = ['[A-Z]+_' + digit + '[AB].wav' for digit in digit_names]
        # do tests, print the accuracy
        print(test(models, os.path.join(data_path, 'test'), file_patterns))

    if args.action[0] == 'record':
        record('test.wav')
        # get all test files using regular expressions
        f = 'test.wav'

        input_audio = load_wav_as_mfcc(f)
        # get all the models from pickle files
        models = []
        for digit in digit_names:
            file = open('models-4gaussians-em-realign/' + digit + '.pkl', 'rb')
            models.append(pickle.load(file))
            file.close()

        best_model = 0
        c = np.inf
        # find the best model
        for i in range(len(models)):
            m = models[i]
            cost = m.evaluate(input_audio)
            if cost < c:
                c = cost
                best_model = i
        print(best_model)


if __name__ == "__main__":
    # cli()

    # continuous speech recognition
    # get all the models from pickle files
    models = []
    for digit in digit_names:
        file = open('models-4gaussians-em-realign/' + digit + '.pkl', 'rb')
        models.append(pickle.load(file))
        file.close()

    n_segments = 5
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
