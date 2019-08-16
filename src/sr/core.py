from sr.audio_capture import *
from sr.feature import *
from sr.recognition import *
from scipy.io import wavfile
import numpy as np
import pickle
import os
import python_speech_features as psf
from config import *
import re
import hashlib


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
    ys = [load_wav_as_mfcc1(filename) for filename in filenames]
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
            input = load_wav_as_mfcc1(f)
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


def aurora_continuous_train():
    models = []
    for digit in digit_names:
        file = open('models-4gaussians-em-realign/' + digit + '.pkl', 'rb')
        models.append(pickle.load(file))
        file.close()

    # get filenames
    sequence_regex = re.compile('(?<=_)[OZ0-9]+(?=[AB])')
    filenames = [f for f in os.listdir('train') if os.path.isfile(os.path.join('train', f))]

    # get transcripts
    sequences = [re.search(sequence_regex, test_filename).group(0) for test_filename in filenames]
    labels = [list(map(lambda x: filename_index_map[x], s)) for s in sequences]

    use_cache = False
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    import inspect
    data_hash = hashlib.md5("".join(filenames).encode()).hexdigest()
    code_hash = hashlib.md5("".join(inspect.getsource(sentence_viterbi).split()).encode()).hexdigest()
    if os.path.isfile('cache/cache.meta'):
        # compare hash value of labels to the value in cache.meta
        meta_file = open('cache/cache.meta', 'rb')
        meta_data = pickle.load(meta_file)
        meta_file.close()
        # the cache is clean only if data_hash didn't change, otherwise cache is dirty
        if data_hash == meta_data['data_hash'] and code_hash == meta_data['code_hash']:
            use_cache = True
    else:
        meta_data = {'data_hash': data_hash, 'code_hash': code_hash}

    if use_cache:
        print('using cache/data.pkl')
        f = open('cache/data.pkl', 'rb')
        data = pickle.load(f)
        f.close()
    else:
        print('loading data')
        data = [load_wav_as_mfcc1(os.path.join('train', f)) for f in filenames]
        f = open('cache/data.pkl', 'wb')
        pickle.dump(data, f)
        f.close()
        # save meta data to file
        meta_file = open('cache/cache.meta', 'wb')
        pickle.dump(meta_data, meta_file)
        meta_file.close()

    continuous_train(data, models, labels, use_cache=use_cache)
