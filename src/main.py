# -*- coding: utf-8 -*-
from sr import *

if __name__ == "__main__":
    models = []
    for digit in range(11):
        file = open('models-continuous-4gaussians-em-norealign/' + str(digit) + '.pkl', 'rb')
        models.append(pickle.load(file))
        file.close()

    print('building model graph(s)...')
    model_graph = LayeredHMMGraph([])
    model_graph.add_non_emitting_state()
    model_graph.add_layer_from_models(models)
    model_graph.add_non_emitting_state()
    model_graph.add_layer_from_models(models)
    model_graph.add_non_emitting_state()
    model_graph.add_layer_from_models(models)
    model_graph.add_non_emitting_state()
    model_graph.add_layer_from_models(models)
    model_graph.add_non_emitting_state()
    model_graph.add_layer_from_models(models)
    model_graph.add_non_emitting_state()
    model_graph.add_layer_from_models(models)
    model_graph.add_non_emitting_state()
    model_graph.add_layer_from_models(models)
    model_graph.add_non_emitting_state(end=True)

    # get all test files using regular expressions
    print('loading data...')
    sequence_regex = re.compile('(?<=_)[OZ0-9]+(?=A)')
    test_filenames = [f for f in os.listdir('test') if os.path.isfile(os.path.join('test', f))]

    sequences = [re.search(sequence_regex, test_filename).group(0) for test_filename in test_filenames]

    labels = [list(map(lambda x: filename_index_map[x], s)) for s in sequences]
    data = [load_wav_as_mfcc('test/' + f) for f in test_filenames]

    print('recognizing...')


    def split_result(seq, val):
        ret = None
        found = False
        for e in seq:
            if e != val:
                if not found:
                    ret = e
                    found = True
            elif ret is not None:
                yield ret
                ret = None
                found = False


    n = len(test_filenames)
    correct = 0
    n_digits = 0
    digit_ndiff = 0
    for x, l in zip(data, labels):
        _, matched = sentence_viterbi(x, model_graph)
        matched = list(split_result(matched, 'NES'))

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
