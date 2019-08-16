from main import *


def cli():
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='CLI for speech recognition functionality.')
    parser.add_argument('action', metavar='ACTION', type=str, nargs=1,
                        choices=['train', 'test', 'record', 'continuous'],
                        help='Action to perform.')
    parser.add_argument("-d", "--model-directory", default='models-4gaussians-em',
                        help="Directory which the trained models are stored, or test models are used.")
    parser.add_argument("-g", "--gmm", help="Use GMM-HMM as the model.", default=False, action='store_true')
    parser.add_argument("-e", "--em", help="Use EM algorithm to train models.", default=False, action='store_true')
    args = parser.parse_args()

    if args.action[0] == 'train':
        print('training...')
        # data folder location
        folder = os.path.join(data_path, 'train')
        for digit in digit_names:
            filenames = [os.path.join(folder, f) for f in os.listdir(folder) if
                         re.match('[A-Z]+_' + digit + '[AB].wav', f)]

            train(filenames, args.model_directory, digit, n_segs=5, use_gmm=args.gmm, use_em=args.em)

    if args.action[0] == 'test':
        print('testing...')
        # get all the models from pickle files
        models = []
        for digit in digit_names:
            file = open(args.model_directory + digit + '.pkl', 'rb')
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
            file = open(args.model_directory + digit + '.pkl', 'rb')
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

    if args.action[0] == 'continuous':
        aurora_continuous_train()


if __name__ == "__main__":
    cli()
