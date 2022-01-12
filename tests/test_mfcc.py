from sr.core import load_wav_as_mfcc1, load_wav_as_mfcc
import os
from matplotlib import pyplot as plt


def test_mfcc_features():
    mfcc = load_wav_as_mfcc(os.path.join('data', 'train', '0_0.wav'))
    mfcc1 = load_wav_as_mfcc1(os.path.join('data', 'train', '0_0.wav'))

    fig, axs = plt.subplots(2, 1)
    fig.set_figwidth(15)
    fig.set_figheight(10)

    axs[0].imshow(mfcc.T)
    axs[0].set_title('Standard implementation')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('feature')

    axs[1].imshow(mfcc1.T)
    axs[1].set_title('My implementation')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('feature')

    plt.show()
