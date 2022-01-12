This repository contains a Python implementation of speech recognition (10 digits in English) using
scipy (numpy).

# Folders
- src/speech: contains the whole project as a Python package.
- src/models: trained models using aurora dataset (the dataset is not open source so it is not included in this repository).
    Note that the models contains two different ways of pronouncing '0' ('zero' and 'ou').
- src/speech/audio_capture: implementation of speech endpointing (voice activity detection)
and file saving
- src/speech/feature: feature computation using typical MFCC procedures.
- src/speech/recognition: implementation of speech recognition using segmental K-means,
 dynamic time warping, GMM-HMM, and Expectation-Maximization algorithm.