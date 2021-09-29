#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Authors: 
#  	- Santiago Donaher: s.donaher@qmul.ac.uk
# 	- Alessio Xompero:  a.xompero@qmul.ac.uk
#
# MIT License

# Copyright (c) 2021 CORSMAL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""Script for testing the environment and setup for the CORSMAL Machine Learning baselines.
"""

import os
import numpy as np

# Test all packages
import librosa
import pandas
import sklearn
import skimage
from tqdm import tqdm
import soundfile as sf

# Load 3 models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize

import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # For pickle warnings, just for the demo

# Globals
svm_pretained_path  = './data/pretrained_models/svm.sav'
rf_pretrained_path  = './data/pretrained_models/rf.sav'
knn_pretrained_path = './data/pretrained_models/knn.sav'

data_path = './data/demo/'
SR = 22050

def parse_label(name):

    fi = int(name[5])
    fu = int(name[9])

    if (fi == 0 or fu == 0): cl = 0
    elif (fi == 1 and fu == 1): cl = 1  # Pasta both fillings
    elif (fi == 1 and fu == 2): cl = 2
    elif (fi == 2 and fu == 1): cl = 3  # Rice both fillings
    elif (fi == 2 and fu == 2): cl = 4
    elif (fi == 3 and fu == 1): cl = 5  # Water both fillings
    elif (fi == 3 and fu == 2): cl = 6
    else: print('Wrong class assignment')

    return cl

def rms_normalization(s):
    return s/np.sqrt(np.sum(np.square((np.abs(s))))/len(s))

def extract_feature_full(file_name):

    # TODO remove for previous librosa
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1: X = X[:,0]
    X = X.T

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

    return mfccs,chroma,mel,contrast,tonnetz

# 2 features, ESC-50 paper
def extract_feature_ESC(file_name):

    # TODO remove for previous librosa
    X, sample_rate = librosa.load(file_name)
    if X.ndim > 1: X = X[:,0]
    X = X.T

    # Zero-crossing rate
    zcr_coef = librosa.feature.zero_crossing_rate(X).T
    zcr_mean = np.mean(zcr_coef, axis=0)
    zcr_stdd = np.std(zcr_coef, axis=0)

    # Mcc (mel-frequency cepstral coefficients)
    mfcc_coef = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T # def. n_mfcc=20
    mfcc_mean = np.mean(mfcc_coef, axis=0)
    mfcc_stdd = np.std(mfcc_coef, axis=0)
    
    # From the 1st to the 12th coefficient + ZCR mean and stdd
    return mfcc_mean[1:13], mfcc_stdd[1:13], zcr_mean, zcr_stdd

# Feature extraction functions
def data_processing():

    data_spectro = []
    data_esc = []
    data_full = []
    labels = []

    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith('.wav'):
            full_signal, sr = librosa.load(os.path.join(data_path, filename), sr=SR)
            signal = rms_normalization(full_signal)

            X = librosa.stft(signal, n_fft=512)
            spectro = librosa.amplitude_to_db(abs(X))
            spectro = resize(spectro, (96, 96))
            spectro = spectro.flatten()
            data_spectro.append(spectro)

            # 5F
            mfccs, chroma, mel, contrast, tonnetz = extract_feature_full(os.path.join(data_path, filename))
            data_full.append(np.hstack([mfccs,chroma,mel,contrast,tonnetz]))

            # ESC-50
            mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd = extract_feature_ESC(os.path.join(data_path, filename))
            data_esc.append(np.hstack([mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd]))


            # Label
            label = parse_label(filename)
            labels.append(label)

    return np.array(data_spectro), np.array(data_esc), np.array(data_full), labels


if __name__ == '__main__':

    print('\n[i] Running ML baselines demo for the CORSMAL Containers Manipulation dataset\n')
    print(' This demo will test:')
    print(' - Environment and dependencies')
    print(' - Data loading and processing')
    print(' - Models initialization')
    print(' - Pre-trained models loading')

    # Test packages
    print('[i] All packages loaded successfully')

    # Load and process data
    print('[i] Data loading and preprocessing...')
    data_spectro, data_esc, data_full, labels = data_processing()
    print('[i] Data loaded and processed successfully')

    # Initialize models
    svm_raw = SVC(C=20.0, gamma=0.00001)
    rf_raw  = RandomForestClassifier()
    knn_raw = KNeighborsClassifier()
    print('[i] Models initialized succesfully')

    # Load pretrained models
    svm_pret = pickle.load(open(svm_pretained_path, 'rb'))
    rf_pret  = pickle.load(open(rf_pretrained_path, 'rb'))
    knn_pret = pickle.load(open(knn_pretrained_path, 'rb'))
    print('[i] Pretrained models loaded succesfully')

    print('\n[i] end_of_script')
