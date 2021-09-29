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

import numpy as np
import librosa
from tqdm import tqdm
import os
import pandas as pd
import soundfile as sf

from skimage.transform import resize

PR_TEST_DIR = 'private_test'

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

def populate_filenames(mode):
    fnames_filling  = ['fi0_fu0', 'fi1_fu1', 'fi1_fu2', 'fi2_fu1', 'fi2_fu2', 'fi3_fu1', 'fi3_fu2']
    fnames_filling2 = ['fi0_fu0', 'fi1_fu1', 'fi1_fu2', 'fi2_fu1', 'fi2_fu2']
    list_filenames = []
    for s in range(0, 3):
        str_s = 's{:d}_'.format(s)
        for b in range(0, 2):
            str_b = '_b{:d}_'.format(b)
            for l in range(0, 2):
                str_l = 'l{:d}'.format(l)
                if mode == 0:
                    for f in fnames_filling:
                        list_filenames.append(str_s + f + str_b + str_l)
                else:
                    for f in fnames_filling2:
                        list_filenames.append(str_s + f + str_b + str_l)

    return list_filenames

# 5 features
def extract_feature_full(file_name):

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


def parse_audio_files(datapath):

    n_feat = 26   # 193 original feat, 26 ESC-50, 42 with all MFCC coefs.
    features, labels = np.empty((0,n_feat)), np.empty(0)

    for objid in range(1, 10):
        print('Container {} of {}'.format(objid, 9))
        containerpath = datapath + 'train/{:d}'.format(objid)
        sequence = 0

        if objid < 7: list_files = populate_filenames(0)
        else: list_files = populate_filenames(1)

        for f in tqdm(list_files, leave=False):

            # try: mfccs, chroma, mel, contrast,tonnetz = extract_feature_orig('{}/audio/{}_audio.wav'.format(containerpath, f))
            try: mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd = extract_feature('{}/audio/{}_audio.wav'.format(containerpath, f))
            except Exception as e:
                print("[Error] extract feature error in %s. %s" % (fn,e))
                continue

            # ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            # print('\nNum. of features', np.hstack([mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd]).shape)
            ext_features = np.hstack([mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd])
            features = np.vstack([features,ext_features])

            # Label
            label = parse_label(f)
            labels = np.append(labels, label)

    return np.array(features), np.array(labels, dtype = np.int)

###########

def train_hfeat(datapath, featset):

    data = []

    # if MAN_ANNOT:
    #     print('[i] USING manual annotations for action segmentation')
    #     man_df = pd.read_csv('./preprocess_data/man_annotations.csv')
    # else: print('[i] NOT using manual annotations for action segmentation')

    for objid in range(1, 10):
        print('Container {} of {}'.format(objid, 9))
        containerpath = datapath + 'train/{:d}'.format(objid)
        sequence = 0

        if objid < 7: list_files = populate_filenames(0)
        else: list_files = populate_filenames(1)

        for f in tqdm(list_files, leave=False):

            if featset=='full':
                mfccs, chroma, mel, contrast, tonnetz = extract_feature_full('{}/audio/{}_audio.wav'.format(containerpath, f))
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

            if featset=='esc':
                mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd = extract_feature_ESC('{}/audio/{}_audio.wav'.format(containerpath, f))
                ext_features = np.hstack([mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd])

            label = parse_label(f)

            data.append([f, int('{:d}'.format(objid)), ext_features, sequence, label])
            sequence += 1

    return np.array(data)
    
def putest_hfeat(CCM_PATH, featset):

    data = []

    for objid in range(10, 13):
        print('Container {} of {}'.format(objid, 12))
        containerpath = CCM_PATH + 'test/{:d}'.format(objid)
        sequence = 0

        list_files = []
        if objid < 12:
            for j in range(0, 84):
                list_files.append('{:04d}'.format(j))
        else:
            for j in range(0, 60):
                list_files.append('{:04d}'.format(j))

        for f in tqdm(list_files, leave=False):

            if featset=='full':
                mfccs, chroma, mel, contrast, tonnetz = extract_feature_full('{}/audio/{}_audio.wav'.format(containerpath, f))
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

            if featset=='esc':
                mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd = extract_feature_ESC('{}/audio/{}_audio.wav'.format(containerpath, f))
                ext_features = np.hstack([mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd])

            data.append([f, int('{:d}'.format(objid)), ext_features, sequence])
            sequence += 1

    return np.array(data)

def prtest_hfeat(CCM_PATH, featset):
    
    data = []

    for objid in range(13, 16):
        print('Container {} of {}'.format(objid, 15))
        # DOUBLECK THE PATH AND DIR NAME FOR PRIVATE TEST DATA
        containerpath = CCM_PATH + '{}/{:d}'.format(PR_TEST_DIR, objid)
        sequence = 0

        list_files = []
        if objid < 15:
            for j in range(0, 84):
                list_files.append('{:04d}'.format(j))
        else:
            for j in range(0, 60):
                list_files.append('{:04d}'.format(j))

        for f in tqdm(list_files, leave=False):

            if featset=='full':
                mfccs, chroma, mel, contrast, tonnetz = extract_feature_full('{}/audio/{}_audio.wav'.format(containerpath, f))
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

            if featset=='esc':
                mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd = extract_feature_ESC('{}/audio/{}_audio.wav'.format(containerpath, f))
                ext_features = np.hstack([mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd])

            data.append([f, int('{:d}'.format(objid)), ext_features, sequence])
            sequence += 1

    return np.array(data)

def second_hfeat(CCM_PATH, featset):
    
    data = []
    sequence = 0

    for subdir, dirs, files in os.walk(CCM_PATH + 'setup2/'):
        for filename in tqdm(files, leave=False):
            filepath = subdir + os.sep + filename
            if filepath.endswith('.wav'):
                
                if featset=='full':
                    mfccs, chroma, mel, contrast, tonnetz = extract_feature_full(filepath)
                    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

                if featset=='esc':
                    mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd = extract_feature_ESC(filepath)
                    ext_features = np.hstack([mfcc_mean, mfcc_stdd, zcr_mean, zcr_stdd])
                
                f_name = os.path.split(filepath)[1]

                # print(f_name)
                label = int(f_name[-5])
                container = int(f_name[1:3])
                
                data.append([os.path.split(filepath)[1], container, ext_features, sequence, label])
                sequence += 1

    return np.array(data)



def main_hand_feat(datasplit, features, CCM_PATH):

    print('\n[i] Extracting {} from {}'.format(features, datasplit))

    if datasplit=='train':
        data = train_hfeat(CCM_PATH, features)
    
    elif datasplit=='val':
        data = train_hfeat(CCM_PATH, features)

    elif datasplit=='putest':
        data = putest_hfeat(CCM_PATH, features)

    elif datasplit=='prtest':
        data = prtest_hfeat(CCM_PATH, features)

    elif datasplit=='sec_setup':
        data = second_hfeat(CCM_PATH, features)

    # # Save files
    data_filename = './data/{}_{}.npy'.format(datasplit, features)
    with open(data_filename, "wb") as fnpy: np.save(fnpy, data)

