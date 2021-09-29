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
from sklearn.decomposition import PCA
from skimage.transform import resize

PR_TEST_DIR = 'private_test'

MAN_ANNOT = True
SR = 22050
SIG_LEN = SR * 10
PCA_COMP = 128

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


# Spectrograms
def rms_normalization(s):
    return s/np.sqrt(np.sum(np.square((np.abs(s))))/len(s))

def train_spectro(datapath):

    data = []
    labels = []

    if MAN_ANNOT:
        print('[i] USING manual annotations for action segmentation')
        man_df = pd.read_csv('./data/man_annotations.csv')
    else: print('[i] NOT using manual annotations for action segmentation')

    for objid in range(1, 10):
        print('Container {} of {}'.format(objid, 9))
        containerpath = datapath + 'train/{:d}'.format(objid)
        sequence = 0

        if objid < 7: list_files = populate_filenames(0)
        else: list_files = populate_filenames(1)

        for f in tqdm(list_files, leave=False):

            full_signal, sr = librosa.load('{}/audio/{}_audio.wav'.format(containerpath, f), sr=SR)

            tmp_df = man_df[(man_df.file_name.str.contains(f)) & (man_df.folder_num == objid)]
            man_beg = tmp_df.start.item()
            man_end = tmp_df.end.item()

            if (man_beg!=-1 and man_end!=-1):
                full_signal = full_signal[int(man_beg*sr):int(man_end*sr)]
                # print('\nOrig {} -> New: {} ({}-{})'.format(len(full_signal)/SR, man_end-man_beg, man_beg, man_end))
            signal = rms_normalization(full_signal)

            if(len(signal) > SIG_LEN):
                signal = signal[0:SIG_LEN]
            elif(len(signal) < SIG_LEN):
                zpad = np.zeros((SIG_LEN - len(signal), 1))
                signal = np.append(signal, zpad)

            X = librosa.stft(signal, n_fft=512)
            spectro = librosa.amplitude_to_db(abs(X))
            spectro = resize(spectro, (96, 96))
            spectro = spectro.flatten()
            # data.append(spectro)

            label = parse_label(f)
            # labels.append(label)

            data.append([f, int('{:d}'.format(objid)), spectro, sequence, label])
            sequence += 1

    return np.array(data)
    # return np.array(data), np.array(labels, dtype = np.int)


def putest_spectro(CCM_PATH):
    
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

            full_signal, sr = librosa.load('{}/audio/{}_audio.wav'.format(containerpath, f), sr=SR)
            signal = rms_normalization(full_signal)

            if(len(signal) > SIG_LEN):
                signal = signal[0:SIG_LEN]
            elif(len(signal) < SIG_LEN):
                zpad = np.zeros((SIG_LEN - len(signal), 1))
                signal = np.append(signal, zpad)
            
            X = librosa.stft(signal, n_fft=512)
            spectro = librosa.amplitude_to_db(abs(X))
            spectro = resize(spectro, (96, 96))
            spectro = spectro.flatten()

            data.append([f, int('{:d}'.format(objid)), spectro, sequence])
            sequence += 1

    return np.array(data)

def prtest_spectro(CCM_PATH):
    
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

            full_signal, sr = librosa.load('{}/audio/{}_audio.wav'.format(containerpath, f), sr=SR)
            signal = rms_normalization(full_signal)

            if(len(signal) > SIG_LEN):
                signal = signal[0:SIG_LEN]
            elif(len(signal) < SIG_LEN):
                zpad = np.zeros((SIG_LEN - len(signal), 1))
                signal = np.append(signal, zpad)

            X = librosa.stft(signal, n_fft=512)
            spectro = librosa.amplitude_to_db(abs(X))
            spectro = resize(spectro, (96, 96))
            spectro = spectro.flatten()

            data.append([f, int('{:d}'.format(objid)), spectro, sequence])
            sequence += 1

    return np.array(data)

def second_spectro(CCM_PATH):
    
    data = []
    sequence = 0

    for subdir, dirs, files in os.walk(CCM_PATH + 'setup2/'):
        for filename in tqdm(files, leave=False):
            filepath = subdir + os.sep + filename
            if filepath.endswith('.wav'):
                full_signal, sr = librosa.load(filepath, sr=SR)
                signal = rms_normalization(full_signal)

                if(len(signal) > SIG_LEN):
                    signal = signal[0:SIG_LEN]
                elif(len(signal) < SIG_LEN):
                    zpad = np.zeros((SIG_LEN - len(signal), 1))
                    signal = np.append(signal, zpad)

                X = librosa.stft(signal, n_fft=512)
                spectro = librosa.amplitude_to_db(abs(X))
                spectro = resize(spectro, (96, 96))
                spectro = spectro.flatten()
                
                f_name = os.path.split(filepath)[1]

                # print(f_name)
                label = int(f_name[-5])
                container = int(f_name[1:3])
                
                data.append([os.path.split(filepath)[1], container, spectro, sequence, label])
                sequence += 1

    return np.array(data)

def main_feat_spectro(datasplit, features, CCM_PATH):

    print('\n[i] Extracting {} from {}'.format(features, datasplit))

    if datasplit=='train':
        data = train_spectro(CCM_PATH)
    
    elif datasplit=='val':
        data = train_spectro(CCM_PATH)

    elif datasplit=='putest':
        data = putest_spectro(CCM_PATH)

    elif datasplit=='prtest':
        data = prtest_spectro(CCM_PATH)

    elif datasplit=='sec_setup':
        data = second_spectro(CCM_PATH)

    # Apply PCA if wanted
    if features == 'pca':
        print('[i] Computing PCA')

        data_tmp = data[:,2]
        spectros = []
        for i in range(len(data_tmp)):
            spectros.append(data_tmp[i])

        pca = PCA(PCA_COMP)
        X_comp = pca.fit_transform(spectros)
        print('[i] Reduced to 128 components')

        pca_data = []
        for i in range(len(X_comp)):
            pca_data.append(np.array(X_comp[i]))

        data[:,2] = pca_data

    # Save files
    data_filename = './data/{}_{}.npy'.format(datasplit, features)
    with open(data_filename, "wb") as fnpy: np.save(fnpy, data)
    