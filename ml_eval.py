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

import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from script.feat_spect import main_feat_spectro
from script.feat_hand import main_hand_feat
from script.export_results import gen_results_csv

CCM_PATH = 'D:/CCM_dataset/'

def eval_knn(x_train, y_train, x_test):

    print('\nFitting KNN...')
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    return clf.predict(x_test)

def eval_svm(x_train, y_train, x_test):

    print('\nFitting SVM...')
    clf = SVC(C=20.0, gamma=0.00001)
    clf.fit(x_train, y_train)

    return clf.predict(x_test)

def eval_rf(x_train, y_train, x_test):

    print('\nFitting Random Forest...')
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    return clf.predict(x_test)

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--features', default='spectro', type=str, choices=['spectro', 'pca', 'esc', 'full'])
    parser.add_argument('--data', default='val', type=str, choices=['val', 'putest', 'prtest', 'sec_setup'])
    args = parser.parse_args()

    features = args.features
    datasplit = args.data

    # Data filenames: [datasplit]_[feature]_[x/y].npy
    train_data_filename = './data/train_{}.npy'.format(features)
    data_filename = './data/{}_{}.npy'.format(datasplit, features)

    # Training data
    if not os.path.isfile(train_data_filename):
        print('[i] No processed training data found, processing...')
        if (features == 'spectro') or (features == 'pca'):
            main_feat_spectro('train', features, CCM_PATH)
        if (features == 'esc') or (features == 'full'):
            main_hand_feat('train', features, CCM_PATH)

    else: print('[i] Loading processed training data')
    data_train = np.load(train_data_filename, allow_pickle=True)

    # Validation/evaluation data
    if not os.path.isfile(data_filename):
        print('[i] No processed evaluation data found, processing...')
        if (features == 'spectro') or (features == 'pca'):
            main_feat_spectro(datasplit, features, CCM_PATH)
        if (features == 'esc') or (features == 'full'):
            main_hand_feat(datasplit, features, CCM_PATH)

    else: print('[i] Loading processed evaluation data')
    data_eval = np.load(data_filename, allow_pickle=True)

    # Parse all data
    X = []
    for i in range(len(data_train)): X.append(data_train[:,2][i])
    Y = data_train[:,4].tolist()
    x_train, _, y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    x_test = []
    for i in range(len(data_eval)): x_test.append(data_eval[:,2][i])

    # Train and predict with the 3 methods
    y_pred_svm = eval_svm(x_train, y_train, x_test)
    y_pred_knn = eval_knn(x_train, y_train, x_test)
    y_pred_rf = eval_rf(x_train, y_train, x_test)

    # Parse and export results
    outdir = './output/'
    if not os.path.exists(outdir): os.makedirs(outdir)

    pred_csv_svm = gen_results_csv(datasplit, y_pred_svm, data_eval[:,1].tolist(), data_eval[:,3].tolist())
    pred_csv_svm.to_csv('{}{}_{}_SVM.csv'.format(outdir, datasplit, features), index=False)

    pred_csv_knn = gen_results_csv(datasplit, y_pred_knn, data_eval[:,1].tolist(), data_eval[:,3].tolist())
    pred_csv_knn.to_csv('{}{}_{}_KNN.csv'.format(outdir, datasplit, features), index=False)

    pred_csv_rf = gen_results_csv(datasplit, y_pred_rf, data_eval[:,1].tolist(), data_eval[:,3].tolist())
    pred_csv_rf.to_csv('{}{}_{}_RF.csv'.format(outdir, datasplit, features), index=False)

    print('\n[i] Results exported to {}'.format(outdir))
