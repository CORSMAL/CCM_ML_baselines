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
import pandas as pd
from datetime import datetime

def y_pred_parser(y_pred, cont_cap, fill_lev, fill_typ):

    for n in range(len(y_pred)):

        cont_cap.append(-1)

        if y_pred[n] == 0:
            fill_lev.append(0)
            fill_typ.append(0)
        elif y_pred[n] == 1:
            fill_lev.append(1)
            fill_typ.append(1)
        elif y_pred[n] == 2:
            fill_lev.append(2)
            fill_typ.append(1)
        elif y_pred[n] == 3:
            fill_lev.append(1)
            fill_typ.append(2)
        elif y_pred[n] == 4:
            fill_lev.append(2)
            fill_typ.append(2)
        elif y_pred[n] == 5:
            fill_lev.append(1)
            fill_typ.append(3)
        elif y_pred[n] == 6:
            fill_lev.append(2)
            fill_typ.append(3)
        else:
            fill_lev.append(-1)
            fill_typ.append(-1)

    return cont_cap, fill_lev, fill_typ

def gen_results_csv(data_split, y_pred, container_df, sequence_df):

    cont_cap  = []
    fill_lev  = []
    fill_typ  = []
    container = []
    sequence  = []

    if (data_split=='val') or (data_split=='all'):
        cont_cap, fill_lev, fill_typ = y_pred_parser(y_pred, cont_cap, fill_lev, fill_typ)
        [container.append(cont_i) for cont_i in container_df]
        [sequence.append(seq_i)   for seq_i  in sequence_df]
    else:
        for objid in range(1, 10):
            seq = 0
            if objid < 7:
                for j in range(0, 84):
                    cont_cap.append(-1); fill_lev.append(-1); fill_typ.append(-1)
                    container.append(objid)
                    sequence.append(seq)
                    seq += 1
            else:
                for j in range(0, 60):
                    cont_cap.append(-1); fill_lev.append(-1); fill_typ.append(-1)
                    container.append(objid)
                    sequence.append(seq)
                    seq += 1

    if (data_split=='putest') or (data_split=='all'):
        cont_cap, fill_lev, fill_typ = y_pred_parser(y_pred, cont_cap, fill_lev, fill_typ)
        [container.append(cont_i) for cont_i in container_df]
        [sequence.append(seq_i)   for seq_i  in sequence_df]

    else:
        for objid in range(10, 13):
            seq = 0
            if objid < 12:
                for j in range(0, 84):
                    cont_cap.append(-1); fill_lev.append(-1); fill_typ.append(-1)
                    container.append(objid)
                    sequence.append(seq)
                    seq += 1
            else:
                for j in range(0, 60):
                    cont_cap.append(-1); fill_lev.append(-1); fill_typ.append(-1)
                    container.append(objid)
                    sequence.append(seq)
                    seq += 1

    if (data_split=='prtest') or (data_split=='all'):
        cont_cap, fill_lev, fill_typ = y_pred_parser(y_pred, cont_cap, fill_lev, fill_typ)
        [container.append(cont_i) for cont_i in container_df]
        [sequence.append(seq_i)   for seq_i  in sequence_df]
    else:
        for objid in range(13, 16):
            seq = 0
            if objid < 15:
                for j in range(0, 84):
                    cont_cap.append(-1); fill_lev.append(-1); fill_typ.append(-1)
                    container.append(objid)
                    sequence.append(seq)
                    seq += 1
            else:
                for j in range(0, 60):
                    cont_cap.append(-1); fill_lev.append(-1); fill_typ.append(-1)
                    container.append(objid)
                    sequence.append(seq)
                    seq += 1

    if (data_split=='sec_setup') or (data_split=='all'):
        cont_cap, fill_lev, fill_typ = y_pred_parser(y_pred, cont_cap, fill_lev, fill_typ)
        [container.append(cont_i) for cont_i in container_df]
        [sequence.append(seq_i)   for seq_i  in sequence_df]
    else:
        for i in range(21):
            cont_cap.append(-1); fill_lev.append(-1); fill_typ.append(-1)
            sequence.append(i)
            if   i <  7: container.append('16')
            elif i < 13: container.append('17')
            elif i < 19: container.append('18')
            else:        container.append('19')

    # Wrap up
    df_results = pd.DataFrame(list(zip(container, sequence, cont_cap, fill_lev, fill_typ)),
            columns =['Container ID', 'Sequence', 'Container Capacity', 'Filling level', 'Filling type'])

    return df_results
    