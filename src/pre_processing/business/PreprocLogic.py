# this file includes functions for analysis
import os
import pickle

import numpy as np

from src import config
from src.pre_processing.services.ParseEyeLinkAsc import ParseEyeLinkAsc
from src.pre_processing.utils import mean_or_single


def preproc_sub(folder, filename, output_folder):
    # loading data from.asc
    fullpath = os.path.join(folder, filename)
    dfTrial, dfMsg, dfFix, dfSacc, dfBlink, dfSamples = ParseEyeLinkAsc(fullpath)

    ## split the data
    trialstrts = dfMsg[dfMsg['text'].str.contains("strt")]
    trialfins = dfMsg[dfMsg['text'].str.contains("fin")]

    trialfins['content'] = trialfins['text'].str.split().str[1]
    trialstrts['content'] = trialfins['content'].values
    trialfins.content = trialfins.content.astype(np.int64)  # change content into int to sort correctly
    trialstrts.content = trialstrts.content.astype(np.int64)  # change content into int to sort correctly

    order = trialfins.content  # save original order
    # sort both
    trialstrts.sort_values(by='content', inplace=True)
    trialfins.sort_values(by='content', inplace=True)

    # get onset to b e a real col otherwise it causes a bug in cutting
    trialfins.reset_index(inplace=True)
    dfSamples.reset_index(inplace=True)
    trialstrts.reset_index(inplace=True)

    dfSamples.rename(
        columns={'tSample': 'onset', 'RPupil': 'pup_r', 'LPupil': 'pup_l', 'RX': 'x_r', 'RY': 'y_r', 'LX': 'x_l',
                 'LY': 'y_l'}, inplace=True)

    mov_data = []
    for n in np.arange(0, len(trialstrts)):  # runs on onsets of trials
        A = dfSamples[(dfSamples['onset'] > trialstrts['time'][n]) & (dfSamples['onset'] < trialfins['time'][n])]
        mov_data.append(A)

    # get the onset to be a col
    for t in range(0, len(mov_data)):
        mov_data[t].reset_index(inplace=True)

    # add gaze as avg between right and left eye
    for i in np.arange(0, len(mov_data)):
        mov_data[i][config.gaze_X] = mov_data[i][['x_l', 'x_r']].apply(mean_or_single, axis=1)
        mov_data[i][config.gaze_Y] = mov_data[i][['y_l', 'y_r']].apply(mean_or_single, axis=1)

        mov_data[i]["pup_l"].loc[mov_data[i]["pup_l"] < 1] = np.nan
        mov_data[i]["pup_r"].loc[mov_data[i]["pup_r"] < 1] = np.nan
        mov_data[i]['pup'] = mov_data[i][['pup_l', 'pup_r']].apply(mean_or_single, axis=1)

    # save the data
    a, b = filename.split('.')

    # save order
    order_file_name = os.path.join(output_folder, a + '_order.pkl')
    with open(order_file_name, 'wb') as f:
        pickle.dump(order, f)
    # save data
    data_file_name = os.path.join(output_folder, a + '_data' + '.pkl')
    with open(data_file_name, 'wb') as f:
        pickle.dump(mov_data, f)
