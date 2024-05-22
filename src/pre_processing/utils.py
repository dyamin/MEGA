# Define a lambda function to compute the row-wise mean of two columns, or return a non-missing value if only one
# value exists
import os
import pickle

import numpy as np
import pandas as pd

import src.config as config


def mean_or_single(x):
    if pd.notna(x[0]) and pd.isna(x[1]):
        return x[0]
    elif pd.notna(x[1]) and pd.isna(x[0]):
        return x[1]
    else:
        return np.mean(x)


def decentralize_raw_data(origin, target):
    """ this function load pickle of one subject that hold a list of the movies and saves them as different files
    each holding a panda of the
    input:
        origin - fullpath of the file
        target - folder in which to create a folder to put the pandas file in
    """
    with open(origin, 'rb') as f:
        data = pickle.load(f)

    data_file_name = os.path.split(origin)

    sub_ses_name = data_file_name[1].split('_')[:2]
    sub_dir = os.path.join(target, sub_ses_name[0])
    if not os.path.exists(sub_dir): os.mkdir(sub_dir)
    sub_ses_dir = os.path.join(sub_dir, '_'.join(sub_ses_name))
    if not os.path.exists(sub_ses_dir): os.mkdir(sub_ses_dir)

    bs_dict = {}
    for index, mov in enumerate(data):
        if mov.empty:
            continue
        mov_ind = str(index + 1)
        file_name = os.path.join(sub_ses_dir, 'mov' + mov_ind + '_data.pkl')
        mov, baseline_time = reset_data_timestamp(mov)
        mov.to_pickle(file_name)
        bs_dict[mov_ind] = baseline_time

    return bs_dict


def decentralize_features(bs_per_movie, origin, target):
    """ this function load pickle of features (blinks/fixations/saccades) of one subject that hold a list of the movies,
    reset their timestamp according to the baseline, and saves them as different files each holding a panda of the
    input:
        origin - fullpath of the data file
        target - folder in which to create a folder to put the pandas file in
    """

    # Get the file names
    data_file_name = os.path.split(origin)
    prefix_file_name = data_file_name[1].removesuffix('_data.pkl')

    blinks_file_name = os.path.join(data_file_name[0], prefix_file_name + '_blinks.pkl')
    fixations_file_name = os.path.join(data_file_name[0], prefix_file_name + '_fixations.pkl')
    saccades_file_name = os.path.join(data_file_name[0], prefix_file_name + '_saccades.pkl')

    # Get the target directory
    subject_name = prefix_file_name.split('_')[0]

    features_target_dir = os.path.join(target, subject_name, prefix_file_name)

    # Load the file and reset the timestamp
    load_and_reset_features(blinks_file_name, features_target_dir, bs_per_movie, 'blinks')
    load_and_reset_features(fixations_file_name, features_target_dir, bs_per_movie, 'fixations')
    load_and_reset_features(saccades_file_name, features_target_dir, bs_per_movie, 'saccades')


def load_and_reset_features(file_name, target, bs_per_movie, feature_name):
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            features = pickle.load(f)
        for index, mov in enumerate(features):
            if mov.empty:
                continue
            mov_ind = str(index + 1)
            mov = reset_features_timestamp(mov, bs_per_movie[mov_ind])
            fname_dir = os.path.join(target, feature_name)
            if not os.path.exists(fname_dir): os.mkdir(fname_dir)
            file_name = os.path.join(fname_dir, 'mov' + mov_ind + '.pkl')
            mov.to_pickle(file_name)
    else:
        print(f'file {file_name} does not exist')


def reset_features_timestamp(data, baseline_time):
    # Set the onset column to be the index, and its starting time to be 0
    data[config.ONSET] = data[config.ONSET].apply(lambda time: time - baseline_time)
    data[config.LAST_ONSET] = data[config.LAST_ONSET].apply(lambda time: time - baseline_time)
    data.set_index(config.ONSET, inplace=True)
    # remove column "index" that is not needed and return the data
    return data.drop(columns=['index'])


def reset_data_timestamp(raw_data):
    # Set the onset column to be the index, and its starting time to be 0
    baseline_time = raw_data[config.ONSET][0]
    raw_data[config.ONSET] = raw_data[config.ONSET].apply(lambda time: time - baseline_time)
    raw_data.set_index(config.ONSET, inplace=True)
    return raw_data.drop(columns=['index']), baseline_time
