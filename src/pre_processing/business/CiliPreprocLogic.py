# this file includes functions for analysis
import pickle

from cili.cleanup import interp_eyelink_blinks, interp_zeros
from cili.util import *

from src import config
from src.pre_processing.utils import mean_or_single


def preproc_sub(folder, filename, output_folder):
    # loading data from .asc
    fullpath = os.path.join(folder, filename)
    samps, events = load_eyelink_dataset(fullpath)

    # handling blinks and missing data
    samps = interp_eyelink_blinks(samps, events, interp_fields=['pup_l'])
    samps = interp_eyelink_blinks(samps, events, interp_fields=['pup_r'])
    samps = interp_zeros(samps, interp_fields=['pup_l'])
    samps = interp_zeros(samps, interp_fields=['pup_r'])

    blinks = events.EBLINK
    saccades = events.ESACC
    fixations = events.EFIX

    ## split the data
    ev = events.MSG
    trialstrts = ev[ev['label'].str.contains("strt")].copy()
    trialfins = ev[ev['label'].str.contains("fin")].copy()

    trialstrts['content'] = trialfins['content'].values
    trialfins.content = trialfins.content.astype(np.int64)  # change content into int to sort correctly
    trialstrts.content = trialstrts.content.astype(np.int64)  # change content into int to sort correctly

    order = trialfins.content  # save original order
    # sort both
    trialstrts = trialstrts.sort_values(by='content')
    trialfins = trialfins.sort_values(by='content')

    # get onset to be a real col otherwise it causes a bug in cutting
    trialfins.reset_index(inplace=True)
    samps.reset_index(inplace=True)
    trialstrts.reset_index(inplace=True)
    blinks.reset_index(inplace=True)
    saccades.reset_index(inplace=True)
    fixations.reset_index(inplace=True)

    samps = columns_renaming(samps)
    blinks = features_columns_renaming(blinks)
    saccades = features_columns_renaming(saccades)
    fixations = features_columns_renaming(fixations)

    # add columns if they are not there
    for col in ['onset', 'x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r']:
        if col not in samps.columns:
            samps[col] = np.nan

    mov_data = []
    blinks_by_mov = []
    saccades_by_mov = []
    fixations_by_mov = []
    for n in np.arange(0, len(trialstrts)):  # runs on onsets of trials
        d = samps[
            (samps[config.ONSET] > trialstrts[config.ONSET][n]) & (samps[config.ONSET] < trialfins[config.ONSET][n])]
        mov_data.append(d)

        b = get_features_by_movie(blinks, n, trialfins, trialstrts)
        blinks_by_mov.append(b)

        s = get_features_by_movie(saccades, n, trialfins, trialstrts)
        saccades_by_mov.append(s)

        f = get_features_by_movie(fixations, n, trialfins, trialstrts)
        fixations_by_mov.append(f)

    # get the onset to be a col
    for t in range(0, len(mov_data)):
        mov_data[t].reset_index(inplace=True)
        blinks_by_mov[t].reset_index(inplace=True)
        saccades_by_mov[t].reset_index(inplace=True)
        fixations_by_mov[t].reset_index(inplace=True)

    # add gaze and features as avg between right and left eye
    mov_data = averaging_gaze_from_both_eyes(mov_data)
    blinks_by_mov = averaging_features_from_both_eyes(blinks_by_mov, 'blink')
    saccades_by_mov = averaging_features_from_both_eyes(saccades_by_mov, 'saccade')
    fixations_by_mov = averaging_features_from_both_eyes(fixations_by_mov, 'fixation')

    # save the data
    a, b = filename.split('.')

    # save
    save_pickle(a, order, output_folder, 'order')
    save_pickle(a, mov_data, output_folder, 'data')
    save_pickle(a, blinks_by_mov, output_folder, 'blinks')
    save_pickle(a, saccades_by_mov, output_folder, 'saccades')
    save_pickle(a, fixations_by_mov, output_folder, 'fixations')


def save_pickle(a, order, output_folder, label):
    order_file_name = os.path.join(output_folder, a + f'_{label}.pkl')
    with open(order_file_name, 'wb') as f:
        pickle.dump(order, f)


def averaging_features_from_both_eyes(mov_data, feature_name):
    """
    This function averages the features from both eyes
    :param mov_data: list of dataframes
    :return: list of dataframes
    """
    aggregated_data = []
    for mov_idx in np.arange(0, len(mov_data)):

        agg_mov_data = pd.DataFrame(columns=mov_data[mov_idx].columns)

        # check if mov_data[mov_idx] is not empty
        if not mov_data[mov_idx].empty:
            # split the data to left and right eye
            mov_data_left = mov_data[mov_idx][mov_data[mov_idx][config.EYE] == 'L']
            mov_data_right = mov_data[mov_idx][mov_data[mov_idx][config.EYE] == 'R']

            # check if both eyes are not empty
            if not mov_data_left.empty and not mov_data_right.empty:
                # take the bigger dataframe and iterate over it
                if len(mov_data_left) > len(mov_data_right):
                    bigger_df = mov_data_left
                    smaller_df = mov_data_right
                else:
                    bigger_df = mov_data_right
                    smaller_df = mov_data_left

                blinks_count = 0
                # iterate over the bigger dataframe
                for idx, row in bigger_df.iterrows():
                    # check if the onset contained in the smaller dataframe
                    slice_from_the_smaller_df = get_onset_feature_data(row, smaller_df)
                    if slice_from_the_smaller_df.shape[0] > 0:
                        # take the mean of the features and add them to the aggregated dataframe
                        agg_feature_to_add = pd.concat([row.to_frame().transpose(), slice_from_the_smaller_df]) \
                            .mean().to_frame().transpose()
                        agg_mov_data = pd.concat([agg_mov_data, agg_feature_to_add])
                        blinks_count += 1
                # update columns
                agg_mov_data[config.EYE] = 'B'
                agg_mov_data['name'] = feature_name
            else:
                # if one of the eyes is empty, take the other one
                if mov_data_left.empty:
                    agg_mov_data = mov_data_right
                else:
                    agg_mov_data = mov_data_left
        aggregated_data.append(agg_mov_data)
    return aggregated_data


def get_onset_feature_data(this, other):
    return other[((other[config.ONSET] <= this[config.ONSET]) & (other[config.LAST_ONSET] >= this[config.ONSET])) | (
            (other[config.ONSET] <= this[config.LAST_ONSET]) & (other[config.LAST_ONSET] >= this[config.LAST_ONSET]))]


def averaging_gaze_from_both_eyes(mov_data):
    for i in np.arange(0, len(mov_data)):
        if mov_data[i].empty:
            continue

        mov_data[i].loc[:, config.gaze_X] = mov_data[i].loc[:, ['x_l', 'x_r']].apply(mean_or_single, axis=1)
        mov_data[i].loc[:, config.gaze_Y] = mov_data[i].loc[:, ['y_l', 'y_r']].apply(mean_or_single, axis=1)

        mov_data[i]["pup_l"].loc[mov_data[i]["pup_l"] < 1] = np.nan
        mov_data[i]["pup_r"].loc[mov_data[i]["pup_r"] < 1] = np.nan
        mov_data[i].loc[:, 'pup'] = mov_data[i].loc[:, ['pup_l', 'pup_r']].apply(mean_or_single, axis=1)
    return mov_data


def get_features_by_movie(features, n, trialfins, trialstrts):
    return features[((features['onset'] > trialstrts['onset'][n]) & (features['onset'] < trialfins['onset'][n]))
                    | ((features['last_onset'] > trialstrts['onset'][n]) & (
            features['last_onset'] < trialfins['onset'][n]))]


def columns_renaming(samps):
    if config.POPULATION == 'animation' or (
            (config.POPULATION == 'elderly' or config.POPULATION == 'mci' or config.POPULATION == 'ad') and
            samps.columns[0] == 'index'):
        renaming = {'index': 'onset', 'onset': 'x_l', 'x_l': 'y_l', 'y_l': 'pup_l', 'pup_l': 'x_r',
                    'x_r': 'y_r',
                    'y_r': 'pup_r',
                    'pup_r': 'dont_know'}

        if 'x_l' not in samps.columns:
            renaming = {'index': 'onset', 'onset': 'x_r', 'x_r': 'y_r', 'y_r': 'pup_r', 'pup_r': 'dont_know'}

        if 'x_r' not in samps.columns:
            renaming = {'index': 'onset', 'onset': 'x_l', 'x_l': 'y_l', 'y_l': 'pup_l', 'pup_l': 'dont_know'}

        samps.rename(columns=renaming, inplace=True)

    return samps


def features_columns_renaming(features):
    return features.rename(
        columns={'eye': config.EYE, 'onset': config.ONSET, 'last_onset': config.LAST_ONSET,
                 'duration': config.DURATION})
