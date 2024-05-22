import os
import pickle as pkl

import pandas as pd


def numerical_derivative(serie, N: int = 3):
    '''
    Calculates the derivative of serie s based on Engbert's approximation:
    dX/dt = ((s[X+(N-1)] + s[X+(N-2)] + ... + s[X+1]) - (s[X-(N-1)] + s[X-(N-2)] + ... + s[X-1]))/2N
    parameter N determines how many arguments to consider for the derivation.
    NOTE: returns NaN for the first & last (N-1) values of the serie.
    '''
    assert (N >= 2 and N < 0.5 * len(serie)
            ), f'Argument N ({N}) must be at least 2 and no more than half the serie\'s length, i.e. ({len(serie) // 2}).'
    return (serie.rolling(N).sum().shift(1 - N) - serie.rolling(N).sum()) / (N * 2)


def get_subjects_sessions_movies(raw_data_df):
    ''' returns a tuple of 3 sets:
            all unique subject IDs from the given DF, all unique session IDs, and all unique movie IDs '''
    subjects = sorted(raw_data_df.index.get_level_values(0).unique())
    sessions = sorted(raw_data_df.index.get_level_values(1).unique())
    movies = sorted(raw_data_df.index.get_level_values(2).unique())
    return subjects, sessions, movies


def save_df_to_pkl(dataframe, filename: str, dir_path=None, prot=-1):
    '''
    Saves the input 'dataframe' to a pickle file named 'filename', with pickling protocol 'prot'
    If 'dir_path' is defined it is saved to said directory. If not - to the current one.
    '''
    # If needed, create the directory
    if dir_path is None:
        dir_path = ''
    elif not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # Create the pkl files
    fullpath = os.path.join(dir_path, filename + '.pkl')
    with open(fullpath, 'wb') as file:
        pkl.dump(dataframe, file, protocol=prot)
    return


def round_down_to_closest_even(num: float, non_negative=True) -> int:
    if (non_negative and (num <= 0)):
        return 0
    num = int(num)
    if (num % 2 == 1):
        num -= 1
    return num


def convert_dict_to_series(d: dict):
    if not d:
        return pd.Series()
    serie = pd.concat(d.values(), keys=d.keys())
    serie.index.names = ['Movie', 'Subject', 'Session', 'Memory', 'TimeStamp']
    return serie.reorder_levels(['Subject', 'Session', 'Movie', 'Memory', 'TimeStamp']).sort_index()


def cut_series(serie: pd.Series, first_value: int, last_value: int, index_name: str = 'TimeStamp') -> pd.Series:
    ## TODO: OPTIMIZE THIS METHOD!
    assert (
            first_value <= last_value), f'Argument @first_value ({first_value}) must be lesser/equal to @last_value ({last_value}).'
    assert (index_name in serie.index.names), f'Couldn\'t find {index_name} in the given Serie\'s index names.'
    if (first_value == last_value):
        return pd.Series()
    serie_as_dataframe = serie.reset_index(index_name)
    cut_dataframe = serie_as_dataframe[
        (serie_as_dataframe[index_name] >= first_value) & (serie_as_dataframe[index_name] <= last_value)]
    cut_dataframe.set_index(index_name, append=True, inplace=True)
    cut_series = cut_dataframe[
        cut_dataframe.columns[0]]  # treat this as a Series by taking the first (and only) column of the DF
    return cut_series


def get_movie_roi_data(movie: str, distances: pd.Series, RoIs: pd.Series):
    MOVIE_LEVEL, ROI_MEDIAN_TIME_COL, ROI_STDEV_TIME_COL = 'Movie', 't_median', 't_StDev'
    assert (movie in distances.index.unique(level=MOVIE_LEVEL)
            ), f'No distances available for given movie {movie}.'
    assert (movie in RoIs.index.unique()
            ), f'No RoI available for given movie {movie}.'

    movie_distances = distances.xs(movie, level=MOVIE_LEVEL)
    roi_median_time, roi_stdev_time = RoIs.loc[movie, [ROI_MEDIAN_TIME_COL, ROI_STDEV_TIME_COL]]
    return movie_distances, roi_median_time, roi_stdev_time


def downsample_data_single_directoy(orig_dir: str, new_dir: str, rate: int, prot=-1):
    for filename in [filename for filename in os.listdir(orig_dir) if filename.split('.')[1] == 'pkl']:
        fullpath = os.path.join(orig_dir, filename)
        df = pd.read_pickle(fullpath)

        # calculate the downsampled dataframe
        # https://stackoverflow.com/questions/39477393/finding-the-average-of-two-consecutive-rows-in-pandas
        idx = 1 + len(df) - rate if (len(df) % rate) else len(df)
        new_df = df[:idx].groupby(df.index[:idx] // rate).mean()
        new_df['onset'] = new_df['onset'].apply(lambda x: int(x))
        new_df['index'] = new_df['index'].apply(lambda x: int(x))

        # save the new dataframe
        save_df_to_pkl(new_df, filename.split(".")[0], dir_path=new_dir, prot=prot)
    return


def downsample_data(data_dir: str, to_downsample, prot=-1):
    s = pd.Series(to_downsample)
    for directory, rate in s.iteritems():
        subj_dir_path = os.path.join(data_dir, directory)

        # create new directories if needed
        downsample_dir_path = os.path.join(os.path.dirname(data_dir), "downsampled")
        if (not os.path.exists(downsample_dir_path)):
            os.makedirs(downsample_dir_path)
        new_subjdir_path = os.path.join(downsample_dir_path, directory)
        if (not os.path.exists(new_subjdir_path)):
            os.makedirs(new_subjdir_path)

        # downsample and save the new data
        downsample_data_single_directoy(subj_dir_path, new_subjdir_path, rate, prot)
    return


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


def convert_color_to_seaborn(color):
    # assuming color is a tuple like (R, G, B)
    return tuple([val / 255 for val in color])
