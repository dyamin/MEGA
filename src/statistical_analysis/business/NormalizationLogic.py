import pandas as pd

import src.config as g_config


def mega_score_normalization(df: pd.DataFrame, first_viewing_df=None,
                             second_viewing_label=g_config.SESSION_B,
                             first_viewing_label=g_config.SESSION_A, column='Result'):
    # This method is used to normalize the data on each trial, based on the first session
    # The first session is used as a baseline, and the data is normalized based on it
    # The data is normalized by subtracting the mean of the first session from each trial
    # and then dividing by the max value between the first session and the second session

    df['Max'] = df.groupby([g_config.MOVIE, g_config.SUBJECT])[column].transform('max')
    df.set_index([g_config.MOVIE, g_config.SUBJECT], inplace=True)

    if first_viewing_df is None:
        first_viewing_df = df[df[g_config.SESSION] == first_viewing_label]
    else:
        first_viewing_df.set_index([g_config.MOVIE, g_config.SUBJECT], inplace=True)

    second_viewing_df = df[df[g_config.SESSION] == second_viewing_label]
    second_viewing_df['Result'] = (first_viewing_df[column] - second_viewing_df[column])
    second_viewing_df['Result'] = (second_viewing_df['Result'] / second_viewing_df['Max'])
    second_viewing_df = second_viewing_df.reset_index()
    return second_viewing_df.drop('Max', axis=1)
