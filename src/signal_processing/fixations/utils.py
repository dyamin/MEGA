from src.signal_processing import config

from src.signal_processing.utils import get_all_fixations_df


def get_filtered_fixations_df():
    df = get_all_fixations_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    return filtered_df
