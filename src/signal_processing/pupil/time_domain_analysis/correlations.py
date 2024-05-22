import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.signal_processing import config
from src.signal_processing.pupil.utils import get_filtered_pupil_radius_df


# Correlation: To what degree do the signals change together?
# Correlation matrix of pupil radius when 1st/2nd viewing a specific movie

def plot_correlation_matrix(session):
    pupil_df = get_filtered_pupil_radius_df()
    pupil_a_df = pupil_df.xs((session,), level=['Session'])

    for mov in config.valid_movies:
        movie_df = pupil_a_df.xs((mov,), level=['Movie'])
        movie_df = movie_df.unstack(level=0).tail(movie_df.shape[0] - 500)

        corr = movie_df.corr(method="pearson")
        mean = corr.values[np.triu_indices_from(corr.values, 1)].mean()
        if mean > 0.4:
            plt.figure()
            plt.title('(' + session + ',' + mov + '), mean: ' + str(round(mean, 3)))
            sns.heatmap(corr,
                        xticklabels=corr.columns.values,
                        yticklabels=corr.columns.values)
            plt.show()


plot_correlation_matrix('Session A')
plot_correlation_matrix('Session B')
