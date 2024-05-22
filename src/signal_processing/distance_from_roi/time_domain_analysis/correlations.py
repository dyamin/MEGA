import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.signal_processing import config
from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df

"""
Correlation: To what degree do the signals change together?
Correlation matrix between subjects. 
In session A (1st viewing) the correlation is higher than in session B (2nd viewing). 
In session B subjects probably look at the ROI before the event time, but in different times.
"""


def plot_correlation_matrix(session, ):
    distance_df = get_filtered_roi_df()
    distance_a_df = distance_df.xs((session,), level=['Session'])

    for mov in config.valid_movies:
        movie_df = distance_a_df.xs((mov,), level=['Movie'])
        movie_df = movie_df.unstack(level=0)
        corr = movie_df.corr(method="pearson")
        mean = corr.values[np.triu_indices_from(corr.values, 1)].mean()
        if mean > 0.0:
            plt.figure()
            plt.title('(' + session + ',' + mov + '), mean: ' + str(round(mean, 3)))
            sns.heatmap(corr,
                        xticklabels=corr.columns.values,
                        yticklabels=corr.columns.values)
            plt.show()


plot_correlation_matrix('Session A')
plot_correlation_matrix('Session B')
