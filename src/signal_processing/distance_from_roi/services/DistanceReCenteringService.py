import pandas as pd

from src.utils import round_down_to_closest_even, get_movie_roi_data, convert_dict_to_series


class DistanceReCenteringService:
    # index names for queries:
    SUBJECT = 'Subject'
    SESSION = 'Session'
    MOVIE = 'Movie'
    MEMORY = 'Memory'
    TIMESTAMP = 'TimeStamp'

    def __init__(self, gaze_distances, RoIs):

        assert (type(RoIs) is pd.DataFrame), f'The argument @RoIs must be an instance of DataFrame.'
        assert (type(gaze_distances) is pd.Series), f'The argument @gaze_distances must be an instance of Series.'
        self.rois = RoIs
        self.distances = gaze_distances.dropna()

        distances_movies = set(self.distances.index.unique(level=self.MOVIE))
        rois_movies = set(self.rois.index.unique())
        assert (distances_movies == rois_movies
                ), f'Movies in Distances Series do not match the Movies in RoIs data.\n\tDistances Movies: {distances_movies}\n\tRoIs Movies: {rois_movies}'
        return

    def recenter_distances(self, verbose=False):
        distances_dict = dict()
        movies = self.distances.index.unique(level=self.MOVIE)
        for movID in movies:
            distances_dict[movID] = self.recenter_distances_single_movie(movID)
            if (verbose):
                print(f'Finished recentering distance data for Movie {movID}.')
        return convert_dict_to_series(distances_dict)

    def recenter_distances_single_movie(self, movID: str):
        assert (movID in self.distances.index.unique(level=self.MOVIE)), f'Couldn\'t find distances for movie {movID}'
        assert (movID in self.rois.index.unique()), f'Couldn\'t find RoI data for movie {movID}'
        movie_distances, roi_time, roi_std_time = get_movie_roi_data(movID, self.distances, self.rois)
        roi_time = round_down_to_closest_even(roi_time)

        # set the TimeStamp index-level to a column to enable manipulations on it:
        temp_df = movie_distances.reset_index(level=self.TIMESTAMP)
        temp_df[self.TIMESTAMP] = temp_df[self.TIMESTAMP] - roi_time

        # reset the TimeStamp column back to the index and convert the DataFrame to a Series
        temp_series = temp_df.set_index(self.TIMESTAMP, append=True)
        return temp_series[temp_series.columns[0]]
