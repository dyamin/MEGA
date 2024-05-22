import csv
import os
import re

import numpy as np
import pandas as pd

from src import config

# MISTAKE IN THE ORIGINAL DATA: movie 41 event time is 7, not 10
df = pd.read_pickle(os.path.join(config.data_dir, config.VIDEO_DIMS_FILE_PATH))
df.rename(columns={'Duration': 't_median', 'Width': 'X_median', 'Height': 'Y_median'}, inplace=True)
# Remove the last row (movie 65)
df = df.iloc[:-1]

tois = np.array(
    [5.05, 5.05, 5, 5, 6, 7, 8, 6, 6, 6, 5, 6, 10, 9, 8, 9, 5, 9, 9, 8, 5, 10, 10, 9, 6, 10, 5, 7, 6, 8, 7, 10, 8, 8, 7,
     10, 7, 5, 8, 8, 8, 7, 5, 8, 7, 7, 6, 9, 9, 7, 5, 7, 10, 10, 6, 5, 9, 7, 5, 6, 8, 9, 9, 5]) * 1000

rois_x = config.num_repeating_movies * [50]
rois_y = config.num_repeating_movies * [50]
# Init the x,y of the first 4 movies
rois_x[0], rois_y[0] = (1539/1920) * 100, (549/1080) * 100
rois_x[1], rois_y[1] = (266/1920) * 100, (476/1080) * 100
rois_x[2], rois_y[2] = (1670/1920) * 100, (650/1080) * 100
rois_x[3], rois_y[3] = (145/1920) * 100, (10/1080) * 100


rois = []
with open(os.path.join(config.rois_dir, 'studio-rois.csv')) as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        if i % 2 == 0:
            newRow = (row[1],) + tuple(re.findall(r'\b\d+\b', row[0]))
        else:
            newRow = newRow + (row[0].replace('y:', ''),)
            rois.append(newRow)
        i = i + 1

video_dims = pd.read_pickle(config.VIDEO_DIMS_FILE_PATH)
movie_width = 2304
movie_height = 1296

j = 0
for i in range(4, config.num_repeating_movies):
    rois_x[i] = (int(rois[j][1]) / movie_width) * 100
    rois_y[i] = (int(rois[j][2]) / movie_height) * 100
    split = rois[j][0].split(':')
    tois[i] = (int(split[0]) * 1000) + (int(split[1]) / 24) * 1000
    j = j + 1

df['X_mean'] = rois_x
df['X_median'] = rois_x
df['Y_mean'] = rois_y
df['Y_median'] = rois_y
df['t_median'] = tois
df[config.T_STDEV] = config.num_repeating_movies * [500]

df.index.name = config.MOVIE
df.to_pickle(os.path.join(config.rois_dir, config.AGGRGATED_ROI_FILE))
