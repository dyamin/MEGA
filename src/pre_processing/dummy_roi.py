import os

import pandas as pd

from src import config

df = pd.read_pickle(os.path.join(config.data_dir, config.VIDEO_DIMS_FILE_PATH))
df.rename(columns={'Duration': 't_median', 'Width': 'X_median', 'Height': 'Y_median'}, inplace=True)
tois = [5, 5, 5, 5, 6, 7, 8, 6, 6, 6, 5, 6, 10, 9, 8, 9, 5, 9, 9, 8, 5, 10, 10, 9, 6, 10, 5, 7, 6, 8, 7, 10, 8, 8, 7,
        10, 7, 5, 8, 8, 8, 7, 5, 8, 10, 7, 6, 9, 9, 7, 5, 7, 10, 10, 6, 5, 9, 7, 5, 6, 8, 9, 9, 5, 1]
df['t_median'] = tois
rois_x = (config.NUMBER_OF_ANIMATIONS) * [50]
rois_y = (config.NUMBER_OF_ANIMATIONS) * [50]
rois_x[1], rois_y[1] = 20, 20
rois_x[1], rois_y[1] = 20, 20
rois_x[2], rois_y[2] = 20, 20
rois_x[3], rois_y[3] = 20, 20
rois_x[4], rois_y[4] = 50, 50
rois_x[5], rois_y[5] = 30, 20
rois_x[6], rois_y[6] = 20, 80
rois_x[7], rois_y[7] = 80, 80
rois_x[8], rois_y[8] = 80, 60
rois_x[9], rois_y[9] = 20, 80
rois_x[10], rois_y[10] = 100, 80
rois_x[11], rois_y[11] = 30, 100
rois_x[12], rois_y[12] = 80, 40
rois_x[13], rois_y[13] = 0, 80
rois_x[14], rois_y[14] = 10, 20
rois_x[15], rois_y[15] = 90, 20
rois_x[16], rois_y[16] = 40, 100
rois_x[17], rois_y[17] = 10, 10
rois_x[18], rois_y[18] = 20, 20
rois_x[19], rois_y[19] = 20, 20
df['X_mean'] = rois_x
df['X_median'] = rois_x
df['Y_mean'] = rois_y
df['Y_median'] = rois_y
df.index.name = config.MOVIE
df.to_pickle(os.path.join(config.rois_dir, config.AGGRGATED_ROI_FILE))
