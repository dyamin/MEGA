import os

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d as mpl

from src import config as global_config
from src.signal_processing.gaze.utils import get_filtered_gaze_tuple_position_df
from src.signal_processing.utils import get_aggregated_roi_df
from src.visualize import config

gaze_df = get_filtered_gaze_tuple_position_df()
rois = get_aggregated_roi_df()
dims_path = os.path.join(global_config.data_dir, config.dims_file)
videos_dims = pd.read_pickle(dims_path)[:config.last_repeating_movie_ind]

# visualize example subject and movie
subject_movie = gaze_df.xs(('LG3', 'mov73'), level=['Subject', 'Movie'])
rois = rois[rois.index.isin(['mov73'])]

sessionA = subject_movie.xs(('Session A',), level=['Session'])
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
(x_A, y_A) = list(zip(*sessionA))
z_A = list(sessionA.index.values)
ax.plot(x_A, y_A, z_A, label='Session A STC')
# ax.set_xlim([0, 1920])
# ax.set_ylim([0, 1080])
ax.legend()

movie_width, movie_height = videos_dims.loc['mov73', ['Width', 'Height']]
x = (rois['X_mean'] / 100) * movie_height
y = (rois['Y_mean'] / 100) * movie_width
ax.scatter(x, y, rois['t_mean'])

ax_B = fig.gca(projection='3d')
sessionB = subject_movie.xs(('Session B',), level=['Session'])
(x_B, y_B) = list(zip(*sessionB))
z_B = list(sessionB.index.values)
ax_B.plot(x_B, y_B, z_B, label='Session B STC')
# ax_B.set_xlim([0, 1920])
# ax_B.set_ylim([0, 1080])
ax_B.legend()
plt.show()
