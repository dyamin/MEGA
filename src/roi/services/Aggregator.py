import os

import numpy as np
import pandas as pd

from src import config as globalconfig
from src.roi import config
from src.roi.services import DataService


def generate_roi_rectangles(use_median=False, write: bool = False) -> pd.DataFrame:
    """
    Calculates ROI rectangles - X,Y pixels of top-left (tl) & bottom-right (br) corners
    @args:
        rois -> a DataFrame with each video's aggregated RoI statistics
        use_median -> if True, the stats used to calculate the rectangles is the median and not the mean
        write -> if True, writes to file

    $return:
        (or writes) a DataFrame containing the X,Y coordinates of both corners, and also the start-time of the ROI.
    """

    aggregated_rois_path = os.path.join(globalconfig.rois_dir, globalconfig.AGGRGATED_ROI_FILE)
    aggregated_rois = pd.read_pickle(aggregated_rois_path)

    stats = config.stats_median if use_median else config.stats_mean  # (X,Y,t)

    print(f"Generating rectangles for RoIs - START")
    data = _execute_generate_roi_rectangles(aggregated_rois, stats)
    print(f"Generating rectangles for RoIs - DONE")

    if write:
        DataService.write_pickle(data, globalconfig.rois_rects_file)

    return data


def _execute_generate_roi_rectangles(rois: pd.DataFrame, stats: list) -> pd.DataFrame:
    tl_x, tl_y, br_x, br_y = _calc_points(rois, stats)

    # scale to the video's dimensions
    br_x.columns = ['Width']
    br_y.columns = ['Height']
    br_x = pd.concat([br_x, DataService.videos_dims['Width']]).min(level=0, skipna=False).rename('Bottom Right X')
    br_y = pd.concat([br_y, DataService.videos_dims['Height']]).min(level=0, skipna=False).rename('Bottom Right Y')

    # create a DataFrame with the results
    result = pd.DataFrame([rois[stats[2]], rois[globalconfig.T_STDEV],
                           tl_x, tl_y, br_x, br_y]).T
    result.reindex(index=sorted(result.index), copy=False)
    return result.dropna(axis=0)


def _calc_points(rois: pd.DataFrame, stats: list) -> tuple:
    roi_size = globalconfig.roi_size

    tl_x = rois[stats[0]].apply(lambda x: max(0, x / 100 - 0.5 * np.sqrt(roi_size)))
    tl_y = rois[stats[1]].apply(lambda y: max(0, y / 100 - 0.5 * np.sqrt(roi_size)))

    # use top-left corner and roibased size to calculate the bottom right corner
    br_x = tl_x.add(np.sqrt(roi_size))
    br_y = tl_y.add(np.sqrt(roi_size))

    return tl_x.multiply(DataService.videos_dims['Width']).rename('Top Left X'), \
        tl_y.multiply(DataService.videos_dims['Height']).rename('Top Left Y'), \
        br_x.multiply(DataService.videos_dims['Width']).rename('Bottom Right X'), \
        br_y.multiply(DataService.videos_dims['Height']).rename('Bottom Right Y')
