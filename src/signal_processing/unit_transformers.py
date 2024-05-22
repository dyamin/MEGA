from math import atan2, degrees, sqrt

import src.config as config


def calculate_dva(X_gaze, Y_gaze, roi):
    horizontal_dva = calculate_dva_one_dimension(X_gaze, roi[0], config.ORIG_HORIZONTAL_RESULOTION_IN_PXL,
                                                 config.HORIZONTAL_SIZE_IN_CM)
    vertical_dva = calculate_dva_one_dimension(Y_gaze, roi[1], config.ORIG_VERTICAL_RESULOTION_IN_PXL,
                                               config.VERTICAL_SIZE_IN_CM)
    dva = sqrt(horizontal_dva ** 2 + vertical_dva ** 2)
    return dva


def calculate_dva_one_dimension(gaze, roi, res_pxl, size_cm):
    # Calculate the number of degrees that correspond to a single pixel. This will
    # generally be a very small value, something like 0.03.
    deg_per_px = get_deg_per_pxl(res_pxl, size_cm)
    # Calculate the size of the stimulus in degrees
    distance_px = abs(gaze - roi)
    return distance_px * deg_per_px


def get_deg_per_pxl(res_pxl, size_cm):
    return degrees(atan2(.5 * size_cm, config.DISTANCE_IN_CM)) / (.5 * res_pxl)
