import os
from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).parent.parent


POPULATION = 'animation'  # "mci_ad" # "mci" # "yoavdata"  # "elderly"  # "animation"  # "no_nap"  # "multiple_populations" # "nap"

# Indexers & Columns
SUBJECT = 'Subject'
SESSION = 'Session'
MOVIE = 'Movie'
MEMORY = 'Memory'
DISTANCE = 'Distance'
DVA = 'DVA'
SQRT_DVA = 'SQRT_DVA'
PUPIL = 'Mean_Pupil' if POPULATION == 'yoavdata' else 'Pupil radius'
SESSION_A = "Session A" if POPULATION != 'yoavdata' else '1st'
SESSION_B = "Session B" if POPULATION != 'yoavdata' else '2nd'
TIMESTAMP = 'TimeStamp'
gaze_X = "X_gaze"
gaze_Y = "Y_gaze"
CoM_X = "X_gaze" if POPULATION != 'yoavdata' else 'CoM_X'
CoM_Y = "Y_gaze" if POPULATION != 'yoavdata' else 'CoM_Y'
X_MEDIAN = 'X_median'
Y_MEDIAN = 'Y_median'
T_MEDIAN, T_STDEV = 't_median', 't_StDev'
ONSET = 'onset' if POPULATION != 'yoavdata' else 'Start_Time'
LAST_ONSET = 'last_onset'
DURATION = 'Duration'
EYE = 'Measured Eye'
AMPLITUDE = 'vis_angle' if POPULATION != 'yoavdata' else 'Total Distance'
VELOCITY = 'peak_velocity' if POPULATION != 'yoavdata' else 'Velocity'
X_START = 'x_start' if POPULATION != 'yoavdata' else 'Start X'
Y_START = 'y_start' if POPULATION != 'yoavdata' else 'Start Y'
X_END = 'x_end' if POPULATION != 'yoavdata' else 'End_X'
Y_END = 'y_end' if POPULATION != 'yoavdata' else 'End_Y'

# File Names:
demographic_data = 'demographic_data'
memory_report = 'memory_report'
all_subject = 'all_subject_'
gaze = 'gaze'
blinks = 'blinks'
fixations = 'fixations'
saccades = 'saccades'

TXT = '.txt'
PKL = '.pkl'
SVG = '.svg'

# Paths:
memory_reports_dir = os.path.join(get_project_root(), "resources", POPULATION, "memory_reports")
raw_data_dir = os.path.join(get_project_root(), "resources", POPULATION, "raw_data")
decentralized_data_dir = os.path.join(get_project_root(), "resources", POPULATION, "decentralized_data")
videos_dir = os.path.join(get_project_root(), "resources", POPULATION, "videos")
data_dir = os.path.join(get_project_root(), "resources", POPULATION, "data")
log_dir = os.path.join(get_project_root(), "resources", "log", "validation")
rois_dir = os.path.join(get_project_root(), "resources", POPULATION, "roi")

VIDEO_DIMS_FILE_PATH = os.path.join(data_dir, "video_dims.pkl")
plots_dir = os.path.join(get_project_root(), "resources", POPULATION, "plots")
statistical_analysis_resource_dir = os.path.join(get_project_root(), "resources", POPULATION, "statistical_analysis")
classification_resource_dir = os.path.join(get_project_root(), "resources", POPULATION, "classification")

nap_statistical_analysis_resource_dir = os.path.join(get_project_root(), "resources", "nap", "statistical_analysis")
no_nap_statistical_analysis_resource_dir = os.path.join(get_project_root(), "resources", "no_nap",
                                                        "statistical_analysis")
elderly_statistical_analysis_resource_dir = os.path.join(get_project_root(), "resources", "elderly",
                                                         "statistical_analysis")
mci_ad_statistical_analysis_resource_dir = os.path.join(get_project_root(), "resources", "mci_ad",
                                                        "statistical_analysis")

memory_performance_csv = os.path.join(get_project_root(), "resources", 'elderly', "statistical_analysis",
                                      "memory_performance.csv")

AGGRGATED_ROI_FILE = 'aggregated_RoIs.pkl'
RAW_GAZE_FILE = 'all_subject_gaze.pkl'
roi_size = 1 / 9  # 9th of the screen
rois_rects_file = f'rects_{int(1 / roi_size)}th.pkl'

# Experimental Details
number_of_sessions = 2
pickling_protocol = -1
INVALID_DATA_RATIO = 0.3

# Valid Movies
if POPULATION == "animation" or POPULATION == "yoavdata":
    valid_movies = list(
        pd.read_pickle(os.path.join(rois_dir, AGGRGATED_ROI_FILE)).index.get_level_values(MOVIE).unique())
    DATE_FORMAT = '%Y_%b_%d_%H%M'

    VERTICAL_SIZE_IN_CM = 20  # Monitor height in cm
    HORIZONTAL_SIZE_IN_CM = 35  # Monitor height in cm
    ORIG_VERTICAL_RESULOTION_IN_PXL = 2160  # 1080  # Vertical resolution of the monitor
    ORIG_HORIZONTAL_RESULOTION_IN_PXL = 3840  # 1920  # Horizontal resolution of the monitor

    NUMBER_OF_ANIMATIONS = 65 if POPULATION == "animation" else 60
    last_repeating_movie_ind = 64
    total_recorded_movies = 65
    num_repeating_movies = 64

else:
    valid_movies = "mov1, mov2, mov8, mov11, mov12, mov13, mov15, mov16, mov20, mov21," \
                   " mov26, mov27, mov29, mov30, mov32, mov35, mov36, mov37, mov42, mov43, " \
                   "mov45, mov47, mov51, mov52, mov53, mov54, mov55, mov56, mov57, mov58, " \
                   "mov59, mov61, mov62, mov65, mov66, mov67, mov69, mov70, mov71, mov72, mov73, " \
                   "mov74, mov75, mov76, mov77, mov78, mov79, mov80".split(", ")
    DATE_FORMAT = '%Y_%m_%d_%H%M'

    VERTICAL_SIZE_IN_CM = 29  # Monitor height in cm
    HORIZONTAL_SIZE_IN_CM = 51  # Monitor height in cm
    ORIG_VERTICAL_RESULOTION_IN_PXL = 1080  # Vertical resolution of the monitor
    ORIG_HORIZONTAL_RESULOTION_IN_PXL = 1920  # Horizontal resolution of the monitor

    last_repeating_movie_ind = 80
    total_recorded_movies = 101
    num_repeating_movies = 80

DISTANCE_IN_CM = 60  # Distance between monitor and participant in cm

# Colors
pink = (130, 10, 229)  # Pink
green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
light_blue = (255, 120, 0)
dark_green = (0, 0.5, 0, 1)  # Dark Green
light_pink = (1, 0.75, 0.8, 1)  # Light Pink
teal = (0, 0.5, 0.5, .2)  # Teal
salmon = (1, 0.6, 0.6, .2)  # Salmon
olive = (0.5, 0.5, 0, .2)  # Olive
raspberry = (0.7, 0, 0.25, .2)  # Raspberry
