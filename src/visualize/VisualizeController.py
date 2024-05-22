import src.visualize.services.DataService as DataService
import src.visualize.services.VideoService as VideoService
from src import config
from src.visualize.models.Video import Video

"""
Welcome to the visualizer!
This component is responsible for the visualization of the data.
Multiple cmd parameters are optional in order to determine the visualizer mode:
    * movie - the number of a movie from videos directory you would like to visualize

(Important! Directories and other key parameters are set in the config file)
"""


def run():
    DataService.init()
    for index in range(52, 75):
        video_name, video_indexer = _naming_convention(index)
        roi = DataService.get_video_roi(video_indexer)
        roi_point = DataService.get_video_roi_point(video_indexer)

        video = Video(config.videos_dir, video_name, video_indexer, roi, roi_point)

        VideoService.draw_on_video(video, write=True)


def _naming_convention(ind) -> (str, str):
    if ind < 10:
        name = f"00{ind}p"  # convention
    elif ind < 100:
        name = f"0{ind}p"  # convention
    else:
        name = f"{ind}p"  # convention
    indexer = f"mov{ind}"

    return name, indexer


run()
