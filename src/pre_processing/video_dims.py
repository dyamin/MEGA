import os

import cv2
import pandas as pd

from src import config

df = pd.DataFrame(columns=['Width', 'Height', 'Duration'])

directory_name = config.videos_dir
directory = os.fsencode(directory_name)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".mp4"):
        file_path = os.path.join(directory_name, filename)
        vid = cv2.VideoCapture(file_path)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = vid.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) * 1000  # in milliseconds
        mov_num = filename[0:3].lstrip("0")
        mov = f"mov{mov_num}"
        df.loc[mov] = [width, height, duration]
    else:
        continue

df.to_pickle(os.path.join(config.data_dir, "video_dims.pkl"))
