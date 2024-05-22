import os

import cv2

from src.features_extraction.models.Point import Point
from src.visualize.models.Rectangle import Rectangle


class Video:

    def __init__(self, dir: str, name: str, indexer: str, ROI: Rectangle, roi_point: Point):

        self.name = name
        self.path = os.path.join(dir, f"{name}.mp4")
        self.indexer = indexer
        self.ROI = ROI
        self.roi_point = roi_point
        self.cap = None
        self.width = None
        self.height = None
        self.num_frames = None
        self.fps = None
        self.duration = None
        self.frame_duration = None
        self.marked_frames = None
        self._get_meta_data()

    def is_open(self) -> bool:
        if self.cap is None:
            return False
        return self.cap.isOpened()

    def _get_meta_data(self) -> None:

        self.open()

        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = float(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = float(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.num_frames / self.fps  # seconds
        self.frame_duration = 1000 * self.duration / self.num_frames  # ms

        self.close()

    def open(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.is_open():
            raise Exception("cv2 could not opened the following video: {}".format(self.path))

    def close(self):
        if self.is_open():
            self.cap.release()  # release cv2.VideoCapture
            cv2.destroyAllWindows()  # close all frames

    def read(self):
        return self.cap.read()

    def get(self, param):
        return self.cap.get(param)
