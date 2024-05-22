import cv2
import numpy as np

from src import config
from src.features_extraction.models.Rectangle import Rectangle
from src.visualize.models.Point import Point


class Drawer:

    @staticmethod
    def draw_roi_on_frame(frame: np.ndarray, p: Point, inplace=True) -> np.ndarray or None:

        new_frame = cv2.circle(frame, p.as_tuple(), 7, config.red, thickness=5)

        if inplace:
            frame = new_frame
        else:
            return new_frame

    @staticmethod
    def draw_rect_on_frame(frame: np.ndarray, rect: Rectangle, inplace=True) -> np.ndarray or None:

        new_frame = cv2.rectangle(frame, rect.tl.as_tuple(), rect.br.as_tuple(),
                                  config.red, thickness=3)

        if inplace:
            frame = new_frame
        else:
            return new_frame

    @staticmethod
    def draw_point_on_frame(frame: np.ndarray, point: Point, color: tuple = (255, 0, 0),
                            radius=1, thickness=5, inplace=True):

        new_frame = cv2.circle(frame, point.as_tuple(), radius, color, thickness=thickness)

        if inplace:
            frame = new_frame

        else:
            return new_frame

    @staticmethod
    def draw_agd_on_frame(frame: np.ndarray, text_a, text_b, frameSize, color: tuple = config.pink, inplace=True):

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 3
        lineType = 2
        location_a = (int(0.05 * frameSize[0]), int(0.1 * frameSize[1]))
        location_b = (int(0.05 * frameSize[0]), int(0.2 * frameSize[1]))

        # Draw bar plots for the mean distance of the 1st and 2nd viewing with the values as text
        new_frame = cv2.putText(frame, 'AGD(1st)=' + text_a,
                                location_a,
                                font,
                                fontScale,
                                config.light_blue,
                                thickness,
                                lineType)
        new_frame = cv2.rectangle(new_frame, (location_a[0], location_a[1] + 10),
                                  (location_a[0] + int(text_a)*20, location_a[1] + 30),
                                  config.light_blue, thickness=-1)
        new_frame = cv2.putText(new_frame, 'AGD(2nd)=' + text_b,
                                location_b,
                                font,
                                fontScale,
                                config.green,
                                thickness,
                                lineType)
        new_frame = cv2.rectangle(new_frame, (location_b[0], location_b[1] + 10),
                                    (location_b[0] + int(text_b)*20, location_b[1] + 30),
                                    config.green, thickness=-1)



        if inplace:
            frame = new_frame
        else:
            return new_frame