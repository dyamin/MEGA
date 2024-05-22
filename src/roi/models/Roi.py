from src.roi.models.Point import *


class Roi:

    def __init__(self, width: float, height: float, mid_x: float, mid_y: float,
                 std_x: float, std_y: float):
        self.width = width
        self.height = height
        self.mid = Point(mid_x, mid_y)
        self.std = Point(std_x, std_y)
        self.left = None
        self.right = None
        self.bottom = None
        self.top = None

    def __str__(self):
        return f"right:{self.right}\nleft:{self.left}\ntop:{self.top}\nbottom:{self.bottom}"
