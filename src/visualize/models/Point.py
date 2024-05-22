import math


class Point:

    def __init__(self, x: float, y: float):
        self.x = int(x) if not math.isnan(x) else 0
        self.y = int(y) if not math.isnan(y) else 0

    def __str__(self):
        return "({},{})".format(self.x, self.y)

    def as_tuple(self) -> (int, int):
        return self.x, self.y
