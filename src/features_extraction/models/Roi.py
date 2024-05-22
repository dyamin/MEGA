from src.features_extraction.models.Point import Point
from src.features_extraction.models.Rectangle import Rectangle


class Roi:

    def __init__(self, tl: Point, br: Point, start: float, end: float):
        self.rect = Rectangle(tl, br)
        self.start = start
        self.end = end

    def __str__(self):
        return f"Start: {self.start}\n" \
               f"End: {self.end}\n" \
               f"Rectangle:{self.rect}"
