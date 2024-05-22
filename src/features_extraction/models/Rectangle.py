from src.features_extraction.models.Point import Point


class Rectangle:

    def __init__(self, tl: Point, br: Point):
        self.tl = tl
        self.br = br

    def __str__(self):
        return f"Top-left:{self.tl}\nBottom-right:{self.br}"
