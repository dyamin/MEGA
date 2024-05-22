from src.visualize.models.Point import Point


class Rectangle:

    def __init__(self, **measurements):

        self.tl = measurements.get('tl')
        br = measurements.get('br')
        if br:
            self.br = br
            self.width = abs(self.br.x - self.tl.x)
            self.height = abs(self.br.y - self.tl.y)
        else:
            width = measurements.get('width')
            height = measurements.get('height')
            self.width = width
            self.height = height
            self.br = Point(self.tl.x + width, self.tl.y + height)

    def __str__(self):
        return "Top-Left: {}\n" \
               "Bottom-Right {}\n".format(self.tl, self.br)
