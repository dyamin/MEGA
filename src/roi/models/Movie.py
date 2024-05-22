class Movie:
    """
        @note: (0,0) is the upper left corner of a movie
    """

    def __init__(self, width: float, height: float, left: float, right: float, top: float, bottom: float):
        self.width = width
        self.height = height
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __str__(self):
        return f"right:{self.right}, left:{self.left}, top:{self.top}, bottom:{self.bottom}"
