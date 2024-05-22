from src.visualize.models.Point import Point


class RelativePoint(Point):
    """

    A relative point is a point where (x,y) fits the sizes of the original screen it was measured on.
    Hence, conversion is needed

    """

    def __init__(self, x: float, y: float, mov: str):
        # absolute_x = x / config.original_screen_size[0] * GetSystemMetrics(0)
        # absolute_y = y / config.original_screen_size[1] * GetSystemMetrics(1)
        absolute_x = x  # (x / config.ORIG_HORIZONTAL_RESULOTION_IN_PXL) * DataService.videos_dims.loc[mov, 'Width']
        absolute_y = y  # (y / config.ORIG_VERTICAL_RESULOTION_IN_PXL) * DataService.videos_dims.loc[mov, 'Height']
        Point.__init__(self, absolute_x, absolute_y)
