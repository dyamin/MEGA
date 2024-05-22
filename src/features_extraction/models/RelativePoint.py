from src.features_extraction.models.Point import Point


class RelativePoint(Point):
    """

    A relative point is a point where (x,y) fits the sizes of the original screen it was measured on.
    Hence, conversion is needed

    """

    def __init__(self, x: float, y: float, mov: str):
        absolute_x = x  # / g_config.ORIG_HORIZONTAL_RESULOTION_IN_PXL * DataService.videos_dims.loc[mov, 'Width']
        absolute_y = y  # / g_config.ORIG_VERTICAL_RESULOTION_IN_PXL * DataService.videos_dims.loc[mov, 'Height']
        Point.__init__(self, absolute_x, absolute_y)
