from src import config as g_config
from src.visualize.models.RelativePoint import RelativePoint


class Fixation(RelativePoint):

    def __init__(self, x: float, y: float, t0: float, duration: float, session: str, mov: str):
        RelativePoint.__init__(self, x, y, mov)
        self.t0 = t0  # ms
        self.duration = duration  # ms
        self.session = session.lower()
        self.color = self._determine_color()

    def _determine_color(self) -> (int, int, int):
        if self.session == "session a":
            return g_config.light_blue
        elif self.session == "session b":
            return g_config.green
