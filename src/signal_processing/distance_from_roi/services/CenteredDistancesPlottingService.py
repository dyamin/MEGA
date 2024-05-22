## TODO: make plotting methods more generic
import pandas as pd
from src.signal_processing.distance_from_roi.services.DataSplittingService import DataSplittingService

from src.signal_processing.distance_from_roi.services.DistanceDifferenceCalculatorService import \
    DistanceDifferenceCalculatorService
from src.signal_processing.distance_from_roi.services.DistancePlottingService import DistancePlottingService
from src.signal_processing.distance_from_roi.services.DistanceReCenteringService import DistanceReCenteringService


class CenteredDistancesPlottingService:
    # index names for queries:
    SUBJECT = 'Subject'
    SESSION = 'Session'
    MOVIE = 'Movie'
    MEMORY = 'Memory'
    TIMESTAMP = 'TimeStamp'

    # plotting constants:
    JPG = '.jpg'
    FIG_SIZE = (20, 11)
    RED, GREEN, BLACK = 'r', 'g', 'k'
    SESSION_A_COLOR1, SESSION_A_COLOR2, SESSION_B_COLOR1, SESSION_B_COLOR2 = '#ff3333', '#ff9999', '#33ff33', '#99ff99'
    EVENT_TIME_LINE_COLOR, EVENT_TIME_FACE_COLOR = '#668cff', '#99b3ff'
    SUP_TITLE_FONTSIZE, SUB_TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE = 22, 18, 14
    LIGHT_LINESTYLE, MEDIUM_LINESTYLE, DASHED_LINESTYLE = ':', "-.", "--"
    X_LABEL, Y_LABEL = 'Time from Event (ms)', 'Distance (px)'

    def __init__(self, gaze_distances, RoIs, plots_dir: str, roi_duration: float = 1500):
        import os
        import pandas as pd

        if not (os.path.isdir(plots_dir)):
            os.mkdir(plots_dir)
        self.plots_dir = plots_dir

        assert (type(RoIs) is pd.DataFrame), f'The argument @RoIs must be an instance of DataFrame.'
        assert (type(gaze_distances) is pd.Series), f'The argument @gaze_distances must be an instance of Series.'
        recentering_logic = DistanceReCenteringService(gaze_distances, RoIs)
        self.centered_distances = recentering_logic.recenter_distances(verbose=False)
        self.rois = RoIs
        self.roi_duration = roi_duration

        self.splitter = DataSplittingService(self.centered_distances)
        self.plotter = DistancePlottingService(self.plots_dir)
        return

    def plot_by_session(self, **kwargs):
        roi_focused, add_errors, _, _, _, filename = self.__unpack_kwargs(**kwargs)
        ses_distances = self.splitter.split_by_sessions()
        errors = [self._calculate_subjects_mean_sem(dists) for dists in ses_distances if len(dists) > 0] if (
            add_errors) else None
        filename = filename if (filename is not None) else 'Distance_from_Event-Sessions'
        return self.plotter.plot_distances_impl(ses_distances, ['Session A', 'Session B'], [self.RED, self.GREEN],
                                                0, self.roi_duration, errors=errors,
                                                sup_title="Mean Distance from Event", ax_title='Grouped by Session',
                                                x_label=self.X_LABEL, y_label=self.Y_LABEL,
                                                filename=filename, ignore_below_min_samples=roi_focused)

    def plot_by_session_and_confidence(self, **kwargs):
        roi_focused, add_errors, include_session_A, include_low_confidence, _, filename = self.__unpack_kwargs(**kwargs)
        sesA_high, sesA_low, sesB_high, sesB_low = self.splitter.split_by_session_and_confidence()

        distances = [sesB_high]
        labels = ['Session B - High Confidence']
        colors = [self.SESSION_B_COLOR1]
        if (include_low_confidence):
            distances += [sesB_low]
            labels += ['Session B - Low Confidence']
            colors += [self.SESSION_B_COLOR2]
        if (include_session_A):
            distances += [sesA_high]
            labels += ['Session A - High Confidence']
            colors += [self.SESSION_A_COLOR1]
            if (include_low_confidence):
                distances += [sesA_low]
                labels += ['Session A - Low Confidence']
                colors += [self.SESSION_A_COLOR2]

        errors = [self._calculate_subjects_mean_sem(dists) for dists in distances if len(dists) > 0] if (
            add_errors) else None
        filename = filename if (filename is not None) else 'Distance_from_Event-Sessions_and_Confidence'
        return self.plotter.plot_distances_impl(distances, labels, colors, 0, self.roi_duration,
                                                errors=errors, sup_title="Mean Distance from Event",
                                                ax_title='Grouped by Session & Confidence', x_label=self.X_LABEL,
                                                y_label=self.Y_LABEL,
                                                filename=filename, ignore_below_min_samples=roi_focused)

    def plot_by_session_and_memory(self, **kwargs):
        roi_focused, add_errors, include_session_A, _, include_not_remembered, filename = self.__unpack_kwargs(**kwargs)
        sesA_remembered, sesA_not_remembered, sesB_remembered, sesB_not_remembered = self.splitter.split_by_session_and_memory()

        distances = [sesB_remembered]
        labels = ['Session B - Remembered']
        colors = [self.SESSION_B_COLOR1]
        if (include_not_remembered):
            distances += [sesB_not_remembered]
            labels += ['Session B - Forgot']
            colors += [
                self.SESSION_A_COLOR1]  # even though this is sesB data, since it's not_remembered, we draw the 2nd line in red (sesA color)
        if (include_session_A):
            distances += [sesA_not_remembered]
            labels += ['Session A - Not Remembered']
            colors += [self.SESSION_B_COLOR2]  # see above comment regarding colors
            if (include_not_remembered):
                distances += [sesA_remembered]
                labels += ['Session A - Remembered']
                colors += [self.SESSION_A_COLOR2]

        errors = [self._calculate_subjects_mean_sem(dists) for dists in distances if len(dists) > 0] if (
            add_errors) else None
        filename = filename if (filename is not None) else 'Distance_from_Event-Sessions_and_Memory'
        return self.plotter.plot_distances_impl(distances, labels, colors, 0, self.roi_duration, errors=errors,
                                                sup_title="Mean Distance from Event",
                                                ax_title='Grouped by Session & Memory',
                                                x_label=self.X_LABEL, y_label=self.Y_LABEL,
                                                filename=filename, ignore_below_min_samples=roi_focused)

    def plot_by_memory_and_confidence_session_B(self, **kwargs):
        roi_focused, add_errors, _, include_low_confidence, include_not_remembered, filename = self.__unpack_kwargs(
            **kwargs)
        _, sesB = self.splitter.split_by_sessions()
        mem_and_conf_splitter = DataSplittingService(sesB)
        remembered_high_conf, remembered_low_conf, not_remembered_high_conf, not_remembered_low_conf = mem_and_conf_splitter.split_by_memory_and_confidence()

        distances = [remembered_high_conf]
        labels = ['Session B - High Confidence - Remembered']
        colors = [self.SESSION_B_COLOR1]

        if (include_low_confidence):
            distances += [remembered_low_conf]
            labels += ['Session B - Low Confidence - Remembered']
            colors += [self.SESSION_B_COLOR2]
        if (include_not_remembered):
            distances += [not_remembered_high_conf]
            labels += ['Session B - High Confidence - Forgot']
            colors += [self.SESSION_A_COLOR1]
            if (include_low_confidence):
                distances += [not_remembered_low_conf]
                labels += ['Session B - Low Confidence - Forgot']
                colors += [self.SESSION_A_COLOR2]

        errors = [self._calculate_subjects_mean_sem(dists) for dists in distances if len(dists) > 0] if (
            add_errors) else None
        filename = filename if (filename is not None) else 'Distance_from_Event-Session_B-Confidence_and_Memory'
        return self.plotter.plot_distances_impl(distances, labels, colors, 0, self.roi_duration,
                                                errors=errors, sup_title="Mean Distance from Event",
                                                ax_title='Grouped by Confidence & Memory', x_label=self.X_LABEL,
                                                y_label=self.Y_LABEL,
                                                filename=filename, ignore_below_min_samples=roi_focused)

    def plot_by_memory_session_differences(self, **kwargs):
        roi_focused, add_errors, _, _, include_not_remembered, filename = self.__unpack_kwargs(**kwargs)
        remembered, not_remembered = self._calculate_session_difference_by_memory(False)

        distances = [remembered]
        labels = ['Remembered']
        colors = [self.SESSION_B_COLOR1]
        if (include_not_remembered):
            distances += [not_remembered]
            labels += ['Not Remembered']
            colors += [self.SESSION_A_COLOR1]

        errors = [self._calculate_subjects_mean_sem(dists) for dists in distances if len(dists) > 0] if (
            add_errors) else None
        fig = self.plotter.plot_distances_impl(distances, labels, colors, 0, self.roi_duration, errors=errors,
                                               sup_title="Mean Distance from Event: Session A - Session B",
                                               ax_title='Grouped by Memory',
                                               x_label=self.X_LABEL, y_label='Distance Difference (px)',
                                               ignore_below_min_samples=roi_focused)
        # add extra graphics to the plot & save:
        self._add_session_differences_graphics_to_figure(fig)
        filename = filename if (filename is not None) else 'Distance_from_Event-Session_Comparison-Memory'
        self.plotter.save_figure(fig, filename)
        return fig

    def plot_by_memory_and_confidence_session_differences(self, **kwargs):
        roi_focused, add_errors, _, include_low_confidence, include_not_remembered, filename = self.__unpack_kwargs(
            **kwargs)
        remembered_high_conf, remembered_low_conf, not_remembered_high_conf, not_remembered_low_conf = self._calculate_session_difference_by_memory(
            True)

        distances = [remembered_high_conf]
        labels = ['High Confidence - Remembered']
        colors = [self.SESSION_B_COLOR1]

        if (include_low_confidence):
            distances += [remembered_low_conf]
            labels += ['Low Confidence - Remembered']
            colors += [self.SESSION_B_COLOR2]
        if (include_not_remembered):
            distances += [not_remembered_high_conf]
            labels += ['High Confidence - Forgot']
            colors += [self.SESSION_A_COLOR1]
            if (include_low_confidence):
                distances += [not_remembered_low_conf]
                labels += ['Low Confidence - Forgot']
                colors += [self.SESSION_A_COLOR2]

        errors = [self._calculate_subjects_mean_sem(dists) for dists in distances if len(dists) > 0] if (
            add_errors) else None
        fig = self.plotter.plot_distances_impl(distances, labels, colors, 0, self.roi_duration,
                                               errors=errors,
                                               sup_title="Mean Distance from Event: Session A - Session B",
                                               ax_title='Grouped by Confidence & Memory', x_label=self.X_LABEL,
                                               y_label='Distance Difference (px)', ignore_below_min_samples=roi_focused)
        # add extra graphics to the plot & save:
        self._add_session_differences_graphics_to_figure(fig)
        filename = filename if (
                filename is not None) else 'Distance_from_Event-Session_Comparison-Confidence_and_Memory'
        self.plotter.save_figure(fig, filename)
        return fig

    ###   HELPER METHODS   ###

    @staticmethod
    def __unpack_kwargs(**kwargs):
        roi_focused = kwargs.pop('roi_focused', False)
        add_errors = kwargs.pop('add_errors', True)
        include_session_A = kwargs.pop('include_session_A', True)
        include_low_confidence = kwargs.pop('include_low_confidence', True)
        include_not_remembered = kwargs.pop('include_not_remembered', True)
        filename = kwargs.pop('filename', None)
        return roi_focused, add_errors, include_session_A, include_low_confidence, include_not_remembered, filename

    def _calculate_session_difference_by_memory(self, should_split_by_confidence_too):
        sesA, sesB = self.splitter.split_by_sessions()
        sesA = sesA.droplevel('Memory')
        mem_and_conf_splitter = DataSplittingService(sesB)
        if (should_split_by_confidence_too):
            sesB_distances = mem_and_conf_splitter.split_by_memory_and_confidence()  # remembered_high_conf_B, remembered_low_conf_B, not_remembered_high_conf_B, not_remembered_low_conf_B
        else:
            sesB_distances = mem_and_conf_splitter.split_by_memory_response()

        differences = list()
        for dists_B in sesB_distances:
            if (len(dists_B) <= 0):
                pass
            splitter = DataSplittingService(dists_B)
            equivalent_distances_A = splitter.extract_matching_subjects_and_movies(sesA)
            diff_calc = DistanceDifferenceCalculatorService(equivalent_distances_A, dists_B)
            differences.append(diff_calc.calculate_differences())
        return differences

    def _calculate_subjects_mean_sem(self, distances: pd.Series):
        subjects_sem = distances.groupby(level=[self.SUBJECT, self.TIMESTAMP]).sem()
        mean_sem = subjects_sem.groupby(level=self.TIMESTAMP).mean()
        return mean_sem / 2

    def _add_session_differences_graphics_to_figure(self, fig):
        axis = fig.axes[0]
        axis.plot([0], [0], color=self.BLACK, marker='o',
                  markersize=self.AXIS_LABEL_FONTSIZE)  # add a dot at point (0,0)
        axis.axhline(y=0, color=self.BLACK, lw=2)  # add a horizontal line at y=0
        min_x, max_x = axis.get_xlim()
        min_y, max_y = axis.get_ylim()
        axis.text(x=min_x + (max_x - min_x) * 0.01, y=0.25 * max_y, s='Session B closer to RoI',
                  color=self.BLACK, rotation=90, fontsize=self.AXIS_LABEL_FONTSIZE)
        axis.text(x=min_x + (max_x - min_x) * 0.01, y=0.75 * min_y, s='Session A closer to RoI',
                  color=self.BLACK, rotation=90, fontsize=self.AXIS_LABEL_FONTSIZE)
        return
