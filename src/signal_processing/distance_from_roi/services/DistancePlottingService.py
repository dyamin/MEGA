import pandas as pd


class DistancePlottingService:
    # index names for queries:
    SUBJECT = 'Subject'
    SESSION = 'Session'
    MOVIE = 'Movie'
    MEMORY = 'Memory'
    TIMESTAMP = 'TimeStamp'

    # plotting constants:
    JPG = '.jpg'
    FIG_SIZE = (20, 11)
    EVENT_TIME_LINE_COLOR, EVENT_TIME_FACE_COLOR = '#668cff', '#99b3ff'
    SUP_TITLE_FONTSIZE, SUB_TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE = 22, 18, 14
    LIGHT_LINESTYLE, MEDIUM_LINESTYLE, DASHED_LINESTYLE = ':', "-.", "--"

    def __init__(self, plots_dir: str):
        import os
        if not (os.path.isdir(plots_dir)):
            os.mkdir(plots_dir)
        self.plots_dir = plots_dir
        return

    def save_figure(self, figure, file_name: str, directory_name: str = None):
        import os
        assert ((file_name is not None)
                and (len(file_name) > 0)
                and (all([c.isalpha() or c.isdigit() or c == '_' or c == '-' for c in file_name]))
                ), f'Must provide a legal file_name, composed of alphanumeric characters only, {file_name} given.'
        directory_path = self.plots_dir if (directory_name is None) else os.path.join(self.plots_dir, directory_name)
        if not (os.path.isdir(directory_path)):
            os.mkdir(directory_path)
        full_path = os.path.join(directory_path, file_name + self.JPG)
        figure.savefig(full_path)
        return

    def plot_distances_impl(self, distances: list, labels: list, line_colors: list,
                            roi_time: float, roi_duration: float, **kwargs):
        assert (len(distances) == len(labels)
                ), f'All plotted distances must have a matching label. {len(distances)} distances and {len(labels)} provided.'
        assert (len(distances) == len(labels)
                ), f'All plotted distances must have a matching color. {len(distances)} distances and {len(line_colors)} provided.'
        min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory = self.__unpack_plotting_kwargs(
            **kwargs)
        assert ((errors is None)
                or (len(distances) == len(errors))
                or (len(errors) == 0)
                ), f'Length mismatch: must have error-bars for all distances or none of them.'

        import matplotlib.pyplot as plt
        plt.close('all')
        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.FIG_SIZE)

        for i in range(len(distances)):
            dist_series, label, color = distances[i], labels[i], line_colors[i]
            error_series = errors[i] if ((errors != None) and (len(errors))) else pd.Series()
            self._plot_with_regards_to_number_of_samples(ax, dist_series, error_series, color, label,
                                                         min_percent_samples, ignore_below_min_samples)

        self._mark_event_time_on_plot(ax, roi_time, roi_duration)
        fig.suptitle(sup_title, fontsize=self.SUP_TITLE_FONTSIZE, y=0.95)
        self._add_axis_titles(ax, ax_title, x_label, y_label)
        if (filename):
            self.save_figure(fig, filename, directory)
        return fig

    @staticmethod
    def __unpack_plotting_kwargs(**kwargs):
        min_percent_samples = kwargs.pop('min_percent_samples', 70)
        ignore_below_min_samples = kwargs.pop('ignore_below_min_samples', False)
        errors = kwargs.pop('errors', None)
        sup_title = kwargs.pop('sup_title', '')
        ax_title = kwargs.pop('ax_title', '')
        x_label = kwargs.pop('x_label', '')
        y_label = kwargs.pop('y_label', '')
        filename = kwargs.pop('filename', '')
        directory = kwargs.pop('directory', None)
        return min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory

    def _mark_event_time_on_plot(self, plot_axis, roi_time: float, roi_duration: float):
        plot_axis.axvline(x=roi_time, c=self.EVENT_TIME_LINE_COLOR, label='Event Time', lw=2)
        plot_axis.axvline(x=roi_time - roi_duration, c=self.EVENT_TIME_LINE_COLOR, ls='--')
        plot_axis.axvline(x=roi_time + roi_duration, c=self.EVENT_TIME_LINE_COLOR, ls='--')
        plot_axis.axvline(x=roi_time - 2 * roi_duration, c=self.EVENT_TIME_LINE_COLOR, ls=':')
        plot_axis.axvline(x=roi_time + 2 * roi_duration, c=self.EVENT_TIME_LINE_COLOR, ls=':')
        plot_axis.axvspan(roi_time - roi_duration, roi_time + roi_duration, facecolor=self.EVENT_TIME_FACE_COLOR,
                          alpha=0.2)
        plot_axis.axvspan(roi_time - roi_duration, roi_time - 2 * roi_duration, facecolor=self.EVENT_TIME_FACE_COLOR,
                          alpha=0.08)
        plot_axis.axvspan(roi_time + roi_duration, roi_time + 2 * roi_duration, facecolor=self.EVENT_TIME_FACE_COLOR,
                          alpha=0.08)
        return

    def _plot_with_regards_to_number_of_samples(self, plt_axis, distances, errors, line_color: str, line_label: str,
                                                min_percent_samples: float, ignore_below_min_samples: bool):
        '''
        Plots the given mean distance on the given @plt_axis, on timestamps with at lease @min_percent_samples out of max possible samples.
        @args:
            distances: pd.Series of distances, containing an index-level matching self.TIMESTAMP
            plt_axis: pyplot axis object
            min_percent_samples: float within range [0,100]; the minimal percent of samples-per-timestamp necessary to draw that timestamp
            ignore_below_min_samples: bool; If True, only timestamps with more that the threshold sample-size will be plotted. If False, all will be plotted but with varying line-width and style
            line_color, line_label: str; If not provided, will use plt's default colors and none-label
        '''
        assert ((min_percent_samples >= 0) and (
                min_percent_samples <= 100)), f'Argument @min_percent_samples must be between 0 and 100, {min_percent_samples} given.'
        if (len(distances) <= 0):
            return
        grouped_distances = distances.groupby(level=self.TIMESTAMP)
        samples_counts = grouped_distances.size()
        mean_distances = grouped_distances.mean()
        max_possible_samples = samples_counts.max()

        mean_distance_with_high_sample_count = mean_distances[
            samples_counts >= max_possible_samples * min_percent_samples / 100]
        mean_distance_with_mediun_sample_count = mean_distances[
            samples_counts >= max_possible_samples * min_percent_samples / (100 * 2)]
        mean_distance_with_low_sample_count = mean_distances

        plt_axis.plot(mean_distance_with_high_sample_count, color=line_color, label=line_label, lw=2.5)
        if not (ignore_below_min_samples):
            # add lighter lines where the data doesn't have enough samples
            plt_axis.plot(mean_distance_with_mediun_sample_count, color=line_color, ls="-.", lw=1.7, alpha=0.85)
            plt_axis.plot(mean_distance_with_low_sample_count, color=line_color, ls=":", alpha=0.65)
        if (len(errors)):
            errors_with_high_samples_count = errors[samples_counts >= max_possible_samples * min_percent_samples / 100]
            plt_axis.plot(mean_distance_with_high_sample_count + errors_with_high_samples_count, color=line_color,
                          lw=0.8, alpha=0.5)
            plt_axis.plot(mean_distance_with_high_sample_count - errors_with_high_samples_count, color=line_color,
                          lw=0.8, alpha=0.5)
        return

    def _add_axis_titles(self, plt_axis, axis_title: str, x_label: str, y_label: str):
        ''' Adds a title, legend and x&y labels to the plt_axis '''
        plt_axis.set_title(axis_title, fontsize=self.SUB_TITLE_FONTSIZE, y=0.95)
        plt_axis.set_xlabel(x_label, fontsize=self.AXIS_LABEL_FONTSIZE)
        plt_axis.set_ylabel(y_label, fontsize=self.AXIS_LABEL_FONTSIZE)
        plt_axis.legend(fontsize=self.AXIS_LABEL_FONTSIZE)
        return
