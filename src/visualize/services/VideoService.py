import os

from src import config as global_config
from src.visualize import config as vis_config
from src.visualize.business.DrawerLogic import *
from src.visualize.models.Fixation import Fixation
from src.visualize.models.Gaze import Gaze
from src.visualize.models.Video import Video
from src.visualize.services import DataService
import statistics


def draw_on_video(video: Video, write: bool = False):
    try:
        speed = vis_config.speed
    except AttributeError:
        speed = 1.0

    video.open()
    subject = _draw(video)
    video.close()
    _display(video, speed)
    if write:
        _write(video, subject)


def _draw(video: Video) -> str or None:
    """
    :param video: video object

    For each frame in video, this functions finds the relevant data in the frame (gaze, fixations),
    and draws it on it.
    At the end, the function updates the video's marked_frames attr with the marked frames

    @:returns name of subject iff num_subjects == 1
    """

    print("Drawing video {} - START".format(video.name))

    frame_ind = 0

    subjects = vis_config.subjects_to_show
    if subjects == []:
        try:
            num_subjects = vis_config.num_subjects_if_random
        except AttributeError:
            num_subjects = 5
        subjects = DataService.generate_subjects(num_subjects)

    DataService.set_data_for_movie_and_subjects(video.indexer, list(subjects))
    DataService.map_data_to_frame(video.frame_duration)

    frames = list()
    all_fixation = list()
    dva_sesA, dva_sesB = [0], [0]

    while video.is_open():
        mem_set = set()
        ret, frame = video.read()
        if not ret:
            print("Drawing video {} - DONE".format(video.name))
            print("\n*******************************\n")
            break

        if vis_config.draw_roi:
            Drawer.draw_roi_on_frame(frame, video.roi_point)

        if vis_config.draw_rect:
            Drawer.draw_rect_on_frame(frame, video.ROI)

        fixations = DataService.get_data_in_frame(frame_ind, ttype="fixations")
        gazes = DataService.get_data_in_frame(frame_ind, ttype="gaze")
        for ind, attrs in fixations.iterrows():
            # ind = (Subject_Name, Session, Order)
            if vis_config.draw_fixations:
                session = ind[1]
                fixation = Fixation(attrs[vis_config.fixation_X], attrs[vis_config.fixation_Y],
                                    attrs[vis_config.fixation_t0], attrs[config.DURATION], session,
                                    video.indexer)
                if session in vis_config.sessions_to_show:
                    if vis_config.draw_all_fixations:
                        all_fixation.append(fixation)
                    else:
                        Drawer.draw_point_on_frame(frame, fixation, color=fixation.color, radius=10)

        for ind, attrs in gazes.iterrows():
            # ind = (Subject_Name, Session, Timestamp, Confidence)
            session = ind[1]
            memory = ind[3]

            if vis_config.draw_raw:
                gaze = Gaze(attrs[global_config.gaze_X], attrs[global_config.gaze_Y],
                            ind[2], session, video.indexer)

                if session in vis_config.sessions_to_show:
                    Drawer.draw_point_on_frame(frame, gaze, color=gaze.color)

            # if config.draw_pupil:
            #     Drawer.draw_memory_on_frame(frame, str(mem_set), frameSize=(int(video.width), int(video.height)),
            #                                 color=config.green)

            if session == global_config.SESSION_B:
                mem_set.add(memory)
                dva_sesB.append(round(attrs[global_config.DVA]))
            else:
                dva_sesA.append(round(attrs[global_config.DVA]))

        if vis_config.draw_memory and len(mem_set) > 0:
            Drawer.draw_agd_on_frame(frame, str(mem_set), frameSize=(int(video.width), int(video.height)),
                                     color=config.green)

        if vis_config.draw_agd:  # and len(dva_sesB) > 0:
            # Draw dva on frame top left corner, 1st viewing and below that 2nd viewing, as bar plots with different
            # colors
            Drawer.draw_agd_on_frame(frame, str(int(statistics.mean(dva_sesA))), str(int(statistics.mean(dva_sesB))),
                                     frameSize=(int(video.width), int(video.height)))

        frames.append(frame)

        frame_ind += 1

    DataService.reset_data_for_movie_and_subjects()

    video.marked_frames = frames

    if vis_config.draw_all_fixations:
        for fra in video.marked_frames:
            for fix in all_fixation:
                Drawer.draw_point_on_frame(fra, fix, color=fixation.color, radius=10)

    if len(subjects) == 1:
        return list(subjects)[0]


def _display(video: Video, speed: float) -> None:
    _print_instructions(speed)

    frames = video.marked_frames

    for frame in frames:
        frame = cv2.resize(frame, (960, 540))  # Resize image
        cv2.imshow('Gaze video', frame)  # display
        key = cv2.waitKey(int(video.frame_duration / speed))  # display frame for ${frame_duration} ms
        if key == ord('q'):  # press q to quit
            print("Video {} - STOPPED".format(video.name))
            print("\n*******************************\n")
            return

    cv2.destroyAllWindows()


def _write(video: Video, subject: str = None) -> None:
    path = _generate_path(video, subject)

    print("Saving video to path {} - START".format(path))

    frames = video.marked_frames

    writer = cv2.VideoWriter(path, apiPreference=cv2.CAP_FFMPEG,
                             fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                             fps=video.fps, frameSize=(int(video.width), int(video.height)))

    for frame in frames:
        writer.write(frame)

    writer.release()

    print("Saving video to path {} - DONE".format(path))


def _print_instructions(speed: float) -> None:
    print(f"Video will be presented in x{speed} speed")
    print("Red dots stands for Session A, and Green dots for Session B")
    print("Small dots represents a Gaze, and Big dots represents a Fixation")
    print("Press q at any moment to quite")
    print("\n*******************************\n")


def _generate_path(video: Video, subject: str) -> str:
    path = os.path.join(global_config.videos_dir, vis_config.marked_videos_dir, video.name)

    if not os.path.exists(path):
        print(f"Creates new path {path}")
        os.makedirs(path)

    if subject is None:
        return os.path.join(path, f"{video.name}.mp4")
    else:
        return os.path.join(path, f"{video.name}_{subject}.mp4")
