from src import config as global_config
from src.post_processing.services.PostProcessingService import process


def run(args):
    memory_report_dir = _extract_path(args, 0, global_config.memory_reports_dir)
    gaze_dir = _extract_path(args, 1, global_config.decentralized_data_dir)
    return process(memory_report_dir, gaze_dir)


def _extract_path(args: list, index: int, fallback: str) -> str:
    if (len(args) > index) and (len(args[index]) > 0):
        return args[index]
    return fallback


run([])
