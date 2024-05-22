from src.roi.services import Aggregator, DataService, Validator

def run(args: list):
    calculate, validate, log, metric, load = _extract_args(args)

    if calculate is False and validate is False:
        raise AttributeError("Please choose if you would to calculate and\or validate RoIs")

    if calculate:
        DataService.init()
        Aggregator.generate_roi_rectangles(write=True, use_median=True)

    if validate:
        if metric is None:
            raise AttributeError("Please choose a metric to validate the data by")
        DataService.init(with_rois=True)
        Validator.validate(metric, log, load)


def _extract_args(args: list) -> (bool, bool, bool, str, bool):
    calculate = "calculate" in args
    validate = "valid" in args or "validate" in args
    log = "log" in args
    load = "load" in args
    if "rect 9" in args:
        metric = "rect 9"
    elif "rect 16" in args:
        metric = "rect 16"
    elif "time" in args:
        metric = "time"
    elif "graph" in args:
        metric = "graph"
    else:
        metric = None

    return calculate, validate, log, metric, load


run(["calculate"])
