import logging
import time
from logging import Logger
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_co_axes(dim_total: int, axes: List[int]) -> List[int]:
    axes_co = list(set(range(dim_total)).difference(set(axes)))
    return axes_co


def create_default_logger(
    project_path: Optional[Path], prefix: str, stream_level: int = logging.INFO
) -> Logger:
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(stream_level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if project_path is not None:
        # create file handler
        timestr = "_" + time.strftime("%Y%m%d%H%M%S")
        log_dir_path = project_path / "log"
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir_path / (prefix + timestr + ".log")

        fh = logging.FileHandler(str(log_file_path))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        logger.addHandler(fh)

        log_sym_path = log_dir_path / ("latest_" + prefix + ".log")

        logger.info("create log symlink :{0} => {1}".format(log_file_path, log_sym_path))
        if log_sym_path.is_symlink():
            log_sym_path.unlink()
        log_sym_path.symlink_to(log_file_path)
    return logger
