import logging
import threading
from pathlib import Path

from rich import get_console
from rich.logging import RichHandler
from wdtagger import Tagger

db_path: None | Path = None

target_dir: None | Path = None
pictoria_dir: None | Path = None
thumbnails_dir: None | Path = None

should_watch = True
stop_event = threading.Event()

console = get_console()


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
    )

    logger = logging.getLogger("pictoria")
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()

tagger: Tagger | None = None
