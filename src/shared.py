import logging
import threading
from pathlib import Path

from rich import get_console
from rich.logging import RichHandler
from wdtagger import Tagger

from ai import OpenAIImageAnnotator

db_path: None | Path = None

target_dir: Path = Path()
pictoria_dir: Path = Path()
thumbnails_dir: Path = Path()

should_watch = True
stop_event = threading.Event()

console = get_console()

openai_key: None | str = None
caption_annotator: None | OpenAIImageAnnotator = None


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
    )

    logger = logging.getLogger("pictoria")
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()

tagger: Tagger | None = None
