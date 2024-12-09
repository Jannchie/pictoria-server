import logging
import threading
import typing
from pathlib import Path
from typing import Optional

from rich import get_console
from rich.logging import RichHandler
from wdtagger import Tagger

if typing.TYPE_CHECKING:
    from ai import OpenAIImageAnnotator

db_path = Path()
vec_path = Path()

target_dir = Path()
pictoria_dir = Path()
thumbnails_dir = Path()
should_watch = True
stop_event = threading.Event()

console = get_console()

openai_key: None | str = None

caption_annotator: Optional["OpenAIImageAnnotator"] = None


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
