import signal
import threading
import time
from pathlib import Path

from sqlalchemy import create_engine
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

import shared
from processors import process_post
from shared import logger
from utils import get_session, remove_post, remove_post_in_path


class Watcher:
    def __init__(self, directory_to_watch: Path) -> None:
        self.DIRECTORY_TO_WATCH = directory_to_watch
        self.observer = Observer()

    def run(self) -> None:
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        logger.info("Starting watcher")
        self.observer.start()
        while not shared.stop_event.is_set():
            time.sleep(1)
        logger.info("Stopping watcher")
        self.observer.stop()
        self.observer.join()

    def stop(self) -> None:
        shared.should_watch = False


class Handler(FileSystemEventHandler):
    def __init__(self, debounce_time: int = 1) -> None:
        super().__init__()
        self.last_event_times = {}
        self.debounce_time = debounce_time
        self.lock = threading.Lock()
        self.engine = create_engine(f"sqlite:///{shared.db_path}", echo=False)

    def on_any_event(self, event: FileSystemEvent):

        if event.src_path.startswith(str(shared.pictoria_dir)):
            return
        if event.is_directory:
            return

        event_key = (event.src_path, event.event_type)
        current_time = time.time()
        with self.lock:
            last_event_time = self.last_event_times.get(event_key, 0)
            if current_time - last_event_time < self.debounce_time:
                return
            self.last_event_times[event_key] = current_time

        try:
            session = get_session()
            if event.event_type == "created":
                logger.debug(f"Received created event - {event.src_path}")
                process_post(session, Path(event.src_path))
            elif event.event_type == "modified":
                logger.debug(f"Received modified event - {event.src_path}")
                process_post(session, Path(event.src_path))
            elif event.event_type == "deleted":
                logger.debug(f"Received deleted event - {event.src_path}")
                if Path(event.src_path).is_file():
                    remove_post(session, Path(event.src_path))
                else:
                    remove_post_in_path(session, Path(event.src_path))
            # self.sync_metadata_folder()
        except Exception as e:
            logger.error(f"Error processing event: {event}")
            logger.exception(e)


def watch_target_dir() -> None:
    w = Watcher(shared.target_dir)
    threading.Thread(target=w.run).start()


def signal_handler(*_: tuple) -> None:
    logger.info("Exit signal received, stopping threads...")
    shared.stop_event.set()


signal.signal(signal.SIGINT, signal_handler)
