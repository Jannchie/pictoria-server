import argparse
import contextlib
import hashlib
import multiprocessing
import os
import signal
import threading
import time
from pathlib import Path

import wdtagger
from fastapi import FastAPI
from fastapi.routing import APIRoute
from PIL import Image
from rich.progress import Progress
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import shared
from alembic import command
from alembic.config import Config
from models import Folder, Post, PostHasTag, Tag, TagGroup
from shared import logger


def initialize_directories(args):
    shared.target_dir = get_target_dir(args)
    shared.pictoria_dir = get_pictoria_directory()
    shared.db_path = get_db_path()
    init_thumbnails_directory()


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """
    Simplify operation IDs so that generated API clients have simpler function
    names.

    Should be called only after all routes have been added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


def migrate_db(db_path):

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    try:
        command.upgrade(alembic_cfg, "head")
    except Exception as e:
        logger.error(f"Error while migrating database: {e}")
        exit(1)
    logger.info("Database migration successful")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target_dir", type=str, default=".")
    args = parser.parse_args()
    return args


def get_pictoria_directory():
    pictoria_dir = shared.target_dir / ".pictoria"
    if not pictoria_dir.exists():
        pictoria_dir.mkdir()
        logger.info(f'Created directory "{pictoria_dir}"')
    return pictoria_dir


def validate_path(target_path):
    if not target_path.exists():
        logger.info(f'Error: Path "{target_path}" does not exist', style="bold red")
        exit(1)


def get_target_dir(args):
    target_dir_str = args.target_dir
    logger.info(f'Target path: "{target_dir_str}"')
    target_dir = Path(target_dir_str).resolve()
    validate_path(target_dir)
    return target_dir


def get_db_path():
    db_path = shared.pictoria_dir / "metadata.db"
    logger.info(f"Using database file: {db_path}")
    return db_path


def execute_database_migration():
    task = multiprocessing.Process(target=migrate_db, args=(shared.db_path,))
    task.start()
    task.join()


def init_thumbnails_directory():
    shared.thumbnails_dir = shared.pictoria_dir / "thumbnails"
    if not shared.thumbnails_dir.exists():
        shared.thumbnails_dir.mkdir()
        logger.info(f'Created directory "{shared.thumbnails_dir}"')


def remove_deleted_files(session, *, os_tuples_set, db_tuples_set):
    deleted_files = db_tuples_set - os_tuples_set
    if deleted_files:
        logger.info(f"Detected {len(deleted_files)} files have been deleted")
        for file_path in deleted_files:
            delete_by_file_path_and_ext(session, file_path)
        session.commit()
        logger.info("Deleted files from database")


def delete_by_file_path_and_ext(session, file_path_and_ext):
    session.query(Post).filter(Post.file_path == file_path_and_ext[0], Post.extension == file_path_and_ext[1]).delete()
    if file_path_and_ext[1]:
        relative_path = Path(file_path_and_ext[0]).with_suffix(f".{file_path_and_ext[1]}")
    else:
        relative_path = Path(file_path_and_ext[0])
    thumbnails_path = shared.thumbnails_dir / relative_path
    if thumbnails_path.exists():
        os.remove(thumbnails_path)


def add_new_files(session, *, os_tuples_set, db_tuples_set):
    new_files = os_tuples_set - db_tuples_set
    if new_files:
        logger.info(f"Detected {len(new_files)} new files")
        for file_path in new_files:
            image = Post(file_path=file_path[0], extension=file_path[1])
            session.add(image)
        session.commit()
        logger.info("Added new files to database")


def sync_metadata():
    threading.Thread(
        target=_sync_metadata,
    ).start()


def _sync_metadata():

    os_tuples = find_files_in_directory(shared.target_dir)

    session = get_session()
    rows = session.query(Post.file_path, Post.extension).all()
    db_tuples = [(row[0], row[1]) for row in rows]
    logger.info(f"Found {len(db_tuples)} files in database")

    db_tuples_set = set(db_tuples)
    os_tuples_set = set(os_tuples)

    remove_deleted_files(session, os_tuples_set=os_tuples_set, db_tuples_set=db_tuples_set)
    add_new_files(session, os_tuples_set=os_tuples_set, db_tuples_set=db_tuples_set)
    process_posts()


def get_session(engine=None):
    if engine is None:
        engine = create_engine(f"sqlite:///{shared.db_path}", echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


def find_files_in_directory(target_dir):
    os_tuples = []
    for file_path in target_dir.rglob("*"):
        relative_path = file_path.relative_to(target_dir)
        if file_path.is_file() and not relative_path.parts[0].startswith("."):
            # 对于所有 relative_path, 需要将其拆分成路径 str 和扩展名 str 的 tuple。然后将其转换为 set。
            ext = file_path.suffix[1:]
            # 移除尾部的扩展名，强制使用正斜杠作为路径分隔符。
            name_without_ext = str(relative_path.with_suffix("")).replace("\\", "/")
            os_tuples.append((name_without_ext, ext))
    logger.info(f"Found {len(os_tuples)} files in target directory")
    return os_tuples


def calculate_md5(file: bytes) -> str:
    # 读取文件的内容并计算 md5 值。
    md5 = hashlib.md5()
    md5.update(file)
    return md5.hexdigest()


def create_thumbnail(input_image_path, output_image_path, max_width=400):
    with Image.open(input_image_path) as img:
        width, height = img.size
        if width > max_width:
            new_width = max_width
            new_height = int((new_width / width) * height)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.save(output_image_path)


def process_posts(all=False):
    """Process posts in the database. Including calculating MD5 hash, size, and creating thumbnails.

    Args:
        all (bool, optional): Process all posts or only those without an MD5 hash. Defaults to False.
    """
    db_path = shared.db_path
    target_dir = shared.target_dir
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    if not all:
        posts = session.query(Post).filter(Post.md5.is_(None)).all()
    else:
        posts = session.query(Post).all()
    with Progress(console=shared.console) as progress:

        if not posts:
            logger.info("No posts to process")
            return
        task = progress.add_task("Processing posts...", total=len(posts))
        for post in posts:
            # 从 file_path 和 extension 属性构建文件的完整路径。
            file_abs_path = target_dir / post.file_path
            file_abs_path = file_abs_path.with_suffix(f".{post.extension}")
            process_post(session, file_abs_path, post)
            progress.update(task, advance=1)


def get_relative_path_and_extension(file_path: Path):
    # 如果时绝对路径，则将其转换为相对路径。
    if file_path.is_absolute():
        basic_path = file_path.relative_to(shared.target_dir)
    else:
        basic_path = file_path
    ext = file_path.suffix[1:]
    name_without_ext = str(basic_path.with_suffix("")).replace("\\", "/")
    return name_without_ext, ext


def process_post(session, file_abs_path=None, post=None):
    if post is None:
        file_path, extension = get_relative_path_and_extension(file_abs_path)
        post = session.query(Post).filter(Post.file_path == file_path, Post.extension == extension).first()
        if post is None:
            post = Post(file_path=file_path, extension=extension)
    if post is None:
        logger.info(f"Post not found in database: {file_abs_path}")
        return
    try:
        logger.debug(f"Processing file: {file_abs_path}")
        with file_abs_path.open("rb") as file:
            file_data = file.read()
            file.seek(0)  # 重置文件指针位置
            with Image.open(file) as img:
                img.verify()  # 验证图像文件的有效性。
                # 读取长宽并更新数据库。
                post.width, post.height = img.size
                post.aspect_ratio = post.width / post.height
                relative_path = file_abs_path.relative_to(shared.target_dir)
                thumbnails_path = shared.thumbnails_dir / relative_path
                if not thumbnails_path.exists():
                    os.makedirs(thumbnails_path.parent, exist_ok=True)
                    create_thumbnail(
                        file_abs_path,
                        thumbnails_path,
                    )

    except Exception as e:
        logger.warn(f"Error processing file: {file_abs_path}")
        logger.exception(e)
    finally:
        post.md5 = calculate_md5(file_data)
        post.size = file_abs_path.stat().st_size

        # 从 file_path 获取所有上级目录，存到列表里
        folder = str(Path(post.file_path).parents[0]).replace("\\", "/")

        # 查询 Folder 表，如果不存在则创建
        folder_record = session.query(Folder).filter(Folder.path == folder).first()
        if folder_record is None:
            folder_record = Folder(path=folder, file_count=0)
            session.add(folder_record)
        folder_record.file_count += 1

        session.add(post)
        session.commit()


def remove_post(session, file_abs_path=None, post=None, auto_commit=True):
    if post is None:
        file_path, extension = get_relative_path_and_extension(file_abs_path)
        post = session.query(Post).filter(Post.file_path == file_path, Post.extension == extension).first()
    else:
        file_abs_path = shared.target_dir / post.file_path
        file_abs_path = file_abs_path.with_suffix(f".{post.extension}")
    if post is None:
        logger.info(f"Post not found in database: {file_abs_path}")
        return
    logger.info(f"Removing post: {post.file_path}.{post.extension}")
    relative_path = file_abs_path.relative_to(shared.target_dir)
    thumbnails_path = shared.thumbnails_dir / relative_path
    if thumbnails_path.exists():
        os.remove(thumbnails_path)
        logger.info(f"Removed thumbnail: {thumbnails_path}")
    session.delete(post)
    if auto_commit:
        session.commit()
    logger.info(f"Removed post from database: {file_abs_path}")


def remove_post_in_path(session, file_path: Path):
    # 如果 file_path 是绝对路径，则转换为相对路径
    if file_path.is_absolute():
        file_path = file_path.relative_to(shared.target_dir)
    file_path = str(file_path).replace("\\", "/")
    # 删除数据库中前缀和 file_path 相同的所有文件
    posts = session.query(Post).filter(Post.file_path.like(f"{file_path}%")).all()
    for post in posts:
        remove_post(session, post=post, auto_commit=False)
    session.commit()


class Watcher:
    def __init__(self, directory_to_watch):
        self.DIRECTORY_TO_WATCH = directory_to_watch
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        while not shared.stop_event.is_set():
            time.sleep(1)
        logger.info("Stopping watcher")
        self.observer.stop()
        self.observer.join()

    def stop(self):
        shared.should_watch = False


class Handler(FileSystemEventHandler):
    def __init__(self, debounce_time=1):
        super().__init__()
        self.last_event_times = {}
        self.debounce_time = debounce_time
        self.lock = threading.Lock()
        self.engine = create_engine(f"sqlite:///{shared.db_path}", echo=False)
        self.session = get_session(self.engine)

    def on_any_event(self, event):

        if event.src_path.startswith(str(shared.pictoria_dir)):
            return None
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
            session = get_session(self.engine)
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


def watch_target_dir():
    w = Watcher(shared.target_dir)
    threading.Thread(target=w.run).start()


def signal_handler(*_):
    logger.info("Exit signal received, stopping threads...")
    shared.stop_event.set()


signal.signal(signal.SIGINT, signal_handler)


def from_rating_to_int(rating):
    """
    0: Not Rated
    1: general
    2. sensitive
    3: questionable
    4: explicit
    """
    if rating == "general":
        return 1
    elif rating == "sensitive":
        return 2
    elif rating == "questionable":
        return 3
    elif rating == "explicit":
        return 4
    else:
        return 0


def attach_tags_to_post(session, post: Post, resp: wdtagger.Result, is_auto=False):
    tags = resp.general_tags
    # 查看是否有名为 general 或者 character 的 TagGroup，如果没有则创建
    group_names = ["general", "character"]
    colors = {
        "general": "#006192",
        "character": "#8243ca",
    }
    for tag_group_name in group_names:
        tag_group = session.query(TagGroup).filter(TagGroup.name == tag_group_name).first()
        if tag_group is None:
            tag_group = TagGroup(name=tag_group_name)
            session.add(tag_group)

    for i, tags in enumerate([resp.general_tags, resp.character_tags]):
        name = group_names[i]
        tag_group: TagGroup = session.query(TagGroup).filter(TagGroup.name == name).first()
        existing_tag_records = {
            tag_record.tag_name for tag_record in session.query(PostHasTag).filter(PostHasTag.tag_name.in_(tags)).all()
        }
        new_tags = set(tags) - existing_tag_records

        # 如果既存的 tag 不属于任何 tag group，则将其放到对应 name 的 tag_group 中
        for tag_name in existing_tag_records:
            tag: Tag = session.query(Tag).filter(Tag.name == tag_name).first()
            if tag.group_id is None:
                tag.group_id = tag_group.id
                session.add(tag)

        for tag_name in new_tags:
            new_tag = Tag(name=tag_name, group_id=tag_group.id)
            session.add(new_tag)

        for tag_name in tags:
            if tag_name in existing_tag_records:
                continue
            postHasTag = PostHasTag(post_id=post.id, tag_name=tag_name, is_auto=is_auto)
            session.add(postHasTag)

    # 提交更改
    session.commit()
