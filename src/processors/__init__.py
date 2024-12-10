import threading
from io import BufferedReader
from pathlib import Path

from PIL import Image
from rich.progress import Progress
from sqlalchemy.orm import Session
from wdtagger import Tagger

import shared
from db import get_img_vec
from models import Post, PostHasColor
from shared import logger
from tools.colors import get_palette_ints
from utils import (
    add_new_files,
    attach_tags_to_post,
    compute_image_properties,
    find_files_in_directory,
    from_rating_to_int,
    get_path_name_and_extension,
    get_session,
    remove_deleted_files,
    update_file_metadata,
)


def sync_metadata():
    threading.Thread(
        target=_sync_metadata,
    ).start()


def _sync_metadata() -> None:
    os_tuples = find_files_in_directory(shared.target_dir)

    session = get_session()
    rows = session.query(Post.file_path, Post.file_name, Post.extension).all()
    db_tuples = [(row[0], row[1], row[2]) for row in rows]
    logger.info(f"Found {len(db_tuples)} files in database")

    db_tuples_set = set(db_tuples)
    os_tuples_set = set(os_tuples)

    remove_deleted_files(session, os_tuples_set=os_tuples_set, db_tuples_set=db_tuples_set)
    add_new_files(session, os_tuples_set=os_tuples_set, db_tuples_set=db_tuples_set)
    process_posts()


def process_posts(*, all_posts: bool = False):
    """Process posts in the database. Including calculating MD5 hash, size, and creating thumbnails.

    Args:
        all (bool, optional): Process all posts or only those without an MD5 hash. Defaults to False.
    """
    target_dir = shared.target_dir
    session = get_session()
    posts = session.query(Post).all() if all_posts else session.query(Post).filter(Post.md5.is_("")).all()
    with Progress(console=shared.console) as progress:
        if not posts:
            logger.info("No posts to process")
            return
        task = progress.add_task("Processing posts...", total=len(posts))
        for post in posts:
            # 构建文件的完整路径。
            file_abs_path = target_dir / post.file_path / f"{post.file_name}.{post.extension}"
            process_post(session, file_abs_path, post)
            progress.update(task, advance=1)


process_post_lock = threading.Lock()


def process_post(session: Session, file_abs_path: Path | None = None, post: Post | None = None):
    with process_post_lock:
        _process_post(session, file_abs_path, post)


def _process_post(session: Session, file_abs_path: Path | None = None, post: Post | None = None) -> None:
    if post is None:
        file_path, file_name, extension = get_path_name_and_extension(file_abs_path)
        post = (
            session.query(Post)
            .filter(Post.file_path == file_path, Post.file_name == file_name, Post.extension == extension)
            .first()
        )
    if post is None:
        logger.info(f"Post not found in database: {file_abs_path}")
        return
    if post.md5:
        logger.info(f"Skipping post: {file_abs_path}")
        return
    file_data = None
    try:
        if file_abs_path is None:
            file_abs_path = shared.target_dir / post.file_path / post.file_name
            file_abs_path = file_abs_path.with_suffix(f".{post.extension}")
        if file_abs_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".avif",
        ]:
            logger.debug(f"Skipping not image file: {file_abs_path}")
            return
        logger.info(f"Processing post: {file_abs_path}")
        with file_abs_path.open("rb") as file:
            file_data = file.read()
            file.seek(0)  # 重置文件指针位置
            with Image.open(file) as img:
                compute_image_properties(img, post, file_abs_path)
            file.seek(0)
            set_post_colors(post, file)

    except Exception as e:
        logger.warning(f"Error processing file: {file_abs_path}")
        logger.exception(e)
    finally:
        if file_data:
            update_file_metadata(file_data, post, file_abs_path)
        session.add(post)
        session.commit()

    threading.Thread(target=get_img_vec, args=(post,)).start()

    def add_tags() -> None:
        abs_path = post.absolute_path
        if shared.tagger is None:
            shared.tagger = Tagger(model_repo="SmilingWolf/wd-vit-large-tagger-v3")
        resp = shared.tagger.tag(abs_path)
        logger.info(resp)
        post.rating = from_rating_to_int(resp.rating)
        attach_tags_to_post(session, post, resp, is_auto=True)
        session.commit()

    threading.Thread(target=add_tags).start()


def set_post_colors(post: Post, file: None | BufferedReader = None):
    if post.colors:
        return
    colors = get_palette_ints(post.absolute_path.as_posix()) if file is None else get_palette_ints(file)
    post.colors.extend(PostHasColor(post_id=post.id, order=i, color=color) for i, color in enumerate(colors))
