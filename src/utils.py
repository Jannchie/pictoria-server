import argparse
import hashlib
import os
import sqlite3
import sys
import time
from collections.abc import Callable
from functools import cache, wraps
from pathlib import Path
from typing import Any, TypeVar

import sqlite_vec
import wdtagger
from fastapi import FastAPI
from fastapi.routing import APIRoute
from PIL import Image
from sqlalchemy import create_engine, insert, select
from sqlalchemy.orm import Session, sessionmaker

import shared
from alembic import command
from alembic.config import Config
from db import get_vec_db
from models import Post, PostHasTag, Tag, TagGroup
from shared import logger

# 定义泛型变量，用于注释被装饰的可调用对象的返回类型
R = TypeVar("R")


def timer(func: Callable[..., R]) -> Callable[..., R]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:  # noqa: ANN401
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in: {execution_time:.4f} seconds")
        return result

    return wrapper


def initialize(target_dir: os.PathLike, openai_key: str | None = None) -> None:
    prepare_paths(Path(target_dir))
    prepare_openai_api(openai_key)
    init_thumbnails_directory()


def prepare_openai_api(openai_key: str | None) -> None:
    if not shared.pictoria_dir:
        logger.warning("Pictoria directory not set, skipping OpenAI API key setup")
        return
    if shared.pictoria_dir.joinpath("OPENAI_API_KEY").exists():
        with shared.pictoria_dir.joinpath("OPENAI_API_KEY").open() as f:
            shared.openai_key = f.read().strip()
    if openai_key:
        shared.openai_key = openai_key


def prepare_paths(target_path: Path) -> None:
    shared.target_dir = get_target_dir(target_path)
    shared.pictoria_dir = get_pictoria_directory()
    shared.db_path = get_db_path()
    shared.vec_path = get_vec_path()


def get_vec_path() -> Path:
    vec_path = shared.pictoria_dir / "vec.db"
    logger.info(f"Vector Database path: {vec_path}")
    return vec_path


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """
    Simplify operation IDs so that generated API clients have simpler function
    names.

    Should be called only after all routes have been added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


def migrate_db(db_path: Path) -> None:
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    try:
        logger.info("Migrating database...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migration successful")
    except Exception as e:
        logger.error(f"Error while migrating database: {e}")
        sys.exit(1)
    logger.info("Database migration successful")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4777)
    parser.add_argument("--target_dir", type=str, default=".")
    parser.add_argument("--openai_key", type=str, default=None)
    return parser.parse_args()


def get_pictoria_directory():
    pictoria_dir = shared.target_dir / ".pictoria"
    if not pictoria_dir.exists():
        pictoria_dir.mkdir()
        logger.info(f'Created directory "{pictoria_dir}"')
    return pictoria_dir


def validate_path(target_path: Path):
    if not target_path.exists():
        logger.info(f'Error: Path "{target_path}" does not exist')
        sys.exit(1)


def get_target_dir(target_path: Path) -> Path:
    target_dir = target_path.resolve()
    validate_path(target_dir)
    logger.info(f"Target directory: {target_dir}")
    return target_dir


def get_db_path():
    db_path = shared.pictoria_dir / "metadata.db"
    logger.info(f"Database path: {db_path}")
    logger.info(f"Using database file: {db_path}")
    return db_path


def execute_database_migration():
    # task = multiprocessing.Process(target=migrate_db, args=(shared.db_path,))
    migrate_db(shared.db_path)
    # task.start()
    # task.join()


def init_thumbnails_directory():
    shared.thumbnails_dir = shared.pictoria_dir / "thumbnails"
    logger.info(f"Thumbnails directory: {shared.thumbnails_dir}")
    if not shared.thumbnails_dir.exists():
        shared.thumbnails_dir.mkdir()
        logger.info(f'Created directory "{shared.thumbnails_dir}"')


def remove_deleted_files(
    session: Session,
    *,
    os_tuples_set: set[tuple[str, str, str]],
    db_tuples_set: set[tuple[str, str, str]],
):
    if deleted_files := db_tuples_set - os_tuples_set:
        logger.info(f"Detected {len(deleted_files)} files have been deleted")
        for file_path in deleted_files:
            delete_by_file_path_and_ext(session, file_path)
        session.commit()
        logger.info("Deleted files from database")


def delete_by_file_path_and_ext(session: Session, path_name_and_ext: tuple[str, str, str]):
    session.query(Post).filter(
        Post.file_path == path_name_and_ext[0],
        Post.file_name == path_name_and_ext[1],
        Post.extension == path_name_and_ext[2],
    ).delete()
    if path_name_and_ext[2]:
        relative_path = (Path(path_name_and_ext[0]) / path_name_and_ext[1]).with_suffix(f".{path_name_and_ext[2]}")
    else:
        relative_path = Path(path_name_and_ext[0]) / path_name_and_ext[1]
    file_path = shared.target_dir / relative_path
    thumbnails_path = shared.thumbnails_dir / relative_path
    if thumbnails_path.exists():
        thumbnails_path.unlink()
    if file_path.exists():
        file_path.unlink()
    # delete vector data
    db = get_vec_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM post_vecs WHERE post_id = ?", (path_name_and_ext[0],))
    db.commit()
    cursor.close()


def add_new_files(
    session: Session,
    *,
    os_tuples_set: set[tuple[str, str, str]],
    db_tuples_set: set[tuple[str, str, str]],
):
    if new_file_tuples := os_tuples_set - db_tuples_set:
        logger.info(f"Detected {len(new_file_tuples)} new files")
        for file_tuple in new_file_tuples:
            image = Post(file_path=file_tuple[0], file_name=file_tuple[1], extension=file_tuple[2])
            session.add(image)
        session.commit()
        logger.info("Added new files to database")


def load_extension(dbapi_connection: sqlite3.Connection, *args) -> None:  # noqa: ANN002, ARG001
    # 只有当使用 SQLite 时才可以加载扩展
    if isinstance(dbapi_connection, sqlite3.Connection):
        logger.info("Loading SQLite extensions")
        dbapi_connection.enable_load_extension(True)  # noqa: FBT003
        sqlite_vec.load(dbapi_connection)
        dbapi_connection.enable_load_extension(False)  # noqa: FBT003
        logger.info("SQLite extensions loaded")


@cache
def get_engine():
    return create_engine(
        f"sqlite:///{shared.db_path}",
        echo=False,
        pool_size=100,
        max_overflow=200,
        connect_args={"timeout": 10},
    )


def get_session():
    engine = get_engine()
    my_session = sessionmaker(bind=engine, expire_on_commit=False, autoflush=True)
    return my_session()


def get_relative_path(file_path: Path, target_dir: Path) -> str:
    return file_path.relative_to(target_dir).parent.as_posix()


def get_file_name(file_path: Path) -> str:
    return file_path.stem


def get_file_extension(file_path: Path) -> str:
    return file_path.suffix[1:]


def find_files_in_directory(target_dir: Path) -> list[tuple[str, str, str]]:
    os_tuples: list[tuple[str, str, str]] = []
    for file_path in target_dir.rglob("*"):
        relative_path = file_path.relative_to(target_dir)
        if file_path.is_file() and not relative_path.parts[0].startswith("."):
            path = get_relative_path(file_path, target_dir)
            name = get_file_name(file_path)
            ext = get_file_extension(file_path)
            os_tuples.append((path, name, ext))
    logger.info(f"Found {len(os_tuples)} files in target directory")
    return os_tuples


def calculate_md5(file: bytes) -> str:
    # 读取文件的内容并计算 md5 值。
    md5 = hashlib.sha256()
    md5.update(file)
    return md5.hexdigest()


def create_thumbnail(input_image_path: Path, output_image_path: Path, max_width: int = 400):
    with Image.open(input_image_path) as img:
        create_thumbnail_by_image(img, output_image_path, max_width)


def create_thumbnail_by_image(img: Image.Image, output_image_path: Path, max_width: int = 400):
    width, height = img.size
    if width > max_width:
        new_width = max_width
        new_height = int((new_width / width) * height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img.save(output_image_path)


def get_path_name_and_extension(file_path: Path) -> tuple[str, str, str]:
    # 如果是绝对路径，则将其转换为相对路径，相对于target_dir
    basic_path = file_path.relative_to(shared.target_dir) if file_path.is_absolute() else file_path

    path = basic_path.parent.as_posix()
    name = basic_path.stem  # 不包含扩展名的文件名
    ext = file_path.suffix[1:]  # 扩展名（不含点）

    return path, name, ext


def update_file_metadata(file_data: bytes, post: Post, file_abs_path: Path):
    post.md5 = calculate_md5(file_data)
    post.size = file_abs_path.stat().st_size

    # 从 file_path 获取所有上级目录，存到列表里
    # folder = str(Path(post.file_path).parents[0]).replace("\\", "/")

    # # 查询 Folder 表，如果不存在则创建
    # folder_record = session.query(Folder).filter(Folder.path == folder).first()
    # if folder_record is None:
    #     folder_record = Folder(path=folder, file_count=0)
    #     session.add(folder_record)
    # folder_record.file_count += 1


def compute_image_properties(img: Image.Image, post: Post, file_abs_path: Path):
    img.verify()
    post.width, post.height = img.size
    relative_path = file_abs_path.relative_to(shared.target_dir)
    thumbnails_path = shared.thumbnails_dir / relative_path
    if not thumbnails_path.exists():
        thumbnails_path.parent.mkdir(parents=True, exist_ok=True)
        create_thumbnail(
            file_abs_path,
            thumbnails_path,
        )


def remove_post(
    session: Session,
    file_abs_path: Path | None = None,
    post: Post | None = None,
    *,
    auto_commit: bool = True,
):
    if post is None:
        file_path, file_name, extension = get_path_name_and_extension(file_abs_path)
        post = (
            session.query(Post)
            .filter(Post.file_path == file_path, Post.file_name == file_name, Post.extension == extension)
            .first()
        )
    else:
        file_abs_path = shared.target_dir / post.file_path
        file_abs_path = file_abs_path.with_suffix(f".{post.extension}")
    if post is None:
        logger.info(f"Post not found in database: {file_abs_path}")
        return
    logger.info(f"Removing post: {post.file_path}.{post.extension}")
    if not file_abs_path:
        return
    relative_path = file_abs_path.relative_to(shared.target_dir)
    thumbnails_path = shared.thumbnails_dir / relative_path
    if thumbnails_path.exists():
        thumbnails_path.unlink()
        logger.info(f"Removed thumbnail: {thumbnails_path}")
    session.delete(post)
    if auto_commit:
        session.commit()
    logger.info(f"Removed post from database: {file_abs_path}")


def remove_post_in_path(session: Session, file_path: Path):
    # 如果 file_path 是绝对路径，则转换为相对路径
    if file_path.is_absolute():
        file_path = file_path.relative_to(shared.target_dir)
    file_path = str(file_path).replace("\\", "/")
    # 删除数据库中前缀和 file_path 相同的所有文件
    posts = session.query(Post).filter(Post.file_path.like(f"{file_path}%")).all()
    for post in posts:
        remove_post(session, post=post, auto_commit=False)
    session.commit()


def from_rating_to_int(rating: str) -> int:
    # sourcery skip: assign-if-exp, reintroduce-else
    """
    0: Not Rated
    1: general
    2. sensitive
    3: questionable
    4: explicit
    """
    if rating == "general":
        return 1
    if rating == "sensitive":
        return 2
    if rating == "questionable":
        return 3
    if rating == "explicit":
        return 4
    return 0


def attach_tags_to_post(session: Session, post: Post, resp: wdtagger.Result, *, is_auto: bool = False):
    # 统一查看是否有名为 general 或者 character 的 TagGroup，如果没有则创建
    group_names = ["general", "character"]
    colors = {
        "general": "#006192",
        "character": "#8243ca",
    }
    for tag_group_name in group_names:
        tag_group = session.scalar(select(TagGroup).where(TagGroup.name == tag_group_name))
        if tag_group is None:
            tag_group = TagGroup(name=tag_group_name, color=colors[tag_group_name])
            session.add(tag_group)
    # 遍历标签并进行处理
    for i, tag_names in enumerate([resp.general_tags, resp.character_tags]):
        name = group_names[i]
        tag_group = session.execute(select(TagGroup).where(TagGroup.name == name)).scalar_one()

        existing_tags = session.scalars(select(Tag).where(Tag.name.in_(tag_names))).all()
        existing_tag_names = {tag.name for tag in existing_tags}
        for existing_tag in existing_tags:
            if not existing_tag.group_id:
                existing_tag.group_id = tag_group.id
                session.add(existing_tag)
        if new_tags := set(tag_names) - existing_tag_names:
            session.execute(
                insert(Tag).values(
                    [
                        {
                            "name": tag_name,
                            "group_id": tag_group.id,
                        }
                        for tag_name in new_tags
                    ],
                ),
            )

        post_existing_tags = session.scalars(
            select(PostHasTag).where(PostHasTag.tag_name.in_(tag_names) & (PostHasTag.post_id == post.id)),
        ).all()
        post_existing_tag_names = {tag_record.tag_name for tag_record in post_existing_tags}
        if post_new_tags := set(tag_names) - post_existing_tag_names:
            session.execute(
                insert(PostHasTag).values(
                    [
                        {
                            "post_id": post.id,
                            "tag_name": tag_name,
                            "is_auto": is_auto,
                        }
                        for tag_name in post_new_tags
                    ],
                ),
            )

    session.add(post)
    session.commit()
