import io
import os
import pathlib
import shutil
import tomllib
from datetime import UTC, datetime
from typing import Annotated

import fastapi
import httpx
import pillow_avif  # noqa: F401
import uvicorn
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Path, UploadFile
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from rich import get_console
from rich.progress import track
from sqlalchemy import Select, func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session, joinedload
from starlette.convertors import Convertor, register_url_convertor
from starlette.middleware.gzip import GZipMiddleware

import shared
from ai import OpenAIImageAnnotator
from danbooru import DanbooruClient
from db import find_similar_posts, get_img_vec, init_vec_db
from models import Post, PostHasColor, PostHasTag, Tag, TagGroup
from processors import process_post, process_posts, set_post_colors, sync_metadata
from scheme import PostPublic, PostWithTagPublic, TagGroupWithTagsPublic, TagWithGroupPublic
from utils import (
    attach_tags_to_post,
    create_thumbnail,
    delete_by_file_path_and_ext,
    from_rating_to_int,
    get_path_name_and_extension,
    get_session,
    get_tagger,
    initialize,
    logger,
    parse_arguments,
    use_route_names_as_operation_ids,
)

# from watch import watch_target_dir

with pathlib.Path("pyproject.toml").open("rb") as f:
    pyproject = tomllib.load(f)


@asynccontextmanager
async def my_lifespan(_: FastAPI):
    load_dotenv()
    initialize(target_dir="demo")
    sync_metadata()
    init_vec_db()
    # watch_target_dir()
    host = "localhost"
    port = 4777
    doc_url = f"http://{host}:{port}/docs"
    shared.logger.info(f"API Document: {doc_url}")
    yield


def analyze_palettes():
    with get_session() as session:
        posts = session.scalars(
            select(Post).outerjoin(PostHasColor).where(Post.width > 0, PostHasColor.color.is_(None)),
        ).all()
        for post in track(posts, description="Analyzing palettes...", console=console):
            try:
                set_post_colors(post)
                session.add(post)
                session.commit()
            except Exception as e:
                logger.exception(e)
                session.rollback()


console = get_console()
app = fastapi.FastAPI(
    default_response_class=ORJSONResponse,
    title="Pictoria",
    version=pyproject["project"]["version"],
    lifespan=my_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的请求来源，可以设为 ["*"] 以允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许的 HTTP 方法，可以指定特定方法如 ["GET", "POST"]
    allow_headers=["*"],  # 允许的 HTTP 头，可以指定特定头如 ["Authorization", "Content-Type"]
)

# 启用 GZip 中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)  # minimum_size 表示最小压缩大小，单位为字节


class PathConvertor(Convertor):
    regex = r".+?"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("pathlike", PathConvertor())


def get_post_by_id(post_id: int, session: Session):
    post = session.query(Post).options(joinedload(Post.tags).joinedload(PostHasTag.tag_info).joinedload(Tag.group)).filter_by(id=post_id).one_or_none()
    if not post:
        return None
    def get_group_sort_key(tag: PostHasTag) -> int:  # noqa: PLR0911
        if tag.tag_info.group is None:
            return 1
        group_name = tag.tag_info.group.name
        if group_name == "artist":
            return 0
        if group_name == "copyright":
            return 1
        if group_name == "character":
            return 2
        if group_name == "general":
            return 3
        if group_name == "meta":
            return 4
        return 1

    post.tags = sorted(post.tags, key=get_group_sort_key)

    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return post


class PostFilter(BaseModel):
    rating: list[int] | None = []
    score: list[int] | None = []
    tags: list[str] | None = []
    extension: list[str] | None = []
    folder: str | None = None


@app.post(
    "/v1/posts",
    response_model=list[PostPublic],
    tags=["Post"],
)
def v1_list_posts(
    *,
    limit: int | None = None,
    offset: int = 0,
    filter: PostFilter = PostFilter(),  # noqa: A002
    ascending: bool = False,
):
    session = get_session()
    stmt = apply_filtered_query(filter, select(Post)).order_by(Post.id.asc() if ascending else Post.id.desc()).limit(limit).offset(offset)
    return session.scalars(stmt)


@app.delete("/v1/posts/{post_id}", tags=["Post"])
def v1_delete_post(post_id: int):
    session = get_session()
    post = session.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    delete_by_file_path_and_ext(session=session, path_name_and_ext=[post.file_path, post.file_name, post.extension])
    session.commit()


def apply_filtered_query(filter: PostFilter, stmt: Select) -> Select:  # noqa: A002
    if filter.rating:
        stmt = stmt.filter(Post.rating.in_(filter.rating))
    if filter.score:
        stmt = stmt.filter(Post.score.in_(filter.score))
    if filter.tags:
        stmt = stmt.join(Post.tags).filter(PostHasTag.tag_name.in_(filter.tags))
    if filter.extension:
        stmt = stmt.filter(Post.extension.in_(filter.extension))
    if filter.folder and filter.folder != "":
        stmt = stmt.filter(Post.file_path == filter.folder)
    return stmt


class PostCountResponse(BaseModel):
    count: int


@app.get("/v1/posts/count", response_model=PostCountResponse, tags=["Post"])
def v1_total_posts_count():
    session = get_session()
    count = session.query(Post).count()
    return {"count": count}


class RatingCountResponse(BaseModel):
    rating: int
    count: int


@app.post("/v1/posts/count/rating", response_model=list[RatingCountResponse], tags=["Post"])
def v1_count_group_by_rating(
    filter: PostFilter = Body(...),  # noqa: A002
    session: Session = Depends(get_session),
):
    stmt = select(Post.rating, func.count()).group_by(Post.rating)
    stmt = apply_filtered_query(filter, stmt)
    resp = session.execute(stmt).all()
    return [RatingCountResponse(rating=row[0], count=row[1]) for row in resp]


class ScoreCountResponse(BaseModel):
    score: int
    count: int


@app.post("/v1/posts/count/score", response_model=list[ScoreCountResponse], tags=["Post"])
def v1_count_group_by_score(
    session: Annotated[Session, Depends(get_session)],
    filter: Annotated[PostFilter, Body(...)],  # noqa: A002
):
    query = apply_filtered_query(filter, select(Post.score, func.count()).group_by(Post.score))
    resp = session.execute(query).all()
    return [ScoreCountResponse(score=row[0] if row[0] is not None else 0, count=row[1]) for row in resp]


class ExtensionCountResponse(BaseModel):
    extension: str
    count: int


@app.post("/v1/posts/count/extension", response_model=list[ExtensionCountResponse], tags=["Post"])
def v1_count_group_by_extension(
    session: Annotated[Session, Depends(get_session)],
    filter: Annotated[PostFilter, Body(...)],  # noqa: A002
):
    query = select(Post.extension, func.count()).group_by(Post.extension)
    query = apply_filtered_query(filter, query)
    resp = session.execute(query).all()
    return [ExtensionCountResponse(extension=row[0], count=row[1]) for row in resp]


class ScoreUpdate(BaseModel):
    score: Annotated[int, Field(ge=0, le=5)]


@app.put("/v1/posts/{post_id}/score", response_model=PostPublic, tags=["Post"])
def v1_update_post_score(post_id: Annotated[int, Path(gt=0)], score_update: ScoreUpdate):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.score = score_update.score
    session.commit()
    session.refresh(post)
    return post


class RatingUpdate(BaseModel):
    rating: Annotated[int, Field(ge=0, le=5)]


@app.put("/v1/posts/{post_id}/rating", response_model=PostPublic, tags=["Post"])
def v1_update_post_rating(
    post_id: Annotated[int, Path(gt=0)],
    rating_update: RatingUpdate,
    session: Annotated[Session, Depends(get_session)],
):
    post = session.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.rating = rating_update.rating
    session.commit()
    session.refresh(post)
    return post


@app.put("/v1/posts/{post_id}/source", response_model=PostPublic, tags=["Post"])
def v1_update_post_source(
    post_id: Annotated[int, Path(gt=0)],
    source: str,
    session: Annotated[Session, Depends(get_session)],
):
    post = session.get(Post, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.source = source
    session.commit()
    session.refresh(post)
    return post


@app.put("/v1/posts/{post_id}/caption", response_model=PostPublic, tags=["Post"])
def v1_update_post_caption(post_id: Annotated[int, Path(gt=0)], caption: str):
    session = get_session()
    post = session.get(Post, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.caption = caption
    session.commit()
    session.refresh(post)
    return post


@app.get("/v1/posts/{post_id}", response_model=PostWithTagPublic, tags=["Post"])
def v1_get_post(post_id: int, session: Annotated[Session, Depends(get_session)]):
    session.begin()
    post = get_post_by_id(post_id, session)
    session.commit()
    return post


@app.get("/v1/posts/{post_id}/similar", response_model=list[PostPublic], tags=["Post"])
def v1_get_similar_posts(post_id: int, session: Annotated[Session, Depends(get_session)]):
    session.begin()
    post = get_post_by_id(post_id, session)
    vec = get_img_vec(post)
    resp = find_similar_posts(vec)
    return [get_post_by_id(row.post_id, session) for row in resp]


@app.post("/v1/cmd/posts/features", tags=["Command"])
def v1_cmd_calculate_features():
    session = get_session()
    posts = session.query(Post).all()
    for post in track(posts, description="Calculating features...", console=console):
        get_img_vec(post)
    return {"status": "ok"}


@app.get("/v1/images/{post_path:pathlike}", tags=["Image"])
async def v1_get_post_by_path(post_path: str) -> fastapi.responses.FileResponse:
    if not shared.target_dir:
        raise HTTPException(status_code=400, detail="Target directory is not set")
    abs_path = shared.target_dir / post_path
    return fastapi.responses.FileResponse(abs_path)


@app.get("/v1/thumbnails/{post_path:pathlike}", tags=["Image"])
async def v1_get_thumbnail(post_path: str) -> fastapi.responses.FileResponse:
    thumbnail_file_path = shared.thumbnails_dir / post_path
    original_file_path = shared.target_dir / post_path
    if not thumbnail_file_path.exists():
        create_thumbnail(original_file_path, thumbnail_file_path)
    return fastapi.responses.FileResponse(thumbnail_file_path)


@app.put("/v1/cmd/posts/{post_id}/rotate", response_model=PostPublic, tags=["Command"])
def v1_cmd_rotate_image(post_id: int, *, clockwise: bool = True, session: Session = Depends(get_session)):
    post = session.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    post.rotate(session, clockwise=clockwise)
    return post


class TagWithCountPublic(BaseModel):
    count: int
    tag_info: TagWithGroupPublic


@app.get("/v1/tags", response_model=list[TagWithCountPublic], tags=["Tag"])
def v1_get_tags(language: str = "en"):
    session = get_session()
    stmt = select(Tag).options(joinedload(Tag.group))

    count_stmt = select(PostHasTag.tag_name, func.count()).group_by(PostHasTag.tag_name)
    resp = session.scalars(stmt).all()
    count_resp = session.execute(count_stmt).all()
    count_dict = {row[0]: row[1] for row in count_resp}

    result = [TagWithCountPublic(tag_info=row, count=count_dict.get(row.name, 0)) for row in resp]
    for row in result:
        row.tag_info.name = shared.i18n.t(language, row.tag_info.name)
    return result


@app.post("/v1/tag/{tag_name}", response_model=Tag, tags=["Tag"])
def v1_create_tag(tag_name: str):
    session = get_session()
    tag = Tag(name=tag_name)
    session.add(tag)
    session.commit()
    return tag


@app.delete("/v1/tag/{tag_name}", response_model=Tag, tags=["Tag"])
def v1_delete_tag(tag_name: str):
    session = get_session()
    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    session.delete(tag)
    session.commit()
    return tag


@app.get("/v1/tags/{tag_name}", response_model=Tag, tags=["Tag"])
def v1_get_tag(tag_name: str):
    session = get_session()
    return session.query(Tag).filter(Tag.name == tag_name).first()


@app.put("/v1/tags/{tag_name}", response_model=Tag, tags=["Tag"])
def v1_update_tag(tag_name: str, new_tag_name: str):
    session = get_session()
    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    tag.name = new_tag_name
    session.commit()
    return tag


@app.post("/v1/posts/{post_id}/tags/{tag_name}", response_model=PostWithTagPublic, tags=["Tag"])
def v1_add_tag_to_post(post_id: int, tag_name: str):
    session = get_session()
    post = session.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    tag = session.get(Tag, tag_name)
    # 如果 tag 不存在，创建一个新的 tag
    if tag is None:
        tag = Tag(name=tag_name)
        session.add(tag)

    # 如果已经存在，直接返回
    # post_has_tag = (
    #     session.query(PostHasTag).filter(PostHasTag.post_id == post_id, PostHasTag.tag_name == tag_name).first()
    # )
    post_has_tag = session.get(PostHasTag, (post_id, tag_name))
    if post_has_tag is None:
        post_has_tag = PostHasTag(post=post, tag_info=tag, is_auto=False)
        session.add(post_has_tag)
        session.commit()
    return get_post_by_id(post_id, session)


@app.delete("/v1/posts/{post_id}/tags/{tag_name}", response_model=PostWithTagPublic, tags=["Tag"])
def v1_remove_tag_from_post(post_id: int, tag_name: str):
    session = get_session()
    post = session.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    # tag_record = session.query(PostHasTag).filter(PostHasTag.tag_name == tag_name).first()
    tag_record = session.get(PostHasTag, (post_id, tag_name))
    if tag_record is not None:
        session.delete(tag_record)
        session.commit()
    return get_post_by_id(post_id, session)


@app.get("/v1/tag-groups", response_model=list[TagGroupWithTagsPublic], tags=["Tag"])
def v1_get_tag_groups(session: Annotated[Session, Depends(get_session)]):
    return session.scalars(select(TagGroup))


@app.put("/v1/posts/move", response_model=PostWithTagPublic, tags=["Post"])
def v1_move_posts(post_ids: list[int], new_path: str, session: Annotated[Session, Depends(get_session)]):
    for post_id in post_ids:
        post = session.get(Post, post_id)
        if post is None:
            raise HTTPException(status_code=404, detail="Post not found")
        post.move(session, new_path)


@app.post("/v1/cmd/process-posts", tags=["Command"])
def v1_cmd_process_posts():
    process_posts(all_posts=True)
    return {"status": "ok"}


@app.get("/v1/cmd/auto-tags/{post_id}", response_model=PostWithTagPublic, tags=["Command"])
def v1_cmd_auto_tags(post_id: int, session: Annotated[Session, Depends(get_session)]):
    post = session.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    abs_path = post.absolute_path
    tagger = get_tagger()
    resp = tagger.tag(abs_path)
    logger.info(resp)
    post.rating = from_rating_to_int(resp.rating)
    attach_tags_to_post(session, post, resp, is_auto=True)
    session.commit()

    return get_post_by_id(post_id, session)


@app.get("/v1/cmd/auto-tags", tags=["Command"])
def v1_cmd_auto_tags_all(session: Annotated[Session, Depends(get_session)]):
    posts = session.query(Post).all()
    tagger = get_tagger()

    # 使用 rich 进度条
    for post in track(posts, description="Processing posts...", console=console):
        try:
            abs_path = post.absolute_path
            resp = tagger.tag(abs_path)
            post.rating = from_rating_to_int(resp.rating)
            attach_tags_to_post(session, post, resp, is_auto=True)
        except Exception as e:
            shared.logger.error(f"Error processing post {post.id}: {e}")
            continue
    session.commit()
    return {"status": "ok"}


@app.get("/v1/cmd/auto-caption/{post_id}", response_model=PostPublic, tags=["Command"])
def v1_cmd_auto_caption(post_id: int):
    session = get_session()
    post: Post = session.query(Post).filter(Post.id == post_id).first()
    if shared.openai_key is None:
        raise HTTPException(status_code=400, detail="OpenAI API key is not set")

    if shared.caption_annotator is None:
        shared.caption_annotator = OpenAIImageAnnotator(shared.openai_key)

    post.caption = shared.caption_annotator.annotate_image(post.absolute_path)
    session.add(post)
    session.commit()
    session.refresh(post)
    return post


class CountResponse(BaseModel):
    count: int


@app.get("/v1/posts/count", response_model=CountResponse, tags=["Post"])
def v1_get_posts_count():
    session = get_session()
    count = session.query(Post).count()
    return {"count": count}


@app.get("/v1/tags/count", response_model=CountResponse, tags=["Tag"])
def v1_get_tags_count():
    session = get_session()
    count = session.query(Tag).count()
    return {"count": count}


class DirectorySummary(BaseModel):
    name: str
    path: str
    file_count: int
    children: list["DirectorySummary"] = Field(default_factory=list)


DirectorySummary.model_rebuild()


def get_directory_summary(path_data: str | pathlib.Path) -> DirectorySummary:
    full_path = pathlib.Path(path_data)
    relative_path = full_path.relative_to(shared.target_dir)
    summary = DirectorySummary(
        name=relative_path.name,
        path=relative_path.as_posix(),
        file_count=0,
        children=[],
    )

    ignore_dirs = shared.pictoria_dir
    with os.scandir(shared.target_dir / path_data) as entries:
        for entry in entries:
            if entry.name == ignore_dirs.name:
                continue
            if entry.is_dir():
                subtree = get_directory_summary(entry.path)
                summary.children.append(subtree)
                summary.file_count += subtree.file_count
            else:
                summary.file_count += 1

    return summary


@app.get("/v1/folders", response_model=DirectorySummary, tags=["Folder"])
def v1_get_folders():
    target_path = shared.target_dir
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    return get_directory_summary(target_path)


@app.post("/v1/upload", tags=["Upload"])
def v1_upload_file(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    path: str | None = Form(None),
    source: str = Form(None),
):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="Either file or url must be provided")
    if file is None:
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }
        if url and "pximg.net" in url:
            headers["referer"] = "https://www.pixiv.net/"
        with httpx.Client() as client:
            resp = client.get(url, headers=headers)

        file_io = io.BytesIO(resp.content)
    else:
        file_io = file.file

    if path is None and file is not None and file.filename:
        path = file.filename
    elif path and file is not None and file.filename:
        path = f"{path}/{file.filename}"
    else:
        path = path or (url.split("/")[-1] if url else "")
    abs_path = shared.target_dir / path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    file_path, file_name, file_ext = get_path_name_and_extension(abs_path)
    if abs_path.exists():
        raise HTTPException(status_code=400, detail="File already exists")
    logger.info(f"Saving file to: {abs_path}")
    post = Post(file_path=file_path, file_name=file_name, extension=file_ext, source=source)
    session = get_session()
    session.add(post)
    session.commit()
    session.refresh(post)
    with abs_path.open("wb") as f:
        shutil.copyfileobj(file_io, f)
    process_post(abs_path, post)
    return ORJSONResponse(content={"filename": path})


@app.post("/v1/cmd/update-openai-key", tags=["Command"])
def v1_update_openai_key(key: str):
    shared.openai_key = key
    shared.pictoria_dir.joinpath("OPENAI_API_KEY").write_text(key)
    return {"status": "ok"}


@app.post("/v1/cmd/apply-danbooru-tags", tags=["Command"])
def v1_cmd_apply_danbooru_tags(session: Session = Depends(get_session)):
    import json

    json_path = pathlib.Path("data/tag_group_gt_100.json")
    json_data = json.loads(json_path.read_text())
    groups = ["artist", "meta", "character", "general", "copyright"]
    for group in groups:
        if not session.execute(select(TagGroup).filter(TagGroup.name == group)).scalar():
            tag_group = TagGroup(name=group)
            session.add(tag_group)
    session.commit()

    groups = {group: session.execute(select(TagGroup).filter(TagGroup.name == group)).scalar() for group in groups}
    values = []
    for group in groups:
        for tag in json_data[f"tag_string_{group}"]:
            values.extend(
                [{"name": tag.replace("_", " ").replace("(", R"\(").replace(")", R"\)"), "group_id": groups[group].id}],
            )

    chunk_size = 1000
    for i in range(0, len(values), chunk_size):
        chunk = values[i : i + chunk_size]
        insert_stmt = insert(Tag).values(chunk)
        update_columns = {col.name: col for col in insert_stmt.excluded}
        session.execute(insert_stmt.on_conflict_do_update(index_elements=["name"], set_=update_columns))
        session.commit()


@app.get("/v1/cmd/download-from-danbooru", tags=["Command"])
def v1_cmd_download_from_danbooru(*, tags: str, session: Session = Depends(get_session)):
    client = DanbooruClient(os.getenv("DANBOORU_API_KEY"), os.getenv("DANBOORU_USER_NAME"))
    danbooru_dir = shared.target_dir / "danbooru"
    save_dir = danbooru_dir / tags
    posts_orig = client.get_posts(tags=tags, limit=99999)
    posts = [post for post in posts_orig if post.file_url]
    logger.info(f"Fetched {len(posts)} avaliable posts ({len(posts_orig)} total)")

    general_group_id = session.execute(select(TagGroup).filter(TagGroup.name == "general")).scalar().id
    character_group_id = session.execute(select(TagGroup).filter(TagGroup.name == "character")).scalar().id
    artist_group_id = session.execute(select(TagGroup).filter(TagGroup.name == "artist")).scalar().id
    meta_group_id = session.execute(select(TagGroup).filter(TagGroup.name == "meta")).scalar().id

    types = ["general", "character", "artist", "meta"]
    type_to_group_id = {
        "general": general_group_id,
        "character": character_group_id,
        "artist": artist_group_id,
        "meta": meta_group_id,
    }

    for post in posts:
        if not post.file_url:
            continue
        now = datetime.now(UTC)
        resp = session.execute(
            insert(Post)
            .values(
                {
                    "file_path": save_dir.relative_to(shared.target_dir).as_posix(),
                    "file_name": str(post.id),
                    "extension": post.file_ext,
                    "source": f"https://danbooru.donmai.us/posts/{post.id}",
                    "rating": from_rating_to_int(post.rating),
                    "updated_at": now,
                    "created_at": now,
                    "published_at": post.created_at,
                },
            )
            .on_conflict_do_update(
                index_elements=["file_path", "file_name", "extension"],
                set_={
                    "rating": from_rating_to_int(post.rating),
                    "updated_at": now,
                    "published_at": post.created_at,
                    "source": f"https://danbooru.donmai.us/posts/{post.id}",
                },
            )
            .returning(Post.id),
        )
        post_id = resp.scalar()
        if post_id is None:
            continue
        for t in types:
            for tag_str in getattr(post, f"tag_string_{t}").split(" "):
                tag_name = tag_str.replace("_", " ")
                session.execute(
                    insert(Tag)
                    .values(
                        {
                            "name": tag_name,
                            "group_id": type_to_group_id[t],
                        },
                    )
                    .on_conflict_do_nothing(),
                )
                session.execute(
                    insert(PostHasTag)
                    .values(
                        {
                            "post_id": post_id,
                            "tag_name": tag_name,
                            "is_auto": False,
                        },
                    )
                    .on_conflict_do_nothing(),
                )
    session.commit()
    client.download_posts(posts, save_dir)
    process_posts()


@app.get("/")
def root():
    return {
        "message": "Pictoria Server",
        "version": pyproject["project"]["version"],
        "description": pyproject["project"]["description"],
        "author": pyproject["project"]["authors"],
    }


use_route_names_as_operation_ids(app)
if __name__ == "__main__":
    args = parse_arguments()
    initialize(target_dir=args.target_dir, openai_key=args.openai_key)
    host = args.host or "localhost"
    uvicorn.run(
        "main:app",
        host=host,
        port=args.port,
        reload=True,
        log_config=None,
    )
