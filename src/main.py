import io
import os
import pathlib
import shutil
import tomllib
from typing import Annotated, Optional

import fastapi
import httpx
import uvicorn
from fastapi import Body, File, Form, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from rich import get_console
from rich.progress import track
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from starlette.convertors import Convertor, register_url_convertor
from starlette.middleware.gzip import GZipMiddleware
from wdtagger import Tagger

import shared
from ai import OpenAIImageAnnotator
from models import Post, PostBase, PostHasTag, PostWithTag, Tag, TagGroup, TagPublic
from utils import (
    attach_tags_to_post,
    delete_by_file_path_and_ext,
    execute_database_migration,
    from_rating_to_int,
    get_path_name_and_extension,
    get_session,
    initialize,
    parse_arguments,
    process_posts,
    sync_metadata,
    use_route_names_as_operation_ids,
    watch_target_dir,
)

pyproject = tomllib.load(open("pyproject.toml", "rb"))

console = get_console()
app = fastapi.FastAPI(default_response_class=ORJSONResponse, title="Pictoria", version=pyproject["project"]["version"])

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


def get_post_by_id(post_id, session) -> PostWithTag:
    post = (
        session.query(Post)
        .filter(Post.id == post_id)
        .options(joinedload(Post.tags).joinedload(PostHasTag.tag_info))
        .first()
    )
    for tag in post.tags:
        assert isinstance(tag, PostHasTag)
    return post


class PostFilter(BaseModel):
    rating: Optional[list[int]] = []
    score: Optional[list[int]] = []
    tags: Optional[list[str]] = []
    extension: Optional[list[str]] = []
    folder: Optional[str] = None


@app.post(
    "/v1/posts",
    response_model=list[PostBase],
)
def v1_get_posts(
    limit: int | None = None,
    offset: int = 0,
    filter: PostFilter = PostFilter(),
):
    session = get_session()
    query = apply_filtered_query(filter, session.query(Post)).limit(limit).offset(offset)
    return query.all()


@app.delete("/v1/posts/{post_id}")
def v1_delete_post(post_id: int):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    delete_by_file_path_and_ext(session=session, path_name_and_ext=[post.file_path, post.extension])
    session.commit()
    return post


def apply_filtered_query(filter: PostFilter, query: fastapi.Query):
    if filter.rating:
        query = query.filter(Post.rating.in_(filter.rating))
    if filter.score:
        query = query.filter(Post.score.in_(filter.score))
    if filter.tags:
        query = query.join(Post.tags).filter(PostHasTag.tag_name.in_(filter.tags))
    if filter.extension:
        query = query.filter(Post.extension.in_(filter.extension))
    if filter.folder and filter.folder != "":
        query = query.filter(Post.file_path == filter.folder)
    return query


class PostCountResponse(BaseModel):
    count: int


@app.get("/v1/posts/count", response_model=PostCountResponse)
def v1_total_posts_count():
    session = get_session()
    count = session.query(Post).count()
    return {"count": count}


class RatingCountResponse(BaseModel):
    rating: int
    count: int


@app.post("/v1/posts/count/rating", response_model=list[RatingCountResponse])
def v1_count_group_by_rating(
    filter: PostFilter = Body(...),
):

    session = get_session()
    query = session.query(Post.rating, func.count()).group_by(Post.rating)
    query = apply_filtered_query(filter, query)
    resp = query.all()
    return [RatingCountResponse(rating=row[0] if row[0] is not None else 0, count=row[1]) for row in resp]


class ScoreCountResponse(BaseModel):
    score: int
    count: int


@app.post("/v1/posts/count/score", response_model=list[ScoreCountResponse])
def v1_count_group_by_score(
    filter: PostFilter = Body(...),
):
    session = get_session()
    query = session.query(Post.score, func.count()).group_by(Post.score)
    query = apply_filtered_query(filter, query)
    resp = query.all()
    return [ScoreCountResponse(score=row[0] if row[0] is not None else 0, count=row[1]) for row in resp]


class ExtensionCountResponse(BaseModel):
    extension: str
    count: int


@app.post("/v1/posts/count/extension", response_model=list[ExtensionCountResponse])
def v1_count_group_by_extension(
    filter: PostFilter = Body(...),
):
    session = get_session()
    query = session.query(Post.extension, func.count()).group_by(Post.extension)
    query = apply_filtered_query(filter, query)
    resp = query.all()
    return [ExtensionCountResponse(extension=row[0], count=row[1]) for row in resp]


class ScoreUpdate(BaseModel):
    score: Annotated[int, Field(ge=0, le=5)]


@app.put("/v1/posts/{post_id}/score", response_model=Post)
def v1_update_post_score(post_id: Annotated[int, Path(gt=0)], score_update: ScoreUpdate):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.score = score_update.score
    session.commit()
    return post


class RatingUpdate(BaseModel):
    rating: Annotated[int, Field(ge=0, le=5)]


@app.put("/v1/posts/{post_id}/rating", response_model=Post)
def v1_update_post_rating(post_id: Annotated[int, Path(gt=0)], rating_update: RatingUpdate):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.rating = rating_update.rating
    session.commit()
    return post


@app.put("/v1/posts/{post_id}/source", response_model=Post)
def v1_update_post_source(post_id: Annotated[int, Path(gt=0)], source: str):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.source = source
    session.commit()
    return post


@app.put("/v1/posts/{post_id}/caption", response_model=Post)
def v1_update_post_caption(post_id: Annotated[int, Path(gt=0)], caption: str):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.caption = caption
    session.commit()
    return post


@app.get("/v1/posts/{post_id}", response_model=PostWithTag)
def v1_get_post(post_id: int):
    session = get_session()
    return get_post_by_id(post_id, session)


@app.get("/v1/images/{post_path:pathlike}")
def v1_get_post_by_path(post_path: str):
    abs_path = shared.target_dir / post_path
    return fastapi.responses.FileResponse(abs_path)


@app.get("/v1/thumbnails/{post_path:pathlike}")
def v1_get_thumbnail(post_path: str):
    abs_path = shared.thumbnails_dir / post_path
    return fastapi.responses.FileResponse(abs_path)


class TagResponse(BaseModel):
    count: int
    tag_info: TagPublic


@app.get("/v1/tags", response_model=list[TagResponse])
def v1_get_tags():
    session = get_session()
    # 从 PostHasTag 表中查询所有的 tag_name 和 count，从 Tag 表中查询 tag_name 对应的 Tag 实例
    query = (
        session.query(PostHasTag.tag_name, func.count(), Tag)
        .group_by(PostHasTag.tag_name)
        .join(Tag, PostHasTag.tag_name == Tag.name)
    )
    resp = query.all()
    return [
        TagResponse(
            count=row[1],
            tag_info=TagPublic(name=row[0], group_id=row[2].group_id),
        )
        for row in resp
    ]


@app.post("/v1/tag/{tag_name}", response_model=Tag)
def v1_create_tag(tag_name: str):
    session = get_session()
    tag = Tag(name=tag_name)
    session.add(tag)
    session.commit()
    return tag


@app.delete("/v1/tag/{tag_name}", response_model=Tag)
def v1_delete_tag(tag_name: str):
    session = get_session()
    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    session.delete(tag)
    session.commit()
    return tag


@app.get("/v1/tags/{tag_name}", response_model=Tag)
def v1_get_tag(tag_name: str):
    session = get_session()
    return session.query(Tag).filter(Tag.name == tag_name).first()


@app.put("/v1/tags/{tag_name}", response_model=Tag)
def v1_update_tag(tag_name: str, new_tag_name: str):
    session = get_session()
    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    tag.name = new_tag_name
    session.commit()
    return tag


@app.post("/v1/posts/{post_id}/tags/{tag_name}", response_model=PostWithTag)
def v1_add_tag_to_post(post_id: int, tag_name: str):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    # 如果 tag 不存在，创建一个新的 tag
    if tag is None:
        tag = Tag(name=tag_name)
        session.add(tag)

    # 如果已经存在，直接返回
    post_has_tag = (
        session.query(PostHasTag).filter(PostHasTag.post_id == post_id, PostHasTag.tag_name == tag_name).first()
    )
    if post_has_tag is None:
        postHasTag = PostHasTag(post_id=post_id, tag_name=tag_name, is_auto=False)
        session.add(postHasTag)
        session.commit()
    post = get_post_by_id(post_id, session)
    return post


@app.delete("/v1/posts/{post_id}/tags/{tag_name}", response_model=PostWithTag)
def v1_remove_tag_from_post(post_id: int, tag_name: str):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    tag_record = session.query(PostHasTag).filter(PostHasTag.tag_name == tag_name).first()
    if tag_record is not None:
        session.delete(tag_record)
        session.commit()
    post = get_post_by_id(post_id, session)
    return post


@app.get("/v1/tag-groups", response_model=list[TagGroup])
def v1_get_tag_groups():
    session = get_session()
    return session.query(TagGroup).all()


@app.post("/v1/cmd/process-posts")
def v1_cmd_process_posts():
    process_posts(True)
    return {"status": "ok"}


@app.get("/v1/cmd/auto-tags/{post_id}", response_model=Post)
def v1_cmd_auto_tags(post_id: int):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()

    abs_path = post.absolute_path
    if shared.tagger is None:
        shared.tagger = Tagger(model_repo="SmilingWolf/wd-vit-large-tagger-v3", slient=True)
    resp = shared.tagger.tag(abs_path)
    shared.logger.info(resp)
    post.rating = from_rating_to_int(resp.rating)
    attach_tags_to_post(session, post, resp, is_auto=True)
    session.commit()

    post = get_post_by_id(post_id, session)
    return post


@app.get("/v1/cmd/auto-tags")
def v1_cmd_auto_tags_all():
    session = get_session()
    posts = session.query(Post).all()
    # posts = session.query(Post).filter(~Post.tags.any()).all()
    if shared.tagger is None:
        shared.tagger = Tagger(model_repo="SmilingWolf/wd-vit-large-tagger-v3", slient=True)

    # 使用 rich 进度条
    for post in track(posts, description="Processing posts...", console=console):
        try:
            abs_path = post.absolute_path
            resp = shared.tagger.tag(abs_path)
            post.rating = from_rating_to_int(resp.rating)
            attach_tags_to_post(session, post, resp, is_auto=True)
        except Exception as e:
            shared.logger.error(f"Error processing post {post.id}: {e}")
            continue
    session.commit()
    return {"status": "ok"}


@app.get("/v1/cmd/auto-caption/{post_id}", response_model=Post)
def v1_cmd_auto_caption(post_id: int):
    session = get_session()
    post: Post = session.query(Post).filter(Post.id == post_id).first()
    if shared.openai_key is None:
        raise HTTPException(status_code=400, detail="OpenAI API key is not set")

    if shared.caption_annotator is None:
        shared.caption_annotator = OpenAIImageAnnotator(shared.openai_key)

    resp = shared.caption_annotator.annotate_image(post.absolute_path)
    caption = resp.get("caption", "")
    post.caption = caption
    session.commit()
    post = get_post_by_id(post_id, session)
    return post


class CountResponse(BaseModel):
    count: int


@app.get("/v1/posts/count", response_model=CountResponse)
def v1_get_posts_count():
    session = get_session()
    count = session.query(Post).count()
    return {"count": count}


@app.get("/v1/tags/count", response_model=CountResponse)
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


def get_directory_summary(path: str) -> DirectorySummary:
    if not path.startswith(str(shared.target_dir)):
        path = shared.target_dir / path
    else:
        path = pathlib.Path(path).relative_to(shared.target_dir)
    path = str(path)
    path = path.replace("\\", "/")
    summary = DirectorySummary(
        name=os.path.basename(path),
        path=path,
        file_count=0,
        children=[],
    )

    ignore_dirs = shared.pictoria_dir
    with os.scandir(shared.target_dir / path) as entries:
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


@app.get("/v1/folders", response_model=DirectorySummary)
def v1_get_folders():
    target_path = str(shared.target_dir)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Directory not found")
    if not os.path.isdir(target_path):
        raise HTTPException(status_code=400, detail="Path is not a directory")

    return get_directory_summary(target_path)


@app.post("/v1/upload")
async def v1_upload_file(
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
        if "pximg.net" in url:
            headers["referer"] = "https://www.pixiv.net/"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)

        file_io = io.BytesIO(resp.content)
    else:
        file_io = file.file

    if path is None and file is not None:
        path = file.filename
    else:
        path = path or url.split("/")[-1]

    abs_path = shared.target_dir / path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    file_path, file_name, file_ext = get_path_name_and_extension(abs_path)

    console.log(f"Saving file to: {abs_path}")
    post = Post(file_path=file_path, file_name=file_name, extension=file_ext, source=source)
    session = get_session()
    session.add(post)
    session.commit()
    with open(abs_path, "wb") as f:
        shutil.copyfileobj(file_io, f)
    return ORJSONResponse(content={"filename": path})


@app.post("/v1/cmd/update-openai-key")
def v1_update_openai_key(key: str):
    shared.openai_key = key
    shared.pictoria_dir.joinpath("OPENAI_API_KEY").write_text(key)
    return {"status": "ok"}


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
    initialize(args)
    execute_database_migration()
    sync_metadata()
    watch_target_dir()
    host = args.host or "localhost"
    doc_url = f"http://{host}:{args.port}/docs"
    shared.logger.info(f"API Document: {doc_url}")
    uvicorn.run(
        "main:app",
        host=host,
        port=args.port,
        reload=args.reload,
        log_config=None,
    )
