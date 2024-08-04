import os
import pathlib
import shutil
import tomllib
from typing import Annotated, Optional

import fastapi
import uvicorn
from fastapi import File, Form, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rich import get_console
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from starlette.convertors import Convertor, register_url_convertor
from wdtagger import Tagger

import shared
from models import Post, PostHasTag, PostPublic, Tag
from utils import (
    attach_tags_to_post,
    execute_database_migration,
    from_rating_to_int,
    get_session,
    initialize_directories,
    parse_arguments,
    process_posts,
    sync_metadata,
    use_route_names_as_operation_ids,
    watch_target_dir,
)

pyproject = tomllib.load(open("pyproject.toml", "rb"))

console = get_console()
app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的请求来源，可以设为 ["*"] 以允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许的 HTTP 方法，可以指定特定方法如 ["GET", "POST"]
    allow_headers=["*"],  # 允许的 HTTP 头，可以指定特定头如 ["Authorization", "Content-Type"]
)


class PathConvertor(Convertor):
    regex = r".+?"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("pathlike", PathConvertor())


def get_post_by_id(post_id, session):
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
    folder: Optional[str]


@app.post("/v1/posts", response_model=list[Post])
def v1_get_posts(
    limit: int | None = None,
    offset: int = 0,
    filter: PostFilter = PostFilter(),
):

    session = get_session()
    query = session.query(Post)
    if filter.rating:
        query = query.filter(Post.rating.in_(filter.rating))
    if filter.score:
        query = query.filter(Post.score.in_(filter.score))
    if filter.tags:
        query = query.join(Post.tags).filter(PostHasTag.tag_name.in_(filter.tags))
    if filter.folder and filter.folder != "." and filter.folder != "":
        query = query.filter(Post.file_path.startswith(filter.folder))
    return query.limit(limit).offset(offset).all()


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


@app.get("/v1/posts/count/rating", response_model=list[RatingCountResponse])
def v1_count_group_by_rating():
    session = get_session()
    query_result = session.query(Post.rating, func.count()).group_by(Post.rating).all()
    # Transform the query result into a list of RatingCountResponse instances
    response_data = [
        RatingCountResponse(rating=row[0] if row[0] is not None else 0, count=row[1]) for row in query_result
    ]
    return response_data


class ScoreCountResponse(BaseModel):
    score: int
    count: int


@app.get("/v1/posts/count/score", response_model=list[ScoreCountResponse])
def v1_count_group_by_score():
    session = get_session()
    query_result = session.query(Post.score, func.count()).group_by(Post.score).all()
    response_data = [
        ScoreCountResponse(score=row[0] if row[0] is not None else 0, count=row[1]) for row in query_result
    ]
    return response_data


class ScoreUpdate(BaseModel):
    score: Annotated[int, Field(ge=0, le=5)]


@app.put("/v1/posts/{post_id}/score")
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


@app.put("/v1/posts/{post_id}/rating")
def v1_update_post_rating(post_id: Annotated[int, Path(gt=0)], rating_update: RatingUpdate):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post.rating = rating_update.rating
    session.commit()
    return post


@app.get("/v1/posts/{post_id}", response_model=PostPublic)
def v1_get_post(post_id: int):
    session = get_session()
    post = get_post_by_id(post_id, session)
    return post


@app.get("/v1/images/{post_path:pathlike}")
def v1_get_post_by_path(post_path: str):
    abs_path = shared.target_dir / post_path
    return fastapi.responses.FileResponse(abs_path)


@app.get("/v1/thumbnails/{post_path:pathlike}")
def v1_get_thumbnail(post_path: str):
    abs_path = shared.thumbnails_dir / post_path
    return fastapi.responses.FileResponse(abs_path)


@app.get("/v1/tags")
def v1_get_tags():
    session = get_session()
    tags = session.query(Tag).all()
    return tags


@app.post("/v1/tag/{tag_name}")
def v1_create_tag(tag_name: str):
    session = get_session()
    tag = Tag(name=tag_name)
    session.add(tag)
    session.commit()
    return tag


@app.delete("/v1/tag/{tag_name}")
def v1_delete_tag(tag_name: str):
    session = get_session()
    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    session.delete(tag)
    session.commit()
    return tag


@app.get("/v1/tags/{tag_name}")
def v1_get_tag(tag_name: str):
    session = get_session()
    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    return tag


@app.put("/v1/tags/{tag_name}")
def v1_update_tag(tag_name: str, new_tag_name: str):
    session = get_session()
    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    tag.name = new_tag_name
    session.commit()
    return tag


@app.post("/v1/posts/{post_id}/tags/{tag_name}")
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


@app.delete("/v1/posts/{post_id}/tags/{tag_name}")
def v1_remove_tag_from_post(post_id: int, tag_name: str):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()
    tag_record = session.query(PostHasTag).filter(PostHasTag.tag_name == tag_name).first()
    if tag_record is not None:
        session.delete(tag_record)
        session.commit()
    post = get_post_by_id(post_id, session)
    return post


@app.post("/v1/cmd/process-posts")
def v1_cmd_process_posts():
    process_posts(True)
    return {"status": "ok"}


@app.get("/v1/cmd/auto-tags/{post_id}")
def v1_cmd_auto_tags(post_id: int):
    session = get_session()
    post = session.query(Post).filter(Post.id == post_id).first()

    abs_path = post.absolute_path
    if shared.tagger is None:
        shared.tagger = Tagger(slient=True)
    resp = shared.tagger.tag(abs_path)
    post.rating = from_rating_to_int(resp.rating)
    all_tags = resp.all_tags
    attach_tags_to_post(session, post, all_tags, is_auto=True)
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


DirectorySummary.update_forward_refs()


def get_directory_summary(path: str) -> DirectorySummary:
    summary = DirectorySummary(
        name=os.path.basename(path),
        path=str(pathlib.Path(path).relative_to(shared.target_dir)),
        file_count=0,
        children=[],
    )

    ignore_dirs = shared.pictoria_dir
    with os.scandir(path) as entries:
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

    directory_summary = get_directory_summary(target_path)
    return directory_summary


@app.post("/v1/upload")
async def v1_upload_file(file: UploadFile = File(...), path: str = Form(...)):
    file_location = shared.target_dir / path
    file_location.parent.mkdir(parents=True, exist_ok=True)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse(content={"filename": path})


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

    initialize_directories(args)
    execute_database_migration()
    sync_metadata()
    watch_target_dir()

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=args.port,
        reload=args.reload,
        log_config=None,
    )
