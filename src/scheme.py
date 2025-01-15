from datetime import datetime

from pydantic import BaseModel


class TagGroupPublic(BaseModel):
    id: int
    name: str
    color: str

    class Config:
        from_attributes = True


class TagPublic(BaseModel):
    name: str

    class Config:
        from_attributes = True


class TagGroupWithTagsPublic(TagGroupPublic):
    tags: list["TagPublic"]


class TagWithGroupPublic(TagPublic):
    group: TagGroupPublic | None


class PostHasTagPublic(BaseModel):
    is_auto: bool
    tag_info: TagWithGroupPublic

    class Config:
        from_attributes = True


class PostHasColorPublic(BaseModel):
    order: int
    color: int


class PostPublic(BaseModel):
    id: int
    file_path: str
    file_name: str
    extension: str
    full_path: str
    width: int | None
    height: int | None
    aspect_ratio: float | None
    updated_at: datetime
    created_at: datetime
    score: int
    rating: int
    description: str
    meta: str
    md5: str
    size: int
    source: str
    caption: str
    colors: list[PostHasColorPublic]
    published_at: datetime | None

    class Config:
        from_attributes = True


class PostWithTagPublic(PostPublic):
    tags: list[PostHasTagPublic]
