from pydantic import BaseModel


class TagGroupPublic(BaseModel):
    id: int
    name: str
    color: str

    class Config:
        from_attributes = True


class TagPublic(BaseModel):
    name: str
    count: int

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


class PostPublic(BaseModel):
    id: int
    file_path: str
    file_name: str
    extension: str
    full_path: str
    width: int | None
    height: int | None
    aspect_ratio: float | None
    updated_at: int
    created_at: int
    score: int
    rating: int
    description: str
    meta: str
    md5: str
    size: int
    source: str
    caption: str

    class Config:
        from_attributes = True


class PostWithTagPublic(PostPublic):
    tags: list[PostHasTagPublic]
