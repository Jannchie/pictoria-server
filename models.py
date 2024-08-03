from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import Column, Index, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlmodel import Field, Relationship, SQLModel

import shared

Base = declarative_base()
SQLModel.metadata = Base.metadata


class TagGroup(SQLModel, table=True):
    __tablename__ = "tag_groups"

    id: Optional[int] = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    name: str = Field(nullable=False, index=True)

    tags: List["Tag"] = Relationship(back_populates="group")


class TagBase(SQLModel):
    count: int = Field(default=0)


class Tag(TagBase, table=True):
    __tablename__ = "tags"
    name: str = Field(primary_key=True)
    group: Optional["TagGroup"] = Relationship(back_populates="tags")
    group_id: Optional[int] = Field(default=None, foreign_key="tag_groups.id")


class TagPublic(TagBase):
    name: str


class Folder(SQLModel, table=True):
    __tablename__ = "folders"

    path: str = Field(primary_key=True)
    file_count: int = Field(default=0)


class PostBase(SQLModel):
    file_path: str = Field(index=True)
    extension: str = Field(index=True)

    width: Optional[int] = Field(default=None, index=True)
    height: Optional[int] = Field(default=None, index=True)
    # 由于 Alembic 对 Computed 的支持不够好，所以这里不使用 Computed
    # aspect_ratio: float = Field(sa_column=Column("aspect_ratio", sa.Float, Computed("width * 1.0 / NULLIF(height, 0)")))
    aspect_ratio: Optional[float] = Field(default=None, index=True)
    score: int = Field(default=0, index=True)
    rating: int = Field(default=0, index=True)
    description: Optional[str] = None
    updated_at: int = Field(
        nullable=False,
        index=True,
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
        sa_column_kwargs={"onupdate": int(datetime.now(UTC).timestamp())},
    )
    created_at: int = Field(
        nullable=False,
        index=True,
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
    )
    meta: Optional[str] = None
    md5: Optional[str] = Field(default=None, index=True)
    size: Optional[int] = Field(default=None, index=True)


class Post(PostBase, table=True):
    __tablename__ = "posts"
    __table_args__ = (Index("idx_file_path_extension", "file_path", "extension", unique=True),)

    id: Optional[int] = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    tags: List["PostHasTag"] = Relationship(back_populates="posts")

    @property
    def relative_path(self):
        return f"{self.file_path}.{self.extension}"

    @property
    def absolute_path(self):
        return f"{shared.target_dir}/{self.relative_path}"


class PostHasTagBase(SQLModel):
    post_id: int = Field(foreign_key="posts.id", primary_key=True)
    tag_name: str = Field(foreign_key="tags.name", primary_key=True)
    is_auto: bool = Field(default=False)


class PostHasTag(PostHasTagBase, table=True):
    __tablename__ = "post_has_tag"
    posts: "Post" = Relationship(back_populates="tags")
    tag_info: "Tag" = Relationship()


class PostHasTagPublic(SQLModel):
    is_auto: bool
    tag_info: TagPublic


class PostPublic(PostBase):
    id: int
    tags: List[PostHasTagPublic] = []
