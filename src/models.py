from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy import Column, Computed, Float, Index, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlmodel import Field, Relationship, SQLModel

import shared

Base = declarative_base()
SQLModel.metadata = Base.metadata


class TagGroup(SQLModel, table=True):
    __tablename__ = "tag_groups"

    id: int = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    name: str = Field(nullable=False, index=True)
    color: str = Field(default="#000000")

    tags: List["Tag"] = Relationship(back_populates="group")


class TagBase(SQLModel):
    count: int = Field(default=0)


class Tag(TagBase, table=True):
    __tablename__ = "tags"
    name: str = Field(primary_key=True)
    group: Optional[TagGroup] = Relationship(back_populates="tags")
    group_id: Optional[int] = Field(default=None, foreign_key="tag_groups.id")


class TagPublic(TagBase):
    name: str
    group_id: Optional[int]


class Folder(SQLModel, table=True):
    __tablename__ = "folders"

    path: str = Field(primary_key=True)
    file_count: int = Field(default=0)


class PostBase(SQLModel):
    id: int = Field(sa_column=Column(Integer, primary_key=True, autoincrement=True, nullable=False))
    file_path: str = Field(index=True)
    file_name: str = Field(index=True)
    extension: str = Field(index=True)

    width: Optional[int] = Field(default=None, index=True)
    height: Optional[int] = Field(default=None, index=True)
    aspect_ratio: float = Field(sa_column=Column("aspect_ratio", Float, Computed("width * 1.0 / NULLIF(height, 0)")))
    # aspect_ratio: Optional[float] = Field(default=None, index=True)
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
    meta: Optional[str] = Field(sa_column=Column("meta", nullable=False, default="", server_default="", index=True))
    md5: Optional[str] = Field(sa_column=Column("md5", nullable=False, default="", server_default="", index=True))
    size: Optional[int] = Field(sa_column=Column("size", nullable=False, default=0, server_default="0", index=True))
    source: Optional[str] = Field(sa_column=Column("source", nullable=False, default="", server_default="", index=True))
    caption: str = Field(sa_column=Column("caption", nullable=False, default="", server_default=""))


class Post(PostBase, table=True):
    __tablename__ = "posts"
    __table_args__ = (Index("idx_file_path_name_extension", "file_path", "file_name", "extension", unique=True),)

    tags: List["PostHasTag"] = Relationship(back_populates="posts")

    @property
    def relative_path(self):
        return f"{self.file_path}/{self.file_name}.{self.extension}"

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


class PostWithTag(PostBase):
    tags: List[PostHasTagPublic] = []
