from datetime import UTC, datetime
from typing import List, Optional

from PIL import Image
from pydantic import BaseModel
from sqlalchemy import (
    Boolean,
    Column,
    Computed,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    Session,
    mapped_column,
    relationship,
)

import shared


class Base(DeclarativeBase, MappedAsDataclass):
    pass


class TagGroup(Base):
    __tablename__ = "tag_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False, default=None)
    name: Mapped[str] = mapped_column(String(120), index=True, nullable=False, default="", server_default="")
    color: Mapped[str] = mapped_column(String(9), nullable=False, default="", server_default="")

    tags: Mapped[list["Tag"]] = relationship("Tag", back_populates="group", default_factory=list)


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


class Tag(Base):
    __tablename__ = "tags"
    name: Mapped[str] = mapped_column(String(120), primary_key=True, nullable=False)
    group_id: Mapped[int] = mapped_column(ForeignKey("tag_groups.id"), nullable=True, default=None)
    count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    group: Mapped["TagGroup"] = relationship("TagGroup", back_populates="tags", default=None, lazy="select")
    posts: Mapped[list["PostHasTag"]] = relationship(
        "PostHasTag", back_populates="tag_info", default_factory=list, lazy="select"
    )


class TagWithGroupPublic(TagPublic):
    group: TagGroupPublic
    pass


class Post(Base):
    __tablename__ = "posts"
    __table_args__ = (Index("idx_file_path_name_extension", "file_path", "file_name", "extension", unique=True),)

    id: Mapped[Optional[int]] = mapped_column(
        Integer, primary_key=True, autoincrement=True, nullable=False, default=None
    )
    file_path: Mapped[str] = mapped_column(String, index=True, default="")
    file_name: Mapped[str] = mapped_column(String, index=True, default="")
    extension: Mapped[str] = mapped_column(String, index=True, default="")

    full_path: Mapped[str] = mapped_column(
        String,
        Computed("file_path || '/' || file_name || '.' || extension"),
        init=False,
    )
    aspect_ratio: Mapped[float] = mapped_column(Float, Computed("width * 1.0 / NULLIF(height, 0)"), init=False)

    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=False, index=True, default=0, server_default="0")
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=False, index=True, default=0, server_default="0")

    updated_at: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
        onupdate=int(datetime.now(UTC).timestamp()),
    )
    created_at: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
    )
    score: Mapped[int] = mapped_column(Integer, default=0, index=True, server_default="0")
    rating: Mapped[int] = mapped_column(Integer, default=0, index=True, server_default="0")

    description: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="")
    meta: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="", index=True)
    md5: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="", index=True)
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0", index=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="", index=True)
    caption: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="")
    tags: Mapped[list["PostHasTag"]] = relationship(
        "PostHasTag", back_populates="post", default_factory=list, lazy="select"
    )

    @property
    def absolute_path(self):
        return f"{shared.target_dir}/{self.full_path}"

    @property
    def thumbnail_path(self):
        return f"{shared.thumbnails_dir}/{self.full_path}"

    def rotate(self, session: Session, clockwise: bool = True):
        from utils import calculate_md5, create_thumbnail_by_image

        image = Image.open(self.absolute_path)
        image = image.rotate(90 if clockwise else -90, expand=True)
        image.save(self.absolute_path)
        create_thumbnail_by_image(image, self.thumbnail_path)
        file_data = image.tobytes()
        self.md5 = calculate_md5(file_data)
        self.width, self.height = image.size
        session.add(self)
        session.commit()
        session.refresh(self)


class PostHasTag(Base):
    __tablename__ = "post_has_tag"

    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), primary_key=True)
    tag_name: Mapped[str] = mapped_column(ForeignKey("tags.name"), primary_key=True)

    post: Mapped["Post"] = relationship("Post", back_populates="tags", lazy="select", default=None)
    tag_info: Mapped["Tag"] = relationship("Tag", back_populates="posts", lazy="select", default=None)
    is_auto: Mapped[bool] = mapped_column(Boolean, default=False)


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
    width: Optional[int]
    height: Optional[int]
    aspect_ratio: Optional[float]
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
