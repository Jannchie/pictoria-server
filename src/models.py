import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from sqlalchemy import Boolean, Computed, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    Session,
    mapped_column,
    relationship,
)

import shared


class Base(
    DeclarativeBase,
    MappedAsDataclass,
):
    pass


class TagGroup(Base):
    __tablename__ = "tag_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False, init=False)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("tag_groups.id", ondelete="SET NULL"), nullable=True, default=None)
    name: Mapped[str] = mapped_column(String(120), index=True, nullable=False, default="", server_default="")
    color: Mapped[str] = mapped_column(String(9), nullable=False, default="", server_default="")

    tags: Mapped[list["Tag"]] = relationship(back_populates="group", default_factory=list)


class Tag(Base):
    __tablename__ = "tags"
    name: Mapped[str] = mapped_column(String(120), primary_key=True, nullable=False)
    group_id: Mapped[int | None] = mapped_column(ForeignKey("tag_groups.id", ondelete="SET NULL"), nullable=True, default=None)
    group: Mapped[Optional["TagGroup"]] = relationship(back_populates="tags", lazy="select", init=False)
    posts: Mapped[list["PostHasTag"]] = relationship(back_populates="tag_info", default_factory=list, lazy="select", init=False)


class PostHasColor(Base):
    __tablename__ = "post_has_color"

    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True)
    order: Mapped[int] = mapped_column(Integer, primary_key=True)
    color: Mapped[int] = mapped_column(Integer, nullable=False)
    post: Mapped["Post"] = relationship(back_populates="colors", init=False)


class Post(Base):
    __tablename__ = "posts"
    __table_args__ = (Index("idx_file_path_name_extension", "file_path", "file_name", "extension", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, nullable=False, init=False)
    file_path: Mapped[str] = mapped_column(String, index=True, default="")
    file_name: Mapped[str] = mapped_column(String, index=True, default="")
    extension: Mapped[str] = mapped_column(String, index=True, default="")

    full_path: Mapped[str] = mapped_column(
        String,
        Computed("file_path || '/' || file_name || '.' || extension"),
        init=False,
        nullable=True,
    )
    aspect_ratio: Mapped[float | None] = mapped_column(Float, Computed("width * 1.0 / NULLIF(height, 0)"), init=False)

    width: Mapped[int | None] = mapped_column(Integer, nullable=False, index=True, default=0, server_default="0")
    height: Mapped[int | None] = mapped_column(Integer, nullable=False, index=True, default=0, server_default="0")

    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        index=True,
        default_factory=lambda: datetime.now(UTC),
        onupdate=datetime.now(UTC),
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        index=True,
        default_factory=lambda: datetime.now(UTC),
    )
    published_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True, index=True, default=None)
    score: Mapped[int] = mapped_column(Integer, default=0, index=True, server_default="0")
    rating: Mapped[int] = mapped_column(Integer, default=0, index=True, server_default="0")

    description: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="")
    meta: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="", index=True)
    md5: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="", index=True)
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0", index=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="", index=True)
    caption: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="")
    tags: Mapped[list["PostHasTag"]] = relationship(back_populates="post", default_factory=list, lazy="select")
    colors: Mapped[list["PostHasColor"]] = relationship(back_populates="post", default_factory=list, lazy="select")

    @property
    def absolute_path(self) -> Path:
        return shared.target_dir / self.full_path

    @property
    def thumbnail_path(self) -> Path:
        return shared.thumbnails_dir / self.full_path

    def rotate(self, session: Session, *, clockwise: bool = True) -> None:
        from utils import calculate_md5, create_thumbnail_by_image

        image = Image.open(self.absolute_path)
        image = image.rotate(-90 if clockwise else 90, expand=True)
        image.save(self.absolute_path)
        create_thumbnail_by_image(image, self.thumbnail_path)
        file_data = image.tobytes()
        self.md5 = calculate_md5(file_data)
        self.width, self.height = image.size
        self.commit(session)

    def move(self, session: Session, new_path: str) -> None:
        def move_file(src: Path, dst: Path) -> None:
            if src.is_dir():
                if not dst.exists():
                    dst.mkdir(parents=True, exist_ok=True)

                # Iterate over all files and directories in the source directory
                for item in os.listdir(src):
                    s = src / item
                    d = dst / item
                    # Recursively call move_file for subdirectories and files
                    move_file(s, d)

                # Remove the source directory after its contents have been moved
                src.rmdir()
            else:
                if not dst.parent.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)

        new_path = new_path.strip("/")
        new_full_path = Path(new_path) / self.file_name
        new_full_path = new_full_path.with_suffix(self.extension)
        new_thumbnail_path = shared.thumbnails_dir / new_full_path
        move_file(self.absolute_path, shared.target_dir / new_full_path)
        move_file(self.thumbnail_path, new_thumbnail_path)
        self.file_path = new_path
        self.commit(session)

    def commit(self, session: Session) -> None:
        session.add(self)
        session.commit()
        session.refresh(self)


class PostHasTag(Base):
    __tablename__ = "post_has_tag"

    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True, init=False)
    tag_name: Mapped[str] = mapped_column(ForeignKey("tags.name", ondelete="CASCADE"), primary_key=True, init=False)

    post: Mapped["Post"] = relationship(back_populates="tags", lazy="select")
    tag_info: Mapped["Tag"] = relationship(back_populates="posts", lazy="select")

    is_auto: Mapped[bool] = mapped_column(Boolean, default=False)
