"""init

Revision ID: a4751fd79514
Revises: 
Create Date: 2024-08-01 23:05:59.759189

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a4751fd79514"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "folders",
        sa.Column("path", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("file_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("path"),
    )
    op.create_table(
        "posts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("file_path", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("extension", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("aspect_ratio", sa.Float(), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("meta", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("md5", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("size", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("posts", schema=None) as batch_op:
        batch_op.create_index("idx_file_path_extension", ["file_path", "extension"], unique=True)
        batch_op.create_index(batch_op.f("ix_posts_aspect_ratio"), ["aspect_ratio"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_extension"), ["extension"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_height"), ["height"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_md5"), ["md5"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_rating"), ["rating"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_score"), ["score"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_size"), ["size"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_updated_at"), ["updated_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_width"), ["width"], unique=False)

    op.create_table(
        "tag_groups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("tag_groups", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_tag_groups_name"), ["name"], unique=False)

    op.create_table(
        "tags",
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["tag_groups.id"],
        ),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "post_has_tag",
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("tag_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("is_auto", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["post_id"],
            ["posts.id"],
        ),
        sa.ForeignKeyConstraint(
            ["tag_name"],
            ["tags.name"],
        ),
        sa.PrimaryKeyConstraint("post_id", "tag_name"),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("post_has_tag")
    op.drop_table("tags")
    with op.batch_alter_table("tag_groups", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_tag_groups_name"))

    op.drop_table("tag_groups")
    with op.batch_alter_table("posts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_posts_width"))
        batch_op.drop_index(batch_op.f("ix_posts_updated_at"))
        batch_op.drop_index(batch_op.f("ix_posts_size"))
        batch_op.drop_index(batch_op.f("ix_posts_score"))
        batch_op.drop_index(batch_op.f("ix_posts_rating"))
        batch_op.drop_index(batch_op.f("ix_posts_md5"))
        batch_op.drop_index(batch_op.f("ix_posts_height"))
        batch_op.drop_index(batch_op.f("ix_posts_extension"))
        batch_op.drop_index(batch_op.f("ix_posts_created_at"))
        batch_op.drop_index(batch_op.f("ix_posts_aspect_ratio"))
        batch_op.drop_index("idx_file_path_extension")

    op.drop_table("posts")
    op.drop_table("folders")
    # ### end Alembic commands ###
