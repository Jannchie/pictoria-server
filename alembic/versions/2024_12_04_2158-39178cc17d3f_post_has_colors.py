"""post has colors

Revision ID: 39178cc17d3f
Revises: 8212d6a8c4a4
Create Date: 2024-12-04 21:58:36.127389

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "39178cc17d3f"
down_revision: str | None = "8212d6a8c4a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "post_has_color",
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("color", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["post_id"],
            ["posts.id"],
        ),
        sa.PrimaryKeyConstraint("post_id", "order"),
    )
    with op.batch_alter_table("posts", schema=None) as batch_op:
        batch_op.create_index("idx_file_path_name_extension", ["file_path", "file_name", "extension"], unique=True)
        batch_op.create_index(batch_op.f("ix_posts_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_extension"), ["extension"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_file_name"), ["file_name"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_file_path"), ["file_path"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_height"), ["height"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_md5"), ["md5"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_meta"), ["meta"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_rating"), ["rating"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_score"), ["score"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_size"), ["size"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_source"), ["source"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_updated_at"), ["updated_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_posts_width"), ["width"], unique=False)

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("posts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_posts_width"))
        batch_op.drop_index(batch_op.f("ix_posts_updated_at"))
        batch_op.drop_index(batch_op.f("ix_posts_source"))
        batch_op.drop_index(batch_op.f("ix_posts_size"))
        batch_op.drop_index(batch_op.f("ix_posts_score"))
        batch_op.drop_index(batch_op.f("ix_posts_rating"))
        batch_op.drop_index(batch_op.f("ix_posts_meta"))
        batch_op.drop_index(batch_op.f("ix_posts_md5"))
        batch_op.drop_index(batch_op.f("ix_posts_height"))
        batch_op.drop_index(batch_op.f("ix_posts_file_path"))
        batch_op.drop_index(batch_op.f("ix_posts_file_name"))
        batch_op.drop_index(batch_op.f("ix_posts_extension"))
        batch_op.drop_index(batch_op.f("ix_posts_created_at"))
        batch_op.drop_index("idx_file_path_name_extension")

    op.drop_table("post_has_color")
    # ### end Alembic commands ###
