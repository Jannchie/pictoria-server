"""add color to tag group

Revision ID: 0a86fd70529f
Revises: 13ab1f9ff043
Create Date: 2024-08-04 22:02:23.337300

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0a86fd70529f"
down_revision: Union[str, None] = "13ab1f9ff043"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("tag_groups", schema=None) as batch_op:
        batch_op.add_column(sa.Column("color", sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("tag_groups", schema=None) as batch_op:
        batch_op.drop_column("color")

    # ### end Alembic commands ###