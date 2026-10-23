"""baseline: existing translations table

Revision ID: fcd65d39a795
Revises: 
Create Date: 2026-08-11 16:17:41.220637

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fcd65d39a795'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # No-op starting point: `translations` predates Alembic and is created by init_db() at startup.
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
