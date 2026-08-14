"""create acronym_mapping table

Revision ID: f13a664a31b6
Revises: fcd65d39a795
Create Date: 2026-08-11 16:21:22.416872

"""
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert

# revision identifiers, used by Alembic.
revision: str = 'f13a664a31b6'
down_revision: Union[str, Sequence[str], None] = 'fcd65d39a795'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# data/acronyms.csv, relative to repo root (migrations/versions/<file> -> repo root is 2 up)
SEED_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "acronyms.csv"


def _split_expansions(raw: str) -> list:
    """Pipe-separated -> deduped list, order preserved (spec §5/§7)."""
    seen = set()
    result = []
    for part in raw.split("|"):
        expansion = part.strip()
        if expansion and expansion not in seen:
            seen.add(expansion)
            result.append(expansion)
    return result


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'acronym_mapping',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('acronym', sa.String(length=32), nullable=False),
        sa.Column(
            'expansions', JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column(
            'is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
        ),
        sa.UniqueConstraint('acronym', name='uq_acronym'),
    )
    op.create_index(
        'idx_acronym_active', 'acronym_mapping', ['acronym', 'is_active']
    )

    acronym_table = sa.table(
        'acronym_mapping',
        sa.column('acronym', sa.String),
        sa.column('expansions', JSONB),
        sa.column('description', sa.Text),
        sa.column('is_active', sa.Boolean),
        sa.column('created_at', sa.DateTime),
        sa.column('updated_at', sa.DateTime),
    )

    with open(SEED_CSV_PATH, newline='', encoding='utf-8') as f:
        now = datetime.now(timezone.utc)
        rows = [
            {
                'acronym': row['acronym'].strip().upper(),
                'expansions': _split_expansions(row['expansions']),
                'description': (row.get('description') or '').strip() or None,
                'is_active': True,
                'created_at': now,
                'updated_at': now,
            }
            for row in csv.DictReader(f)
        ]

    if rows:
        stmt = pg_insert(acronym_table).values(rows)
        # ON CONFLICT upsert keeps this migration idempotent — safe to re-run
        # against a DB that already has seed data (e.g. re-running on a stale env).
        stmt = stmt.on_conflict_do_update(
            index_elements=['acronym'],
            set_={
                'expansions': stmt.excluded.expansions,
                'description': stmt.excluded.description,
                'updated_at': stmt.excluded.updated_at,
            },
        )
        op.get_bind().execute(stmt)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_acronym_active', table_name='acronym_mapping')
    op.drop_table('acronym_mapping')
