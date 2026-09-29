from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models.db_models import Base, TranslationRecord

engine = create_engine(
    settings.DATABASE_URL,
    # Without connect_timeout, a blackholed Postgres host hangs a connection
    # attempt until the OS-level TCP timeout (minutes) — confirmed live. Every
    # caller of SessionLocal()/get_db() benefits from this, not just acronym
    # lookups (acronym_service.get_expansions_batch() also caches the
    # resulting failure so it doesn't repeat this — now-bounded — cost for
    # every candidate token in the query).
    #
    # connect_timeout only covers OPENING a connection though — once one is
    # established, nothing bounded a QUERY running on it. "options": "-c
    # statement_timeout=..." sets that Postgres-side GUC for every session
    # opened through this engine (bare integer = milliseconds, Postgres's
    # default unit), so a connection that accepts new connections fine but
    # hangs on an actual query now gets that query cancelled instead of
    # blocking the request forever. A query that legitimately needs more
    # room can override it per-transaction with `SET LOCAL statement_timeout
    # = '30s'` right before running — that doesn't touch this default for
    # anything else.
    connect_args={
        "connect_timeout": settings.POSTGRES_CONNECT_TIMEOUT,
        "options": f"-c statement_timeout={settings.POSTGRES_STATEMENT_TIMEOUT_MS}",
    },
)
SessionLocal = sessionmaker(bind=engine)

# Only `translations` predates Alembic and is created this way (see
# migrations/versions/fcd65d39a795_baseline_existing_translations_table.py).
# Every other table on Base (e.g. acronym_mapping) is Alembic-managed
# exclusively — scoping create_all() to just this table prevents it from
# racing Alembic's own CREATE TABLE on first boot against a fresh database.
Base.metadata.create_all(bind=engine, tables=[TranslationRecord.__table__])

def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
