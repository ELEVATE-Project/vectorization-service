from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models.db_models import Base, TranslationRecord

engine = create_engine(
    settings.DATABASE_URL,
    # Fail fast instead of hanging when Postgres does not respond:
    # - connect_timeout (seconds): limit for opening a connection.
    # - statement_timeout (ms): limit for each query. A query that needs more
    #   time can run `SET LOCAL statement_timeout = '30s'` first.
    connect_args={
        "connect_timeout": settings.POSTGRES_CONNECT_TIMEOUT,
        "options": f"-c statement_timeout={settings.POSTGRES_STATEMENT_TIMEOUT_MS}",
    },
)
SessionLocal = sessionmaker(bind=engine)

# Only `translations` is created here (it existed before Alembic). Alembic creates
# all other tables, so create_all() must not touch them.
Base.metadata.create_all(bind=engine, tables=[TranslationRecord.__table__])

def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
