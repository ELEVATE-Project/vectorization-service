from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models.db_models import Base, TranslationRecord

engine = create_engine(
    settings.DATABASE_URL,
    # Limits: connect_timeout (s) to open a connection, statement_timeout (ms) per query.
    connect_args={
        "connect_timeout": settings.POSTGRES_CONNECT_TIMEOUT,
        "options": f"-c statement_timeout={settings.POSTGRES_STATEMENT_TIMEOUT_MS}",
    },
)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create the `translations` table if missing; called once at app startup.

    Only this table: it predates Alembic, which creates all the others.
    """
    Base.metadata.create_all(bind=engine, tables=[TranslationRecord.__table__])


def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
