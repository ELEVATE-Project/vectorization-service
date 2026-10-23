from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class TranslationRecord(Base):
    __tablename__ = "translations"

    chunk_id = Column(String, primary_key=True)
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AcronymMapping(Base):
    __tablename__ = "acronym_mapping"
    __table_args__ = (
        UniqueConstraint("acronym", name="uq_acronym"),
        # JSONB alone also accepts {}, "text" or null; expansions must be a list.
        CheckConstraint("jsonb_typeof(expansions) = 'array'", name="ck_acronym_expansions_array"),
        Index("idx_acronym_active", "acronym", "is_active"),
    )

    # Auto-incrementing integer primary key.
    id = Column(BigInteger, Identity(always=True), primary_key=True)
    # Stored uppercase per spec §3/§8 — detection normalizes the query token to
    # uppercase before lookup, so the dictionary key must be uppercase too.
    acronym = Column(String(32), nullable=False)
    expansions = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    created_by = Column(String(64), nullable=False, server_default=text("'SYSTEM'"))
    updated_by = Column(String(64), nullable=False, server_default=text("'SYSTEM'"))
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        onupdate=lambda: datetime.now(timezone.utc),
    )
