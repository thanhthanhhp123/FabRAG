"""
Database layer: SQLAlchemy models mirroring schema.sql, plus a couple of
small write helpers used by ingest.py.

Models are kept in sync with schema.sql by hand for now (no Alembic yet —
fine for an MVP with one schema version; revisit once the schema needs to
change after real data exists).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

load_dotenv()

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    manufacturer: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    num_pages: Mapped[int | None] = mapped_column(Integer)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("filename"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Text)
    chunking_strategy: Mapped[str] = mapped_column(Text, default="fixed")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")


def get_engine():
    url = os.environ["DATABASE_URL"]
    return create_engine(url, future=True)


def get_session() -> Session:
    return Session(get_engine())


def upsert_document(
    session: Session,
    *,
    filename: str,
    title: str | None = None,
    manufacturer: str | None = None,
    source_url: str | None = None,
    num_pages: int | None = None,
) -> Document:
    """Insert the document row, or return the existing one for this filename.

    Ingestion is re-run often during development — this makes re-running
    idempotent for the `documents` row (chunk replacement is handled by the
    caller via delete-then-insert, see ingest.py).
    """
    existing = session.query(Document).filter_by(filename=filename).one_or_none()
    if existing is not None:
        existing.title = title or existing.title
        existing.manufacturer = manufacturer or existing.manufacturer
        existing.source_url = source_url or existing.source_url
        existing.num_pages = num_pages or existing.num_pages
        return existing

    doc = Document(
        filename=filename,
        title=title,
        manufacturer=manufacturer,
        source_url=source_url,
        num_pages=num_pages,
    )
    session.add(doc)
    session.flush()  # populate doc.id without committing
    return doc
