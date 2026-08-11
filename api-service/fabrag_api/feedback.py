"""Persistence boundary for answer feedback."""

from __future__ import annotations

import uuid

from src.db import Feedback, get_session


def store_feedback(answer_id: uuid.UUID, rating: str, comment: str | None) -> None:
    with get_session() as session:
        session.add(Feedback(answer_id=answer_id, rating=rating, comment=comment))
        session.commit()
