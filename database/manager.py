"""
Database session management and helpers.
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, url: str, echo: bool = False):
        self.engine = create_engine(url, echo=echo, pool_pre_ping=True)
        self._Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)
        logger.info("All tables created / verified")

    @contextmanager
    def session(self) -> Session:
        s = self._Session()
        try:
            yield s
            s.commit()
        except SQLAlchemyError:
            s.rollback()
            raise
        finally:
            s.close()

    def upsert(self, session: Session, model_instance):
        """merge (insert-or-update) a single ORM instance."""
        return session.merge(model_instance)
