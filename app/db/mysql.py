import logging
from pathlib import Path
from typing import Any

import pymysql
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.models import Base, Conversation, Feedback, Message

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: Engine | None = None
        self.available = False
        self.backend = "offline"

    def initialize(self) -> bool:
        try:
            sqlalchemy_url = self.settings.sqlalchemy_url
            if sqlalchemy_url.startswith("mysql"):
                self._ensure_mysql_database()
            self.engine = self._create_engine(sqlalchemy_url)
            Base.metadata.create_all(self.engine)
            self.available = True
            self.backend = self._backend_name(sqlalchemy_url)
            return True
        except Exception as exc:
            logger.warning("Primary database unavailable: %s", exc)
            if self.settings.database_url:
                self.available = False
                return False
            try:
                fallback_path = Path(self.settings.fallback_database_path)
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                self.engine = self._create_engine(f"sqlite:///{fallback_path.as_posix()}")
                Base.metadata.create_all(self.engine)
                self.available = True
                self.backend = "sqlite-fallback"
                logger.warning("Using SQLite fallback at %s", fallback_path)
                return True
            except Exception as fallback_exc:
                logger.error("Fallback database unavailable: %s", fallback_exc)
                self.available = False
                return False

    def _create_engine(self, url: str) -> Engine:
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
        else:
            kwargs["pool_recycle"] = 280
        return create_engine(url, **kwargs)

    @staticmethod
    def _backend_name(url: str) -> str:
        if url.startswith("mysql"):
            return "mysql"
        if url.startswith("postgresql") or url.startswith("postgres"):
            return "postgresql"
        return "sqlite"

    def _ensure_mysql_database(self) -> None:
        database_name = self.settings.mysql_database.replace("`", "")
        connection = pymysql.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            user=self.settings.mysql_user,
            password=self.settings.mysql_password,
            autocommit=True,
            connect_timeout=3,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        finally:
            connection.close()

    def _session(self) -> Session | None:
        if not self.available or self.engine is None:
            return None
        return Session(self.engine)

    def save_conversation(self, session_id: str, title: str) -> None:
        session = self._session()
        if session is None:
            return
        try:
            conversation = session.scalar(select(Conversation).where(Conversation.session_id == session_id))
            if conversation is None:
                session.add(Conversation(session_id=session_id, title=title[:160]))
            else:
                conversation.title = title[:160]
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("Could not save conversation: %s", exc)
        finally:
            session.close()

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        entities: dict | list | None = None,
        evidence: list | None = None,
        grounded: bool = False,
        latency_ms: float | None = None,
    ) -> int | None:
        session = self._session()
        if session is None:
            return None
        try:
            message = Message(
                session_id=session_id,
                role=role,
                content=content,
                intent=intent,
                entities=entities,
                evidence=evidence,
                grounded=grounded,
                latency_ms=latency_ms,
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            return message.id
        except Exception as exc:
            session.rollback()
            logger.warning("Could not save message: %s", exc)
            return None
        finally:
            session.close()

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        session = self._session()
        if session is None:
            return []
        try:
            messages = session.scalars(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.asc(), Message.id.asc())
                .limit(limit)
            ).all()
            return [
                {
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                    "intent": message.intent,
                    "entities": message.entities or [],
                    "evidence": message.evidence or [],
                    "grounded": message.grounded,
                    "latency_ms": message.latency_ms,
                    "created_at": message.created_at.isoformat() if message.created_at else None,
                }
                for message in messages
            ]
        except Exception as exc:
            logger.warning("Could not load messages: %s", exc)
            return []
        finally:
            session.close()

    def save_feedback(self, session_id: str, message_id: int | None, rating: int, comment: str | None) -> None:
        session = self._session()
        if session is None:
            return
        try:
            session.add(Feedback(session_id=session_id, message_id=message_id, rating=rating, comment=comment))
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("Could not save feedback: %s", exc)
        finally:
            session.close()

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
