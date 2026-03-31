"""
PostgreSQL chat logging (Render). Uses DATABASE_URL from the environment.
Reads DATABASE_URL lazily so it is always current for the running process.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import BigInteger, Column, DateTime, Text, create_engine, func, inspect, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


class ChatLog(Base):
    """ORM model — PostgreSQL table name: chat_logs"""

    __tablename__ = "chat_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    visitor_id = Column(Text, nullable=True)
    submitted_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    user_message = Column(Text, nullable=False)
    ai_reply = Column(Text, nullable=False)
    source = Column(Text, nullable=True)
    project_type = Column(Text, nullable=True)
    city = Column(Text, nullable=True)
    name = Column(Text, nullable=True)
    phone = Column(Text, nullable=True)
    email = Column(Text, nullable=True)
    meta = Column(JSONB, nullable=True)


def _raw_database_url() -> str:
    """Read URL at call time (not only at import)."""
    return (os.getenv("DATABASE_URL") or "").strip()


def _normalize_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


def get_engine():
    global _engine
    raw = _raw_database_url()
    if not raw:
        return None
    url = _normalize_url(raw)
    if _engine is None:
        logger.info("database: creating SQLAlchemy engine (pool_pre_ping=True)")
        try:
            _engine = create_engine(url, pool_pre_ping=True)
            logger.info("database: SQLAlchemy engine created successfully")
        except Exception:
            logger.exception("database: failed to create SQLAlchemy engine")
            raise
    return _engine


def init_db() -> None:
    """
    Create chat_logs (and any other registered models) if missing.
    Logs success/failure; does not raise — app should keep running.
    """
    logger.info("database: init_db started")
    raw = _raw_database_url()
    if not raw:
        logger.warning(
            "database: DATABASE_URL is missing or empty — skipping init_db "
            "(set DATABASE_URL on your Render **Web Service**, not only on the DB)"
        )
        return

    logger.info(
        "database: DATABASE_URL detected (length=%s, starts_with=%s)",
        len(raw),
        raw[:16] + "..." if len(raw) > 16 else raw,
    )

    try:
        engine = get_engine()
        if engine is None:
            logger.error("database: get_engine() returned None after URL was non-empty")
            return

        table_names = list(Base.metadata.tables.keys())
        if "chat_logs" not in table_names:
            logger.error(
                "database: ChatLog model not registered on Base.metadata (tables=%s)",
                table_names,
            )
        logger.info("database: Base.metadata.create_all(checkfirst=True) STARTING now")
        logger.info("database: tables to ensure: %s", table_names)
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("database: Base.metadata.create_all FINISHED")

        insp = inspect(engine)
        public_tables: List[str] = insp.get_table_names(schema="public")
        if "chat_logs" in public_tables:
            logger.info(
                "database: chat_logs table exists (public schema). All tables: %s",
                public_tables,
            )
        else:
            logger.error(
                "database: create_all finished but chat_logs NOT in public schema. "
                "Tables found: %s",
                public_tables,
            )
    except Exception:
        logger.exception("database: init_db failed with exception")
        # Do not re-raise — keep FastAPI running per product requirement


def save_chat_log(
    *,
    user_message: str,
    ai_reply: str,
    visitor_id: Optional[str] = None,
    source: Optional[str] = None,
    project_type: Optional[str] = None,
    city: Optional[str] = None,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Insert one row. Returns True on commit success.
    Logs failures with full traceback; never raises (does not break /ask).
    """
    logger.info("database: save_chat_log attempting insert (visitor_id=%r)", visitor_id)
    engine = get_engine()
    if not engine:
        logger.warning("database: save_chat_log skipped — no engine (DATABASE_URL unset?)")
        return False

    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _t(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

    row = ChatLog(
        visitor_id=_t(visitor_id),
        user_message=user_message or "",
        ai_reply=ai_reply or "",
        source=_t(source),
        project_type=_t(project_type),
        city=_t(city),
        name=_t(name),
        phone=_t(phone),
        email=_t(email),
        meta=meta if meta is not None else None,
    )

    session = _SessionLocal()
    try:
        session.add(row)
        session.commit()
        # Refresh to log assigned id (BIGSERIAL)
        try:
            session.refresh(row)
            rid = row.id
        except Exception:
            rid = None
        logger.info("database: save_chat_log succeeded (id=%s)", rid)
        return True
    except Exception:
        logger.exception("database: save_chat_log failed")
        session.rollback()
        return False
    finally:
        session.close()


def db_health_check() -> Dict[str, Any]:
    """
    Diagnostics for GET /db-test. Does not expose full DATABASE_URL.
    """
    raw = _raw_database_url()
    out: Dict[str, Any] = {
        "database_url_configured": bool(raw),
        "database_url_length": len(raw) if raw else 0,
        "engine_created": False,
        "connectivity_ok": False,
        "chat_logs_exists": False,
        "public_tables": None,
        "error": None,
    }
    if not raw:
        out["error"] = "DATABASE_URL not set on this process"
        return out

    try:
        engine = get_engine()
        out["engine_created"] = engine is not None
        if not engine:
            out["error"] = "get_engine() returned None"
            return out

        # SQLAlchemy 2.x: use begin() so the connection transaction is committed/closed cleanly
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        out["connectivity_ok"] = True

        insp = inspect(engine)
        tables = insp.get_table_names(schema="public")
        out["public_tables"] = tables
        out["chat_logs_exists"] = "chat_logs" in tables

        if out["chat_logs_exists"]:
            with engine.begin() as conn:
                n = conn.execute(text("SELECT COUNT(*) FROM chat_logs")).scalar_one()
            out["chat_logs_row_count"] = int(n)
        else:
            out["chat_logs_row_count"] = None
    except Exception as e:
        out["error"] = repr(e)
        logger.exception("database: db_health_check failed")

    return out


def debug_insert_test_row() -> Dict[str, Any]:
    """Temporary helper for POST /debug-insert-chat."""
    health_before = db_health_check()
    logger.info("database: debug_insert_test_row — health before insert: %s", health_before)
    ok = save_chat_log(
        user_message="[debug-insert-chat] test user_message",
        ai_reply="[debug-insert-chat] test ai_reply",
        visitor_id="debug-insert-chat",
        source="debug",
        meta={"debug": True, "kind": "manual_test"},
    )
    health_after = db_health_check()
    logger.info("database: debug_insert_test_row — save_chat_log=%s, health after: %s", ok, health_after)

    before_n = health_before.get("chat_logs_row_count")
    after_n = health_after.get("chat_logs_row_count")
    delta = None
    if isinstance(before_n, int) and isinstance(after_n, int):
        delta = after_n - before_n

    return {
        "success": ok,
        "save_chat_log_returned": ok,
        "health_before": health_before,
        "health_after": health_after,
        "chat_logs_row_count_delta": delta,
        "message": "If success is false, check Render logs for database: save_chat_log failed",
    }
