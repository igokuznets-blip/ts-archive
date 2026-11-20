"""Подключение к PostgreSQL и применение миграций."""
from __future__ import annotations

import atexit
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings

log = logging.getLogger(__name__)
_pool: ConnectionPool | None = None

MIGRATIONS_DIR = Path(os.environ.get(
    "MIGRATIONS_DIR", Path(__file__).resolve().parent.parent.parent / "db" / "migrations"))


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url, min_size=1, max_size=int(os.environ.get("DB_POOL_SIZE", "10")),
            kwargs={"row_factory": dict_row, "autocommit": False}, open=True,
        )
        atexit.register(_pool.close)
    return _pool


@contextmanager
def tx():
    """Транзакция: commit при успехе, rollback при исключении."""
    with pool().connection() as conn:
        with conn.transaction():
            yield conn


def migrate() -> None:
    """Применяет SQL-файлы из db/migrations по порядку имени. Безопасно при параллельном запуске."""
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(727001)")
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                         "name text PRIMARY KEY, applied_at timestamptz DEFAULT now())")
            done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
            for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if f.name in done:
                    continue
                log.info("применение миграции %s", f.name)
                with conn.transaction():
                    conn.execute(f.read_text(encoding="utf-8"))
                    conn.execute("INSERT INTO schema_migrations(name) VALUES (%s)", (f.name,))
        finally:
            conn.execute("SELECT pg_advisory_unlock(727001)")
