"""Этап 5. Служебная очистка: зависшие задачи, истёкшие выгрузки, временные файлы.

Исходные TS не упаковываются и не перемещаются: они и есть файлы для воспроизведения
(HLS byte-range), поэтому их пути и размеры в БД неизменны."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .config import settings
from .db import tx

log = logging.getLogger(__name__)


def cleanup_once() -> int:
    n = 0
    with tx() as conn:
        # возврат «зависших» задач упавших воркеров
        n += conn.execute(
            "UPDATE fragment SET status='new', claimed_at=NULL WHERE status='probing' "
            "AND claimed_at < now() - %s * interval '1 second'", (settings.stale_claim_sec,)).rowcount
        n += conn.execute(
            "UPDATE segment SET status='ready', claimed_at=NULL WHERE status='processing' "
            "AND claimed_at < now() - %s * interval '1 second'", (settings.stale_claim_sec,)).rowcount
        n += conn.execute(
            "UPDATE segment SET detect_status='pending', detect_claimed_at=NULL WHERE detect_status='processing' "
            "AND detect_claimed_at < now() - %s * interval '1 second'", (settings.stale_claim_sec,)).rowcount
        expired = conn.execute(
            "DELETE FROM export WHERE expires_at < now() RETURNING rel_path").fetchall()
    for e in expired:
        if e["rel_path"]:
            (settings.archive_root / e["rel_path"]).unlink(missing_ok=True)
            n += 1
    # временные файлы старше суток
    now = datetime.now(timezone.utc).timestamp()
    for p in settings.tmp.glob("*"):
        try:
            if now - p.stat().st_mtime > 86400:
                if p.is_dir():
                    import shutil
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
    return n
