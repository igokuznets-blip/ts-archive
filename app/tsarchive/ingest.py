"""Этап 1. Приём фрагментов из incoming/ → raw/, регистрация в БД.

Ожидаемая структура входного каталога (настраивается FILENAME_REGEX):
    incoming/{vehicle}/{camera}/[любые подкаталоги/]{YYYYMMDD_HHMMSS[_mmm]}*.ts
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg

from .config import settings
from .db import pool, tx

log = logging.getLogger(__name__)
_rx = re.compile(settings.filename_regex)
_tz = ZoneInfo(settings.filename_tz)


def parse_ts(s: str) -> datetime:
    digits = re.sub(r"[^0-9]", "", s)
    base = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    frac = digits[14:]
    micro = int((frac + "000000")[:6]) if frac else 0
    return base.replace(microsecond=micro, tzinfo=_tz)


def parse_rel(rel: str) -> tuple[str, str, datetime] | None:
    m = _rx.match(rel)
    if not m:
        return None
    gd = m.groupdict()
    ts = parse_ts(gd["ts"]) if gd.get("ts") else None
    if ts is None:
        return None
    return gd["vehicle"], gd["camera"], ts


def sha256_file(path: Path) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError:
        shutil.move(str(src), str(dst))


def _quarantine(src: Path, reason: str, rel: str) -> None:
    dst = settings.quarantine / reason / rel
    _move(src, dst)
    log.warning("карантин (%s): %s", reason, rel)


def _ensure_camera(conn, vehicle: str, camera: str, cache: dict) -> int:
    key = (vehicle, camera)
    if key in cache:
        return cache[key]
    vid = conn.execute(
        "INSERT INTO vehicle(code) VALUES (%s) ON CONFLICT (code) DO UPDATE SET code=EXCLUDED.code RETURNING id",
        (vehicle,)).fetchone()["id"]
    cid = conn.execute(
        "INSERT INTO camera(vehicle_id, code) VALUES (%s,%s) "
        "ON CONFLICT (vehicle_id, code) DO UPDATE SET code=EXCLUDED.code RETURNING id",
        (vid, camera)).fetchone()["id"]
    cache[key] = cid
    return cid


def _iter_ready_files(limit: int):
    now = time.time()
    n = 0
    for root, _dirs, files in os.walk(settings.incoming):
        for name in sorted(files):
            if name.startswith(".") or name.endswith((".part", ".tmp", ".filepart")):
                continue  # файл ещё докачивается
            p = Path(root) / name
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            if now - st.st_mtime < settings.stable_seconds:
                continue
            yield p, st
            n += 1
            if n >= limit:
                return


INGEST_LOCK = 727003


def run_once() -> int:
    """Принимает порцию файлов. Выполняется одним воркером одновременно (advisory lock)."""
    with pool().connection() as lock_conn:
        lock_conn.autocommit = True
        if not lock_conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (INGEST_LOCK,)).fetchone()["ok"]:
            return 0
        try:
            return _ingest_batch()
        finally:
            lock_conn.execute("SELECT pg_advisory_unlock(%s)", (INGEST_LOCK,))


def _ingest_batch() -> int:
    settings.ensure_dirs()
    cache: dict = {}
    processed = 0
    for path, st in _iter_ready_files(settings.ingest_batch):
        rel = path.relative_to(settings.incoming).as_posix()
        processed += 1
        if not rel.lower().endswith(".ts"):
            _quarantine(path, "not_ts", rel)
            continue
        parsed = parse_rel(rel)
        if parsed is None:
            _quarantine(path, "bad_name", rel)
            continue
        vehicle, camera, ts = parsed
        digest = sha256_file(path)
        raw_rel = Path("raw") / vehicle / camera / ts.strftime("%Y/%m/%d") / path.name
        dst = settings.archive_root / raw_rel
        if dst.exists():
            raw_rel = raw_rel.with_name(f"{path.stem}_{digest[:4].hex()}{path.suffix}")
            dst = settings.archive_root / raw_rel
        _move(path, dst)
        try:
            with tx() as conn:
                cid = _ensure_camera(conn, vehicle, camera, cache)
                conn.execute(
                    "INSERT INTO fragment(camera_id, rel_path, orig_name, sha256, size_bytes, ts_start) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (cid, raw_rel.as_posix(), rel, digest, st.st_size, ts))
        except psycopg.errors.UniqueViolation:
            # дубликат по содержимому — повторная выгрузка того же файла
            if settings.delete_duplicates:
                dst.unlink(missing_ok=True)
            else:
                _quarantine(dst, "duplicate", rel)
            log.info("дубликат пропущен: %s", rel)
        except Exception:
            # возвращаем файл, чтобы повторить приём
            _move(dst, path)
            raise
    _cleanup_empty_dirs(settings.incoming)
    return processed


def _cleanup_empty_dirs(root: Path) -> None:
    for d, _subdirs, files in os.walk(root, topdown=False):
        p = Path(d)
        # удаляем только давно неизменявшиеся пустые каталоги, чтобы не мешать загрузке
        if p != root and not files and not _subdirs and time.time() - p.stat().st_mtime > 3600:
            try:
                p.rmdir()
            except OSError:
                pass
