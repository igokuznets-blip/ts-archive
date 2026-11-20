"""Воркер конвейера. Этапы включаются переменной WORKER_STAGES, воркеры масштабируются репликами:
все очереди реализованы в PostgreSQL через SELECT ... FOR UPDATE SKIP LOCKED."""
from __future__ import annotations

import logging
import signal
import time

from . import grouper, indexer, ingest, maintenance, probe
from .config import settings
from .db import migrate

log = logging.getLogger("tsarchive.worker")

STAGES = {
    "ingest": ingest.run_once,
    "probe": probe.run_once,
    "group": grouper.run_once,
    "index": indexer.run_once,
    "cleanup": maintenance.cleanup_once,
    "detect": lambda: _detect_run(),
}


def _detect_run() -> int:
    # numpy/OpenVINO нужны только образу с детектором (target detect в Dockerfile)
    from . import detect
    return detect.run_once()
# служебные этапы не нужно выполнять на каждой итерации
PERIODIC = {"cleanup": 300.0}

_stop = False


def _sigterm(*_):
    global _stop
    _stop = True
    log.info("получен сигнал остановки, завершаю текущую задачу")


def run_pipeline_once() -> int:
    """Один проход всех этапов (используется в тестах и CLI)."""
    total = 0
    for name in settings.worker_stages:
        total += STAGES[name]()
    return total


def main() -> None:
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    migrate()
    settings.ensure_dirs()
    unknown = set(settings.worker_stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"неизвестные этапы: {unknown}")
    log.info("воркер запущен, этапы: %s", ",".join(settings.worker_stages))
    last_run: dict[str, float] = {}
    while not _stop:
        busy = 0
        for name in settings.worker_stages:
            if _stop:
                break
            period = PERIODIC.get(name)
            if period and time.monotonic() - last_run.get(name, 0) < period:
                continue
            try:
                done = STAGES[name]()
                busy += done
                if period and not done:
                    last_run[name] = time.monotonic()
            except Exception:  # noqa: BLE001
                log.exception("ошибка на этапе %s", name)
                time.sleep(1)
        if not busy:
            time.sleep(settings.poll_interval)
