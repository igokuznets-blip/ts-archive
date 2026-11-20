"""Очередь разметки: постановка сегментов на (пере)разметку. Без numpy/OpenVINO — используется API."""
from __future__ import annotations

from .db import tx


def requeue(seg_id: int | None = None, camera_id: int | None = None, priority: int = 5) -> int:
    """Поставить на (пере)разметку: сегмент, все сегменты камеры или весь архив (CLI/API)."""
    where, args = ["status='done'"], []
    if seg_id is not None:
        where.append("id=%s")
        args.append(seg_id)
    if camera_id is not None:
        where.append("camera_id=%s")
        args.append(camera_id)
    with tx() as conn:
        return conn.execute(
            f"UPDATE segment SET detect_status='pending', detect_attempts=0, detect_error=NULL, "
            f"detect_priority=greatest(detect_priority, %s) WHERE {' AND '.join(where)} "
            f"AND detect_status <> 'processing'", (priority, *args)).rowcount
