"""CLI: python -m tsarchive <команда>

  worker                     запустить воркер конвейера
  api                        запустить HTTP API (uvicorn)
  migrate                    применить миграции БД
  once                       один проход всех этапов конвейера
  reindex <segment_id>       заново проиндексировать сегмент (шкала, превью, прокси)
  detect [seg|cam|all] [id]  поставить сегмент / камеру / весь архив на (пере)разметку людей
  quantize <src.onnx> <dst.xml> [кадров]  INT8-квантование модели на кадрах из архива (NNCF)
  helmet-crops <camera_id> <каталог> [кадров]  квадраты голов из архива для проверки/дообучения модели касок
  head-frames <camera_id> <каталог> [кадров]   кадры + предразметка голов (COCO) для дообучения детектора голов
  extract <fragment_id> <out.ts>  скопировать оригинальный фрагмент с проверкой sha256
  verify [N]                 проверить sha256 у N случайных оригиналов (по умолчанию 100)
"""
import hashlib
import os
import sys
from pathlib import Path

from .config import settings


def _stream_fragment(f: dict, out=None) -> bytes:
    """Возвращает sha256, при необходимости одновременно пишет в out."""
    h = hashlib.sha256()
    left = f["size_bytes"]
    with open(settings.archive_root / f["rel_path"], "rb") as src:
        while left > 0:
            buf = src.read(min(4 << 20, left))
            if not buf:
                break
            h.update(buf)
            if out:
                out.write(buf)
            left -= len(buf)
    return h.digest()


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "worker":
        from .worker import main as wmain
        wmain()
    elif cmd == "api":
        import uvicorn
        from .db import migrate
        migrate()
        uvicorn.run("tsarchive.api.main:app", host="0.0.0.0", port=int(os.environ.get("API_PORT", "8000")),
                    workers=int(os.environ.get("API_WORKERS", "2")), proxy_headers=True,
                    forwarded_allow_ips="*")
    elif cmd == "migrate":
        from .db import migrate
        migrate()
    elif cmd == "once":
        from .db import migrate
        from .worker import run_pipeline_once
        migrate()
        settings.ensure_dirs()
        while run_pipeline_once():
            pass
    elif cmd == "reindex":
        from .indexer import reindex
        reindex(int(args[0]))
        print(f"сегмент {args[0]} поставлен в очередь на индексацию")
    elif cmd == "detect":
        from .detect_queue import requeue
        kind = args[0] if args else "all"
        ident = int(args[1]) if len(args) > 1 else None
        n = requeue(seg_id=ident if kind == "seg" else None, camera_id=ident if kind == "cam" else None)
        print(f"на разметку поставлено сегментов: {n}")
    elif cmd == "helmet-crops":
        from .helmet_crops import export
        print(export(int(args[0]), Path(args[1]), int(args[2]) if len(args) > 2 else 300))
    elif cmd == "head-frames":
        from .head_frames import export
        print(export(int(args[0]), Path(args[1]), int(args[2]) if len(args) > 2 else 300))
    elif cmd == "quantize":
        from .quantize import quantize
        quantize(args[0], args[1], int(args[2]) if len(args) > 2 else 300)
    elif cmd in ("extract", "verify"):
        from .db import tx
        with tx() as conn:
            if cmd == "extract":
                f = conn.execute("SELECT * FROM fragment WHERE id=%s", (int(args[0]),)).fetchone()
                if not f:
                    print("фрагмент не найден")
                    return 1
                with open(args[1], "wb") as out:
                    digest = _stream_fragment(f, out)
                ok = digest == bytes(f["sha256"])
                print(f"записано {f['size_bytes']} байт, sha256 {'совпадает' if ok else 'НЕ СОВПАДАЕТ'}")
                return 0 if ok else 2
            n = int(args[0]) if args else 100
            rows = conn.execute("SELECT * FROM fragment WHERE status='grouped' ORDER BY random() LIMIT %s",
                                (n,)).fetchall()
        bad = 0
        for f in rows:
            try:
                if _stream_fragment(f) != bytes(f["sha256"]):
                    bad += 1
                    print("НЕ СОВПАДАЕТ:", f["id"], f["rel_path"])
            except FileNotFoundError:
                bad += 1
                print("ОТСУТСТВУЕТ:", f["id"], f["rel_path"])
        print(f"проверено {len(rows)}, ошибок {bad}")
        return 1 if bad else 0
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
