"""Конфигурация приложения. Все параметры задаются переменными окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


DEFAULT_FILENAME_REGEX = (
    r"^(?P<vehicle>[^/]+)/(?P<camera>[^/]+)/(?:.*/)?"
    r"(?P<ts>\d{8}[_T-]?\d{6}(?:[_.]\d{1,6})?)[^/]*\.ts$"
)


@dataclass(frozen=True)
class Settings:
    database_url: str = _env("DATABASE_URL", "postgresql://tsa:tsa@localhost:5432/tsarchive")
    archive_root: Path = Path(_env("ARCHIVE_ROOT", "/archive"))

    # --- приём ---
    filename_regex: str = _env("FILENAME_REGEX", DEFAULT_FILENAME_REGEX)
    filename_tz: str = _env("FILENAME_TZ", "Europe/Moscow")
    stable_seconds: float = _float("INGEST_STABLE_SECONDS", 30)
    ingest_batch: int = _int("INGEST_BATCH", 200)
    delete_duplicates: bool = _bool("INGEST_DELETE_DUPLICATES", True)

    # --- анализ ---
    probe_batch: int = _int("PROBE_BATCH", 20)
    probe_threads: int = _int("PROBE_THREADS", 4)

    # --- группировка ---
    gap_threshold_sec: float = _float("GAP_THRESHOLD_SEC", 3.0)
    overlap_tolerance_sec: float = _float("OVERLAP_TOLERANCE_SEC", 1.0)
    segment_max_sec: float = _float("SEGMENT_MAX_SEC", 3600)
    idle_close_sec: float = _float("IDLE_CLOSE_SEC", 600)

    # --- воспроизведение исходных TS (HLS byte-range, без склеенных копий) ---
    hls_chunk_sec: float = _float("HLS_CHUNK_SEC", 6.0)       # целевая длина куска плейлиста
    pat_search_bytes: int = _int("PAT_SEARCH_BYTES", 1 << 20)  # поиск PAT/PMT перед ключевым кадром
    playlist_cache_sec: int = _int("PLAYLIST_CACHE_SEC", 60)

    # --- сегменты, превью, прокси ---
    # auto: прокси H.264 только если кодек не воспроизводится браузером (MPEG-2 Video и т.п.)
    # never: никогда не перекодировать; always: всегда делать прокси H.264
    proxy_transcode: str = _env("PROXY_TRANSCODE", "auto")
    browser_codecs: tuple = tuple(_env("BROWSER_CODECS", "h264,hevc").split(","))
    audio_mode: str = _env("AUDIO_MODE", "auto")  # для выгрузок MP4: auto (G.711 → AAC) | drop
    thumb_interval_sec: int = _int("THUMB_INTERVAL_SEC", 15)
    thumb_width: int = _int("THUMB_WIDTH", 160)
    thumb_cols: int = _int("THUMB_COLS", 10)
    ffmpeg_bin: str = _env("FFMPEG_BIN", "ffmpeg")
    ffprobe_bin: str = _env("FFPROBE_BIN", "ffprobe")
    ffmpeg_timeout: int = _int("FFMPEG_TIMEOUT", 3600)
    max_attempts: int = _int("MAX_ATTEMPTS", 3)

    # --- выгрузки и очистка ---
    export_ttl_hours: float = _float("EXPORT_TTL_HOURS", 24)
    export_max_sec: int = _int("EXPORT_MAX_SEC", 3600)
    stale_claim_sec: int = _int("STALE_CLAIM_SEC", 3600)

    # --- детекция людей (этап detect, OpenVINO) ---
    detect_model: str = _env("DETECT_MODEL", "/models/yolox_s.onnx")  # .onnx или OpenVINO IR .xml (INT8)
    detect_decoded: bool = _bool("DETECT_DECODED", False)   # выход модели уже декодирован (export --decode_in_inference)
    detect_classes: str = _env("DETECT_CLASSES", "person")  # классы COCO через запятую: person,car,truck
    detect_fps: float = _float("DETECT_FPS", 2.0)           # кадров в секунду на анализ
    detect_conf: float = _float("DETECT_CONF", 0.45)
    detect_nms: float = _float("DETECT_NMS", 0.45)
    detect_min_height: float = _float("DETECT_MIN_HEIGHT", 0.03)  # минимальная высота рамки, доля кадра
    detect_gap_sec: float = _float("DETECT_GAP_SEC", 3.0)   # пауза, после которой интервал закрывается
    detect_min_dur_sec: float = _float("DETECT_MIN_DUR_SEC", 1.0)
    detect_min_hits: int = _int("DETECT_MIN_HITS", 2)       # минимум кадров с объектом в интервале
    detect_device: str = _env("DETECT_DEVICE", "CPU")
    detect_hint: str = _env("DETECT_HINT", "THROUGHPUT")    # THROUGHPUT | LATENCY
    detect_threads: int = _int("DETECT_THREADS", 0)         # 0 — все доступные ядра
    detect_streams: str = _env("DETECT_STREAMS", "")        # пусто — подбирает OpenVINO
    detect_decode_threads: int = _int("DETECT_DECODE_THREADS", 2)
    detect_thumb_width: int = _int("DETECT_THUMB_WIDTH", 320)
    # --- проверка каски (второй шаг detect; включается, если есть файл модели) ---
    helmet_model: str = _env("HELMET_MODEL", "/models/helmet_cls.onnx")
    helmet_yes: float = _float("HELMET_YES_THRESH", 0.6)     # p(каска) ≥ — «в каске»
    helmet_no: float = _float("HELMET_NO_THRESH", 0.7)       # p(без каски) ≥ — «без каски»
    helmet_min_head_px: float = _float("HELMET_MIN_HEAD_PX", 14)  # голова меньше — «не определить»
    helmet_min_hits: int = _int("HELMET_MIN_HITS", 3)        # кадров «без каски» в интервале, минимум
    helmet_max_width: int = _int("HELMET_MAX_WIDTH", 1920)   # кадр для вырезки голов, не шире
    helmet_threads: int = _int("HELMET_THREADS", 2)
    helmet_scales: str = _env("HELMET_SCALES", "1,1.4")     # масштабы квадрата головы (через запятую)
    # Режим проверки касок: heads — детектор голов (основной), crop — квадрат головы из рамки человека
    # + классификатор (запасной), auto — heads, если есть модель, иначе crop; off — выключено.
    helmet_mode: str = _env("HELMET_MODE", "auto")
    heads_model: str = _env("HEADS_MODEL", "/models/heads_nano.onnx")
    heads_conf: float = _float("HEADS_CONF", 0.4)             # голова учитывается от этой уверенности
    heads_no_conf: float = _float("HEADS_NO_CONF", 0.5)       # «без каски» — от этой уверенности
    heads_alone_conf: float = _float("HEADS_ALONE_CONF", 0.5)  # голова без рамки человека считается человеком
    heads_min_px: float = _float("HEADS_MIN_PX", 6)            # сторона головы во входе модели, минимум
    heads_nms: float = _float("HEADS_NMS", 0.45)
    heads_all_frames: bool = _bool("HEADS_ALL_FRAMES", True)   # искать головы и в кадрах без людей

    # --- воркер ---
    worker_stages: tuple = tuple(
        s.strip() for s in _env("WORKER_STAGES", "ingest,probe,group,index,cleanup").split(",") if s.strip()
    )
    poll_interval: float = _float("POLL_INTERVAL", 2.0)

    # --- API ---
    use_x_accel: bool = _bool("USE_X_ACCEL", False)
    x_accel_prefix: str = _env("X_ACCEL_PREFIX", "/protected/")

    @property
    def incoming(self) -> Path:
        return self.archive_root / "incoming"

    @property
    def raw(self) -> Path:
        return self.archive_root / "raw"

    @property
    def proxies(self) -> Path:
        return self.archive_root / "proxies"

    @property
    def thumbs(self) -> Path:
        return self.archive_root / "thumbs"

    @property
    def quarantine(self) -> Path:
        return self.archive_root / "quarantine"

    @property
    def exports(self) -> Path:
        return self.archive_root / "exports"

    @property
    def detections(self) -> Path:
        return self.archive_root / "detections"

    @property
    def tmp(self) -> Path:
        return self.archive_root / "tmp"

    def ensure_dirs(self) -> None:
        for p in (self.incoming, self.raw, self.proxies, self.thumbs,
                  self.quarantine, self.exports, self.detections, self.tmp):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
