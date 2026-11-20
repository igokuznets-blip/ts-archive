-- Предварительная разметка: интервалы «человек в кадре» (этап detect)

ALTER TABLE camera
    ADD COLUMN detect_enabled boolean NOT NULL DEFAULT true,
    ADD COLUMN detect_config  jsonb;    -- {"ignore": [[[x,y],...]], "zones": {"имя": [[x,y],...]}, "min_conf": .., "min_height": ..}

ALTER TABLE segment
    ADD COLUMN detect_status     text NOT NULL DEFAULT 'pending'   -- pending | processing | done | error
              CHECK (detect_status IN ('pending','processing','done','error')),
    ADD COLUMN detect_priority   int NOT NULL DEFAULT 0,           -- >0: открыт в интерфейсе / пересчёт по запросу
    ADD COLUMN detect_attempts   int NOT NULL DEFAULT 0,
    ADD COLUMN detect_error      text,
    ADD COLUMN detect_claimed_at timestamptz,
    ADD COLUMN detect_done_at    timestamptz,
    ADD COLUMN detect_model      text,
    ADD COLUMN detect_path       text,      -- detections/<техника>/<камера>/<день>/<id>: boxes.json + миниатюры
    ADD COLUMN detect_frames     int,
    ADD COLUMN detect_sec        real;      -- время обработки сегмента, с
CREATE INDEX segment_detect_queue_idx ON segment (detect_priority DESC, ts_start DESC)
    WHERE status = 'done' AND detect_status = 'pending';

CREATE TABLE detection (
    id         bigserial PRIMARY KEY,
    camera_id  int NOT NULL REFERENCES camera(id) ON DELETE CASCADE,
    segment_id bigint NOT NULL REFERENCES segment(id) ON DELETE CASCADE,
    label      text NOT NULL DEFAULT 'person',
    ts_start   timestamptz NOT NULL,
    ts_end     timestamptz NOT NULL,
    off_start  double precision NOT NULL,   -- позиция на шкале сегмента (плейлиста), с
    off_end    double precision NOT NULL,
    best_off   double precision,            -- кадр миниатюры
    max_count  int NOT NULL,                -- максимум объектов одновременно
    peak_score real NOT NULL,
    zones      text[] NOT NULL DEFAULT '{}',-- именованные зоны камеры, куда попадали люди
    thumb      text,                        -- путь к миниатюре относительно ARCHIVE_ROOT
    model      text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX detection_cam_range_gist ON detection USING gist (camera_id, tstzrange(ts_start, ts_end, '[]'));
CREATE INDEX detection_cam_label_ts_idx ON detection (camera_id, label, ts_start);
CREATE INDEX detection_segment_idx ON detection (segment_id);
CREATE INDEX detection_zones_gin ON detection USING gin (zones) WHERE zones <> '{}';
