-- Базовая схема архива видеофрагментов MPEG-2 TS
CREATE EXTENSION IF NOT EXISTS btree_gist;

CREATE TABLE vehicle (
    id               serial PRIMARY KEY,
    code             text NOT NULL UNIQUE,          -- бортовой номер / код техники
    name             text,
    type             text,                          -- трубоукладчик, экскаватор, сварочный комплекс...
    pipeline_section text,                          -- участок трубопровода
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE camera (
    id         serial PRIMARY KEY,
    vehicle_id int NOT NULL REFERENCES vehicle(id) ON DELETE CASCADE,
    code       text NOT NULL,                       -- cam1, cabin, boom...
    name       text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (vehicle_id, code)
);

-- Сегмент — непрерывный участок записи (логическая запись: видео остаётся в исходных TS)
CREATE TABLE segment (
    id              bigserial PRIMARY KEY,
    camera_id       int NOT NULL REFERENCES camera(id) ON DELETE CASCADE,
    ts_start        timestamptz NOT NULL,
    ts_end          timestamptz NOT NULL,
    duration        double precision NOT NULL DEFAULT 0,   -- длительность медиа (сумма фрагментов)
    fragments_count int NOT NULL DEFAULT 0,
    codec_sig       text,
    vcodec          text,
    width           int,
    height          int,
    gap_total_sec   double precision NOT NULL DEFAULT 0,   -- суммарные мелкие разрывы внутри
    status          text NOT NULL DEFAULT 'ready'          -- ready | processing | done | error
                    CHECK (status IN ('ready','processing','done','error')),
    attempts        int NOT NULL DEFAULT 0,
    error           text,
    proxy_path      text,      -- прокси H.264 только для кодеков, которые браузер не воспроизводит
    proxy_transcoded boolean NOT NULL DEFAULT false,
    sprite_path     text,
    thumbs          jsonb,     -- {interval, cols, rows, w, h, count}
    size_bytes      bigint,    -- сумма исходных фрагментов (+ прокси, если есть)
    claimed_at      timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now(),
    done_at         timestamptz
);
CREATE INDEX segment_cam_range_gist ON segment USING gist (camera_id, tstzrange(ts_start, ts_end, '[]'));
CREATE INDEX segment_status_idx ON segment (status) WHERE status <> 'done';

-- Исходный фрагмент (~80 МБ, несколько минут записи) — хранится как есть и воспроизводится напрямую
CREATE TABLE fragment (
    id                bigserial PRIMARY KEY,
    camera_id         int NOT NULL REFERENCES camera(id) ON DELETE CASCADE,
    rel_path          text NOT NULL UNIQUE,     -- путь относительно ARCHIVE_ROOT (raw/...)
    orig_name         text NOT NULL,
    sha256            bytea NOT NULL,
    size_bytes        bigint NOT NULL,
    ts_start          timestamptz NOT NULL,     -- абсолютное время начала (из имени файла)
    ts_end            timestamptz,              -- ts_start + duration
    duration          double precision,
    start_pts         double precision,
    vcodec            text,
    width             int,
    height            int,
    fps               text,
    acodec            text,
    codec_sig         text,                     -- сигнатура потоков: склейка -c copy только при совпадении
    starts_with_key   boolean,
    status            text NOT NULL DEFAULT 'new'  -- new | probing | probed | grouped | quarantine
                      CHECK (status IN ('new','probing','probed','grouped','quarantine')),
    error             text,
    segment_id        bigint REFERENCES segment(id) ON DELETE SET NULL,
    offset_in_segment double precision,         -- позиция фрагмента на шкале сегмента (плейлиста), с
    kf_t              double precision[],       -- ключевые кадры: время от начала файла, с
    kf_pos            bigint[],                 -- ключевые кадры: байтовое смещение (с PAT/PMT) для Range
    claimed_at        timestamptz,
    ingested_at       timestamptz NOT NULL DEFAULT now(),
    probed_at         timestamptz
);
CREATE UNIQUE INDEX fragment_sha_uq ON fragment (sha256);
CREATE INDEX fragment_cam_range_gist ON fragment USING gist (camera_id, tstzrange(ts_start, coalesce(ts_end, ts_start), '[]'));
CREATE INDEX fragment_status_idx ON fragment (status, camera_id, ts_start) WHERE status IN ('new','probing','probed');
CREATE INDEX fragment_segment_idx ON fragment (segment_id, offset_in_segment);

-- Метки событий (ручные или от будущей аналитики)
CREATE TABLE event (
    id         bigserial PRIMARY KEY,
    camera_id  int NOT NULL REFERENCES camera(id) ON DELETE CASCADE,
    ts         timestamptz NOT NULL,
    kind       text NOT NULL DEFAULT 'mark',
    note       text,
    payload    jsonb,
    author     text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX event_cam_ts_idx ON event (camera_id, ts);

-- Выгрузки фрагментов архива
CREATE TABLE export (
    id         uuid PRIMARY KEY,
    camera_id  int NOT NULL REFERENCES camera(id) ON DELETE CASCADE,
    ts_from    timestamptz NOT NULL,
    ts_to      timestamptz NOT NULL,
    rel_path   text,
    size_bytes bigint,
    author     text,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL
);

-- Аудит просмотров и выгрузок
CREATE TABLE audit_log (
    id      bigserial PRIMARY KEY,
    ts      timestamptz NOT NULL DEFAULT now(),
    actor   text,
    action  text NOT NULL,
    details jsonb
);
CREATE INDEX audit_log_ts_idx ON audit_log (ts);
