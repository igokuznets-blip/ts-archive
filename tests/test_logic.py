"""Тесты логики без БД и FFmpeg: pytest -q tests/"""
from datetime import datetime, timedelta, timezone

from tsarchive import grouper, ingest, media

T0 = datetime(2026, 1, 15, 7, 0, tzinfo=timezone.utc)


def frag(i, start, dur=1.0, sig="v:h264:640x360"):
    s = T0 + timedelta(seconds=start)
    return {"id": i, "ts_start": s, "ts_end": s + timedelta(seconds=dur), "duration": dur, "codec_sig": sig}


def test_split_by_gap_codec_and_length(monkeypatch):
    monkeypatch.setattr(grouper, "settings", type("S", (), {
        "gap_threshold_sec": 3.0, "overlap_tolerance_sec": 1.0, "segment_max_sec": 10.0})())
    frags = [frag(i, i) for i in range(5)]                     # 0..4 непрерывно
    frags += [frag(10 + i, 60 + i) for i in range(3)]          # разрыв 56 с
    frags += [frag(20, 63, sig="v:h264:1920x1080")]            # смена разрешения
    frags += [frag(30 + i, 64 + i) for i in range(12)]         # длиннее SEGMENT_MAX_SEC
    groups = grouper._split(frags)
    sizes = [(len(g), closed) for g, closed in groups]
    assert sizes == [(5, True), (3, True), (1, True), (10, True), (2, False)]


def test_concat_durations_use_continuous_pts():
    # аудио сдвигает start_time фрагмента на ±1 аудиокадр — берём разницу PTS
    starts = [10.0, 11.024, 11.984, 13.008, None]
    durs = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert [round(x, 3) for x in media.concat_durations(starts, durs)] == [1.024, 0.96, 1.024, 1.0, 1.0]


def test_concat_durations_reset_and_wrap():
    wrap = media.PTS_WRAP
    starts = [100.0, 5.0, wrap - 0.5, 0.5]       # рестарт регистратора, затем переполнение PTS
    out = media.concat_durations(starts, [1.0, 1.0, 1.0, 1.0])
    assert out[0] == 1.0                          # рестарт — длительность видео
    assert abs(out[2] - 1.0) < 1e-6               # переход через 2^33 учтён


def test_filename_parsing():
    v, c, ts = ingest.parse_rel("PT-101/cam1/2026/01/15/20260115_100001_250.ts")
    assert (v, c) == ("PT-101", "cam1")
    assert ts.astimezone(timezone.utc) == datetime(2026, 1, 15, 7, 0, 1, 250000, tzinfo=timezone.utc)
    assert ingest.parse_rel("garbage.ts") is None
    assert ingest.parse_rel("EX-1/boom/20260115T235959.ts")[2].hour == 23


def test_hls_chunks_cover_file_exactly():
    from tsarchive import hls
    kf_t = [0.064 + 2 * i for i in range(80)]
    kf_pos = [188] + [1_000_000 * i - 376 for i in range(1, 80)]
    ch = hls.fragment_chunks(160.0, 83_000_000, kf_t, kf_pos, 6.0)
    assert ch[0][2] == 0                                   # первый кусок с начала файла (PAT/PMT)
    assert abs(sum(d for d, _, _ in ch) - 160.0) < 1e-6    # шкала совпадает с картой фрагментов
    assert sum(n for _, n, _ in ch) == 83_000_000          # байты без пропусков и перекрытий
    assert all(ch[i][2] + ch[i][1] == ch[i + 1][2] for i in range(len(ch) - 1))
    assert all(d >= 6.0 - 1e-6 for d, _, _ in ch[:-1])


def test_hls_playlist_discontinuity_only_on_pts_reset():
    from tsarchive import hls
    f = lambda i, off, pts: {"id": i, "off": off, "size_bytes": 1000, "start_pts": pts, "kf_t": [], "kf_pos": []}
    pl = hls.build_playlist([f(1, 0, 1.4), f(2, 160, 161.4), f(3, 320, 5.0)], 400,
                            lambda x: f"/api/fragments/{x['id']}/media", 6)
    assert pl.count("#EXT-X-DISCONTINUITY") == 1
    assert pl.index("#EXT-X-DISCONTINUITY") > pl.index("/api/fragments/2/media")
    assert "#EXT-X-BYTERANGE:1000@0" in pl and pl.rstrip().endswith("#EXT-X-ENDLIST")
    assert "#EXT-X-TARGETDURATION:160" in pl
