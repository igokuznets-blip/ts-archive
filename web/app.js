"use strict";
/* Веб-интерфейс архива: синхронный просмотр до 4 камер и таймлайн с покрытием записи.
   Видео воспроизводится напрямую из исходных TS по HLS-плейлисту с byte-range (hls.js лежит
   в web/vendor — внешних запросов нет, работает в изолированной сети). */

const MAX_CAMS = 4;
const $ = (id) => document.getElementById(id);
const pad = (n, w = 2) => String(n).padStart(w, "0");
const fmtDT = (ms) => { const d = new Date(ms);
  return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };
const fmtT = (ms) => { const d = new Date(ms); return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };
const fmtDur = (s) => { s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60;
  return h ? `${h} ч ${pad(m)} мин` : m ? `${m} мин ${pad(x)} с` : `${x} с`; };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { let msg = r.statusText; try { msg = (await r.json()).detail || msg; } catch (_) {} throw new Error(msg); }
  return r.json();
}
function toast(html, ms = 6000) {
  const t = $("toast"); t.innerHTML = html; t.classList.remove("hidden");
  clearTimeout(toast._t); if (ms) toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}

/* ---------------------------------------------------------------- состояние */
const state = {
  vehicles: [], cams: new Map(),      // id → {id, code, name, vehicle}
  selected: [],                      // id камер
  t: Date.now(),                     // текущее абсолютное время, мс
  playing: false, speed: 1,
  view: { from: Date.now() - 3600e3, to: Date.now() },
  tl: new Map(),                     // camera_id → {segments, events}
  tlRange: null,
  sel: null,                         // {a, b} выделение для выгрузки
  onlyPeople: false, showBoxes: true,
  detKind: "person",                 // интервалы для переходов и списка: person | no_helmet
};
const boxCache = new Map();          // segment_id → Promise<{fps, frames}> рамки по кадрам
const segCache = new Map();          // segment_id → детали + карта фрагментов
async function getSegment(id) {
  if (!segCache.has(id)) segCache.set(id, api(`/api/segments/${id}`));
  return segCache.get(id);
}
/* карта фрагментов [offset, startMs, dur]: позиция плеера ↔ абсолютное время */
function offsetToWall(seg, off) {
  const f = seg.fragments; let lo = 0, hi = f.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (f[m][0] <= off) lo = m; else hi = m - 1; }
  const [o, st, d] = f[lo]; return st + Math.min(d, Math.max(0, off - o)) * 1000;
}
function wallToOffset(seg, ms) {
  const f = seg.fragments; let lo = 0, hi = f.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (f[m][1] <= ms) lo = m; else hi = m - 1; }
  const [o, st, d] = f[lo]; return o + Math.min(d, Math.max(0, (ms - st) / 1000));
}

/* ближайший размеченный кадр к позиции плеера (кадры анализа идут с шагом 1/fps) */
function nearestFrame(b, t) {
  const f = b.frames; if (!f.length) return null;
  let lo = 0, hi = f.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (f[m][0] < t) lo = m + 1; else hi = m; }
  const cand = [f[lo], f[lo - 1]].filter(Boolean).sort((x, y) => Math.abs(x[0] - t) - Math.abs(y[0] - t))[0];
  return Math.abs(cand[0] - t) <= 0.5 / b.fps + 0.05 ? cand : null;
}

/* ---------------------------------------------------------------- плееры */
const HLS_OK = typeof Hls !== "undefined" && Hls.isSupported();
const bufferFor = (speed) => Math.min(120, 20 * Math.max(1, speed));   // секунд видео вперёд

class Player {
  constructor(cam) {
    this.cam = cam; this.seg = null; this.exact = false; this.nextStart = null; this.busy = false; this.hls = null;
    this.el = document.createElement("div"); this.el.className = "tile";
    this.video = document.createElement("video"); this.video.muted = true; this.video.playsInline = true;
    this.video.preload = "auto";
    this.label = document.createElement("div"); this.label.className = "label";
    this.label.textContent = `${cam.vehicle.code} · ${cam.name || cam.code}`;
    this.norec = document.createElement("div"); this.norec.className = "norec"; this.norec.textContent = "Нет записи";
    this.overlay = document.createElement("canvas"); this.overlay.className = "overlay";
    this.badge = document.createElement("div"); this.badge.className = "badge hidden";
    this.el.append(this.video, this.overlay, this.label, this.badge, this.norec);
    this.boxes = null; this.boxesTried = 0;
    this.video.addEventListener("ended", () => this.onEnded());
    this.video.addEventListener("error", () => { if (!this.hls) toast(`Камера ${cam.code}: ошибка воспроизведения сегмента`); });
  }
  destroy() {
    if (this.hls) { this.hls.destroy(); this.hls = null; }
    this.video.pause(); this.video.removeAttribute("src"); this.video.load();
  }
  /* загрузка сегмента: исходные TS через hls.js (или нативный HLS в Safari), прокси MP4 — напрямую */
  load(seg, startAt) {
    if (this.hls) { this.hls.destroy(); this.hls = null; }
    const v = this.video;
    const ready = new Promise((res) => { v.addEventListener("loadedmetadata", res, { once: true });
      v.addEventListener("error", res, { once: true }); setTimeout(res, 15000); });
    if (seg.urls.hls && HLS_OK) {
      const h = new Hls({ startPosition: startAt, maxBufferLength: bufferFor(state.speed),
        maxMaxBufferLength: 2 * bufferFor(state.speed), backBufferLength: 30, enableWorker: true,
        lowLatencyMode: false });
      let netRetry = 0;
      h.on(Hls.Events.ERROR, (_, d) => {
        if (!d.fatal) return;
        if (d.type === Hls.ErrorTypes.NETWORK_ERROR && netRetry++ < 3) h.startLoad();
        else if (d.type === Hls.ErrorTypes.MEDIA_ERROR) h.recoverMediaError();
        else toast(`Камера ${this.cam.code}: ошибка воспроизведения (${d.details})`);
      });
      h.loadSource(seg.urls.hls); h.attachMedia(v); this.hls = h;
    } else if (seg.urls.hls && v.canPlayType("application/vnd.apple.mpegurl")) {
      v.src = seg.urls.hls;
    } else if (seg.urls.video) {
      v.src = seg.urls.video;
    } else {
      toast("Браузер не поддерживает HLS (MediaSource) — воспроизведение недоступно");
    }
    return ready;
  }
  setBuffer(speed) {
    if (this.hls) { this.hls.config.maxBufferLength = bufferFor(speed); this.hls.config.maxMaxBufferLength = 2 * bufferFor(speed); }
  }
  wall() { return this.seg && this.exact ? offsetToWall(this.seg, this.video.currentTime) : null; }
  /* рамки по кадрам: загружаются, когда сегмент размечен (если ещё нет — повторная проверка раз в 20 с) */
  async ensureBoxes() {
    const seg = this.seg; if (!seg || this.boxes || Date.now() - this.boxesTried < 20e3) return;
    this.boxesTried = Date.now();
    let url = seg.detect && seg.detect.boxes;
    if (!url) {
      const fresh = await api(`/api/segments/${seg.id}`).catch(() => null);
      if (!fresh || !fresh.detect || !fresh.detect.boxes) return;
      segCache.set(seg.id, Promise.resolve(fresh)); url = fresh.detect.boxes;
    }
    if (!boxCache.has(seg.id)) boxCache.set(seg.id, api(url).catch(() => { boxCache.delete(seg.id); return null; }));
    const b = await boxCache.get(seg.id);
    if (b && this.seg === seg) this.boxes = b;
  }
  drawBoxes() {
    const c = this.overlay, v = this.video, W = c.clientWidth, H = c.clientHeight;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const g = c.getContext("2d"); g.clearRect(0, 0, W, H);
    const fr = this.boxes && this.exact && state.showBoxes ? nearestFrame(this.boxes, v.currentTime) : null;
    const heads = fr && fr.length > 2 ? fr[2] : null;   // v3: детектор голов
    const n = !fr ? 0 : heads ? fr[3] : fr[1].length;
    const nh = !fr ? 0 : heads ? fr[4] : fr[1].filter((r) => r[7] === 2).length;
    this.badge.classList.toggle("hidden", !n);
    this.badge.classList.toggle("nh", nh > 0);
    if (!n || !v.videoWidth) return;
    this.badge.textContent = `Людей в кадре: ${n}` + (nh ? ` · без каски: ${nh}` : "");
    // область кадра внутри <video> при object-fit: contain
    const k = Math.min(W / v.videoWidth, H / v.videoHeight), vw = v.videoWidth * k, vh = v.videoHeight * k;
    const ox = (W - vw) / 2, oy = (H - vh) / 2;
    g.font = "12px system-ui, sans-serif"; g.textBaseline = "bottom";
    const placed = [];   // занятые подписями прямоугольники — следующая подпись сдвигается ниже
    const tag = (X, Y, t, col) => {
      const w = g.measureText(t).width + 8;
      for (let i = 0; i < 6 && placed.some((r) => X < r[0] + r[2] && X + w > r[0] && Y - 15 < r[1] && Y > r[1] - 15); i++) Y += 15;
      placed.push([X - 1, Y, w]);
      g.fillStyle = col; g.fillRect(X - 1, Y - 15, w, 15);
      g.fillStyle = "#000"; g.fillText(t, X + 3, Y - 1);
    };
    for (const [x1, y1, x2, y2, sc, code] of heads || []) {
      const X = ox + x1 * vw, Y = oy + y1 * vh, w = (x2 - x1) * vw, h = (y2 - y1) * vh;
      const col = code === 2 ? "#e040fb" : "#3ddc84";
      g.lineWidth = 2; g.strokeStyle = col; g.strokeRect(X, Y, w, h);
      tag(X, Y, (code === 2 ? "без каски " : "каска ") + Math.round(sc * 100) + "%", col);
    }
    for (const [x1, y1, x2, y2, sc, , zone, hel, pno] of fr[1]) {
      const X = ox + x1 * vw, Y = oy + y1 * vh, w = (x2 - x1) * vw, h = (y2 - y1) * vh;
      if (heads) {   // рамка человека тонкая, решение о каске — на рамке головы
        g.lineWidth = 1.5; g.strokeStyle = zone ? "#ff4d4d" : "#ffb020"; g.strokeRect(X, Y, w, h);
        tag(X, Y + h + 15 > oy + vh ? Y + h : Y + h + 15, `${Math.round(sc * 100)}%`, g.strokeStyle);
        continue;
      }
      // код каски: 2 — без каски, 1 — в каске, 0/нет — не определено
      g.lineWidth = 2;
      g.strokeStyle = hel === 2 ? "#e040fb" : zone ? "#ff4d4d" : hel === 1 ? "#3ddc84" : "#ffb020";
      g.strokeRect(X, Y, w, h);
      tag(X, Y, hel === 2 ? `без каски ${Math.round(pno * 100)}%` : hel === 1 ? `каска · ${Math.round(sc * 100)}%`
        : `${Math.round(sc * 100)}%`, g.strokeStyle);
    }
  }
  setNoRec(on, text) { this.norec.classList.toggle("hidden", !on); if (text) this.norec.textContent = text; }
  async seekTo(ms) {
    if (this.busy) return; this.busy = true;
    try {
      const s = await api(`/api/seek?camera_id=${this.cam.id}&ts=${new Date(ms).toISOString()}`).catch(() => null);
      if (!s) { this.exact = false; this.nextStart = null; this.setNoRec(true, "Нет записи"); this.video.pause(); return; }
      if (!s.exact) {
        this.exact = false; this.nextStart = Date.parse(s.segment_start);
        this.setNoRec(true, `Нет записи · следующая ${fmtDT(this.nextStart)}`); this.video.pause(); return;
      }
      const seg = await getSegment(s.segment_id);
      if (!this.seg || this.seg.id !== seg.id) {
        this.seg = seg; this.boxes = null; this.boxesTried = 0; this.ensureBoxes();
        await this.load(seg, wallToOffset(seg, ms));
      }
      this.exact = true; this.nextStart = null; this.setNoRec(false);
      this.video.currentTime = wallToOffset(seg, ms);
      this.video.playbackRate = state.speed;
      if (state.playing) this.video.play().catch(() => {}); else this.video.pause();
    } finally { this.busy = false; }
  }
  onEnded() {
    // сегмент закончился — продолжаем с ближайшей следующей записи
    if (!this.seg) return;
    const end = Date.parse(this.seg.end) + 1;
    this.exact = false; this.seekTo(Math.max(end, state.t));
  }
}
let players = [];

function rebuildPlayers() {
  const box = $("players");
  players.forEach((p) => p.destroy());
  box.innerHTML = "";
  players = state.selected.map((id) => new Player(state.cams.get(id)));
  box.className = "players" + (players.length ? ` n${players.length}` : " empty");
  if (!players.length) { box.innerHTML = '<div class="placeholder">Выберите технику или камеры слева</div>'; return; }
  players.forEach((p) => box.appendChild(p.el));
  seekAll(state.t);
}
function seekAll(ms) {
  state.t = ms; players.forEach((p) => p.seekTo(ms)); drawTimeline(); updateClock();
}
function setPlaying(on) {
  state.playing = on; $("btnPlay").textContent = on ? "❚❚" : "▶";
  players.forEach((p) => { if (on && p.exact) p.video.play().catch(() => {}); else p.video.pause(); });
  lastFrame = performance.now();
}

/* главный цикл: ведущая камера задаёт время, остальные подтягиваются */
let lastFrame = performance.now();
function tick(now) {
  const dt = now - lastFrame; lastFrame = now;
  for (const p of players) { p.drawBoxes(); if (!p.boxes && p.seg) p.ensureBoxes(); }
  if (state.playing && state.onlyPeople && players.length && !players.some((p) => p.busy)) skipToPeople();
  if (state.playing && players.length) {
    const master = players.find((p) => p.exact && !p.busy && !p.video.paused && p.video.readyState >= 2);
    if (master) {
      state.t = master.wall();
      for (const p of players) {
        if (p === master || p.busy) continue;
        if (p.exact) {
          const w = p.wall();
          if (Math.abs(w - state.t) > 1000 * Math.max(1, state.speed / 2)) p.seekTo(state.t);
          else if (p.video.paused) p.video.play().catch(() => {});
          if (state.t > Date.parse(p.seg.end)) p.seekTo(state.t);
        } else if (p.nextStart !== null && state.t >= p.nextStart) p.seekTo(state.t);
      }
    } else if (!players.some((p) => p.busy || (p.exact && p.video.readyState < 3 && !p.video.ended))) {
      // ни у одной камеры нет записи — перескакиваем разрыв к ближайшей записи
      const next = players.map((p) => p.nextStart).filter((x) => x !== null && x > state.t);
      if (next.length) seekAll(Math.min(...next));
      else if (players.every((p) => !p.exact)) { setPlaying(false); toast("Дальше записи нет"); }
      else state.t += dt * state.speed;
    }
    if (state.t > state.view.to - (state.view.to - state.view.from) * 0.05) {
      const w = state.view.to - state.view.from; setView(state.t - w * 0.2, state.t + w * 0.8);
    }
    updateClock(); drawTimeline();
  }
  requestAnimationFrame(tick);
}
function updateClock() { $("clock").textContent = fmtDT(state.t); }

/* ---------------------------------------------------------------- люди в кадре */
const PAD = 1500;   // запас до и после интервала при переходах и в режиме «только с людьми»
const KIND = {
  person: { btn: "люди", only: "только с людьми", title: "Люди в кадре", none: "людей не найдено", what: "интервалов с людьми" },
  no_helmet: { btn: "без каски", only: "только без каски", title: "Люди без каски", none: "людей без каски не найдено",
               what: "интервалов с людьми без каски" },
};
function allDetections(kind = state.detKind) {
  return state.selected.flatMap((id) => (state.tl.get(id) || { detections: [] }).detections
    .filter((d) => d.label === kind).map((d) => ({ ...d, cid: id }))).sort((x, y) => x.a - y.a);
}
function setDetKind(kind) {
  state.detKind = kind; const k = KIND[kind];
  $("btnPrevPerson").textContent = `◀ ${k.btn}`; $("btnNextPerson").textContent = `${k.btn} ▶`;
  for (const b of [$("btnPrevPerson"), $("btnNextPerson")]) b.classList.toggle("nh", kind === "no_helmet");
  $("onlyLabel").textContent = k.only; $("peopleTitle").textContent = k.title;
  renderPeople();
}
function skipToPeople() {
  const ds = allDetections();
  if (ds.some((d) => d.a - PAD <= state.t && state.t <= d.b + PAD)) return;
  const next = ds.find((d) => d.a - PAD > state.t);
  if (!next) { setPlaying(false); toast(`Дальше ${KIND[state.detKind].what} в загруженном диапазоне нет`); return; }
  seekAll(next.a - PAD);
}
function jumpPerson(dir) {
  const ds = allDetections();
  const d = dir > 0 ? ds.find((d) => d.a - PAD > state.t + 500) : ds.filter((d) => d.a - PAD < state.t - 2500).pop();
  if (!d) { toast(`Других ${KIND[state.detKind].what} в загруженном диапазоне нет`); return; }
  const w = state.view.to - state.view.from, t = d.a - PAD;
  if (t < state.view.from || t > state.view.to) setView(t - w / 3, t + 2 * w / 3);
  seekAll(t);
}
function renderPeople() {
  const box = $("people"), ds = allDetections().filter((d) => d.b >= state.view.from && d.a <= state.view.to).reverse();
  $("peopleCount").textContent = ds.length ? `${ds.length}` : "";
  box.innerHTML = "";
  if (!ds.length) { box.innerHTML = `<div class="hint">В видимом интервале ${KIND[state.detKind].none}` +
    (anyPendingDetect() ? " (часть записи ещё размечается)" : "") + "</div>"; return; }
  for (const d of ds.slice(0, 200)) {
    const cam = state.cams.get(d.cid), it = document.createElement("div"); it.className = "person";
    it.innerHTML = `<img loading="lazy" alt=""><div><b></b><span></span></div>`;
    const img = it.querySelector("img"); if (d.thumb) img.src = d.thumb; else img.style.visibility = "hidden";
    it.querySelector("b").textContent = `${fmtT(d.a)} – ${fmtT(d.b)}`;
    it.querySelector("span").textContent = `${cam ? cam.vehicle.code + " " + (cam.name || cam.code) : ""} · ` +
      (d.label === "no_helmet" ? `без каски: ${d.count}` : `людей: ${d.count}`) + (d.zones.length ? ` · ${d.zones.join(", ")}` : "");
    if (d.label === "no_helmet") it.classList.add("nohelmet"); else if (d.zones.length) it.classList.add("zone");
    it.onclick = () => seekAll(d.a - PAD);
    box.appendChild(it);
  }
}
function anyPendingDetect() {
  return state.selected.some((id) => (state.tl.get(id) || { segments: [] }).segments.some((s) => s.status === "done" && s.detect !== "done"));
}

/* ---------------------------------------------------------------- дерево техники */
async function loadVehicles() {
  state.vehicles = await api("/api/vehicles");
  state.cams.clear();
  for (const v of state.vehicles) for (const c of v.cameras) state.cams.set(c.id, { ...c, vehicle: v });
  renderTree();
}
function renderTree() {
  const q = $("filter").value.trim().toLowerCase();
  const tree = $("tree"); tree.innerHTML = "";
  let section = undefined;
  for (const v of state.vehicles) {
    const text = [v.code, v.name, v.type, v.pipeline_section, ...v.cameras.map((c) => c.code + " " + (c.name || ""))]
      .join(" ").toLowerCase();
    if (q && !text.includes(q)) continue;
    if (v.pipeline_section !== section) {
      section = v.pipeline_section;
      const s = document.createElement("div"); s.className = "section";
      s.textContent = section || "Участок не указан"; tree.appendChild(s);
    }
    const vh = document.createElement("div"); vh.className = "veh";
    vh.innerHTML = `<b></b><span class="type"></span>`;
    vh.querySelector("b").textContent = v.name ? `${v.code} — ${v.name}` : v.code;
    vh.querySelector(".type").textContent = v.type || "";
    vh.title = "Показать все камеры машины";
    vh.onclick = () => { state.selected = v.cameras.slice(0, MAX_CAMS).map((c) => c.id); onSelectionChanged(true); };
    tree.appendChild(vh);
    for (const c of v.cameras) {
      const row = document.createElement("label"); row.className = "cam";
      const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = state.selected.includes(c.id);
      cb.onchange = () => {
        if (cb.checked) { if (state.selected.length >= MAX_CAMS) { cb.checked = false; toast(`Не более ${MAX_CAMS} камер`); return; }
          state.selected.push(c.id); } else state.selected = state.selected.filter((x) => x !== c.id);
        onSelectionChanged(state.selected.length === 1);
      };
      const name = document.createElement("span"); name.textContent = c.name || c.code;
      const range = document.createElement("span"); range.className = "range";
      range.textContent = c.data_to ? `до ${fmtDT(Date.parse(c.data_to)).slice(0, 16)}` : "нет записей";
      row.append(cb, name, range); tree.appendChild(row);
    }
  }
}
function selectedRange() {
  const cams = state.selected.map((id) => state.cams.get(id)).filter((c) => c && c.data_from);
  if (!cams.length) return null;
  return { from: Math.min(...cams.map((c) => Date.parse(c.data_from))), to: Math.max(...cams.map((c) => Date.parse(c.data_to))) };
}
function onSelectionChanged(jump) {
  renderTree(); resizeCanvas();
  const r = selectedRange();
  if (jump && r) {
    const w = Math.min(Math.max(r.to - r.from, 600e3) * 1.05, 6 * 3600e3);
    const from = Math.max(r.from - 60e3, r.to - w);
    state.t = Math.max(r.from, from);
    setView(from, from + w);
  } else fetchTimeline(true);
  rebuildPlayers();
}

/* ---------------------------------------------------------------- таймлайн */
const cv = $("timeline"), ctx = cv.getContext("2d");
const RULER = 26, TRACK_H = 38, LABEL_W = 110;
const SEG_Y = 5, SEG_H = 17, DET_Y = 25, DET_H = 9;   // полоса записи и полоса «люди» внутри дорожки
function setView(from, to) {
  const minW = 30e3, maxW = 90 * 86400e3; let w = Math.min(maxW, Math.max(minW, to - from));
  const c = (from + to) / 2; state.view = { from: c - w / 2, to: c + w / 2 };
  drawTimeline(); fetchTimelineDebounced(); renderPeopleDebounced();
}
let fetchTimer = null, peopleTimer = null;
function renderPeopleDebounced() { clearTimeout(peopleTimer); peopleTimer = setTimeout(renderPeople, 250); }
function fetchTimelineDebounced() { clearTimeout(fetchTimer); fetchTimer = setTimeout(() => fetchTimeline(false), 200); }
async function fetchTimeline(force) {
  if (!state.selected.length) { state.tl.clear(); drawTimeline(); return; }
  const { from, to } = state.view, w = to - from;
  const r = state.tlRange;
  if (!force && r && r.from <= from && r.to >= to && r.key === state.selected.join(",") && Date.now() - r.at < 30e3) return;
  const qf = from - w, qt = to + w;
  const qs = state.selected.map((id) => `camera_id=${id}`).join("&");
  try {
    const data = await api(`/api/timeline?${qs}&from=${new Date(qf).toISOString()}&to=${new Date(qt).toISOString()}`);
    state.tl.clear();
    for (const c of data.cameras) state.tl.set(c.camera_id, {
      segments: c.segments.map((s) => ({ ...s, a: Date.parse(s.start), b: Date.parse(s.end) })),
      events: c.events.map((e) => ({ ...e, t: Date.parse(e.ts) })), pending: c.pending_fragments,
      detections: (c.detections || []).map((d) => ({ ...d, a: Date.parse(d.start), b: Date.parse(d.end),
        thumb: `/api/detections/${d.id}/thumb` })) });
    state.tlRange = { from: qf, to: qt, key: state.selected.join(","), at: Date.now() };
    drawTimeline(); renderPeople();
  } catch (e) { toast("Таймлайн: " + e.message); }
}
function resizeCanvas() {
  const h = RULER + Math.max(1, state.selected.length) * TRACK_H + 10;
  cv.style.height = h + "px";
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawTimeline();
}
const xOf = (ms) => LABEL_W + (ms - state.view.from) / (state.view.to - state.view.from) * (cv.clientWidth - LABEL_W);
const msOf = (x) => state.view.from + (x - LABEL_W) / (cv.clientWidth - LABEL_W) * (state.view.to - state.view.from);
const STEPS = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 3 * 3600, 6 * 3600, 12 * 3600, 86400, 2 * 86400, 7 * 86400].map((s) => s * 1000);
function drawTimeline() {
  const W = cv.clientWidth, H = cv.clientHeight; if (!W) return;
  const css = getComputedStyle(document.documentElement), col = (n) => css.getPropertyValue(n).trim();
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = col("--panel"); ctx.fillRect(0, 0, W, H);
  // шкала
  const span = state.view.to - state.view.from, pxPerMs = (W - LABEL_W) / span;
  const step = STEPS.find((s) => s * pxPerMs >= 90) || STEPS[STEPS.length - 1];
  const tzOff = new Date(state.view.from).getTimezoneOffset() * 60e3;
  let t0 = Math.ceil((state.view.from - tzOff) / step) * step + tzOff;
  ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "middle";
  for (let t = t0; t < state.view.to; t += step) {
    const x = xOf(t);
    ctx.fillStyle = col("--line"); ctx.fillRect(x, RULER - 6, 1, H - RULER + 6);
    ctx.fillStyle = col("--muted");
    const d = new Date(t);
    const label = step >= 86400e3 || (d.getHours() === 0 && d.getMinutes() === 0 && d.getSeconds() === 0)
      ? `${pad(d.getDate())}.${pad(d.getMonth() + 1)}` + (step < 86400e3 ? " 00:00" : "")
      : step < 60e3 ? fmtT(t) : `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    ctx.fillText(label, x + 3, RULER / 2);
  }
  // дорожки
  state.selected.forEach((cid, i) => {
    const y = RULER + i * TRACK_H, cam = state.cams.get(cid);
    ctx.fillStyle = i % 2 ? col("--panel") : col("--panel2"); ctx.fillRect(LABEL_W, y, W - LABEL_W, TRACK_H);
    ctx.fillStyle = col("--text"); ctx.textBaseline = "middle";
    ctx.fillText(((cam && (cam.vehicle.code + " " + (cam.name || cam.code))) || cid).slice(0, 16), 8, y + TRACK_H / 2);
    const tl = state.tl.get(cid); if (!tl) return;
    for (const s of tl.segments) {
      if (s.b < state.view.from || s.a > state.view.to) continue;
      const x1 = Math.max(LABEL_W, xOf(s.a)), x2 = Math.min(W, xOf(s.b));
      ctx.fillStyle = s.status === "done" ? col("--ok") : s.status === "error" ? col("--err") : col("--warn");
      ctx.fillRect(x1, y + SEG_Y, Math.max(1, x2 - x1), SEG_H);
      if (s.status === "done" && s.detect !== "done") {   // ещё не размечено — штриховка полосы «люди»
        ctx.fillStyle = col("--line"); ctx.fillRect(x1, y + DET_Y + 3, Math.max(1, x2 - x1), 2);
      }
    }
    for (const kind of ["person", "no_helmet"]) {   // «без каски» рисуется поверх интервалов людей
      for (const d of tl.detections) {
        if (d.label !== kind || d.b < state.view.from || d.a > state.view.to) continue;
        const x1 = Math.max(LABEL_W, xOf(d.a)), x2 = Math.min(W, xOf(d.b));
        ctx.fillStyle = kind === "no_helmet" ? col("--nohelmet") : d.zones.length ? col("--err") : col("--person");
        ctx.fillRect(x1, y + DET_Y, Math.max(2, x2 - x1), DET_H);
      }
    }
    ctx.fillStyle = col("--ev");
    for (const e of tl.events) { const x = xOf(e.t); if (x >= LABEL_W && x <= W) ctx.fillRect(x - 1, y + 2, 3, TRACK_H - 4); }
  });
  // выделение
  if (state.sel) {
    const a = xOf(Math.min(state.sel.a, state.sel.b)), b = xOf(Math.max(state.sel.a, state.sel.b));
    ctx.fillStyle = col("--sel"); ctx.fillRect(Math.max(LABEL_W, a), RULER, Math.min(W, b) - Math.max(LABEL_W, a), H - RULER);
  }
  // курсор
  const cx = xOf(state.t);
  if (cx >= LABEL_W && cx <= W) { ctx.fillStyle = col("--cursor"); ctx.fillRect(cx - 1, 0, 2, H); }
}

/* мышь: клик — перейти, протянуть — сдвиг, Shift+протянуть — выделение, колесо — масштаб */
let drag = null;
cv.addEventListener("mousedown", (e) => {
  const x = e.offsetX; if (x < LABEL_W) return;
  drag = { x0: x, view: { ...state.view }, moved: false, select: e.shiftKey, t0: msOf(x) };
  if (drag.select) state.sel = { a: drag.t0, b: drag.t0 };
});
window.addEventListener("mousemove", (e) => {
  const rect = cv.getBoundingClientRect(), x = e.clientX - rect.left;
  if (drag) {
    if (Math.abs(x - drag.x0) > 3) drag.moved = true;
    if (drag.select) { state.sel.b = msOf(x); updateSel(); drawTimeline(); }
    else if (drag.moved) {
      const dms = (x - drag.x0) / (cv.clientWidth - LABEL_W) * (drag.view.to - drag.view.from);
      state.view = { from: drag.view.from - dms, to: drag.view.to - dms }; drawTimeline(); fetchTimelineDebounced(); renderPeopleDebounced();
    }
  }
  if (e.target === cv) hover(e.offsetX, e.offsetY, e); else $("tooltip").classList.add("hidden");
});
window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  const d = drag; drag = null;
  if (d.select) { if (!d.moved) { state.sel = null; updateSel(); drawTimeline(); } return; }
  if (!d.moved) { state.sel = null; updateSel(); seekAll(d.t0); }
});
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const at = msOf(Math.max(LABEL_W, e.offsetX)), k = e.deltaY > 0 ? 1.25 : 0.8;
  const from = at - (at - state.view.from) * k, to = at + (state.view.to - at) * k;
  if (to - from >= 30e3 && to - from <= 90 * 86400e3) { state.view = { from, to }; drawTimeline(); fetchTimelineDebounced(); renderPeopleDebounced(); }
}, { passive: false });
cv.addEventListener("mouseleave", () => $("tooltip").classList.add("hidden"));

let hoverToken = 0;
async function hover(x, y, e) {
  const tt = $("tooltip"), thumb = $("ttThumb");
  if (x < LABEL_W) { tt.classList.add("hidden"); return; }
  const ms = msOf(x), i = Math.floor((y - RULER) / TRACK_H), cid = state.selected[i];
  $("ttText").textContent = fmtDT(ms);
  thumb.style.display = "none";
  tt.classList.remove("hidden");
  const wrap = cv.parentElement.getBoundingClientRect();
  tt.style.left = Math.min(e.clientX - wrap.left + 12, wrap.width - 190) + "px";
  tt.style.bottom = (wrap.bottom - e.clientY + 12) + "px";
  const tl = cid !== undefined && state.tl.get(cid);
  const yIn = y - RULER - i * TRACK_H, tol = 3 * (state.view.to - state.view.from) / (cv.clientWidth - LABEL_W);
  const hits = tl && yIn >= DET_Y - 2 ? tl.detections.filter((d) => d.a - tol <= ms && d.b + tol >= ms) : [];
  const det = hits.find((d) => d.label === "no_helmet") || hits[0];
  if (det) {
    $("ttText").textContent = (det.label === "no_helmet" ? `Без каски: ${det.count}` : `Людей: ${det.count}`) +
      ` · ${fmtT(det.a)}–${fmtT(det.b)} (${fmtDur((det.b - det.a) / 1000)})` +
      (det.zones.length ? ` · ${det.zones.join(", ")}` : "");
    ++hoverToken;
    thumb.style.display = "block"; thumb.style.width = "240px"; thumb.style.height = "135px";
    thumb.style.backgroundImage = `url(${det.thumb})`; thumb.style.backgroundSize = "contain";
    thumb.style.backgroundPosition = "center"; return;
  }
  thumb.style.backgroundSize = "";
  const s = tl && tl.segments.find((s) => s.a <= ms && s.b >= ms && s.status === "done");
  if (!s) return;
  $("ttText").textContent = `${fmtDT(ms)} · сегмент ${fmtDur(s.duration)}` + (s.gap_total_sec > 0.5 ? ` · пропуски ${s.gap_total_sec.toFixed(1)} с` : "");
  const token = ++hoverToken;
  const seg = await getSegment(s.id);
  if (token !== hoverToken || !seg.thumbs || !seg.urls.sprite) return;
  const th = seg.thumbs, idx = Math.min(th.count - 1, Math.floor(wallToOffset(seg, ms) / th.interval));
  thumb.style.display = "block"; thumb.style.width = th.w + "px"; thumb.style.height = th.h + "px";
  thumb.style.backgroundImage = `url(${seg.urls.sprite})`; thumb.style.backgroundSize = "";
  thumb.style.backgroundPosition = `-${(idx % th.cols) * th.w}px -${Math.floor(idx / th.cols) * th.h}px`;
}
function updateSel() {
  const s = state.sel, btn = $("btnExport");
  if (!s || Math.abs(s.b - s.a) < 1000) { $("selInfo").textContent = ""; btn.disabled = true; return; }
  const a = Math.min(s.a, s.b), b = Math.max(s.a, s.b);
  $("selInfo").textContent = `Выделено: ${fmtDT(a)} — ${fmtT(b)} (${fmtDur((b - a) / 1000)})`;
  btn.disabled = false;
}

/* ---------------------------------------------------------------- кнопки */
$("btnPlay").onclick = () => setPlaying(!state.playing);
$("btnBack").onclick = () => seekAll(state.t - 10e3);
$("btnFwd").onclick = () => seekAll(state.t + 10e3);
$("speed").onchange = (e) => { state.speed = +e.target.value; players.forEach((p) => { p.video.playbackRate = state.speed; p.setBuffer(state.speed); }); };
function jumpRecord(dir) {
  const all = state.selected.flatMap((id) => (state.tl.get(id) || { segments: [] }).segments.filter((s) => s.status === "done"));
  const cand = dir > 0 ? all.filter((s) => s.a > state.t + 1000).sort((x, y) => x.a - y.a)
                       : all.filter((s) => s.a < state.t - 2000).sort((x, y) => y.a - x.a);
  if (!cand.length) { toast("Других записей в загруженном диапазоне нет"); return; }
  const t = cand[0].a; const w = state.view.to - state.view.from;
  if (t < state.view.from || t > state.view.to) setView(t - w / 3, t + 2 * w / 3);
  seekAll(t);
}
$("btnNext").onclick = () => jumpRecord(1);
$("btnPrevPerson").onclick = () => jumpPerson(-1);
$("btnNextPerson").onclick = () => jumpPerson(1);
$("onlyPeople").onchange = (e) => { state.onlyPeople = e.target.checked; if (state.onlyPeople && state.playing) skipToPeople(); };
$("detKind").onchange = (e) => setDetKind(e.target.value);
$("showBoxes").onchange = (e) => { state.showBoxes = e.target.checked; };
$("btnPrev").onclick = () => jumpRecord(-1);
$("btnGoto").onclick = () => {
  const v = $("gotoTime").value; if (!v) return;
  const t = new Date(v).getTime(), w = state.view.to - state.view.from;
  setView(t - w / 2, t + w / 2); seekAll(t);
};
document.querySelectorAll("[data-zoom]").forEach((b) => (b.onclick = () => {
  const w = +b.dataset.zoom * 1000; setView(state.t - w / 2, state.t + w / 2);
}));
$("btnFit").onclick = () => { const r = selectedRange(); if (r) setView(r.from - (r.to - r.from) * 0.03, r.to + (r.to - r.from) * 0.03); };
$("btnMark").onclick = async () => {
  if (!state.selected.length) return;
  const note = prompt(`Метка на ${fmtDT(state.t)}. Комментарий:`); if (note === null) return;
  const cid = (players.find((p) => p.exact) || players[0]).cam.id;
  try { await api("/api/events", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ camera_id: cid, ts: new Date(state.t).toISOString(), kind: "mark", note }) });
    fetchTimeline(true); toast("Метка сохранена", 2500);
  } catch (e) { toast("Ошибка: " + e.message); }
};
$("btnExport").onclick = async () => {
  const s = state.sel; if (!s) return;
  const a = new Date(Math.min(s.a, s.b)).toISOString(), b = new Date(Math.max(s.a, s.b)).toISOString();
  $("btnExport").disabled = true; toast("Готовлю выгрузку…", 0);
  const links = [];
  for (const cid of state.selected) {
    const cam = state.cams.get(cid);
    try {
      const r = await api("/api/exports", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: cid, from: a, to: b }) });
      links.push(`<a href="${r.url}">${r.name}</a> (${(r.size_bytes / 1048576).toFixed(1)} МБ)`);
    } catch (e) { links.push(`${cam.vehicle.code} ${cam.code}: ${e.message}`); }
  }
  toast("Выгрузка готова:<br>" + links.join("<br>"), 0);
  $("btnExport").disabled = false;
};
$("toast").addEventListener("click", (e) => { if (e.target.tagName !== "A") $("toast").classList.add("hidden"); });
$("filter").oninput = renderTree;
window.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
  else if (e.code === "ArrowLeft") seekAll(state.t - (e.shiftKey ? 60e3 : 10e3));
  else if (e.code === "ArrowRight") seekAll(state.t + (e.shiftKey ? 60e3 : 10e3));
  else if (e.code === "KeyP") jumpPerson(e.shiftKey ? -1 : 1);
  else if (e.key === "+" || e.key === "=") { const c = state.t, w = (state.view.to - state.view.from) * 0.8; setView(c - w / 2, c + w / 2); }
  else if (e.key === "-") { const c = state.t, w = (state.view.to - state.view.from) * 1.25; setView(c - w / 2, c + w / 2); }
});
window.addEventListener("resize", resizeCanvas);

/* ---------------------------------------------------------------- статистика */
async function loadStats() {
  try {
    const s = await api("/api/stats");
    const q = (s.fragments.new || 0) + (s.fragments.probing || 0) + (s.fragments.probed || 0);
    const segQ = (s.segments.ready || 0) + (s.segments.processing || 0);
    const gb = (b) => (b / 1073741824).toFixed(1);
    $("stats").textContent = `Сегментов: ${s.segments.done || 0} · очередь: ${q} фрагм. / ${segQ} сегм.` +
      ` · входящих: ${s.incoming_files} · карантин: ${s.fragments.quarantine || 0}` +
      (s.segments.error ? ` · ошибок: ${s.segments.error}` : "") +
      (s.detect ? ` · разметка: ${s.detect.segments.done || 0}/${Object.values(s.detect.segments).reduce((a, b) => a + b, 0)}` +
        (s.detect.realtime_factor_24h ? ` (×${s.detect.realtime_factor_24h})` : "") : "") +
      ` · свободно ${gb(s.disk.free)} ГБ`;
  } catch (_) { $("stats").textContent = "API недоступен"; }
}

(async function init() {
  resizeCanvas();
  await loadVehicles().catch((e) => toast("Не удалось загрузить список техники: " + e.message));
  loadStats(); setInterval(loadStats, 15000);
  setInterval(() => { loadVehicles().catch(() => {}); if (state.selected.length) fetchTimeline(true); }, 60000);
  if (state.vehicles.length) { state.selected = state.vehicles[0].cameras.slice(0, MAX_CAMS).map((c) => c.id); onSelectionChanged(true); }
  requestAnimationFrame(tick);
})();
