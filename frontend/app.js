/* Shroud front end. Vanilla JS, no build step. Talks to the API described in backend/contract.md. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const G = window.Guilloche;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const INK = css("--ink"), GREEN = css("--green"), PAPER = css("--paper"), GRAPHITE = css("--graphite");
  const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const clamp01 = (v) => Math.max(0, Math.min(1, v));

  const state = { strength: "balanced", decoy: null, busy: false, job: null, lastImageJob: null, lab: false, raf: 0 };

  /* ——— static guilloché: logo mark, hero rosette, progress pattern ——— */
  G.draw($("mark"), G.rosette.mark());
  G.draw($("hero-rosette"), G.rosette.hero());
  const stagePaths = G.draw($("stage-guilloche"), G.rosette.full());
  function setPattern(progress) {
    const n = stagePaths.length;
    stagePaths.forEach((p, i) => { p.style.strokeDashoffset = String(1 - clamp01(progress * n - i)); });
  }

  /* ——— helpers ——— */
  function mockSeen(data) { if (data && data.mock) $("mock-banner").hidden = false; return data; }
  async function api(path, opts) {
    let res;
    try { res = await fetch(path, opts); } catch (e) { throw new Error("Can't reach the Shroud server. Is it still running?"); }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "The server refused that file.");
    return mockSeen(data);
  }
  // Numbers count between values instead of jumping.
  function countTo(el, to, ms = 500, digits = 2) {
    const from = parseFloat(el.dataset.v ?? "NaN");
    el.dataset.v = to;
    if (reduced || Number.isNaN(from) || ms === 0) { el.textContent = fmt(to, digits); return; }
    const t0 = performance.now();
    cancelAnimationFrame(el._raf);
    (function tick(now) {
      const k = clamp01((now - t0) / ms), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(from + (to - from) * e, digits);
      if (k < 1) el._raf = requestAnimationFrame(tick);
    })(t0);
  }
  function kindOf(file) {
    const t = file.type || "", ext = (file.name.split(".").pop() || "").toLowerCase();
    if (t.startsWith("image/") || ["jpg", "jpeg", "png", "webp", "bmp", "heic"].includes(ext)) return "image";
    if (t.startsWith("audio/") || ["wav", "mp3", "m4a", "ogg", "flac", "opus", "aac"].includes(ext)) return "audio";
    if (t.startsWith("video/") || ["mp4", "mov", "webm", "mkv", "avi"].includes(ext)) return "video";
    return null;
  }

  /* ——— health ——— */
  api("/api/health").then((h) => {
    state.lab = !!h.lab;
    $("status").textContent = `Local · ${h.device}`;
    $("foot-models").textContent = (h.models || []).join(" · ");
    refreshLab();
  }).catch(() => { $("status").textContent = "Server offline"; $("status").classList.add("down"); });

  /* ——— controls ——— */
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    state.strength = b.dataset.strength;
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  }));
  $("decoy").addEventListener("change", (e) => {
    state.decoy = e.target.files[0] || null;
    $("decoy-name").textContent = state.decoy ? state.decoy.name : "Choose a face or voice to point at";
    $("decoy-clear").hidden = !state.decoy;
  });
  $("decoy-clear").addEventListener("click", () => { $("decoy").value = ""; $("decoy").dispatchEvent(new Event("change")); });

  const drop = $("drop");
  $("file").addEventListener("change", (e) => { if (e.target.files[0]) start(e.target.files[0]); e.target.value = ""; });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) start(f); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  /* ——— record voice ——— */
  let rec = null;
  $("btn-record").addEventListener("click", async () => {
    const btn = $("btn-record");
    if (rec) { rec.stop(); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { return fail("Microphone access was refused. Allow it in the browser, or drop an audio file instead."); }
    const chunks = [], t0 = Date.now();
    rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => chunks.push(e.data);
    const timer = setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000);
      btn.textContent = `Stop recording · 0:${String(s).padStart(2, "0")}`;
      if (s >= 20) rec.stop();
    }, 250);
    rec.onstop = () => {
      clearInterval(timer); stream.getTracks().forEach((t) => t.stop());
      const type = rec.mimeType || "audio/webm"; rec = null;
      btn.textContent = "Record voice"; btn.classList.remove("live");
      start(new File(chunks, "voice-note." + (type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm"), { type }), "audio");
    };
    rec.start(); btn.classList.add("live"); btn.textContent = "Stop recording · 0:00";
  });

  /* ——— webcam ——— */
  let camStream = null;
  function closeCam() { if (camStream) camStream.getTracks().forEach((t) => t.stop()); camStream = null; $("webcam").hidden = true; }
  $("btn-webcam").addEventListener("click", async () => {
    if (camStream) return closeCam();
    try { camStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } }); }
    catch (e) { return fail("Camera access was refused. Allow it in the browser, or drop a photo instead."); }
    $("webcam-video").srcObject = camStream; $("webcam").hidden = false; $("error").hidden = true;
  });
  $("webcam-cancel").addEventListener("click", closeCam);
  $("webcam-snap").addEventListener("click", () => {
    const v = $("webcam-video"), c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((b) => { closeCam(); start(new File([b], "webcam.png", { type: "image/png" }), "image"); }, "image/png");
  });

  /* ——— run a job ——— */
  function fail(message) {
    state.busy = false;
    $("stage").hidden = true; $("intake").hidden = false;
    $("error-text").textContent = message; $("error").hidden = false;
  }
  $("error-retry").addEventListener("click", () => { $("error").hidden = true; $("file").click(); });

  async function start(file, forcedKind) {
    if (state.busy) return;
    const kind = forcedKind || kindOf(file);
    if (!kind) return fail("That file type isn't supported. Use a photo, an audio recording or a video.");
    state.busy = true; closeCam();
    $("error").hidden = true; $("intake").hidden = true; $("stage").hidden = false;
    setPattern(0);
    requestAnimationFrame(() => $("stage").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "end" }));
    const sim = $("ro-sim"); delete sim.dataset.v; sim.textContent = "—"; sim.parentElement.classList.remove("below");
    $("ro-phase").textContent = "Sending"; $("ro-trail").textContent = "—"; $("ro-step").textContent = "—"; $("ro-thr").textContent = "—"; $("ro-elapsed").textContent = "0.0 s";
    preview(file, kind);

    const body = new FormData();
    body.append("file", file, file.name); body.append("strength", state.strength);
    if (state.decoy) body.append("decoy", state.decoy, state.decoy.name);
    try {
      const { job_id } = await api(`/api/protect/${kind}`, { method: "POST", body });
      await poll(job_id);
    } catch (e) { fail(e.message); }
  }

  function preview(file, kind) {
    const box = $("stage-preview"), url = URL.createObjectURL(file);
    box.replaceChildren();
    if (kind === "image") { const el = new Image(); el.src = url; el.alt = ""; box.appendChild(el); }
    else if (kind === "video") { const el = document.createElement("video"); Object.assign(el, { src: url, muted: true, loop: true, autoplay: true, playsInline: true }); box.appendChild(el); }
    else { const c = document.createElement("canvas"); box.appendChild(c); peaksFrom(file).then((p) => drawWave(c, p, { step: 4 })).catch(() => {}); }
  }

  const PHASE = { queued: "Queued", preparing: "Reading the original", shielding: "Shielding", encoding: "Encoding", verifying: "Checking the result", done: "Done" };
  async function poll(id) {
    const trail = [];
    for (;;) {
      const j = await api(`/api/jobs/${id}`);
      if (j.status === "error") return fail(j.error || "Something went wrong while shielding that file.");
      const s = j.scores || {}, cur = j.current_similarity;
      $("ro-phase").textContent = PHASE[j.phase] || j.phase;
      $("ro-elapsed").textContent = `${fmt(j.elapsed_s, 1)} s`;
      if (s.threshold != null) $("ro-thr").textContent = fmt(s.threshold);
      if (j.total_steps) $("ro-step").textContent = `${j.step}/${j.total_steps}`;
      if (s.before != null && !trail.length) trail.push(s.before);
      if (cur != null) {
        countTo($("ro-sim"), cur, 280);
        $("ro-sim").parentElement.classList.toggle("below", s.threshold != null && cur < s.threshold);
        if (trail.length && trail[trail.length - 1] - cur >= 0.14) trail.push(cur);   // keep the big drops
        const mid = trail.length > 1 ? [trail[trail.length - 1]] : [];   // start → last big drop → now
        const shown = [trail[0], ...mid].concat(Math.abs(trail[trail.length - 1] - cur) > 0.005 ? [cur] : []);
        $("ro-trail").textContent = shown.map((v) => fmt(v)).join(" → ");
      }
      setPattern(j.progress || 0);
      if (j.status === "done") { setPattern(1); await new Promise((r) => setTimeout(r, reduced ? 0 : 450)); return finish(j); }
      await new Promise((r) => setTimeout(r, 300));
    }
  }

  /* ——— certificate ——— */
  const certRO = new ResizeObserver(() => {
    const c = $("cert"); if (c.hidden) return;
    G.frame($("cert-border"), c.clientWidth, c.clientHeight, parseFloat(getComputedStyle(c).paddingLeft) - 4);
  });
  certRO.observe($("cert"));

  function row(label, value) {
    const d = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = label; dd.textContent = value; d.append(dt, dd); return d;
  }

  function finish(j) {
    state.busy = false; state.job = j;
    cancelAnimationFrame(state.raf);
    $("stage").hidden = true; $("intake").hidden = false;
    $("result-empty").hidden = true; $("cert").hidden = false;
    const s = j.scores, ok = s.after < s.threshold;

    $("c-serial").textContent = "No. " + String(j.serial || 0).padStart(6, "0");
    const now = new Date();
    $("c-date").textContent = now.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }) + " " + now.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
    $("c-kind").textContent = { image: "Photograph", audio: "Voice recording", video: "Video" }[j.kind];
    $("c-strength").textContent = j.strength;

    // headline numerals
    const before = $("c-before"), after = $("c-after");
    before.textContent = fmt(s.before); delete after.dataset.v; after.dataset.v = s.before;
    after.classList.toggle("fail", !ok);
    countTo(after, s.after, 900);

    // scale bar 0..1 with the threshold tick
    const bar = document.querySelector(".scale-bar");
    bar.querySelectorAll(".scale-tick").forEach((t) => t.remove());
    for (let i = 0; i <= 10; i++) { const t = document.createElement("i"); t.className = "scale-tick"; t.style.left = i * 10 + "%"; bar.appendChild(t); }
    $("scale-thr").style.left = s.threshold * 100 + "%";
    $("scale-before").style.left = clamp01(s.before) * 100 + "%";
    const ma = $("scale-after"); ma.classList.toggle("fail", !ok);
    ma.style.left = clamp01(s.before) * 100 + "%";
    requestAnimationFrame(() => requestAnimationFrame(() => { ma.style.left = clamp01(s.after) * 100 + "%"; }));

    const v = $("verdict"); v.classList.toggle("fail", !ok);
    v.textContent = ok ? "A deepfake tool would read this as someone else." : "Still recognizable — try Strong.";

    // passport-style fields
    const f = $("c-fields"); f.replaceChildren();
    f.append(row("Threshold", `below ${fmt(s.threshold)}`));
    const q = s.quality || {};
    if (q.snr_db != null) f.append(row("Quality", `SNR ${fmt(q.snr_db, 1)} dB`));
    if (q.psnr_db != null) f.append(row("Quality", `PSNR ${fmt(q.psnr_db, 1)} dB · SSIM ${fmt(q.ssim, 3)}`));
    for (const [name, e] of Object.entries(s.per_encoder || {})) f.append(row(name, `${fmt(e.before)} → ${fmt(e.after)}  (thr ${fmt(e.threshold)})`));
    if (s.after_jpeg_q90 != null) f.append(row("After JPEG q90", fmt(s.after_jpeg_q90)));
    if (s.frames != null) f.append(row("Frames", `${s.frames} · face found in ${s.faces}`));
    else if (s.faces != null) f.append(row("Faces", String(s.faces)));
    if (s.audio) f.append(row("Audio track", `${fmt(s.audio.before)} → ${fmt(s.audio.after)} · SNR ${fmt(s.audio.quality.snr_db, 1)} dB`));
    if (s.epsilon != null) f.append(row("Largest change", j.kind === "audio" ? `${(s.epsilon * 100).toFixed(2)}% of full scale` : `${Math.round(s.epsilon * 255)} of 255 levels`));
    f.append(row("Elapsed", `${fmt(j.elapsed_s, 1)} s`));

    const w = $("c-warnings"); w.replaceChildren();
    (j.warnings || []).forEach((t) => { const li = document.createElement("li"); li.textContent = t; w.appendChild(li); });

    const stamp = $("stamp"); stamp.classList.remove("on"); void stamp.offsetWidth; if (ok) stamp.classList.add("on");

    const ext = j.result_url.split(".").pop();
    $("download").href = j.result_url; $("download").download = `shielded-${String(j.serial || 0).padStart(6, "0")}.${ext}`;

    // media
    $("compare").hidden = j.kind === "audio"; $("media-actions").hidden = j.kind === "audio";
    $("waves").hidden = j.kind !== "audio"; $("spark").hidden = j.kind !== "video";
    $("btn-video-play").hidden = j.kind !== "video";
    setShieldView(false);
    if (j.kind === "audio") showWaves(j); else showCompare(j);
    if (j.kind === "video") sparkline(s.per_frame || [], s.threshold);
    if (j.kind === "image") state.lastImageJob = j;
    refreshLab();
    $("result").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  }
  $("again").addEventListener("click", () => { $("shield").scrollIntoView({ behavior: reduced ? "auto" : "smooth" }); });

  /* ——— before/after slider ——— */
  const compare = $("compare");
  let mediaA = null, mediaB = null;   // A = original (top layer, left of the handle), B = shielded
  function showCompare(j) {
    const make = (url) => {
      if (j.kind === "image") { const el = new Image(); el.src = url; el.alt = ""; el.draggable = false; return el; }
      const el = document.createElement("video"); Object.assign(el, { src: url, muted: true, loop: true, autoplay: true, playsInline: true }); return el;
    };
    mediaA = make(j.original_url); mediaB = make(j.result_url);
    $("compare-before").replaceChildren(mediaA); $("compare-after").replaceChildren(mediaB);
    setSplit(50);
    if (j.kind === "video") {
      $("btn-video-play").textContent = "Pause";
      mediaB.addEventListener("timeupdate", () => { if (Math.abs(mediaA.currentTime - mediaB.currentTime) > 0.12) mediaA.currentTime = mediaB.currentTime; });
    }
  }
  function setSplit(pct) {
    pct = Math.max(0, Math.min(100, pct));
    compare.style.setProperty("--split", pct + "%"); $("compare-handle").setAttribute("aria-valuenow", Math.round(pct));
  }
  function dragTo(e) { const r = compare.getBoundingClientRect(); setSplit(((e.clientX - r.left) / r.width) * 100); }
  compare.addEventListener("pointerdown", (e) => { compare.setPointerCapture(e.pointerId); dragTo(e); });
  compare.addEventListener("pointermove", (e) => { if (compare.hasPointerCapture(e.pointerId)) dragTo(e); });
  $("compare-handle").addEventListener("keydown", (e) => {
    const cur = parseFloat(compare.style.getPropertyValue("--split")) || 50;
    if (e.key === "ArrowLeft") setSplit(cur - 4); if (e.key === "ArrowRight") setSplit(cur + 4);
  });
  $("btn-video-play").addEventListener("click", (e) => {
    if (!mediaB || !mediaB.play) return;
    if (mediaB.paused) { mediaA.play(); mediaB.play(); e.target.textContent = "Pause"; } else { mediaA.pause(); mediaB.pause(); e.target.textContent = "Play"; }
  });

  /* ——— "Show the shield": |original − shielded| × 20, printed as green ink. Computed here, in the browser. ——— */
  const gRGB = [parseInt(GREEN.slice(1, 3), 16), parseInt(GREEN.slice(3, 5), 16), parseInt(GREEN.slice(5, 7), 16)];
  const pRGB = [parseInt(PAPER.slice(1, 3), 16), parseInt(PAPER.slice(3, 5), 16), parseInt(PAPER.slice(5, 7), 16)];
  const scratch = [document.createElement("canvas"), document.createElement("canvas")];
  function renderShield() {
    const w0 = mediaB.naturalWidth || mediaB.videoWidth, h0 = mediaB.naturalHeight || mediaB.videoHeight;
    if (!w0) return;
    const k = Math.min(1, 1400 / Math.max(w0, h0)), w = Math.round(w0 * k), h = Math.round(h0 * k);
    const out = $("shield-canvas"); out.width = w; out.height = h;
    const px = [mediaA, mediaB].map((m, i) => {
      const c = scratch[i]; c.width = w; c.height = h;
      const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(m, 0, 0, w, h); return x.getImageData(0, 0, w, h).data;
    });
    const ctx = out.getContext("2d"), img = ctx.createImageData(w, h), o = img.data, [a, b] = px;
    for (let i = 0; i < o.length; i += 4) {
      const d = (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])) / 3;
      const t = Math.min(1, (d * 20) / 255);
      o[i] = pRGB[0] + (gRGB[0] - pRGB[0]) * t; o[i + 1] = pRGB[1] + (gRGB[1] - pRGB[1]) * t; o[i + 2] = pRGB[2] + (gRGB[2] - pRGB[2]) * t; o[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  function setShieldView(on) {
    const btn = $("btn-shield-view");
    btn.setAttribute("aria-pressed", String(on)); btn.textContent = on ? "Hide the shield" : "Show the shield";
    $("shield-canvas").hidden = !on; $("shield-note").hidden = !on; compare.classList.toggle("shield-on", on);
    cancelAnimationFrame(state.raf);
    if (!on || !mediaB) return;
    try { renderShield(); } catch (e) { /* media not decoded yet */ }
    if (mediaB.tagName === "VIDEO") (function loop() { try { renderShield(); } catch (e) {} state.raf = requestAnimationFrame(loop); })();
  }
  $("btn-shield-view").addEventListener("click", (e) => setShieldView(e.target.getAttribute("aria-pressed") !== "true"));

  /* ——— waveforms: thin vertical ink lines ——— */
  async function peaksFrom(src) {
    const buf = src instanceof Blob ? await src.arrayBuffer() : await (await fetch(src)).arrayBuffer();
    const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const audio = await new Ctx(1, 1, 44100).decodeAudioData(buf);
    const d = audio.getChannelData(0), N = 400, per = Math.max(1, Math.floor(d.length / N)), peaks = new Float32Array(N);
    for (let i = 0; i < N; i++) { let m = 0; for (let k = i * per, e = Math.min(d.length, k + per); k < e; k++) { const v = Math.abs(d[k]); if (v > m) m = v; } peaks[i] = m; }
    return peaks;
  }
  function drawWave(canvas, peaks, { step = 3, guilloche = false, playhead = null, scale = null } = {}) {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const c = canvas.getContext("2d"); c.scale(dpr, dpr); c.clearRect(0, 0, w, h);
    if (guilloche) {   // faint woven band behind the protected waveform
      c.strokeStyle = GREEN; c.globalAlpha = 0.38; c.lineWidth = 0.6;
      for (let k = 0; k < 7; k++) {
        c.beginPath();
        for (let x = 0; x <= w; x += 2) { const y = h / 2 + (h / 2 - 2) * Math.sin((x / w) * Math.PI * 2 * (w / 90) + (Math.PI * 2 * k) / 7) * (0.55 + 0.45 * Math.cos((x / w) * Math.PI * 2 * 3)); x ? c.lineTo(x, y) : c.moveTo(x, y); }
        c.stroke();
      }
      c.globalAlpha = 1;
    }
    let max = scale; if (!max) { max = 0.05; for (const p of peaks) if (p > max) max = p; }
    c.strokeStyle = INK; c.lineWidth = 1; c.beginPath();
    const cols = Math.floor(w / step);
    for (let i = 0; i < cols; i++) {
      const p = peaks[Math.floor((i / cols) * peaks.length)] / max, x = i * step + 0.5, half = Math.max(0.5, (p * (h - 8)) / 2);
      c.moveTo(x, h / 2 - half); c.lineTo(x, h / 2 + half);
    }
    c.stroke();
    if (playhead != null) { c.strokeStyle = GREEN; c.lineWidth = 1.5; c.beginPath(); c.moveTo(playhead * w, 0); c.lineTo(playhead * w, h); c.stroke(); }
    return max;
  }
  const wave = { orig: null, prot: null, scale: null };
  function redrawWaves() {
    const a = $("audio-orig"), b = $("audio-prot");
    if (wave.orig) wave.scale = drawWave($("wave-orig"), wave.orig, { scale: wave.scale, playhead: a.currentTime && a.duration ? a.currentTime / a.duration : null });
    if (wave.prot) drawWave($("wave-prot"), wave.prot, { guilloche: true, scale: wave.scale, playhead: b.currentTime && b.duration ? b.currentTime / b.duration : null });
  }
  async function showWaves(j) {
    $("audio-orig").src = j.original_url; $("audio-prot").src = j.result_url;
    wave.orig = wave.prot = wave.scale = null;
    document.querySelectorAll("[data-play]").forEach((b) => (b.textContent = "Play"));
    try { [wave.orig, wave.prot] = await Promise.all([peaksFrom(j.original_url), peaksFrom(j.result_url)]); redrawWaves(); } catch (e) { /* browser can't decode: players still work */ }
  }
  document.querySelectorAll("[data-play]").forEach((btn) => {
    const el = $(btn.dataset.play === "orig" ? "audio-orig" : "audio-prot"), other = $(btn.dataset.play === "orig" ? "audio-prot" : "audio-orig");
    btn.addEventListener("click", () => { if (el.paused) { other.pause(); el.play(); } else el.pause(); });
    el.addEventListener("play", () => (btn.textContent = "Pause"));
    ["pause", "ended"].forEach((t) => el.addEventListener(t, () => { btn.textContent = "Play"; redrawWaves(); }));
    el.addEventListener("timeupdate", redrawWaves);
  });
  window.addEventListener("resize", redrawWaves);

  /* ——— per-frame sparkline ——— */
  function sparkline(frames, thr) {
    const svg = $("spark-svg"), W = 600, H = 96, NS = "http://www.w3.org/2000/svg";
    svg.replaceChildren();
    const seen = frames.filter((v) => v != null), top = Math.max(2 * thr, ...seen.map((v) => v * 1.15));
    const y = (v) => (H - 2) * (1 - clamp01(v / top)) + 1;
    const line = (x1, y1, x2, y2, stroke, dash) => { const l = document.createElementNS(NS, "line"); Object.entries({ x1, y1, x2, y2, stroke, "stroke-width": 1, "vector-effect": "non-scaling-stroke" }).forEach(([k, v]) => l.setAttribute(k, v)); if (dash) l.setAttribute("stroke-dasharray", dash); svg.appendChild(l); };
    line(0, H, W, H, css("--rule")); line(0, y(thr), W, y(thr), INK, "4 4");
    let d = "", pen = false;                      // gaps where no face was found in the decoded frame
    frames.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + ((i / Math.max(1, frames.length - 1)) * W).toFixed(1) + " " + y(v).toFixed(1); pen = true; });
    const p = document.createElementNS(NS, "path");
    Object.entries({ d, fill: "none", stroke: GREEN, "stroke-width": 1.25, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }).forEach(([k, v]) => p.setAttribute(k, v));
    svg.appendChild(p);
    $("spark-thr").textContent = `Dashed: threshold ${fmt(thr)}`;
    $("spark-last").textContent = `Frame ${frames.length} · peak ${seen.length ? fmt(Math.max(...seen)) : "—"}`;
  }

  /* ——— attack lab ——— */
  function refreshLab() {
    const btn = $("lab-run"), note = $("lab-note");
    if (!state.lab) { btn.disabled = true; note.textContent = "Face swapper not installed on this server"; return; }
    btn.disabled = !state.lastImageJob; note.textContent = state.lastImageJob ? "Runs locally, a few seconds" : "Shield a photo first";
  }
  $("lab-run").addEventListener("click", async () => {
    const btn = $("lab-run"), note = $("lab-note"); btn.disabled = true; note.textContent = "Swapping faces";
    try {
      const body = new FormData(); body.append("job_id", state.lastImageJob.job_id);
      const r = await api("/api/lab/faceswap", { method: "POST", body });
      $("lab-target").src = r.target_url; $("lab-a").src = r.swap_original_url; $("lab-b").src = r.swap_shielded_url;
      const put = (el, v) => { el.textContent = `Match to you ${fmt(v)} · ${v >= r.threshold ? "reads as you" : "reads as someone else"}`; el.className = v >= r.threshold ? "bad" : "ok"; };
      put($("lab-a-sim"), r.similarity_original); put($("lab-b-sim"), r.similarity_shielded);
      $("lab-out").hidden = false; note.textContent = `Threshold ${fmt(r.threshold)}`;
    } catch (e) { note.textContent = e.message; }
    btn.disabled = false;
  });
})();
