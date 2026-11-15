/* Shroud extension — content script.
   Watches every file input. When you pick a photo or audio file, the page's own handlers are held back,
   the file is shielded by your local Shroud server, swapped into the input, and the change event is re-sent. */
(() => {
  const DEFAULTS = { enabled: true, backend: "http://localhost:8000", strength: "balanced" };
  const ours = new WeakSet();          // change events we dispatched ourselves
  const fmt = (v) => (v == null ? "—" : Number(v).toFixed(2));
  const kindOf = (f) => (f.type.startsWith("image/") ? "image" : f.type.startsWith("audio/") ? "audio" : null);

  /* ——— toast: paper, ink, one green, a small guilloché rosette ——— */
  let toast, line, sub, paths = [];
  function ring(R, A, n, K) {
    const out = [];
    for (let k = 0; k < K; k++) {
      let d = "";
      for (let i = 0; i <= 240; i++) {
        const t = (Math.PI * 2 * i) / 240, r = R + A * Math.sin(n * t + (Math.PI * 2 * k) / K);
        d += (i ? "L" : "M") + (r * Math.cos(t)).toFixed(2) + " " + (r * Math.sin(t)).toFixed(2);
      }
      out.push(d + "Z");
    }
    return out;
  }
  function ensureToast() {
    if (toast) return;
    const font = document.createElement("style");   // @font-face must live in the page document, not the shadow root
    font.textContent = `@font-face{font-family:"Shroud Mono";src:url("${chrome.runtime.getURL("fonts/plex-mono-normal-400.woff2")}") format("woff2");}`;
    document.documentElement.appendChild(font);
    const host = document.createElement("div");
    host.style.cssText = "all:initial;position:fixed;right:24px;bottom:24px;z-index:2147483647;";
    const root = host.attachShadow({ mode: "closed" });
    root.innerHTML = `<style>
      .t{display:flex;gap:12px;align-items:center;background:#F3EFE6;color:#15171A;border:1px solid #15171A;border-radius:2px;padding:12px 16px 12px 12px;
         font:400 12px/1.5 "Shroud Mono",ui-monospace,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;font-variant-numeric:tabular-nums;
         opacity:0;transform:translateY(8px);transition:opacity 200ms ease-out,transform 200ms ease-out;max-width:360px}
      .t.on{opacity:1;transform:none}
      svg{width:40px;height:40px;flex:none}
      path{fill:none;stroke:#1E5C47;stroke-width:.6px;vector-effect:non-scaling-stroke;stroke-dasharray:1;stroke-dashoffset:1;transition:stroke-dashoffset 400ms linear}
      .s{color:#5B5E63;border-top:1px solid #D8D2C4;margin-top:4px;padding-top:4px}
      .t.bad .l{color:#8A2B22}
      .t.ok .l b{color:#1E5C47;font-weight:400}
      @media (prefers-reduced-motion:reduce){.t,path{transition:none}}
    </style>
    <div class="t"><svg viewBox="-20 -20 40 40"></svg><div><div class="l"></div><div class="s"></div></div></div>`;
    toast = root.querySelector(".t"); line = root.querySelector(".l"); sub = root.querySelector(".s");
    const svg = root.querySelector("svg");
    for (const d of [...ring(13, 4.5, 9, 6), ...ring(6, 2.6, 6, 4)]) {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", d); p.setAttribute("pathLength", "1"); svg.appendChild(p); paths.push(p);
    }
    document.documentElement.appendChild(host);
  }
  function show(html, note, progress, cls = "") {
    ensureToast();
    line.innerHTML = html; sub.textContent = note; toast.className = "t on " + cls;
    paths.forEach((p, i) => { p.style.strokeDashoffset = String(1 - Math.max(0, Math.min(1, progress * paths.length - i))); });
  }
  let hideTimer;
  function hideLater(ms) { clearTimeout(hideTimer); hideTimer = setTimeout(() => toast && toast.classList.remove("on"), ms); }

  /* ——— one file → the local server (through the service worker) → shielded file ——— */
  const readDataUrl = (file) => new Promise((ok, no) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = no; r.readAsDataURL(file); });
  function shield(file, kind, idx, total) {
    return new Promise(async (resolve, reject) => {
      const port = chrome.runtime.connect({ name: "shroud" });
      const n = total > 1 ? ` ${idx + 1}/${total}` : "";
      port.onDisconnect.addListener(() => reject(new Error("Shroud stopped responding.")));
      port.onMessage.addListener(async (m) => {
        if (m.type === "progress") {
          const j = m.job, s = j.scores || {};
          show(`Shielding${n} · match ${fmt(s.before)} → ${fmt(j.current_similarity)}`, j.total_steps ? `Step ${j.step}/${j.total_steps} · ${file.name}` : file.name, j.progress || 0);
        } else if (m.type === "done") {
          const blob = await (await fetch(m.dataUrl)).blob();
          const ext = m.job.result_url.split(".").pop();
          resolve({ job: m.job, file: new File([blob], file.name.replace(/\.[^.]+$/, "") + "." + ext, { type: m.mime || blob.type, lastModified: Date.now() }) });
          port.disconnect();
        } else if (m.type === "error") { reject(new Error(m.message)); port.disconnect(); }
      });
      show(`Shielding${n}`, file.name, 0);
      port.postMessage({ kind, name: file.name, dataUrl: await readDataUrl(file) });
    });
  }

  document.addEventListener("change", async (e) => {
    const input = e.target;
    if (!(input instanceof HTMLInputElement) || input.type !== "file" || ours.has(e) || !input.files.length) return;
    const files = [...input.files];
    if (!files.some(kindOf)) return;                       // nothing we can shield: leave the page alone
    e.stopImmediatePropagation();                          // hold the page back before any await, so it never sees the raw file
    const cfg = await chrome.storage.local.get(DEFAULTS).catch(() => DEFAULTS);
    const release = () => { const ev = new Event("change", { bubbles: true }); ours.add(ev); input.dispatchEvent(ev); };
    if (!cfg.enabled) return release();

    const dt = new DataTransfer(), targets = files.filter(kindOf);
    let last = null, n = 0;
    try {
      for (const f of files) {
        const kind = kindOf(f);
        if (!kind) { dt.items.add(f); continue; }
        const r = await shield(f, kind, n++, targets.length);
        dt.items.add(r.file); last = r.job;
      }
      input.files = dt.files;
      const s = last.scores;
      const ok = s.after < s.threshold;
      show(`<b>${ok ? "Shielded" : "Shielded, weakly"}</b> · match ${fmt(s.before)} → ${fmt(s.after)}`, `${targets.length > 1 ? targets.length + " files" : files[0].name} · swapped into the upload`, 1, ok ? "ok" : "bad");
      hideLater(6000);
    } catch (err) {
      show("Not shielded", `${err.message} The original file was passed through.`, 0, "bad");
      hideLater(8000);
    }
    release();
  }, true);
})();
