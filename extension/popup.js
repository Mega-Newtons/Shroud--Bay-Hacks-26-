/* Shroud extension — popup: on/off, strength, server address. Settings live in chrome.storage.local. */
const DEFAULTS = { enabled: true, backend: "http://localhost:8000", strength: "balanced" };
const $ = (id) => document.getElementById(id);

// the same small rosette as the site's logo mark
(function mark() {
  const ring = (R, A, n, K) => Array.from({ length: K }, (_, k) => {
    let d = "";
    for (let i = 0; i <= 240; i++) { const t = (Math.PI * 2 * i) / 240, r = R + A * Math.sin(n * t + (Math.PI * 2 * k) / K); d += (i ? "L" : "M") + (r * Math.cos(t)).toFixed(2) + " " + (r * Math.sin(t)).toFixed(2); }
    return d + "Z";
  });
  for (const d of [...ring(13, 4.5, 9, 6), ...ring(6, 2.6, 6, 4)]) { const p = document.createElementNS("http://www.w3.org/2000/svg", "path"); p.setAttribute("d", d); $("mark").appendChild(p); }
})();

function seg(id, value, onPick) {
  $(id).querySelectorAll("button").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.v === value));
    b.onclick = () => { seg(id, b.dataset.v, onPick); onPick(b.dataset.v); };
  });
}
async function ping(url) {
  const el = $("state");
  try {
    const h = await (await fetch(url.replace(/\/+$/, "") + "/api/health")).json();
    el.textContent = `Server ready · ${h.device}${h.mock ? " · mock data" : ""}`; el.className = "label";
  } catch (e) { el.textContent = "Server not reachable"; el.className = "label down"; }
}
chrome.storage.local.get(DEFAULTS).then((cfg) => {
  seg("enabled", cfg.enabled ? "on" : "off", (v) => chrome.storage.local.set({ enabled: v === "on" }));
  seg("strength", cfg.strength, (v) => chrome.storage.local.set({ strength: v }));
  $("backend").value = cfg.backend;
  $("backend").addEventListener("change", (e) => { chrome.storage.local.set({ backend: e.target.value.trim() }); ping(e.target.value.trim()); });
  ping(cfg.backend);
});
