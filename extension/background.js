/* Shroud extension — service worker. Talks to the local Shroud server so pages never need CORS or mixed-content exceptions.
   A content script opens a port, sends one file, and receives progress messages and finally the shielded file. */
const DEFAULTS = { enabled: true, backend: "http://localhost:8000", strength: "balanced" };

async function toDataUrl(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return `data:${blob.type || "application/octet-stream"};base64,${btoa(bin)}`;
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "shroud") return;
  let open = true;
  port.onDisconnect.addListener(() => { open = false; });
  port.onMessage.addListener(async (msg) => {
    try {
      const cfg = await chrome.storage.local.get(DEFAULTS);
      const base = cfg.backend.replace(/\/+$/, "");
      const body = new FormData();
      body.append("file", await (await fetch(msg.dataUrl)).blob(), msg.name);
      body.append("strength", cfg.strength);
      let res = await fetch(`${base}/api/protect/${msg.kind}`, { method: "POST", body });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "The Shroud server refused that file.");
      const { job_id } = await res.json();
      while (open) {
        const job = await (await fetch(`${base}/api/jobs/${job_id}`)).json();
        if (job.status === "error") throw new Error(job.error);
        if (job.status === "done") {
          const blob = await (await fetch(base + job.result_url)).blob();
          port.postMessage({ type: "done", job, dataUrl: await toDataUrl(blob), mime: blob.type });
          return;
        }
        port.postMessage({ type: "progress", job });
        await new Promise((r) => setTimeout(r, 400));
      }
    } catch (e) {
      if (open) port.postMessage({ type: "error", message: e instanceof TypeError ? "Can't reach the Shroud server. Is it running?" : e.message });
    }
  });
});
