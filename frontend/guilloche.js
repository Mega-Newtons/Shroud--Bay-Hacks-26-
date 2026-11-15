/* Guilloché: the fine woven line patterns of banknotes and passports, generated as SVG paths.
   Two primitives:
   - ring():  K phase-shifted rosette curves  r(θ) = R + A·sin(nθ + φk)  → a woven circular band
   - band():  K phase-shifted sine waves along a straight strip           → a woven border */
(function () {
  const TAU = Math.PI * 2, NS = "http://www.w3.org/2000/svg";

  function ring(R, A, n, K, twist = 0) {
    const out = [];
    for (let k = 0; k < K; k++) {
      const phi = (TAU * k) / K, pts = Math.max(240, n * 40);
      let d = "";
      for (let i = 0; i <= pts; i++) {
        const t = (TAU * i) / pts;
        const r = R + A * Math.sin(n * t + phi) + twist * Math.sin(2 * n * t - phi);
        d += (i ? "L" : "M") + (r * Math.cos(t)).toFixed(2) + " " + (r * Math.sin(t)).toFixed(2);
      }
      out.push(d + "Z");
    }
    return out;
  }

  // Hypotrochoid: a point at distance d from the centre of a circle r rolling inside a circle R.
  function hypotrochoid(R, r, d, turns) {
    const pts = turns * 180;
    let p = "";
    for (let i = 0; i <= pts; i++) {
      const t = (TAU * turns * i) / pts, k = (R - r) / r;
      p += (i ? "L" : "M") + ((R - r) * Math.cos(t) + d * Math.cos(k * t)).toFixed(2) + " " + ((R - r) * Math.sin(t) - d * Math.sin(k * t)).toFixed(2);
    }
    return p;
  }

  function band(len, h, K, waves, vertical) {
    const out = [], step = 2;
    for (let k = 0; k < K; k++) {
      let d = "";
      for (let x = 0; x <= len; x += step) {
        const y = h / 2 + (h / 2 - 1) * Math.sin((TAU * waves * x) / len + (TAU * k) / K);
        d += (x ? "L" : "M") + (vertical ? y.toFixed(1) + " " + x : x + " " + y.toFixed(1));
      }
      out.push(d);
    }
    return out;
  }

  function draw(svg, paths, attrs = {}) {
    const made = [];
    for (const d of paths) {
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", d); p.setAttribute("class", "g"); p.setAttribute("pathLength", "1");
      for (const k in attrs) p.setAttribute(k, attrs[k]);
      svg.appendChild(p); made.push(p);
    }
    return made;
  }

  const rosette = {
    mark: () => [...ring(13, 4.5, 9, 6), ...ring(6, 2.6, 6, 4)],
    // drawn from the outside in, so the pattern closes on the subject as shielding completes
    full: () => [...ring(92, 5, 36, 6), ...ring(74, 12, 16, 8, 2), ...ring(47, 13, 10, 8), hypotrochoid(30, 11, 14, 11), ...ring(14, 5, 7, 4)],
    hero: () => [...ring(100, 6, 48, 6), ...ring(82, 11, 20, 10, 2), ...ring(56, 14, 12, 10), hypotrochoid(38, 14, 18, 14), ...ring(16, 6, 8, 6)],
  };

  // Woven frame around an element: four bands plus a small rosette on each corner.
  function frame(svg, w, h, t) {
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.replaceChildren();
    const g = (tx, ty) => { const el = document.createElementNS(NS, "g"); el.setAttribute("transform", `translate(${tx} ${ty})`); svg.appendChild(el); return el; };
    const wx = Math.round((w - 2 * t) / 26), wy = Math.round((h - 2 * t) / 26);
    draw(g(t, 0), band(w - 2 * t, t, 5, wx, false)); draw(g(t, h - t), band(w - 2 * t, t, 5, wx, false));
    draw(g(0, t), band(h - 2 * t, t, 5, wy, true)); draw(g(w - t, t), band(h - 2 * t, t, 5, wy, true));
    for (const [x, y] of [[t / 2, t / 2], [w - t / 2, t / 2], [t / 2, h - t / 2], [w - t / 2, h - t / 2]])
      draw(g(x, y), ring(t * 0.3, t * 0.14, 6, 4));
  }

  window.Guilloche = { ring, hypotrochoid, band, draw, rosette, frame };
})();
