/* charts.js — inline SVG charts with no dependencies.
 *
 * Every function here returns a string of SVG markup, because the dashboards
 * build their pages by concatenating HTML strings and handing the result to
 * innerHTML. Nothing is fetched from a CDN: a chart library would be a network
 * dependency and a supply-chain surface for what amounts to four kinds of
 * rectangle.
 *
 * Colours are CSS custom properties defined by the host page, so the charts
 * follow its light/dark theme with no JavaScript involved.
 *
 * This file is byte-identical in phish-drift and ozone-drift.
 */

(function injectChartStyles() {
  if (document.getElementById("ch-style")) return;
  const style = document.createElement("style");
  style.id = "ch-style";
  style.textContent = `
    .ch { margin: .5rem 0 1.1rem; }
    .ch-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .ch svg { display: block; width: 100%; height: auto; }
    .ch-cap { color: var(--muted); font-size: .84rem; line-height: 1.5; margin: .35rem 0 0; }
    .ch-legend { display: flex; flex-wrap: wrap; gap: .25rem 1rem; font-size: .8rem;
                 color: var(--muted); margin-bottom: .3rem; }
    .ch-legend span { display: inline-flex; align-items: center; gap: .35rem; }
    .ch-legend i { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }
    .ch-grid   { stroke: var(--line); stroke-width: 1; }
    .ch-axis   { stroke: var(--line); stroke-width: 1.5; }
    .ch-tick   { fill: var(--muted); font-size: 11px;
                 font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .ch-lab    { fill: var(--ink); font-size: 12px; }
    .ch-sub    { fill: var(--muted); font-size: 10.5px; }
    .ch-val    { fill: var(--ink); font-size: 11.5px; font-weight: 600;
                 font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .ch-title  { fill: var(--muted); font-size: 10.5px; letter-spacing: .06em; }
    .t-accent  { fill: var(--accent); stroke: var(--accent); }
    .t-good    { fill: var(--good);   stroke: var(--good); }
    .t-bad     { fill: var(--bad);    stroke: var(--bad); }
    .t-muted   { fill: var(--muted);  stroke: var(--muted); }
    .ch-ghost  { opacity: .28; }
    .ch-stroke { fill: none; stroke-width: 2; }
    .ch-dash   { stroke-dasharray: 4 4; }
  `;
  document.head.appendChild(style);
})();

const _chEsc = (s) => String(s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* Round tick spacing to something a human would have chosen. */
function _chTicks(lo, hi, want = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / want;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) {
    out.push(Number((Math.round(v / step) * step).toFixed(10)));
  }
  return out;
}

/* Pad a data range out to round numbers, always including zero. */
function _chDomain(values, forceZero = true) {
  const finite = values.filter(v => Number.isFinite(v));
  if (!finite.length) return [0, 1];
  let lo = Math.min(...finite), hi = Math.max(...finite);
  if (forceZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08;
  return [lo - (lo < 0 ? pad : 0), hi + pad];
}

function _chFrame(w, h, inner, caption, legend, aria) {
  return `<figure class="ch">${legend || ""}<div class="ch-scroll">` +
    `<svg viewBox="0 0 ${w} ${h}" role="img" style="min-width:${Math.min(w, 640)}px"` +
    ` aria-label="${_chEsc(aria || "chart")}">${inner}</svg></div>` +
    (caption ? `<figcaption class="ch-cap">${caption}</figcaption>` : "") +
    `</figure>`;
}

function _chLegend(series) {
  const shown = series.filter(s => s.name);
  if (shown.length < 2) return "";
  return `<div class="ch-legend">${shown.map(s =>
    `<span><i style="background:var(--${s.tone === "accent" ? "accent"
      : s.tone === "good" ? "good" : s.tone === "bad" ? "bad" : "muted"})"></i>${_chEsc(s.name)}</span>`
  ).join("")}</div>`;
}

/* Wrap a label onto at most two lines so long names don't overlap. */
function _chWrap(text, max) {
  const words = String(text).split(/\s+/);
  const lines = [""];
  for (const word of words) {
    const candidate = lines[lines.length - 1] ? lines[lines.length - 1] + " " + word : word;
    if (candidate.length <= max || !lines[lines.length - 1]) lines[lines.length - 1] = candidate;
    else lines.push(word);
  }
  return lines.slice(0, 2);
}

/* ------------------------------------------------------------------ bars --
 * Vertical bars. An item may carry:
 *   value    the bar's top (or bottom, if negative)
 *   from     the bar's other end, for a floating waterfall segment
 *   display  text to print instead of the value — a waterfall shows its step
 *   ci       [lo, hi] drawn as a whisker
 *   tone     accent | good | bad | muted
 */
function chartBars({ items, domain, fmt = v => v.toFixed(3), height = 270,
                     caption = "", yLabel = "", slot = 104, aria = "" }) {
  const pad = { t: 26, r: 18, b: 56, l: 56 };
  const pw = Math.max(items.length * slot, 240);
  const w = pad.l + pw + pad.r;
  const h = height;
  const ph = h - pad.t - pad.b;

  const spread = items.flatMap(d => [d.value, d.from ?? 0, ...(d.ci || [])]);
  const [lo, hi] = domain || _chDomain(spread);
  const sy = v => pad.t + ph * (1 - (v - lo) / (hi - lo));
  const ticks = _chTicks(lo, hi, 5);

  const grid = ticks.map(t =>
    `<line class="ch-grid" x1="${pad.l}" x2="${w - pad.r}" y1="${sy(t).toFixed(1)}" y2="${sy(t).toFixed(1)}"/>` +
    `<text class="ch-tick" x="${pad.l - 9}" y="${(sy(t) + 4).toFixed(1)}" text-anchor="end">${fmt(t)}</text>`
  ).join("");

  const zeroLine = (lo < 0 && hi > 0)
    ? `<line class="ch-axis" x1="${pad.l}" x2="${w - pad.r}" y1="${sy(0).toFixed(1)}" y2="${sy(0).toFixed(1)}"/>` : "";

  const bw = Math.min(58, slot * 0.56);
  const bars = items.map((d, i) => {
    const cx = pad.l + slot * i + slot / 2;
    const x = cx - bw / 2;
    const base = d.from ?? 0;
    const y0 = sy(base), y1 = sy(d.value);
    const top = Math.min(y0, y1), boxH = Math.max(Math.abs(y1 - y0), 1.5);
    const tone = `t-${d.tone || "accent"}`;

    const whisker = d.ci ? `
      <line class="${tone} ch-stroke" x1="${cx}" x2="${cx}" y1="${sy(d.ci[0]).toFixed(1)}" y2="${sy(d.ci[1]).toFixed(1)}"/>
      <line class="${tone} ch-stroke" x1="${cx - 7}" x2="${cx + 7}" y1="${sy(d.ci[0]).toFixed(1)}" y2="${sy(d.ci[0]).toFixed(1)}"/>
      <line class="${tone} ch-stroke" x1="${cx - 7}" x2="${cx + 7}" y1="${sy(d.ci[1]).toFixed(1)}" y2="${sy(d.ci[1]).toFixed(1)}"/>` : "";

    // Clear the whisker cap, not just the bar, or the number lands on the interval.
    const capTop = d.ci ? Math.min(sy(d.ci[0]), sy(d.ci[1])) : top;
    const labelY = Math.max(Math.min(top, capTop) - 7, 13);
    const lines = _chWrap(d.label, 14);
    const xlabels = lines.map((ln, k) =>
      `<text class="ch-lab" x="${cx}" y="${h - pad.b + 17 + k * 13}" text-anchor="middle">${_chEsc(ln)}</text>`
    ).join("") + (d.sub
      ? `<text class="ch-sub" x="${cx}" y="${h - pad.b + 17 + lines.length * 13}" text-anchor="middle">${_chEsc(d.sub)}</text>`
      : "");

    return `<rect class="${tone}" x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${bw}" height="${boxH.toFixed(1)}" rx="2"/>
      ${whisker}
      <text class="ch-val" x="${cx}" y="${labelY.toFixed(1)}" text-anchor="middle">${_chEsc(d.display ?? fmt(d.value))}</text>
      ${xlabels}`;
  }).join("");

  const axisTitle = yLabel
    ? `<text class="ch-title" x="${pad.l - 44}" y="${pad.t + ph / 2}" text-anchor="middle"
         transform="rotate(-90 ${pad.l - 44} ${pad.t + ph / 2})">${_chEsc(yLabel)}</text>` : "";

  return _chFrame(w, h, grid + zeroLine + bars + axisTitle, caption, "", aria || caption);
}

/* ---------------------------------------------------------------- barsH --
 * Horizontal bars, for rankings like feature importance where the labels are
 * words rather than short codes.
 */
function chartBarsH({ items, fmt = v => v.toFixed(3), caption = "", labelWidth = 168,
                      barHeight = 20, tone = "accent", aria = "" }) {
  const pad = { t: 8, r: 54, b: 8, l: labelWidth };
  const gap = 8, track = 340;
  const w = pad.l + track + pad.r;
  const h = pad.t + items.length * (barHeight + gap) + pad.b;
  const max = Math.max(...items.map(d => d.value), 1e-9);

  const rows = items.map((d, i) => {
    const y = pad.t + i * (barHeight + gap);
    const len = Math.max((d.value / max) * track, 1);
    return `<text class="ch-lab" x="${pad.l - 10}" y="${y + barHeight * 0.72}" text-anchor="end">${_chEsc(d.label)}</text>
      <rect class="t-muted ch-ghost" x="${pad.l}" y="${y}" width="${track}" height="${barHeight}" rx="3"/>
      <rect class="t-${d.tone || tone}" x="${pad.l}" y="${y}" width="${len.toFixed(1)}" height="${barHeight}" rx="3"/>
      <text class="ch-val" x="${pad.l + track + 8}" y="${y + barHeight * 0.72}">${_chEsc(fmt(d.value))}</text>`;
  }).join("");

  return _chFrame(w, h, rows, caption, "", aria || caption);
}

/* ----------------------------------------------------------------- line --
 * One or more series over a shared categorical x axis. Used for records that
 * grow: days of collection, years of monitoring.
 */
function chartLine({ series, labels, fmt = v => String(v), height = 250, caption = "",
                     yLabel = "", markers = [], area = false, aria = "" }) {
  const pad = { t: 22, r: 18, b: 46, l: 60 };
  const w = Math.max(pad.l + labels.length * 46 + pad.r, 560);
  const h = height, ph = h - pad.t - pad.b, pw = w - pad.l - pad.r;

  const all = series.flatMap(s => s.values.filter(v => Number.isFinite(v)));
  const [lo, hi] = _chDomain(all);
  const sy = v => pad.t + ph * (1 - (v - lo) / (hi - lo));
  const sx = i => labels.length === 1 ? pad.l + pw / 2 : pad.l + pw * (i / (labels.length - 1));
  const ticks = _chTicks(lo, hi, 5);

  const grid = ticks.map(t =>
    `<line class="ch-grid" x1="${pad.l}" x2="${w - pad.r}" y1="${sy(t).toFixed(1)}" y2="${sy(t).toFixed(1)}"/>` +
    `<text class="ch-tick" x="${pad.l - 9}" y="${(sy(t) + 4).toFixed(1)}" text-anchor="end">${_chEsc(fmt(t))}</text>`
  ).join("");

  const every = Math.ceil(labels.length / 12);
  const xlabels = labels.map((l, i) => (i % every === 0 || i === labels.length - 1)
    ? `<text class="ch-tick" x="${sx(i).toFixed(1)}" y="${h - pad.b + 18}" text-anchor="middle">${_chEsc(l)}</text>`
    : "").join("");

  const marks = markers.map(m => {
    const x = sx(m.at);
    return `<line class="ch-grid ch-dash" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${pad.t}" y2="${h - pad.b}" stroke-width="1.5"/>
      <text class="ch-sub" x="${(x + 5).toFixed(1)}" y="${pad.t + 11}">${_chEsc(m.label)}</text>`;
  }).join("");

  const paths = series.map(s => {
    const pts = s.values.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
    const fill = area
      ? `<polygon class="t-${s.tone || "accent"} ch-ghost" points="${pad.l},${sy(lo).toFixed(1)} ${pts} ${sx(s.values.length - 1).toFixed(1)},${sy(lo).toFixed(1)}"/>`
      : "";
    const dots = s.values.length <= 40
      ? s.values.map((v, i) => `<circle class="t-${s.tone || "accent"}" cx="${sx(i).toFixed(1)}" cy="${sy(v).toFixed(1)}" r="2.6"/>`).join("")
      : "";
    return `${fill}<polyline class="t-${s.tone || "accent"} ch-stroke ${s.dashed ? "ch-dash" : ""}" points="${pts}"/>${dots}`;
  }).join("");

  const axisTitle = yLabel
    ? `<text class="ch-title" x="${pad.l - 46}" y="${pad.t + ph / 2}" text-anchor="middle"
         transform="rotate(-90 ${pad.l - 46} ${pad.t + ph / 2})">${_chEsc(yLabel)}</text>` : "";

  return _chFrame(w, h, grid + marks + paths + xlabels + axisTitle, caption, _chLegend(series), aria || caption);
}

/* -------------------------------------------------------------- scatter --
 * A unit square with the no-skill diagonal drawn in. Plotting false-alarm rate
 * against recall puts a model's behaviour somewhere interpretable: the top-left
 * corner is a good detector, the diagonal is a coin flip, and the top-right
 * corner is a model that has simply started saying yes to everything.
 */
function chartScatter({ points, xLabel = "", yLabel = "", caption = "", size = 300, aria = "" }) {
  const pad = { t: 16, r: 108, b: 46, l: 54 };
  const w = pad.l + size + pad.r, h = pad.t + size + pad.b;
  const sx = v => pad.l + size * v;
  const sy = v => pad.t + size * (1 - v);
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  const grid = ticks.map(t => `
    <line class="ch-grid" x1="${pad.l}" x2="${pad.l + size}" y1="${sy(t)}" y2="${sy(t)}"/>
    <line class="ch-grid" x1="${sx(t)}" x2="${sx(t)}" y1="${pad.t}" y2="${pad.t + size}"/>
    <text class="ch-tick" x="${pad.l - 9}" y="${sy(t) + 4}" text-anchor="end">${t}</text>
    <text class="ch-tick" x="${sx(t)}" y="${pad.t + size + 17}" text-anchor="middle">${t}</text>`).join("");

  const diagonal = `<line class="ch-grid ch-dash" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke-width="1.5"/>
    <text class="ch-sub" x="${sx(0.62)}" y="${sy(0.56)}" transform="rotate(-45 ${sx(0.62)} ${sy(0.56)})">no skill</text>`;

  // Cells that behave alike land on top of each other — which is itself the
  // finding, so the dots stay put and only the labels are nudged apart.
  const placed = [];
  const dots = points.map(p => {
    const x = sx(p.x), y = sy(p.y);
    const right = x < pad.l + size * 0.62;
    let ly = y + 4;
    while (placed.some(q => Math.abs(q.y - ly) < 13 && Math.abs(q.x - x) < 76 && q.right === right)) ly += 13;
    placed.push({ x, y: ly, right });
    const leader = Math.abs(ly - (y + 4)) > 2
      ? `<line class="ch-grid" x1="${x.toFixed(1)}" y1="${y.toFixed(1)}"
           x2="${(right ? x + 9 : x - 9).toFixed(1)}" y2="${(ly - 4).toFixed(1)}"/>` : "";
    return `<circle class="t-${p.tone || "accent"}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill-opacity=".85"/>
      ${leader}
      <text class="ch-sub" x="${(right ? x + 11 : x - 11).toFixed(1)}" y="${ly.toFixed(1)}"
        text-anchor="${right ? "start" : "end"}">${_chEsc(p.label)}</text>`;
  }).join("");

  const axes = `
    <text class="ch-title" x="${pad.l + size / 2}" y="${h - 8}" text-anchor="middle">${_chEsc(xLabel)}</text>
    <text class="ch-title" x="${pad.l - 40}" y="${pad.t + size / 2}" text-anchor="middle"
      transform="rotate(-90 ${pad.l - 40} ${pad.t + size / 2})">${_chEsc(yLabel)}</text>`;

  return _chFrame(w, h, grid + diagonal + dots + axes, caption, "", aria || caption);
}
