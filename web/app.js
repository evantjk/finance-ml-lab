/* Pairs — front-end logic (ES module).
   Loads the precomputed snapshot, connects to the live API when available, and
   renders one conviction-ranked market-neutral pair per sector.
   anime.js v4 is loaded from CDN and used to *enhance* — if it fails to load the
   page still works (content is visible by default; we only hide when anime is in). */

// Same-origin: the FastAPI app serves this page *and* the API. Override with
// ?api=http://localhost:8000 when running the front-end against a separate API.
const API = new URLSearchParams(location.search).get("api") || "";
const STATE = { pairs: [], sort: "conviction", hiOnly: false, q: "" };

// hide animatable elements up-front (removed if anime never arrives)
document.documentElement.classList.add("preload");

let A = null;
try {
  A = await import("https://cdn.jsdelivr.net/npm/animejs/+esm");
} catch (e) {
  document.documentElement.classList.remove("preload");
}
const { animate, createTimeline, stagger, utils, svg } = A || {};

const $ = (s) => document.querySelector(s);
const pct = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
const pp = (x) => (x == null ? "—" : x.toFixed(1));

/* ---------- data ---------- */
async function loadSnapshot() {
  const r = await fetch("data/snapshot.json", { cache: "no-store" });
  const d = await r.json();
  STATE.pairs = d.pairs;
  $("#meta").textContent =
    `trained ${d.trained_on} · snapshot ${d.generated_at.replace("T", " ").replace("+00:00", "Z")}`;
  renderSummary();
  heroIntro(d);
  render();
}

function renderSummary() {
  const c = { High: 0, Medium: 0, Low: 0 };
  STATE.pairs.forEach((p) => { c[p.conviction_tier]++; });
  $("#summary").innerHTML =
    `<span class="chip high"><i></i><span class="count">${c.High}</span>&nbsp;High conviction</span>
     <span class="chip medium"><i></i><span class="count">${c.Medium}</span>&nbsp;Medium</span>
     <span class="chip low"><i></i><span class="count">${c.Low}</span>&nbsp;Low</span>`;
}

async function pingAPI() {
  const s = $("#status"), t = $("#statusText");
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2500) });
    const h = await r.json();
    if (h.status === "ok" && h.ranker_loaded) {
      s.className = "status live"; t.textContent = "live model connected"; return;
    }
  } catch (e) { /* offline */ }
  s.className = "status offline"; t.textContent = "static snapshot";
}

async function refreshFromModel() {
  const btn = $("#refresh"), icon = btn.querySelector(".rfx");
  if (A) animate(icon, { rotate: "+=360", duration: 700, ease: "inOut(3)" });
  try {
    const r = await fetch(`${API}/pairs`, { signal: AbortSignal.timeout(8000) });
    const d = await r.json();
    const bySector = Object.fromEntries(STATE.pairs.map((p) => [p.sector, p]));
    d.pairs.forEach((np) => {
      const o = bySector[np.sector];
      if (!o) return;
      Object.assign(o, {
        edge: np.edge, conviction: np.conviction, conviction_tier: np.conviction_tier,
      });
      o.long.signal = np.long.signal; o.long.percentile = np.long.percentile; o.long.ticker = np.long.ticker;
      o.short.signal = np.short.signal; o.short.percentile = np.short.percentile; o.short.ticker = np.short.ticker;
    });
    render();
  } catch (e) {
    alert("Live model not reachable. Start it with:\n\nuvicorn api.main:app --port 8000");
  }
}

/* ---------- render ---------- */
function visible() {
  let p = [...STATE.pairs];
  if (STATE.hiOnly) p = p.filter((x) => x.conviction_tier === "High");
  if (STATE.q) {
    const q = STATE.q;
    p = p.filter((x) => x.sector.toLowerCase().includes(q) ||
      x.long.ticker.toLowerCase().includes(q) || x.short.ticker.toLowerCase().includes(q));
  }
  const radj = (x) => x.edge / Math.max(((x.long.vol_annual || .5) + (x.short.vol_annual || .5)) / 2, .05);
  const by = {
    conviction: (a, b) => b.conviction - a.conviction,
    edge: (a, b) => b.edge - a.edge,
    risk: (a, b) => radj(b) - radj(a),
    sector: (a, b) => a.sector.localeCompare(b.sector),
  }[STATE.sort];
  return p.sort(by);
}

function sparkSVG(leg) {
  const s = leg.spark;
  if (!s || s.length < 2) return `<div class="spark"></div>`;
  const w = 100, h = 32, n = s.length;
  const min = Math.min(...s), max = Math.max(...s), rng = (max - min) || 1;
  const X = (i) => (i / (n - 1)) * w;
  const Y = (v) => h - 2 - ((v - min) / rng) * (h - 4);
  const d = "M" + s.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" L");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${d}"/></svg>`;
}

function legHTML(side, leg) {
  const chg = leg.spark && leg.spark.length > 1
    ? ((leg.spark[leg.spark.length - 1] / leg.spark[0] - 1) * 100) : null;
  const chgTxt = chg == null ? "" :
    `<span class="chg">${chg >= 0 ? "▲" : "▼"} ${Math.abs(chg).toFixed(1)}% · 6mo</span>`;
  return `
    <div class="leg ${side}">
      <div class="leg-tag">${side === "long" ? "Long" : "Short"}</div>
      <div class="tk">${leg.ticker}</div>
      <div class="subind">${leg.sub_industry || ""}</div>
      ${sparkSVG(leg)}${chgTxt}
      <div class="metrics">
        <div class="m">
          <div class="m-row"><span>Rank percentile</span><b>${pp(leg.percentile)}</b></div>
          <div class="bar"><i data-w="${leg.percentile || 0}"></i></div>
        </div>
        <div class="m"><div class="m-row"><span>Forecast vol (21d)</span><b>${pct(leg.vol_annual)}</b></div></div>
        <div class="m"><div class="m-row"><span>P(up next day)</span><b>${pct(leg.prob_up)}</b></div></div>
      </div>
    </div>`;
}

function render() {
  const grid = $("#grid");
  grid.innerHTML = "";
  const rows = visible();
  if (!rows.length) { grid.innerHTML = `<p class="empty">No pairs match this filter.</p>`; return; }
  rows.forEach((p) => {
    const tier = p.conviction_tier.toLowerCase();
    const el = document.createElement("article");
    el.className = `card ${tier}`;
    if (A) el.style.opacity = 0;
    el.innerHTML = `
      <div class="card-top">
        <div class="sector">${p.sector}</div>
        <div class="edge">
          <div class="edge-num">+${(p.edge * 100).toFixed(1)}%</div>
          <div class="edge-lab">EST. MONTHLY SPREAD</div>
        </div>
      </div>
      <div class="legs">${legHTML("long", p.long)}<div class="vs">vs</div>${legHTML("short", p.short)}</div>
      <div class="conv-row">
        <span class="badge ${tier}"><i></i>${p.conviction_tier} conviction</span>
        <span class="conv-val">${p.conviction}</span>
      </div>
      <div class="conv-meter"><i data-w="${p.conviction}"></i></div>`;
    grid.appendChild(el);
  });
  revealCards();
}

/* ---------- motion ---------- */
function fillBars(scope) {
  scope.querySelectorAll("[data-w]").forEach((b) => { b.style.width = b.dataset.w + "%"; });
}

function heroIntro(d) {
  const doCount = () => {
    countUp($("#sN"), d.n_sectors, 0);
    countUp($("#uN"), d.universe_size, 0);
    const avg = d.pairs.reduce((s, p) => s + p.edge, 0) / d.pairs.length;
    countUp($("#eN"), avg * 100, 1, "+", "%");
  };
  if (!A) { document.documentElement.classList.remove("preload"); doCount(); return; }
  document.documentElement.classList.remove("preload");
  utils.set([".eyebrow", "h1", ".lede", ".stat"], { opacity: 0 });
  createTimeline({ defaults: { ease: "out(3)", duration: 760 } })
    .add(".eyebrow", { opacity: [0, 1], y: [14, 0], duration: 600 })
    .add("h1", { opacity: [0, 1], y: [26, 0] }, "-=420")
    .add(".lede", { opacity: [0, 1], y: [16, 0] }, "-=520")
    .add(".stat", { opacity: [0, 1], y: [18, 0], delay: stagger(95) }, "-=440")
    .call(doCount, "-=200");
  // soft breathing halo
  animate("#halo", { scale: [0.92, 1.06], opacity: [0.35, 0.6], duration: 6000,
    loop: true, alternate: true, ease: "inOut(2)" });
}

let io;
function revealCards() {
  const cards = document.querySelectorAll(".card");
  if (!A) { cards.forEach(fillBars); return; }
  if (io) io.disconnect();
  io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      const c = e.target;
      animate(c, { opacity: [0, 1], y: [22, 0], duration: 640, ease: "out(3)" });
      setTimeout(() => fillBars(c), 160);
      const paths = Array.from(c.querySelectorAll(".spark path"));
      if (paths.length && svg) {
        try {
          animate(svg.createDrawable(paths),
            { draw: ["0 0", "0 1"], duration: 900, delay: 140, ease: "inOut(2)" });
        } catch (err) { /* leave drawn */ }
      }
      io.unobserve(c);
    });
  }, { threshold: 0.1, rootMargin: "0px 0px -6% 0px" });
  cards.forEach((c) => io.observe(c));
}

/* ---------- backtest equity chart ---------- */
async function loadEquity() {
  let d;
  try { d = await (await fetch("data/equity.json", { cache: "no-store" })).json(); }
  catch (e) { $("#perf")?.remove(); return; }

  $("#perfSub").textContent =
    `Top ${d.top_k} sectors by conviction, monthly · ${d.span[0]} → ${d.span[1]} · net of ${d.cost_bps}bps`;
  const s = d.stats_high;
  $("#perfStats").innerHTML = [
    ["Total", `${s["Total%"] >= 0 ? "+" : ""}${Math.round(s["Total%"])}%`],
    ["CAGR", `${s["CAGR%"].toFixed(1)}%`],
    ["Sharpe", s["Sharpe"].toFixed(2)],
    ["Max DD", `${Math.round(s["MaxDD%"])}%`],
  ].map(([l, v]) => `<div class="ps"><div class="ps-num">${v}</div><div class="ps-lab">${l}</div></div>`).join("");
  $("#chartLegend").innerHTML =
    `<span class="lg high"><i></i>High-conviction sleeve (${s["Sharpe"].toFixed(2)} Sharpe)</span>
     <span class="lg all"><i></i>All-sector blend (${d.stats_all["Sharpe"].toFixed(2)} Sharpe, smoother)</span>`;

  drawEquity(d);
}

function drawEquity(d) {
  const el = $("#equityChart");
  const W = 1000, H = 300, pL = 44, pR = 14, pT = 14, pB = 24;
  const hi = d.high, al = d.all, n = hi.length;
  const vals = [...hi, ...al].filter((v) => v > 0);
  const lo = Math.min(...vals), mx = Math.max(...vals);
  const lg = (v) => Math.log10(Math.max(v, 1e-6));
  const y0 = lg(lo), y1 = lg(mx);
  const X = (i) => pL + (i / (n - 1)) * (W - pL - pR);
  const Y = (v) => pT + (1 - (lg(v) - y0) / (y1 - y0)) * (H - pT - pB);
  const path = (a) => "M" + a.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" L");

  const ticks = [0.5, 1, 2, 3, 5, 10, 15, 20, 30].filter((t) => t >= lo * 0.85 && t <= mx * 1.1);
  let g = "";
  ticks.forEach((t) => {
    const y = Y(t).toFixed(1);
    g += `<line class="grid-line" x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}"/>`;
    g += `<text class="axis-txt" x="2" y="${(+y + 3).toFixed(1)}">${t}×</text>`;
  });
  // year labels
  let lastY = "", xl = "";
  d.dates.forEach((ym, i) => {
    const yr = ym.slice(0, 4);
    if (yr !== lastY) { lastY = yr; xl += `<text class="axis-txt" x="${X(i).toFixed(1)}" y="${H - 6}" text-anchor="middle">${yr}</text>`; }
  });
  const areaHigh = path(hi) + ` L${X(n - 1).toFixed(1)},${H - pB} L${X(0).toFixed(1)},${H - pB} Z`;

  el.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el.innerHTML =
    `<defs><linearGradient id="gHigh" x1="0" y1="0" x2="0" y2="1">
       <stop offset="0%" stop-color="var(--area-grad)"/><stop offset="100%" stop-color="transparent"/>
     </linearGradient></defs>
     ${g}${xl}
     <path class="area-high" d="${areaHigh}"/>
     <path class="ln-all" d="${path(al)}"/>
     <path class="ln-high" d="${path(hi)}"/>`;

  // ---- interactive crosshair tooltip (works with or without anime.js) ----
  const wrap = el.closest(".chart-wrap");
  let tip = wrap.querySelector(".eq-tip");
  if (!tip) { tip = document.createElement("div"); tip.className = "eq-tip"; wrap.appendChild(tip); }
  el.insertAdjacentHTML("beforeend",
    `<line class="eq-cross" y1="${pT}" y2="${H - pB}"/>
     <circle class="eq-dot al" r="4"/><circle class="eq-dot hi" r="4"/>`);
  const cross = el.querySelector(".eq-cross"), dotHi = el.querySelector(".eq-dot.hi"), dotAl = el.querySelector(".eq-dot.al");
  const fmtDate = (ym) => { const [y, m] = ym.split("-"); return new Date(+y, +m - 1, 1).toLocaleDateString(undefined, { month: "short", year: "numeric" }); };
  let cur = -1;
  const showAt = (i) => {
    i = Math.max(0, Math.min(n - 1, i)); cur = i;
    const ux = X(i), yh = Y(hi[i]), ya = Y(al[i]);
    cross.setAttribute("x1", ux); cross.setAttribute("x2", ux);
    dotHi.setAttribute("cx", ux); dotHi.setAttribute("cy", yh);
    dotAl.setAttribute("cx", ux); dotAl.setAttribute("cy", ya);
    tip.innerHTML =
      `<div class="d">${fmtDate(d.dates[i])}</div>
       <div class="r hi"><span class="k"><i></i>High-conviction</span><b>${hi[i].toFixed(2)}×</b></div>
       <div class="r al"><span class="k"><i></i>All-sector</span><b>${al[i].toFixed(2)}×</b></div>`;
    const r = el.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
    const sx = r.left + (ux / W) * r.width - wr.left;
    const sy = r.top + (Math.min(yh, ya) / H) * r.height - wr.top;
    tip.style.left = Math.max(74, Math.min(wr.width - 74, sx)) + "px";
    tip.style.top = Math.max(46, sy) + "px";
    wrap.classList.add("show-tip");
  };
  const hide = () => { wrap.classList.remove("show-tip"); cur = -1; };
  el.addEventListener("pointermove", (e) => {
    const r = el.getBoundingClientRect();
    const ux = ((e.clientX - r.left) / r.width) * W;
    showAt(Math.round((ux - pL) / (W - pL - pR) * (n - 1)));
  });
  el.addEventListener("pointerleave", hide);
  el.setAttribute("tabindex", "0");
  el.setAttribute("role", "img");
  el.setAttribute("aria-label",
    `Backtested growth of $1, ${d.span[0]} to ${d.span[1]}: high-conviction sleeve ends near ${hi[n - 1].toFixed(1)}×, all-sector blend near ${al[n - 1].toFixed(1)}×. Use arrow keys to inspect points.`);
  el.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); showAt((cur < 0 ? n - 1 : cur) + (e.key === "ArrowRight" ? 1 : -1)); }
    else if (e.key === "Escape") { hide(); el.blur(); }
  });
  el.addEventListener("focus", () => { if (cur < 0) showAt(n - 1); });
  el.addEventListener("blur", hide);

  const lines = el.querySelectorAll("path.ln-high, path.ln-all");
  const area = el.querySelector(".area-high");
  if (!A) return;
  if (area) area.style.opacity = 0;
  const obs = new IntersectionObserver((es) => {
    es.forEach((e) => {
      if (!e.isIntersecting) return;
      try { animate(svg.createDrawable(Array.from(lines)), { draw: ["0 0", "0 1"], duration: 1600, ease: "inOut(2)" }); } catch (err) { /* */ }
      if (area) animate(area, { opacity: [0, 1], duration: 1200, delay: 500, ease: "out(2)" });
      obs.disconnect();
    });
  }, { threshold: 0.25 });
  obs.observe($("#perf"));
}

function countUp(el, to, dec, prefix = "", suffix = "") {
  if (!A) { el.textContent = prefix + to.toFixed(dec) + suffix; return; }
  const o = { v: 0 };
  animate(o, { v: to, duration: 1100, ease: "out(4)",
    onUpdate: () => { el.textContent = prefix + o.v.toFixed(dec) + suffix; },
    onComplete: () => { el.textContent = prefix + to.toFixed(dec) + suffix; } });
}

/* ---------- wire up ---------- */
$("#sortSeg").addEventListener("click", (e) => {
  const b = e.target.closest(".seg-btn"); if (!b) return;
  $("#sortSeg .active")?.classList.remove("active");
  b.classList.add("active"); STATE.sort = b.dataset.sort; render();
});
$("#hiOnly").addEventListener("change", (e) => { STATE.hiOnly = e.target.checked; render(); });
$("#search").addEventListener("input", (e) => { STATE.q = e.target.value.trim().toLowerCase(); render(); });
$("#refresh").addEventListener("click", refreshFromModel);
addEventListener("scroll", () => $("#nav").classList.toggle("scrolled", scrollY > 8), { passive: true });

/* ---------- theme toggle (dark mode) ---------- */
$("#themeBtn").addEventListener("click", () => {
  const root = document.documentElement;
  const next = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try { localStorage.setItem("fulcrum-theme", next); } catch (e) { /* private mode */ }
  if (A) animate("#themeBtn", { scale: [1, 0.85, 1], duration: 420, ease: "out(3)" });
});

loadSnapshot();
loadEquity();
pingAPI();
