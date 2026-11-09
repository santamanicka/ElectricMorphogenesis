"use strict";
const $ = id => document.getElementById(id);
const A_UM = 5.0;                                   // cell radius, um
const PITCH_UM = 2 * A_UM;

let D = null, info = null;
let MODE = BUILD.mode;                             // 'server' (Python backend) or 'js' (in-browser core); 'auto' is resolved in main()
let curV = null, curG = null, curE = null;
let running = false, runToken = 0, dirtyState = false, needInit = true, stepNow = 0;
let maskG = null, maskV = null, clampShown = null, clampLeft = 0;   // painted pixels since the last sync; the clamp outline; steps left of the running clamp
let hist = { t: [], V: [], fh: [], g: [] };
let lastReadoutH = 0, maskVer = -1, devPreview = false, devKey = '', devCached = null, devThumbs = [], lastStage = -2, lastDevStatus = null;   // tissue-mask version held by the page; development state
let curMask = null;                                // mask of the initialised tissue (Uint8Array, 1 = tissue) or null for the plain rectangle
let customMask = null, customDims = null, prevShape = 'rectangle', pendKey = '', pendCache = null;   // a drawn shape; the cache of the shape Initialise would build
let effSpeed = null;
const RANGES = { V: [-55, -5], g: [0, 2], e: [0, 2] };
let view = 'V';

// ---------- helpers ----------
function setStatus(text, cls) { const s = $('status'); s.textContent = text; s.className = cls || ''; }
function u8(x) { if (typeof x !== 'string') return Uint8Array.from(x); const bin = atob(x), a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return a; }
function f32(x) { if (typeof x !== 'string') return x; const b64 = x, bin = atob(b64), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(u8.buffer); }
function toB64u8(a) { let s = ''; for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode.apply(null, a.subarray(i, i + 0x8000)); return btoa(s); }
function toB64(a) { const u8 = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); }
async function api(path, body) {
  if (MODE === 'js') return jsApi(path, body);
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ error: 'bad response' }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

// ---------- in-browser backend: same endpoints and replies as visualizer.py, running core.js ----------
const STIG = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true };  // data/StigmergicModelParameters.dat
const JS = { core: null, V: null, g: null, n: 0, cfg: null, diverged: false, clamp: null };
const numOr = (x, d) => (Number.isFinite(+x) && x !== null && x !== '') ? +x : d;
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function randn(rng) { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); }
// Cost model of a step (core.js, PotentiaCore.estimate; fitted to Node/V8 timings, within ~15% on the machine it was fitted on): per-pixel work plus the cheaper field engine,
// the direct sum over the kernel (probes x kernel pairs) or the FFT convolution (grid size). The core picks the same engine.
function estStepMs(nx, ny, m, Rs, vector, periodic) { return PotentiaCoreLib.PotentiaCore.estimate(nx, ny, m, Rs, vector, !!periodic).ms; }
const HEAVY_MS = 80;                                // the in-browser version refuses settings estimated above this many ms per step
// largest field range (R_s, in the model's units) whose estimated step time stays within limitMs for this tissue and resolution (0 if even a tiny range is too heavy)
function maxRsFor(nx, ny, m, vector, periodic, limitMs) {
  let lo = 0.05, hi = 500; if (estStepMs(nx, ny, m, lo, vector, periodic) > limitMs) return 0;
  for (let i = 0; i < 40; i++) { const mid = 0.5 * (lo + hi); if (estStepMs(nx, ny, m, mid, vector, periodic) > limitMs) hi = mid; else lo = mid; }
  return lo;
}
// what to do about a too-heavy setting: the range is usually the cost (kernel pairs per probe grow with its square), so say which range fits
const isSmooth235 = n => { for (const q of [2, 3, 5]) while (n % q === 0) n /= q; return n === 1; };
const nearestSmoothCells = (n, m) => { for (let d = 0; d < n; d++) for (const c of [n + d, n - d]) if (c >= 2 && isSmooth235(c * m)) return c; return n; };
function heavyAdvice(nx, ny, m, Rs, vector, periodic) {
  const R = A_UM * Math.SQRT2 * (Rs + 0.001), Rpx = R / (PITCH_UM / m), L = Math.min(nx, ny) * PITCH_UM, mx = maxRsFor(nx, ny, m, vector, periodic, 10);               // 10 ms per step = about 1 simulated s per wall s at dt = 0.01 s
  const here = `field range ${R.toFixed(0)} µm = ${Rpx.toFixed(0)} px (${(100 * R / L).toFixed(1)}% of the ${L} µm shorter side)`, parts = [];
  if (mx > 0) { const Rmax = A_UM * Math.SQRT2 * (mx + 0.001); parts.push(`at this size a range up to about ${Rmax.toFixed(0)} µm (${(100 * Rmax / L).toFixed(1)}% of the shorter side) runs at roughly 1 simulated s per wall s (estimate)`); }
  if (periodic && !(isSmooth235(nx * m) && isSmooth235(ny * m))) {                // the in-browser FFT is only fast when the pixel count has no prime factor above 5
    const ax = nearestSmoothCells(nx, m), ay = nearestSmoothCells(ny, m), ms = estStepMs(ax, ay, m, Rs, vector, true);
    if (ms < HEAVY_MS) parts.push(`a periodic tissue of ${ax}×${ay} cells (its pixel count has only the factors 2, 3 and 5, so the FFT is fast) would take about ${ms.toFixed(0)} ms per step at this range`);
  }
  parts.push('lower m or the tissue size, or use the Python server version (its FFT copes with any tissue size)');
  const txt = parts.join('; or ');
  return { here, fix: txt };
}
function estSpeed(nx, ny, m, Rs, vector, GJ, periodic) { const gjc = GJ * 10 * m * m, dt = gjc > 0 ? Math.min(0.01, 0.2 / gjc) : 0.01; return dt * 1000 / estStepMs(nx, ny, m, Rs, vector, periodic); }
// The field range is controlled as a percentage of the tissue's shorter side. The model parameter R_s (in cell half-diagonals; R = a*sqrt(2)*(R_s + 0.001)) is kept in the hidden
// input #Rs and derived from the percentage; RPCT is the control's exact value, so changing the tissue size keeps the percentage and the physical range scales with it.
let RPCT = 0;
const tissueMinUm = () => Math.min(parseInt($('nx').value) || 0, parseInt($('ny').value) || 0) * PITCH_UM;
const rangeUm = Rs => A_UM * Math.SQRT2 * (Rs + 0.001);
function showPct() { $('Rpct').value = RPCT.toFixed(1); $('Rpct_sl').value = RPCT; }
function syncPctFromRs() { const L = tissueMinUm(); RPCT = L > 0 ? 100 * rangeUm(num('Rs')) / L : 0; showPct(); }
function syncRsFromPct() { const L = tissueMinUm(); if (L > 0 && RPCT > 0) $('Rs').value = (RPCT / 100 * L) / (A_UM * Math.SQRT2) - 0.001; }
function jsFrame(wantE, done, wallMs, clientVer) {
  const c = JS.core, n = c.Np0 * c.Np1, V = new Float32Array(n), g = new Float32Array(n), mk = c.mask;
  let s = 0, mn = Infinity, mx = -Infinity, hp = 0, sg = 0, nt = 0;
  for (let k = 0; k < n; k++) {
    const v = JS.V[k] * 1e3; V[k] = v; g[k] = JS.g[k];
    if (mk && !mk[k]) continue;                                                      // statistics over the tissue only
    nt++; s += v; if (v < mn) mn = v; if (v > mx) mx = v; if (v < -30) hp++; sg += JS.g[k];
  }
  const out = { t: JS.n * c.dt, step: JS.n, Np: [c.Np0, c.Np1], V, g, steps_done: done, wall_ms: wallMs, diverged: JS.diverged,
                stats: { meanV: s / nt, minV: mn, maxV: mx, fh: hp / nt, meang: sg / nt },
                clamp_left: JS.clamp ? Math.max(0, JS.clamp.steps - (JS.n - JS.clamp.start)) : 0 };
  if (wantE) { out.ebar = Float32Array.from(c.perceivedOverBias(JS.V)); if (mk) for (let k = 0; k < n; k++) if (!mk[k]) out.ebar[k] = 0; }
  out.mask_ver = JS.maskVer; if (JS.devo) out.dev = { ...JS.devStatus };
  if (clientVer !== JS.maskVer && mk) out.mask = mk;                                  // the page's copy of the tissue mask is older: send it
  return out;
}
async function jsApi(path, body) {
  if (path === '/api/defaults') return { params: STIG, cell_um: PITCH_UM, max_side_px: 512, max_total_px: 200000, ic: { V0_mV: -9.2, g0: 1.0 } };
  if (path === '/api/init') {
    const size = body.size_cells.map(x => parseInt(x)), m = parseInt(body.m), pr = body.params;
    if (!(m >= 1 && m <= 16)) throw new Error('m must be between 1 and 16');
    if (!(Math.min(...size) >= 2) || Math.max(...size) * m > 512 || size[0] * size[1] * m * m > 200000)
      throw new Error(`grid ${size[0] * m}x${size[1] * m} pixels is outside the allowed range (each side <= 512, total <= 200000); lower m or the tissue size`);
    const dc = body.dev && body.dev.enabled ? body.dev : null; let devo = null, rs0 = +pr.fieldScreenSize, rsMax = rs0, mask0 = body.mask || null, rngCfg = null;
    if (dc) {                                                                          // development: the tissue starts as the egg; the range follows its schedule
      if (body.periodic) throw new Error('development needs periodic boundaries off');
      devo = new PotentiaCoreLib.PotentiaCore.Development(size[0] * m, size[1] * m, numOr(dc.duration, 300), numOr(dc.update, 0.5));
      if (!(devo.D > 0 && devo.update > 0)) throw new Error('development duration and update interval must be positive');
      const rc = dc.range || {}, mode = rc.mode || 'const', hUm = PITCH_UM / m;
      if (!['const', 'prop', 'stage'].includes(mode) || (mode === 'stage' && !(rc.stage_um && rc.stage_um.length === devo.stages.length))) throw new Error("development range: mode must be 'const', 'prop' or 'stage' (with one value per stage)");
      mask0 = devo.maskAt(0);
      rngCfg = { mode, constUm: A_UM * Math.SQRT2 * (+pr.fieldScreenSize + 0.001), pct: numOr(rc.pct, 25), stageUm: rc.stage_um };
      if (mode !== 'const') rs0 = devo.rangeUm(mode, 0, mask0, hUm, 0, rngCfg.pct, rngCfg.stageUm) / (A_UM * Math.SQRT2) - 0.001;
      let rmax = 0; for (const t of [0, 0.25, 0.5, 0.75, 1]) rmax = Math.max(rmax, devo.rangeUm(mode, t, devo.maskAt(t), hUm, rngCfg.constUm, rngCfg.pct, rngCfg.stageUm));
      rsMax = Math.max(rs0, rmax / (A_UM * Math.SQRT2) - 0.001);                       // the largest range of the schedule decides whether the run is affordable
    }
    const ms = estStepMs(size[0], size[1], m, rsMax, pr.fieldVector, !!body.periodic);
    if (ms > HEAVY_MS) { const a = heavyAdvice(size[0], size[1], m, rsMax, pr.fieldVector, !!body.periodic); throw new Error(`too heavy for the in-browser version: about ${ms.toFixed(0)} ms per step at a ${a.here}. ${a.fix[0].toUpperCase() + a.fix.slice(1)}.`); }
    const fq = (typeof location !== 'undefined' && /[?&]field=(direct|fft)\b/.exec(location.search || '')), field = fq ? fq[1] : 'auto';      // ?field=direct or ?field=fft forces the field engine (default: the cheaper one)
    const core = new PotentiaCoreLib.PotentiaCore({ sizeCells: size, m, params: { ...pr, fieldScreenSize: rs0 }, periodic: !!body.periodic, field, mask: mask0 });
    const ic = body.ic || { mode: 'homogeneous' }, n = core.Np0 * core.Np1, g0 = numOr(ic.g0, 1), branch = ic.branch === 'hyp' ? 'hyp' : 'dep';
    const g = new Float64Array(n);                                                   // initial conditions are set on G_pol; Vmem follows it
    if (ic.mode === 'random') {
      const rng = mulberry32((parseInt(ic.seed) || 0) + 12345), sG = numOr(ic.sigmaG, 0.3);
      for (let k = 0; k < n; k++) g[k] = ic.uniform ? 2 * rng() : g0 + sG * randn(rng);
    } else g.fill(g0);
    for (let k = 0; k < n; k++) g[k] = Math.min(2, Math.max(0, g[k]));
    const V = PotentiaCoreLib.PotentiaCore.equilibriumV(g, branch, 1.5);
    if (core.outIdx) for (let q = 0; q < core.outIdx.length; q++) { V[core.outIdx[q]] = 0; g[core.outIdx[q]] = 0; }            // nothing outside the tissue shape
    Object.assign(JS, { core, V, g, n: 0, cfg: body, diverged: false, clamp: null, devo, maskVer: (JS.maskVer || 0) + 1, devStatus: null });
    const info = core.info(); info.shaped = !!core.mask; info.developing = !!devo;
    if (devo) {
      const st = devo.stageAt(0);
      Object.assign(JS, { devRng: rngCfg, devFill: [PotentiaCoreLib.PotentiaCore.equilibriumV(g0, branch, 1.5), g0], devK: Math.max(1, Math.round(devo.update / core.dt)), devStatus: { tau: 0, stage: st.name, nf: st.nf, tissue: mask0.reduce((a, b) => a + b, 0), rangeUm: core.R * 1e6, range_um: core.R * 1e6, changed: true } });
    }
    return { info, frame: jsFrame(!!body.want_ebar, 0, 0, undefined) };
  }
  if (!JS.core) throw new Error('initialise first');
  if (path === '/api/state') {
    JS.V = Float64Array.from(body.V, v => v * 1e-3); JS.g = Float64Array.from(body.g, v => Math.min(2, Math.max(0, v))); JS.diverged = false; JS.clamp = null;
    if (JS.core.outIdx) for (let q = 0; q < JS.core.outIdx.length; q++) { JS.V[JS.core.outIdx[q]] = 0; JS.g[JS.core.outIdx[q]] = 0; }
    const cl = body.clamp;                       // hold the painted pixels for cl.steps steps (see visualizer.py: Session.set_state)
    if (cl && cl.steps > 0) {
      const iV = [], iG = [];
      for (let k = 0; k < JS.V.length; k++) { if (cl.maskV && cl.maskV[k]) iV.push(k); if (cl.maskG && cl.maskG[k]) iG.push(k); }
      if (iV.length || iG.length) JS.clamp = { start: JS.n, steps: Math.floor(cl.steps), iV: Int32Array.from(iV), iG: Int32Array.from(iG), vV: Float64Array.from(iV, k => JS.V[k]), vG: Float64Array.from(iG, k => JS.g[k]) };
    }
    return { frame: jsFrame(!!body.want_ebar, 0, 0, body.mask_ver) };
  }
  if (path === '/api/step') {
    const c = JS.core; let steps = Math.max(0, Math.floor(numOr(body.steps, 1)));
    if (body.until !== null && body.until !== undefined) steps = Math.min(steps, Math.max(0, Math.round(body.until / c.dt) - JS.n));
    const t0 = performance.now(), maxMs = numOr(body.max_ms, 150); let done = 0;
    const cl = JS.clamp;
    while (done < steps) {
      for (let i = 0, q = Math.min(25, steps - done); i < q; i++) {
        c.step(JS.V, JS.g); done++;
        if (JS.devo && (JS.n + done) % JS.devK === 0) {                                  // development: grow the tissue (and move the range) on schedule
          const st = PotentiaCoreLib.PotentiaCore.devUpdate(c, JS.devo, JS.n + done, JS.V, JS.g, JS.devRng, JS.devFill[0], JS.devFill[1]);
          JS.devStatus = { ...st, range_um: st.rangeUm }; if (st.changed) JS.maskVer++;
        }
        if (cl) { const cc = JS.n + done - cl.start; if (cc > 0 && cc < cl.steps) { for (let k = 0; k < cl.iV.length; k++) JS.V[cl.iV[k]] = cl.vV[k]; for (let k = 0; k < cl.iG.length; k++) JS.g[cl.iG[k]] = cl.vG[k]; } }   // hold the painted values for the next step
      }
      if (performance.now() - t0 > maxMs) break;
    }
    JS.n += done; if (cl && JS.n - cl.start >= cl.steps) JS.clamp = null;                    // released
    for (let k = 0; k < JS.V.length; k++) if (!Number.isFinite(JS.V[k]) || !Number.isFinite(JS.g[k])) { JS.diverged = true; break; }
    return { frame: jsFrame(!!body.want_ebar, done, performance.now() - t0, body.mask_ver) };
  }
  throw new Error('not found');
}

const num = id => parseFloat($(id).value);
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------- parameters <-> controls ----------
function fillParams(p) {
  $('Rs').value = p.fieldScreenSize; syncPctFromRs();
  $('bias').value = p.bias; $('fs').value = p.fieldStrength; $('gain').value = p.gain;
  $('weight').value = p.weight; $('tau').value = p.tau; $('GJ').value = p.GJStrength; $('GJ_sl').value = Math.min(1, p.GJStrength);
  $('vec').checked = p.fieldVector; syncBiasSlider(); updateGeo();
}
function syncBiasSlider() { if (D) $('bias_sl').value = Math.log10(num('bias') / D.params.bias); }
function leakValue() { const m = $('leakmode').value; return m === 'impl' ? null : m === 'one' ? 1.0 : num('leakv'); }
function gatherCfg() {
  const mode = document.querySelector('input[name=ic]:checked').value;
  return {
    size_cells: [parseInt($('nx').value), parseInt($('ny').value)], m: parseInt($('m').value), periodic: $('periodic').checked,
    shape: shapeName(), shapeP: shapeParam(), mask: $('devOn').checked ? devObj().maskAt(0) : pendingMask(), dev: devCfg(),
    params: { fieldScreenSize: num('Rs'), fieldStrength: num('fs'), gain: num('gain'), weight: num('weight'), bias: num('bias'),
              tau: num('tau'), GJStrength: num('GJ'), fieldVector: $('vec').checked, leak: leakValue() },
    ic: { mode, g0: num('g0'), branch: $('branch').value, sigmaG: num('sigG'), seed: parseInt($('seed').value) || 0, uniform: $('uniform').checked },
    want_ebar: view === 'e'
  };
}
function updateGeo() {
  const nx = parseInt($('nx').value) || 0, ny = parseInt($('ny').value) || 0, m = parseInt($('m').value) || 1;
  const Rs = num('Rs'), h = PITCH_UM / m, R = rangeUm(Rs);
  $('range-note').textContent = `= ${R.toFixed(1)} µm of the ${tissueMinUm()} µm shorter side · R_s = ${Rs.toFixed(2)} in the model's units (cell half-diagonals)`;
  let txt = `tissue ${nx * PITCH_UM} × ${ny * PITCH_UM} µm · grid ${nx * m} × ${ny * m} px · pixel ${h.toFixed(2)} µm · field range ${R.toFixed(1)} µm = ${(R / h).toFixed(1)} px`;
  const tot = nx * ny * m * m, side = Math.max(nx, ny) * m;
  let warn = false;
  if (D && (side > D.max_side_px || tot > D.max_total_px)) { txt += ' — too large: lower m or the tissue size'; warn = true; }
  else if (R / h < 3) { txt += ' — range under ~3 px: not resolved at this m (raise m)'; warn = true; }
  if (MODE === 'js' && !warn && nx >= 2 && ny >= 2 && estStepMs(nx, ny, m, Rs, $('vec').checked, $('periodic').checked) > HEAVY_MS) { const a = heavyAdvice(nx, ny, m, Rs, $('vec').checked, $('periodic').checked); txt += ` — too heavy for the in-browser version (est. over ${HEAVY_MS} ms per step); ${a.fix}`; warn = true; }
  else if (MODE === 'js' && !warn && nx >= 2 && ny >= 2) { const sp = estSpeed(nx, ny, m, Rs, $('vec').checked, num('GJ') || 0, $('periodic').checked); txt += ` · est. speed ${sp < 10 ? sp.toFixed(1) : sp.toFixed(0)} simulated s per wall s in this browser`; if (sp < 1.5) { txt += ' (slow: lower the field range, m or the tissue size)'; warn = true; } }
  $('geo').textContent = txt; $('geo').className = 'note' + (warn ? ' warn' : '');
  const per = $('periodic').checked, Lmin = tissueMinUm(); let bc = 'Finite tissue: no-flux edges (the default).', bcWarn = false;
  if (per) {
    bc = 'Periodic: field sources, gap junctions and painted pixels wrap around the edges, so there is no edge.';
    if (R > Lmin / 2) { bc += ` The field range (${R.toFixed(0)} µm) exceeds half the shorter side (${(Lmin / 2).toFixed(0)} µm): each cell also senses wrapped images of the sources.`; bcWarn = true; }
    if (selectedIC() === 'homogeneous') { bc += ' A homogeneous start has no asymmetry to grow from, so it stays uniform (up to round-off): use a random start, or paint a perturbation.'; bcWarn = true; }
  }
  if ($('devOn').checked) bc = 'Development: the tissue shape comes from the frog schedule (no-flux edge); periodic boundaries are off.';
  else if (!per && ELONG[shapeName()] && pxDims()[1] < 0.9 * ELONG[shapeName()] * pxDims()[0]) { const m = parseInt($('m').value) || 1; bc = `This shape needs a box at least ${ELONG[shapeName()]} times wider than tall (at least ${Math.ceil(ELONG[shapeName()] * pxDims()[0] / m)} columns for ${pxDims()[0] / m} rows); in this box it is almost a disc.`; bcWarn = true; }
  else if (!per && shapeName() !== 'rectangle') { const pm = pendingMask(); let nt = 0; if (pm) for (let k = 0; k < pm.length; k++) nt += pm[k]; bc = `Tissue shape: ${nt} of ${pm ? pm.length : 0} pixels (${pm ? (100 * nt / pm.length).toFixed(0) : 0}% of the box); its boundary is a no-flux gap-junction edge. Periodic boundaries need the plain rectangle.`; }
  $('bc-note').textContent = bc; $('bc-note').className = 'note' + (bcWarn ? ' warn' : '');
  devNote();
  if (D) { const b = num('bias') / D.params.bias; $('bias-note').textContent = `bias = ${isFinite(b) ? b.toFixed(3) : '–'} × the Stigmergic bias (${D.params.bias.toExponential(3)})`; }
}
function markDirty() { needInit = true; setStatus('settings changed: press Initialise', 'warn'); updateGeo(); if (info) drawClamp(); }

// ---------- rendering ----------
function arrFor(v) { return v === 'V' ? curV : v === 'g' ? curG : curE; }
function resizeStage() {
  if (!info) return;
  const [R, C] = info.Np, availW = Math.min($('viewgroup').clientWidth, 760), narrow = window.matchMedia('(max-width: 900px)').matches;
  // desktop: the stage gets whatever height the left column has once the colour bar, readouts (including the development line) and time series have their own; measured, so it
  // also holds on short windows and when a readout wraps (a fixed reserve cut the time series off at heights around 600 px)
  const left = $('left'), other = ($('viewgroup').offsetHeight - $('stage').offsetHeight) + $('ts').offsetHeight + ($('aux').hidden ? 0 : $('aux').offsetHeight) + 16;    // everything in the column except the stage
  const maxH = narrow ? Math.max(180, Math.round(window.innerHeight * 0.36)) : Math.max(150, left.clientHeight - other);
  let w = availW, h = w * R / C; if (h > maxH) { h = maxH; w = h * C / R; }
  $('stage').style.width = w + 'px'; $('stage').style.height = h + 'px';
  for (const id of ['grid', 'cursor', 'mask']) { const c = $(id), dpr = window.devicePixelRatio || 1; c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  drawGrid(); drawClamp(); drawTS();
}
function draw() {
  if (!info || !curV) return;
  const [R, C] = info.Np, cv = $('view');
  if (cv.width !== C || cv.height !== R) { cv.width = C; cv.height = R; }
  const ctx = cv.getContext('2d'), img = ctx.createImageData(C, R), arr = arrFor(view);
  const [lo, hi] = [num('vmin'), num('vmax')], lut = view === 'g' ? MAGMA : VIRIDIS, span = hi - lo || 1;
  const mk = curMask && curMask.length === R * C ? curMask : null;
  if (arr) for (let k = 0; k < R * C; k++) {
    const o = 4 * k;
    if (mk && !mk[k]) { img.data[o + 3] = 0; continue; }                              // outside the tissue shape: transparent (the stage shows a checkerboard)
    let u = (arr[k] - lo) / span; u = u < 0 ? 0 : u > 1 ? 1 : u;
    const i3 = Math.round(u * 255) * 3;
    img.data[o] = lut[i3]; img.data[o + 1] = lut[i3 + 1]; img.data[o + 2] = lut[i3 + 2]; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  drawColorbar(lut, lo, hi);
}
function drawColorbar(lut, lo, hi) {
  const c = $('cbar'), ctx = c.getContext('2d'), img = ctx.createImageData(c.width, 1);
  for (let x = 0; x < c.width; x++) { const i3 = Math.round(x / (c.width - 1) * 255) * 3; img.data.set([lut[i3], lut[i3 + 1], lut[i3 + 2], 255], 4 * x); }
  for (let y = 0; y < c.height; y++) ctx.putImageData(img, 0, y);
  $('cb-lo').textContent = (+lo).toPrecision(3); $('cb-hi').textContent = (+hi).toPrecision(3);
  $('cb-unit').textContent = view === 'V' ? 'mV' : view === 'g' ? 'G_pol / G_ref' : '|E| sensed / bias';
}
function outlinePath(ctx, m, R, C, sx, sy) {            // path along the boundary of the set of pixels in mask m
  const per = info && info.periodic;                       // periodic tissue: a painted region continues across the edge
  const on = (i, j) => { if (per) { i = (i + R) % R; j = (j + C) % C; } return i >= 0 && j >= 0 && i < R && j < C && m[i * C + j]; };
  ctx.beginPath();
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (m[i * C + j]) {
    if (!on(i - 1, j)) { ctx.moveTo(j * sx, i * sy); ctx.lineTo((j + 1) * sx, i * sy); }
    if (!on(i + 1, j)) { ctx.moveTo(j * sx, (i + 1) * sy); ctx.lineTo((j + 1) * sx, (i + 1) * sy); }
    if (!on(i, j - 1)) { ctx.moveTo(j * sx, i * sy); ctx.lineTo(j * sx, (i + 1) * sy); }
    if (!on(i, j + 1)) { ctx.moveTo((j + 1) * sx, i * sy); ctx.lineTo((j + 1) * sx, (i + 1) * sy); }
  }
}
function drawClamp() {         // dashed outline: pixels currently held by a clamp; solid outline: pixels painted but not applied yet (visible in every view)
  const c = $('mask'), ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height);
  if (!info) return;
  const [R, C] = info.Np, sx = c.width / C, sy = c.height / R;
  const om = overlayMask();
  if (om) {                                                                          // preview of the shape Initialise will build: dim outside, cyan outline
    const off = document.createElement('canvas'); off.width = C; off.height = R; const o = off.getContext('2d'), img = o.createImageData(C, R);
    for (let k = 0; k < R * C; k++) if (!om[k]) { img.data[4 * k] = 15; img.data[4 * k + 1] = 15; img.data[4 * k + 2] = 30; img.data[4 * k + 3] = 175; }
    o.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, c.width, c.height);
    outlinePath(ctx, om, R, C, sx, sy); ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.stroke(); ctx.lineWidth = 1.6; ctx.strokeStyle = '#00d2ff'; ctx.stroke();
  }
  if (clampShown && clampLeft > 0) {
    outlinePath(ctx, clampShown, R, C, sx, sy);
    ctx.setLineDash([6, 4]); ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.stroke(); ctx.lineWidth = 1.6; ctx.strokeStyle = '#ffffff'; ctx.stroke(); ctx.setLineDash([]);
  }
  if (maskG && maskV) {
    const pend = Uint8Array.from(maskG, (v, k) => v | maskV[k]);
    if (pend.some(v => v)) { outlinePath(ctx, pend, R, C, sx, sy); ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.stroke(); ctx.lineWidth = 1.6; ctx.strokeStyle = '#ffd400'; ctx.stroke(); }
  }
}
let BIST = null;
const bistable = () => BIST || (BIST = PotentiaCoreLib.PotentiaCore.bistableWindow(1.5));        // [g_low, g_high] where two Vmem branches exist
function paintHint() {                                    // what was just painted and when it takes effect (the paint may not be visible in every view)
  let n = 0; for (let k = 0; k < maskG.length; k++) if (maskG[k] | maskV[k]) n++;
  if (!n) return;
  const tgt = $('ptarget').value, val = num('pval'), w = bistable(), inWin = tgt === 'g' && val >= w[0] && val <= w[1];
  let msg = `painted ${tgt === 'g' ? 'G_pol = ' + val : 'Vmem = ' + val + ' mV'} on ${n} pixels (yellow outline); applied when you press Run or +1 s`;
  if (inWin && view === 'V') msg += '. G_pol in the bistable window hardly changes Vmem: see the G_pol view';
  if (view === 'e') msg += '. The sensed-field view updates once the tissue runs';
  if (needInit) msg += '. Settings have changed since the last Initialise, so Run would restart from the background and discard this paint: press Initialise first, then paint';
  setStatus(msg);
}
function updateClampNote() {
  const N = Math.floor(num('clampN')), dt = info ? info.dt : 0.01;
  $('clampNote').textContent = N > 0 ? `${N} steps = ${(N * dt).toFixed(2)} s at dt = ${dt} s. The painted G_pol (and the matching Vmem, where Vmem follows it) is held for that many steps from the moment the paint is applied (Run or +1 s), then released. Painting again replaces a clamp that is still running.` : 'Enter a number of steps (1 or more).';
}
function drawGrid() {
  const c = $('grid'), ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height);
  if (!info || !$('showgrid').checked) return;
  const [R, C] = info.Np, m = info.m; ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1; ctx.beginPath();
  for (let j = 0; j <= C; j += m) { const x = Math.round(j / C * c.width) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, c.height); }
  for (let i = 0; i <= R; i += m) { const y = Math.round(i / R * c.height) + 0.5; ctx.moveTo(0, y); ctx.lineTo(c.width, y); }
  ctx.stroke();
}
function drawTS() {
  const c = $('ts'), dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const cs = getComputedStyle(document.documentElement), ink = cs.getPropertyValue('--ink').trim(), ink2 = cs.getPropertyValue('--ink2').trim(), line = cs.getPropertyValue('--line').trim();
  const panels = [['mean Vmem (mV)', hist.V, 'var(--s1)', null], ['hyperpolarised fraction (V < −30 mV)', hist.fh, 'var(--s2)', [0, 1]], ['mean g = G_pol / G_ref', hist.g, 'var(--s3)', [0, 2]]];
  const tmax = Math.max(num('T') || 1, hist.t.length ? hist.t[hist.t.length - 1] : 1), left = 46, right = 8, ph = (H - 22) / 3;
  ctx.font = '11px system-ui, sans-serif';
  panels.forEach(([title, ys, col, fixed], k) => {
    const top = 4 + k * ph, bot = top + ph - 14;
    let lo, hi;
    if (fixed) [lo, hi] = fixed; else { lo = Infinity; hi = -Infinity; for (const v of ys) { lo = Math.min(lo, v); hi = Math.max(hi, v); } if (!isFinite(lo)) { lo = -60; hi = 0; } const pad = (hi - lo) * 0.1 || 1; lo -= pad; hi += pad; }
    ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.strokeRect(left + 0.5, top + 0.5, W - left - right, bot - top);
    ctx.fillStyle = ink2; ctx.textAlign = 'right'; ctx.fillText(hi.toPrecision(3), left - 4, top + 9); ctx.fillText(lo.toPrecision(3), left - 4, bot);
    ctx.textAlign = 'left'; ctx.lineJoin = 'round'; ctx.lineWidth = 3.5; ctx.strokeStyle = cs.getPropertyValue('--bg').trim(); ctx.strokeText(title, left + 6, top + 11); ctx.fillStyle = ink; ctx.fillText(title, left + 6, top + 11);
    ctx.strokeStyle = cs.getPropertyValue(col.slice(4, -1)).trim();
    ctx.lineWidth = 1.6; ctx.beginPath();
    ys.forEach((v, i) => { const x = left + (hist.t[i] / tmax) * (W - left - right), y = bot - (v - lo) / (hi - lo || 1) * (bot - top); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  });
  ctx.fillStyle = ink2; ctx.textAlign = 'center'; ctx.fillText(`time (s)  0 … ${tmax.toPrecision(3)}`, left + (W - left - right) / 2, H - 3);
}

// ---------- frames ----------
function applyFrame(f) {
  if (f.mask_ver !== undefined) maskVer = f.mask_ver;
  if (f.mask) curMask = u8(f.mask);                                                  // the tissue changed (development): take the new mask
  if (f.dev) showDev(f.dev); else setDevReadout(false);
  curV = f32(f.V); curG = f32(f.g); curE = f.ebar ? f32(f.ebar) : curE;
  dirtyState = false;
  stepNow = f.step; clampLeft = f.clamp_left || 0; $('r-clamp').hidden = clampLeft <= 0; $('r-clampn').textContent = clampLeft; drawClamp();
  $('r-t').textContent = f.t.toFixed(2); $('r-step').textContent = f.step;
  $('r-v').textContent = f.stats.meanV.toFixed(1); $('r-fh').textContent = (100 * f.stats.fh).toFixed(0) + '%'; $('r-g').textContent = f.stats.meang.toFixed(2);
  if (hist.t.length === 0 || f.t > hist.t[hist.t.length - 1] + 1e-9) {
    hist.t.push(f.t); hist.V.push(f.stats.meanV); hist.fh.push(f.stats.fh); hist.g.push(f.stats.meang);
    if (hist.t.length > 3000) for (const k in hist) hist[k] = hist[k].filter((_, i) => i % 2 === 0);
  } else if (f.t < hist.t[hist.t.length - 1]) { hist = { t: [f.t], V: [f.stats.meanV], fh: [f.stats.fh], g: [f.stats.meang] }; }
  draw(); drawTS();
  const rh = $('readout').offsetHeight; if (rh !== lastReadoutH) { lastReadoutH = rh; if (info) resizeStage(); }      // a readout line appeared, disappeared or wrapped: the stage gets the height that is left
}

async function initialise() {
  stopRun();
  devPreview = false; lastStage = -2; maskVer = -1;
  const cfg = gatherCfg(), pm = cfg.mask;
  if (MODE !== 'js' && pm) cfg.mask = toB64u8(pm);                                    // the Python server takes the shape as base64 bytes
  try {
    setStatus('initialising…');
    const r = await api('/api/init', cfg);
    curMask = pm ? Uint8Array.from(pm) : null;
    info = r.info; hist = { t: [], V: [], fh: [], g: [] }; effSpeed = null; $('r-speed').textContent = '–';
    $('dtnote').textContent = info.dt.toExponential(2) + ' s' + (info.dt < 0.01 ? ' (reduced for GJ stability)' : '');
    maskG = new Uint8Array(info.Np[0] * info.Np[1]); maskV = new Uint8Array(maskG.length); clampShown = null; clampLeft = 0; updateClampNote();
    needInit = false; resizeStage(); applyFrame(r.frame); updateRangeInputs(false);
    $('bc').textContent = info.periodic ? 'periodic tissue (torus)' : 'finite tissue';
    if (MODE === 'js') $('backend').textContent = `· runs in this browser (${info.field === 'fft' ? 'FFT field' : PotentiaCoreLib.PotentiaCore.simdAvailable() && $('vec').checked ? 'direct field sum, WebAssembly SIMD' : 'direct field sum, JavaScript'})`;
    setStatus(`ready · grid ${info.Np[0]}×${info.Np[1]} px${info.periodic ? ' · periodic' : ''}`);
  } catch (e) { setStatus(e.message, 'bad'); }
}
async function syncState() {
  if (!dirtyState) return;
  const N = Math.floor(num('clampN')), any = maskG.some(v => v) || maskV.some(v => v), hold = $('clampOn').checked && N > 0 && any;
  const enc = a => MODE === 'js' ? a.slice() : toB64u8(a);
  const body = MODE === 'js' ? { V: curV, g: curG, want_ebar: view === 'e', mask_ver: maskVer } : { V: toB64(curV), g: toB64(curG), want_ebar: view === 'e', mask_ver: maskVer };
  if (hold) body.clamp = { steps: N, maskV: enc(maskV), maskG: enc(maskG) };
  const r = await api('/api/state', body);
  clampShown = hold ? Uint8Array.from(maskG, (v, k) => v | maskV[k]) : null;                    // outline of what is held
  maskG.fill(0); maskV.fill(0);
  clampLeft = r.frame.clamp_left || 0; $('r-clamp').hidden = clampLeft <= 0; $('r-clampn').textContent = clampLeft; drawClamp();
  dirtyState = false;
}
async function advance(seconds, token) {
  const f = await api('/api/step', { steps: Math.max(1, Math.round(seconds / info.dt)), until: null, max_ms: 1e9, want_ebar: view === 'e', mask_ver: maskVer });
  applyFrame(f.frame);
}
function stopRun() { running = false; runToken++; $('b-run').textContent = 'Run'; }
async function runLoop() {
  const token = ++runToken; running = true; $('b-run').textContent = 'Pause';
  try {
    if (needInit) await initialise();
    if (!info) return;
    running = true; runToken = token; $('b-run').textContent = 'Pause';
    await syncState();
    let T = num('T'); if (hist.t.length && hist.t[hist.t.length - 1] >= T - 1e-9) { T = hist.t[hist.t.length - 1] + Math.max(10, T); $('T').value = T; }
    devPreview = false; if (info) drawClamp();
    setStatus('running…');
    let lastT = hist.t.length ? hist.t[hist.t.length - 1] : 0, lastW = performance.now();
    while (running && token === runToken) {
      const frameMs = 40, t0 = performance.now(), speed = Math.max(0.1, num('speedN') || 20);
      const steps = Math.max(1, Math.round(speed * frameMs / 1000 / info.dt));
      const r = await api('/api/step', { steps, until: T, max_ms: MODE === 'js' ? 34 : 120, want_ebar: view === 'e', mask_ver: maskVer });
      if (token !== runToken) break;
      applyFrame(r.frame);
      const now = performance.now(), inst = (r.frame.t - lastT) / ((now - lastW) / 1000);
      effSpeed = effSpeed === null ? inst : 0.8 * effSpeed + 0.2 * inst; lastT = r.frame.t; lastW = now;
      $('r-speed').textContent = effSpeed.toFixed(1);
      if (r.frame.diverged) { setStatus('diverged (non-finite values): reduce parameters or press Initialise', 'bad'); break; }
      if (r.frame.t >= T - 1e-9) { setStatus(`finished at t = ${r.frame.t.toFixed(1)} s`); break; }
      await sleep(Math.max(0, frameMs - (performance.now() - t0)));   // always yield so the page stays responsive
    }
  } catch (e) { setStatus(e.message, 'bad'); }
  if (token === runToken) { running = false; $('b-run').textContent = 'Run'; if ($('status').textContent === 'running…') setStatus('paused'); }
}

// ---------- painting ----------
function pixelAt(ev) {
  const r = $('cursor').getBoundingClientRect(), [R, C] = info.Np;
  return [Math.floor((ev.clientY - r.top) / r.height * R), Math.floor((ev.clientX - r.left) / r.width * C)];
}
function paintAt(i0, j0) {
  const [R, C] = info.Np, rad = num('prad'), tgt = $('ptarget').value, val = num('pval');
  const arr = tgt === 'V' ? curV : curG, v = tgt === 'g' ? Math.min(2, Math.max(0, val)) : val;
  // Vmem follows the painted G_pol on a fresh tissue, and on a running one wherever g is monostable (Vmem would relax there anyway); inside the bistable window a running cell keeps its branch
  const mono = v < bistable()[0] || v > bistable()[1], follow = tgt === 'g' && (stepNow === 0 || mono), vEq = follow ? PotentiaCoreLib.PotentiaCore.equilibriumV(v, $('branch').value, 1.5) * 1e3 : 0;
  let n = 0;                                                                         // pixels actually painted (pixels outside the tissue are ignored)
  const ci = [i0], cj = [j0];
  if ($('miry').checked) ci.push(R - 1 - i0); if ($('mirx').checked) cj.push(C - 1 - j0);
  const rr = Math.ceil(rad);
  for (const ic of ci) for (const jc of cj)
    for (let di = -rr; di <= rr; di++) for (let dj = -rr; dj <= rr; dj++) {
      if (di * di + dj * dj > rad * rad) continue;
      let i = ic + di, j = jc + dj;
      if (info.periodic) { i = (i % R + R) % R; j = (j % C + C) % C; } else if (i < 0 || j < 0 || i >= R || j >= C) continue;   // the brush wraps around in a periodic tissue
      const k = i * C + j; if (curMask && !curMask[k]) continue;                                   // nothing to paint outside the tissue
      n++; arr[k] = v; if (follow) curV[k] = vEq;
      if (tgt === 'g') { maskG[k] = 1; if (follow) maskV[k] = 1; } else maskV[k] = 1;                   // remembered for the clamp (held together with the Vmem that follows G_pol)
    }
  if (n) dirtyState = true;
  return n;
}
function drawBrush(ev) {
  const c = $('cursor'), ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height);
  const tool = paintEnabled() || $('shapeEdit').checked;
  if (!ev || !info || !tool || (running && !$('shapeEdit').checked)) return;
  const r = c.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * c.width, y = (ev.clientY - r.top) / r.height * c.height;
  const [R, C] = info.Np, rad = num('prad') / C * c.width;
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, Math.max(rad, 2), 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 0.5; ctx.stroke();
}
function selectedIC() { return document.querySelector('input[name=ic]:checked').value; }
// ---------- development (schematic frog sequence) ----------
const stageInputs = () => Array.from(document.querySelectorAll('#devStageInputs input')).map(i => parseFloat(i.value));
function devCfg() { return $('devOn').checked ? { enabled: true, duration: num('devD'), update: num('devU'), range: { mode: $('devRange').value, pct: num('devPct'), stage_um: stageInputs() } } : null; }
function devObj() {                                // the schedule for the pending grid (masks and distance fields are rebuilt only when the grid changes)
  const [R, C] = pxDims(), key = R + 'x' + C;
  if (!(R >= 6 && C >= 6)) throw new Error('the grid is too small for the development sequence');
  if (devKey !== key || !devCached) {
    devCached = new PotentiaCoreLib.PotentiaCore.Development(R, C, num('devD') || 300, num('devU') || 0.5); devKey = key;
    devThumbs = devCached.masks.map(mk => { const c = document.createElement('canvas'); c.width = C; c.height = R; const x = c.getContext('2d'), img = x.createImageData(C, R);
      for (let k = 0; k < R * C; k++) if (mk[k]) { img.data[4 * k] = 47; img.data[4 * k + 1] = 127; img.data[4 * k + 2] = 209; img.data[4 * k + 3] = 255; }
      x.putImageData(img, 0, 0); return c; });
  }
  devCached.D = num('devD') || 300; devCached.update = num('devU') || 0.5;
  return devCached;
}
function drawDevStrip(active) {                    // the six keyframe shapes as thumbnails, the current stage outlined
  const c = $('devStrip'); if (!$('devOn').checked || !c) return;
  const [R, C] = pxDims(); if (!(R >= 6 && C >= 6)) return;
  const d = devObj(), dpr = window.devicePixelRatio || 1, W = c.clientWidth || 280, cols = 3, gap = 8, labelH = 15, tw = (W - gap * (cols - 1)) / cols, th = tw * R / C, rows = Math.ceil(d.stages.length / cols), H = rows * (th + labelH + gap);
  c.style.height = H + 'px'; c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
  const x = c.getContext('2d'), cs = getComputedStyle(document.documentElement), ink = cs.getPropertyValue('--ink').trim() || '#222', line = cs.getPropertyValue('--line').trim() || '#999';
  x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, H); x.font = '11px system-ui, sans-serif'; x.textBaseline = 'top';
  d.stages.forEach((st, i) => {
    const px = (i % cols) * (tw + gap), py = Math.floor(i / cols) * (th + labelH + gap);
    x.fillStyle = ink; x.fillText(`${st.name} (${st.nf})`, px, py);
    x.imageSmoothingEnabled = false; x.drawImage(devThumbs[i], px, py + labelH, tw, th);
    x.lineWidth = i === active ? 2.5 : 1; x.strokeStyle = i === active ? '#d97706' : line; x.strokeRect(px + 0.5, py + labelH + 0.5, tw - 1, th - 1);
  });
}
function setDevReadout(on) { if ($('r-dev').hidden === on) { $('r-dev').hidden = !on; if (info) resizeStage(); } }
function showDev(dv) {
  lastDevStatus = dv;
  $('r-stage').textContent = `${dv.stage} (${dv.nf})`; $('r-tissue').textContent = dv.tissue; $('r-range').textContent = (+dv.range_um).toFixed(0);
  setDevReadout(true);
  const idx = devObj().stages.findIndex(st => st.name === dv.stage);
  if (idx !== lastStage) { lastStage = idx; drawDevStrip(idx); }
}
function devNote() {
  if (!$('devOn').checked) return;
  const [R, C] = pxDims(); if (!(R >= 6 && C >= 6)) { $('dev-note').textContent = 'The grid is too small for the development sequence.'; $('dev-note').className = 'note warn'; return; }
  const m = parseInt($('m').value) || 1, d = devObj(), ext = mk => { let r0 = 1e9, r1 = -1, c0 = 1e9, c1 = -1; for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (mk[i * C + j]) { if (i < r0) r0 = i; if (i > r1) r1 = i; if (j < c0) c0 = j; if (j > c1) c1 = j; } return [(r1 - r0 + 1) / m, (c1 - c0 + 1) / m]; };
  const egg = ext(d.masks[0]), fin = ext(d.masks[d.masks.length - 1]), D = num('devD'), T = num('T'); let w = false;
  let txt = `Grid ${R}×${C} px. The egg is about ${egg[1].toFixed(0)} cells across (${(egg[1] * PITCH_UM).toFixed(0)} µm), the tadpole ${fin[1].toFixed(0)} cells (${(fin[1] * PITCH_UM / 1000).toFixed(2)} mm) long; development takes ${D} simulated s and the final shape is then held. Run time T = ${T} s.`;
  if (T < D) { txt += ' T is shorter than the development: raise T to see all stages.'; w = true; }
  if (C / R < 2.6) { txt += ' The sequence assumes a grid at least about 2.7 times wider than tall (try 50 × 130 cells).'; w = true; }
  $('dev-note').textContent = txt; $('dev-note').className = 'note' + (w ? ' warn' : '');
  const tau = parseFloat($('devScrub').value), st = d.stageAt(tau), stale = info && (R !== info.Np[0] || C !== info.Np[1]);
  $('devScrubLabel').textContent = `${st.name} (${st.nf}), ${(tau * D).toFixed(0)} s${stale ? ' (press Initialise to see it on the field)' : ''}`;
}
// ---------- tissue shape ----------
const SHAPE_INFO = { disc: 'circle of the shorter side: blastoderm, explant, micropatterned colony', ellipse: 'inscribed in the box: egg or embryo outline', annulus: 'ring with a hole: gut or vessel cross-section, wound-healing assay',
  stadium: 'capsule along the longer side: elongated body', hexagon: 'regular hexagon: epithelial cluster', triangle: 'equilateral triangle', cross: 'plus-shaped explant',
  dumbbell: 'two discs along the longer side joined by a neck; neck 0 gives two islands with no gap junctions between them, which interact only through the field if they are closer than the field range',
  tadpole: 'round head and a narrow tail along the longer side (use an elongated box)', custom: 'drawn by hand: starts as a copy of the previous shape' };
const ELONG = { stadium: 2.0, dumbbell: 2.5, tadpole: 2.6 };         // columns / rows that these shapes need (in a squarer box they degenerate towards a disc)
const pxDims = () => { const m = parseInt($('m').value) || 1; return [(parseInt($('nx').value) || 0) * m, (parseInt($('ny').value) || 0) * m]; };
const shapeName = () => $('shape').value;
function shapeParam() { const v = parseFloat($('shapeP').value); return Number.isFinite(v) ? v : undefined; }
function pendingMask() {                          // the shape that Initialise would build: Uint8Array on the pixel grid, or null for the plain rectangle
  const name = shapeName(), [R, C] = pxDims();
  if (name === 'rectangle' || !(R > 0 && C > 0) || (name !== 'custom' && !PotentiaCoreLib.PotentiaCore.SHAPES[name])) return null;
  if (name === 'custom') { if (!customMask || customDims[0] !== R || customDims[1] !== C) { customMask = new Uint8Array(R * C).fill(1); customDims = [R, C]; } return customMask; }
  const key = name + '|' + R + 'x' + C + '|' + shapeParam();
  if (pendKey !== key) { pendCache = PotentiaCoreLib.PotentiaCore.shapeMask(name, R, C, shapeParam()); pendKey = key; }
  return pendCache;
}
function overlayMask() {                          // the pending shape, when it should be previewed on the field: while drawing, or when it differs from the tissue on screen
  if (!info) return null;
  const [R, C] = pxDims(); if (R !== info.Np[0] || C !== info.Np[1]) return null;
  if ($('devOn').checked) return devPreview ? devObj().maskAt(parseFloat($('devScrub').value)) : null;     // preview of a stage chosen with the slider
  const pm = pendingMask() || new Uint8Array(R * C).fill(1);
  if (!$('shapeEdit').checked) {
    if (!needInit) return null;
    let same = true; for (let k = 0; k < R * C; k++) if (pm[k] !== (curMask ? curMask[k] : 1)) { same = false; break; }
    if (same) return null;
  }
  return pm;
}
function editShapeAt(i0, j0) {
  const [R, C] = info.Np, [pr, pc] = pxDims();
  if (pr !== R || pc !== C) { setStatus('press Initialise first: the grid size changed, so the shape cannot be drawn on this view', 'warn'); return; }
  if (shapeName() !== 'custom' || !pendingMask()) return;
  const add = $('shapeBrush').value === 'add', rad = num('prad'), rr = Math.ceil(rad);
  for (let di = -rr; di <= rr; di++) for (let dj = -rr; dj <= rr; dj++) {
    if (di * di + dj * dj > rad * rad) continue;
    const i = i0 + di, j = j0 + dj; if (i < 0 || j < 0 || i >= R || j >= C) continue;
    customMask[i * C + j] = add ? 1 : 0;
  }
  markDirty();
}
const paintEnabled = () => $('paintOn').checked;               // painting is a tool on the current tissue, independent of the initial-condition choice
function syncPaintUi() { const on = paintEnabled() || $('shapeEdit').checked; $('stage').style.touchAction = on ? 'none' : 'auto'; if (!on) drawBrush(null); }
let painting = false;
function bindPaint() {
  const c = $('cursor'); let mode = null, last = null;                              // mode: 'paint' or 'shape' during a drag; last: previous pixel of the stroke
  let nPainted = 0;
  const stamp = (i, j) => { if (mode === 'shape') editShapeAt(i, j); else if (!running) nPainted += paintAt(i, j); };
  const stroke = (i, j) => {                                                        // a fast drag is a few widely spaced pointer events: stamp the brush along the line between them
    const [li, lj] = last || [i, j], n = Math.max(Math.abs(i - li), Math.abs(j - lj), 1);
    for (let t = last ? 1 : 0; t <= n; t++) stamp(Math.round(li + (i - li) * t / n), Math.round(lj + (j - lj) * t / n));
    last = [i, j];
    if (mode === 'paint' && !running) { draw(); drawClamp(); }
  };
  c.addEventListener('pointerdown', ev => {
    if (!info) return;
    if ($('shapeEdit').checked) { mode = 'shape'; painting = true; last = null; c.setPointerCapture(ev.pointerId); const [i, j] = pixelAt(ev); stroke(i, j); ev.preventDefault(); return; }   // drawing the shape works while running too
    if (!paintEnabled()) return;
    if (running) { setStatus('pause to paint', 'warn'); return; }
    mode = 'paint'; painting = true; last = null; nPainted = 0; c.setPointerCapture(ev.pointerId); const [i, j] = pixelAt(ev); stroke(i, j); ev.preventDefault();
  });
  c.addEventListener('pointermove', ev => { drawBrush(ev); if (painting) { const [i, j] = pixelAt(ev); stroke(i, j); } });
  c.addEventListener('pointerup', () => { if (painting) { painting = false; if (mode === 'paint') { if (nPainted) paintHint(); else setStatus('nothing painted: that spot is outside the tissue (the checkerboard)' + ($('devOn').checked ? '; during development only the tissue that exists so far can be painted, with the run paused' : ''), 'warn'); } mode = null; last = null; } });
  c.addEventListener('pointerleave', () => drawBrush(null));
}

// ---------- view / ranges ----------
function updateRangeInputs(store) {
  if (store) { RANGES[view] = [num('vmin'), num('vmax')]; }
  [$('vmin').value, $('vmax').value] = RANGES[view];
}
async function setView(v) {
  RANGES[view] = [num('vmin'), num('vmax')]; view = v; [$('vmin').value, $('vmax').value] = RANGES[view];
  if (v === 'e' && info && !running) { try { await syncState(); const f = await api('/api/step', { steps: 0, want_ebar: true, mask_ver: maskVer }); applyFrame(f.frame); } catch (e) { setStatus(e.message, 'bad'); } }
  draw();
}

// ---------- wiring ----------
function bindSliderPair(numId, slId, toNum, toSl) {
  $(slId).addEventListener('input', () => { $(numId).value = toNum(parseFloat($(slId).value)); markDirty(); });
  $(numId).addEventListener('input', () => { $(slId).value = toSl(parseFloat($(numId).value)); });
}
async function main() {
  if (MODE === 'auto') {                       // local page: use the Python server if it answers, otherwise the in-browser core
    MODE = 'js';
    if (location.protocol !== 'file:' && !/[?&]js\b/.test(location.search)) {      // nothing to probe when opened from disk
      try { const ctl = new AbortController(), to = setTimeout(() => ctl.abort(), 1500), r = await fetch('/api/defaults', { signal: ctl.signal }); clearTimeout(to);
            if (r.ok) { const j = await r.json(); if (j && j.params) { D = j; MODE = 'server'; } } } catch (e) { /* no server */ }
    }
  }
  if (MODE === 'js') D = await jsApi('/api/defaults');
  $('backend').textContent = MODE === 'js' ? (PotentiaCoreLib.PotentiaCore.simdAvailable() ? '· runs in this browser (WebAssembly SIMD)' : '· runs in this browser (JavaScript)') : '· Python server';
  $('shape').innerHTML = Object.entries(PotentiaCoreLib.PotentiaCore.SHAPES).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('') + '<option value="custom">custom (draw)</option>';   // before the first updateGeo
  fillParams(D.params);
  // registered first. Resizing the tissue either keeps the range in um (default: the physical model does not change, and the cost of a step stays put) or keeps the percentage (the range scales with the tissue)
  ['nx', 'ny'].forEach(id => $(id).addEventListener('input', () => { if ($('holdpct').checked) syncRsFromPct(); else syncPctFromRs(); }));
  const dirtyIds = ['nx', 'ny', 'm', 'periodic', 'bias', 'fs', 'gain', 'weight', 'tau', 'vec', 'GJ', 'leakmode', 'leakv', 'g0', 'branch', 'sigG', 'seed', 'uniform'];
  dirtyIds.forEach(id => { $(id).addEventListener('input', markDirty); $(id).addEventListener('change', markDirty); });
  bindSliderPair('GJ', 'GJ_sl', x => x, x => Math.min(1, x));
  const onPct = v => { if (v > 0) { RPCT = v; syncRsFromPct(); markDirty(); } };
  $('Rpct').addEventListener('input', () => { onPct(parseFloat($('Rpct').value)); if (RPCT > 0) $('Rpct_sl').value = RPCT; });
  $('Rpct_sl').addEventListener('input', () => { onPct(parseFloat($('Rpct_sl').value)); showPct(); });
  bindSliderPair('bias', 'bias_sl', x => (D.params.bias * Math.pow(10, x)).toPrecision(4), x => Math.log10(x / D.params.bias));
  $('bias_sl').addEventListener('input', updateGeo); $('bias').addEventListener('input', updateGeo);
  bindSliderPair('pval', 'pval_sl', x => x, x => x); bindSliderPair('prad', 'prad_sl', x => x, x => x);
  $('speed').addEventListener('input', () => { $('speedN').value = Math.pow(10, parseFloat($('speed').value)).toFixed(1); });
  $('speedN').addEventListener('input', () => { $('speed').value = Math.log10(Math.max(0.5, num('speedN') || 1)); });
  $('leakmode').addEventListener('change', () => $('leakrow').classList.toggle('on', $('leakmode').value === 'custom'));
  document.querySelectorAll('input[name=ic]').forEach(r => r.addEventListener('change', () => {
    const v = selectedIC(); $('ic-random').classList.toggle('on', v === 'random');
    if (v === 'paint') $('paintOn').checked = true;                 // the 'paint' start = homogeneous background with painting on
    syncPaintUi(); markDirty();
  }));
  document.querySelectorAll('input[name=view]').forEach(r => r.addEventListener('change', () => setView(document.querySelector('input[name=view]:checked').value)));
  ['vmin', 'vmax'].forEach(id => $(id).addEventListener('input', draw));
  $('showgrid').addEventListener('change', drawGrid); $('T').addEventListener('input', () => { drawTS(); devNote(); });
  $('ptarget').addEventListener('change', () => {
    const g = $('ptarget').value === 'g'; $('pval_sl').min = g ? 0 : -60; $('pval_sl').max = g ? 2 : 0; $('pval_sl').step = g ? 0.05 : 0.5;
    $('pval').step = g ? 0.05 : 0.5; $('pval').value = g ? 2 : -55; $('pval_sl').value = $('pval').value;
  });
  $('b-init').addEventListener('click', initialise);
  $('b-run').addEventListener('click', () => { if (running) { stopRun(); setStatus('paused'); } else runLoop(); });
  $('b-step').addEventListener('click', async () => { try { stopRun(); if (needInit) await initialise(); await syncState(); await advance(1.0); setStatus('stepped +1 s'); } catch (e) { setStatus(e.message, 'bad'); } });
  $('b-defaults').addEventListener('click', () => { fillParams(D.params); $('leakmode').value = 'impl'; $('leakrow').classList.remove('on'); markDirty(); });
  $('b-clear').addEventListener('click', async () => { if (!info) return; await initialise(); setStatus('paint cleared: background restored'); });
  const showAux = node => { const box = $('aux-body'); box.replaceChildren(node); $('aux').hidden = false; };
  $('aux-close').addEventListener('click', () => { $('aux').hidden = true; });
  function pngCanvas() { const [R, C] = info.Np, s = Math.max(1, Math.ceil(640 / Math.max(R, C))), o = document.createElement('canvas'); o.width = C * s; o.height = R * s;
    const x = o.getContext('2d'); x.imageSmoothingEnabled = false; x.drawImage($('view'), 0, 0, o.width, o.height); return o; }
  const stateObject = () => ({ config: MODE === 'js' ? { ...JS.cfg, mask: undefined } : null, mask: curMask ? Array.from(curMask) : null, development: lastDevStatus, t: parseFloat($('r-t').textContent), shape: info.Np, V_mV: Array.from(curV), g: Array.from(curG) });
  if (BUILD.target === 'artifact') {           // file saves are blocked inside an artifact: show the PNG in the page, copy the state as text
    $('b-png').textContent = 'Show PNG'; $('b-export').textContent = 'Copy state (JSON)';
    $('b-png').addEventListener('click', () => { if (!info) return; const img = new Image(); img.src = pngCanvas().toDataURL('image/png'); img.alt = 'Potentia field';
      const wrap = document.createElement('div'); const p = document.createElement('p'); p.className = 'note'; p.textContent = 'Right-click (or long-press) the image to save it.'; wrap.append(p, img); showAux(wrap); });
    $('b-export').addEventListener('click', async () => { if (!info) return; const txt = JSON.stringify(stateObject());
      try { await navigator.clipboard.writeText(txt); setStatus('state copied to the clipboard as JSON'); }
      catch (e) { const ta = document.createElement('textarea'); ta.value = txt; ta.readOnly = true; showAux(ta); ta.focus(); ta.select(); setStatus('copy blocked: select the text below and copy it', 'warn'); } });
  } /*LOCAL_ONLY_BEGIN*/ else {
    $('b-png').addEventListener('click', () => { if (!info) return; pngCanvas().toBlob(b => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = `potentia_${view}_t${$('r-t').textContent}s.png`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }); });
    $('b-export').addEventListener('click', async () => {
      try { await syncState(); let b, name;
        if (MODE === 'js') { b = new Blob([JSON.stringify(stateObject())], { type: 'application/json' }); name = `potentia_state_t${$('r-t').textContent}s.json`; }
        else { const r = await fetch('/api/export'); if (!r.ok) throw new Error((await r.json()).error); b = await r.blob(); name = `potentia_state_t${$('r-t').textContent}s.npz`; }
        const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      } catch (e) { setStatus(e.message, 'bad'); }
    });
    if (MODE === 'js') $('b-export').textContent = 'Export state (.json)';
  } /*LOCAL_ONLY_END*/
  document.addEventListener('keydown', ev => { if (ev.code === 'Space' && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(ev.target.tagName)) { ev.preventDefault(); $('b-run').click(); } });
  window.addEventListener('resize', resizeStage);
  { const w = PotentiaCoreLib.PotentiaCore.bistableWindow(1.5), ed = PotentiaCoreLib.PotentiaCore.equilibriumV, lo = ed(0, 'dep') * 1e3, mid = ed(w[0] - 0.01, 'dep') * 1e3, hy = ed(2, 'hyp') * 1e3;
    $('eqnote').textContent = `Vmem is not set separately: it follows G_pol (isolated-cell equilibrium). g below ${w[0].toFixed(2)}: depolarised (${lo.toFixed(0)} to ${mid.toFixed(0)} mV); ${w[0].toFixed(2)} to ${w[1].toFixed(2)}: bistable, so the branch above decides (depolarised is the repo default); above ${w[1].toFixed(2)}: hyperpolarised (about ${hy.toFixed(0)} mV).`; }
  $('clampN').addEventListener('input', updateClampNote); updateClampNote();
  $('paintOn').addEventListener('change', () => {                  // switching the tool on or off never restarts the tissue
    if (paintEnabled()) $('shapeEdit').checked = false;
    syncPaintUi(); if (running) return;
    setStatus(paintEnabled() ? 'painting on: drag on the field; the tissue is not restarted' : 'painting off');
  });
  // ---- tissue shape ----
  { const PC = PotentiaCoreLib.PotentiaCore;
    const onShape = () => {
      let boxNote = '';
      if (ELONG[shapeName()] && !$('devOn').checked) {                              // an elongated shape in a squarer box is almost a disc: set an elongated box
        const m = parseInt($('m').value) || 1, nx = parseInt($('nx').value) || 0, ny = parseInt($('ny').value) || 0, want = Math.min(Math.round(ELONG[shapeName()] * nx), Math.floor((D ? D.max_side_px : 512) / m));
        if (nx >= 2 && ny < 0.9 * ELONG[shapeName()] * nx && want > ny) { $('ny').value = want; $('ny').dispatchEvent(new Event('input')); boxNote = ` The box was changed to ${nx} × ${want} cells because this shape needs a box at least ${ELONG[shapeName()]} times wider than tall.`; }
      }
      const name = shapeName(), sh = PC.SHAPES[name] || {}, [R, C] = pxDims();
      if (name === 'custom' && prevShape !== 'custom') { customMask = prevShape === 'rectangle' ? new Uint8Array(R * C).fill(1) : PC.shapeMask(prevShape, R, C, shapeParam()); customDims = [R, C]; }
      if (sh.plabel) { $('shapeP-label').textContent = sh.plabel; for (const id of ['shapeP', 'shapeP_sl']) { $(id).min = sh.range[0]; $(id).max = sh.range[1]; } $('shapeP').step = $('shapeP_sl').step = 0.01; $('shapeP').value = $('shapeP_sl').value = sh.p; }
      $('shapeP-wrap').classList.toggle('on', !!sh.plabel); $('shape-draw').classList.toggle('on', name === 'custom');
      if (name !== 'custom') { $('shapeEdit').checked = false; syncPaintUi(); }
      const rect = name === 'rectangle', lock = !rect || $('devOn').checked; $('periodic').disabled = lock; if (lock) $('periodic').checked = false;      // a shaped tissue has a no-flux edge, periodic needs the full rectangle
      $('shape-note').textContent = rect ? '' : (SHAPE_INFO[name] || '') + '. Outside the shape there is no tissue: no charge, no gap junctions (a no-flux edge).' + boxNote;
      prevShape = name; markDirty();
    };
    $('shape').addEventListener('change', onShape);
    $('shapeP').addEventListener('input', () => { $('shapeP_sl').value = $('shapeP').value; markDirty(); });
    $('shapeP_sl').addEventListener('input', () => { $('shapeP').value = $('shapeP_sl').value; markDirty(); });
    $('shapeEdit').addEventListener('change', () => { if ($('shapeEdit').checked) $('paintOn').checked = false; syncPaintUi(); if (info) drawClamp(); if (!running) setStatus($('shapeEdit').checked ? 'drawing the shape: drag on the field to ' + ($('shapeBrush').value === 'add' ? 'add' : 'erase') + ' tissue; press Initialise to build it' : 'shape drawing off'); });
    $('shape-fill').addEventListener('click', () => { const pm = pendingMask(); if (pm && shapeName() === 'custom') { pm.fill(1); markDirty(); } });
    $('shape-invert').addEventListener('click', () => { const pm = pendingMask(); if (pm && shapeName() === 'custom') { for (let k = 0; k < pm.length; k++) pm[k] = pm[k] ? 0 : 1; markDirty(); } });
    ['nx', 'ny', 'm'].forEach(id => $(id).addEventListener('input', () => { if (info) drawClamp(); }));
  }
  // ---- development ----
  { const PC = PotentiaCoreLib.PotentiaCore; let shapeBeforeDev = 'rectangle', stageEdited = false;
    $('devStageInputs').innerHTML = PC.STAGES.map((st, i) => `<label>${st.name}<input type="number" min="1" step="1" data-i="${i}" value="28.3"></label>`).join('');
    document.querySelectorAll('#devStageInputs input').forEach(i => i.addEventListener('input', () => { stageEdited = true; markDirty(); }));
    const fillStages = () => { const r = rangeUm(num('Rs')).toFixed(1); document.querySelectorAll('#devStageInputs input').forEach(i => { i.value = r; }); };
    const onDevRange = () => { const mode = $('devRange').value; $('dev-range-prop').classList.toggle('on', mode === 'prop'); $('dev-range-stage').classList.toggle('on', mode === 'stage'); if (mode === 'stage' && !stageEdited) fillStages(); markDirty(); };
    const onDevOn = async () => {
      const on = $('devOn').checked; $('dev-body').classList.toggle('on', on);
      if (on) {
        shapeBeforeDev = shapeName(); $('shapeEdit').checked = false; $('shape').value = 'rectangle'; $('shape').disabled = true; $('shape').dispatchEvent(new Event('change'));
        if ((parseInt($('ny').value) || 0) < 2.6 * (parseInt($('nx').value) || 1)) { $('nx').value = 50; $('ny').value = 130; for (const id of ['nx', 'ny']) $(id).dispatchEvent(new Event('input')); }
        if (num('T') < num('devD')) { $('T').value = num('devD') + 60; drawTS(); }
      } else { $('shape').disabled = false; $('shape').value = shapeBeforeDev; $('shape').dispatchEvent(new Event('change')); setDevReadout(false); }
      syncPaintUi(); lastStage = -2; devPreview = false; markDirty(); try { drawDevStrip(-1); } catch (e) { /* grid too small */ }
      if (on) { await initialise(); setStatus('development on: initialised at the egg; press Run (the stage slider previews the other stages)'); }
    };
    $('devOn').addEventListener('change', onDevOn); $('devRange').addEventListener('change', onDevRange);
    for (const id of ['devD', 'devU', 'devPct']) $(id).addEventListener('input', markDirty);
    $('devD').addEventListener('input', () => { try { drawDevStrip(lastStage); } catch (e) { /* ignore */ } });
    $('devScrub').addEventListener('input', () => { devPreview = true; devNote(); if (info) drawClamp(); });
    ['nx', 'ny', 'm'].forEach(id => $(id).addEventListener('input', () => { try { drawDevStrip(lastStage); } catch (e) { /* grid too small */ } }));
    window.addEventListener('resize', () => { try { drawDevStrip(lastStage); } catch (e) { /* ignore */ } });
  }
  bindPaint(); updateRangeInputs(false); updateGeo();
  await initialise();
}
main();
