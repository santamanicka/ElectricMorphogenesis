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
let effSpeed = null;
const RANGES = { V: [-55, -5], g: [0, 2], e: [0, 2] };
let view = 'V';

// ---------- helpers ----------
function setStatus(text, cls) { const s = $('status'); s.textContent = text; s.className = cls || ''; }
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
// rough cost model of a step (fitted to the benchmarks in core_test.js, within ~10%): per-pixel work plus probes x kernel pairs for the field sum
function estStepMs(nx, ny, m, Rs, vector) {
  const Rpx = A_UM * Math.SQRT2 * (Rs + 0.001) / (PITCH_UM / m), pairs = Math.PI * (Rpx + 0.5) * (Rpx + 0.5) / 2, probes = (nx * m + m) * (ny * m + m);
  const perPair = (vector && PotentiaCoreLib.PotentiaCore.simdAvailable()) ? 4.8e-7 : 1.7e-6;      // ms per probe-pair: WebAssembly SIMD vs JavaScript loop
  return 0.012 + 4.9e-5 * nx * m * ny * m + perPair * probes * pairs;
}
function estSpeed(nx, ny, m, Rs, vector, GJ) { const gjc = GJ * 10 * m * m, dt = gjc > 0 ? Math.min(0.01, 0.2 / gjc) : 0.01; return dt * 1000 / estStepMs(nx, ny, m, Rs, vector); }
// The field range is controlled as a percentage of the tissue's shorter side. The model parameter R_s (in cell half-diagonals; R = a*sqrt(2)*(R_s + 0.001)) is kept in the hidden
// input #Rs and derived from the percentage; RPCT is the control's exact value, so changing the tissue size keeps the percentage and the physical range scales with it.
let RPCT = 0;
const tissueMinUm = () => Math.min(parseInt($('nx').value) || 0, parseInt($('ny').value) || 0) * PITCH_UM;
const rangeUm = Rs => A_UM * Math.SQRT2 * (Rs + 0.001);
function showPct() { $('Rpct').value = RPCT.toFixed(1); $('Rpct_sl').value = RPCT; }
function syncPctFromRs() { const L = tissueMinUm(); RPCT = L > 0 ? 100 * rangeUm(num('Rs')) / L : 0; showPct(); }
function syncRsFromPct() { const L = tissueMinUm(); if (L > 0 && RPCT > 0) $('Rs').value = (RPCT / 100 * L) / (A_UM * Math.SQRT2) - 0.001; }
function jsFrame(wantE, done, wallMs) {
  const c = JS.core, n = c.Np0 * c.Np1, V = new Float32Array(n), g = new Float32Array(n);
  let s = 0, mn = Infinity, mx = -Infinity, hp = 0, sg = 0;
  for (let k = 0; k < n; k++) { const v = JS.V[k] * 1e3; V[k] = v; g[k] = JS.g[k]; s += v; if (v < mn) mn = v; if (v > mx) mx = v; if (v < -30) hp++; sg += JS.g[k]; }
  const out = { t: JS.n * c.dt, step: JS.n, Np: [c.Np0, c.Np1], V, g, steps_done: done, wall_ms: wallMs, diverged: JS.diverged,
                stats: { meanV: s / n, minV: mn, maxV: mx, fh: hp / n, meang: sg / n },
                clamp_left: JS.clamp ? Math.max(0, JS.clamp.steps - (JS.n - JS.clamp.start)) : 0 };
  if (wantE) out.ebar = Float32Array.from(c.perceivedOverBias(JS.V));
  return out;
}
async function jsApi(path, body) {
  if (path === '/api/defaults') return { params: STIG, cell_um: PITCH_UM, max_side_px: 512, max_total_px: 200000, ic: { V0_mV: -9.2, g0: 1.0 } };
  if (path === '/api/init') {
    const size = body.size_cells.map(x => parseInt(x)), m = parseInt(body.m), pr = body.params;
    if (!(m >= 1 && m <= 16)) throw new Error('m must be between 1 and 16');
    if (!(Math.min(...size) >= 2) || Math.max(...size) * m > 512 || size[0] * size[1] * m * m > 200000)
      throw new Error(`grid ${size[0] * m}x${size[1] * m} pixels is outside the allowed range (each side <= 512, total <= 200000); lower m or the tissue size`);
    const ms = estStepMs(size[0], size[1], m, +pr.fieldScreenSize, pr.fieldVector);
    if (ms > 80) throw new Error(`too heavy for the in-browser version (about ${ms.toFixed(0)} ms per step): lower the field range, m or the tissue size, or use the Python server version`);
    const core = new PotentiaCoreLib.PotentiaCore({ sizeCells: size, m, params: pr });
    const ic = body.ic || { mode: 'homogeneous' }, n = core.Np0 * core.Np1, g0 = numOr(ic.g0, 1), branch = ic.branch === 'hyp' ? 'hyp' : 'dep';
    const g = new Float64Array(n);                                                   // initial conditions are set on G_pol; Vmem follows it
    if (ic.mode === 'random') {
      const rng = mulberry32((parseInt(ic.seed) || 0) + 12345), sG = numOr(ic.sigmaG, 0.3);
      for (let k = 0; k < n; k++) g[k] = ic.uniform ? 2 * rng() : g0 + sG * randn(rng);
    } else g.fill(g0);
    for (let k = 0; k < n; k++) g[k] = Math.min(2, Math.max(0, g[k]));
    const V = PotentiaCoreLib.PotentiaCore.equilibriumV(g, branch, 1.5);
    Object.assign(JS, { core, V, g, n: 0, cfg: body, diverged: false, clamp: null });
    return { info: core.info(), frame: jsFrame(!!body.want_ebar, 0, 0) };
  }
  if (!JS.core) throw new Error('initialise first');
  if (path === '/api/state') {
    JS.V = Float64Array.from(body.V, v => v * 1e-3); JS.g = Float64Array.from(body.g, v => Math.min(2, Math.max(0, v))); JS.diverged = false; JS.clamp = null;
    const cl = body.clamp;                       // hold the painted pixels for cl.steps steps (see visualizer.py: Session.set_state)
    if (cl && cl.steps > 0) {
      const iV = [], iG = [];
      for (let k = 0; k < JS.V.length; k++) { if (cl.maskV && cl.maskV[k]) iV.push(k); if (cl.maskG && cl.maskG[k]) iG.push(k); }
      if (iV.length || iG.length) JS.clamp = { start: JS.n, steps: Math.floor(cl.steps), iV: Int32Array.from(iV), iG: Int32Array.from(iG), vV: Float64Array.from(iV, k => JS.V[k]), vG: Float64Array.from(iG, k => JS.g[k]) };
    }
    return { frame: jsFrame(!!body.want_ebar, 0, 0) };
  }
  if (path === '/api/step') {
    const c = JS.core; let steps = Math.max(0, Math.floor(numOr(body.steps, 1)));
    if (body.until !== null && body.until !== undefined) steps = Math.min(steps, Math.max(0, Math.round(body.until / c.dt) - JS.n));
    const t0 = performance.now(), maxMs = numOr(body.max_ms, 150); let done = 0;
    const cl = JS.clamp;
    while (done < steps) {
      for (let i = 0, q = Math.min(25, steps - done); i < q; i++) {
        c.step(JS.V, JS.g); done++;
        if (cl) { const cc = JS.n + done - cl.start; if (cc > 0 && cc < cl.steps) { for (let k = 0; k < cl.iV.length; k++) JS.V[cl.iV[k]] = cl.vV[k]; for (let k = 0; k < cl.iG.length; k++) JS.g[cl.iG[k]] = cl.vG[k]; } }   // hold the painted values for the next step
      }
      if (performance.now() - t0 > maxMs) break;
    }
    JS.n += done; if (cl && JS.n - cl.start >= cl.steps) JS.clamp = null;                    // released
    for (let k = 0; k < JS.V.length; k++) if (!Number.isFinite(JS.V[k]) || !Number.isFinite(JS.g[k])) { JS.diverged = true; break; }
    return { frame: jsFrame(!!body.want_ebar, done, performance.now() - t0) };
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
    size_cells: [parseInt($('nx').value), parseInt($('ny').value)], m: parseInt($('m').value),
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
  if (MODE === 'js' && !warn && nx >= 2 && ny >= 2) { const sp = estSpeed(nx, ny, m, Rs, $('vec').checked, num('GJ') || 0); txt += ` · est. speed ${sp < 10 ? sp.toFixed(1) : sp.toFixed(0)} simulated s per wall s in this browser`; if (sp < 1.5) { txt += ' (slow: lower the field range, m or the tissue size)'; warn = true; } }
  $('geo').textContent = txt; $('geo').className = 'note' + (warn ? ' warn' : '');
  if (D) { const b = num('bias') / D.params.bias; $('bias-note').textContent = `bias = ${isFinite(b) ? b.toFixed(3) : '–'} × the Stigmergic bias (${D.params.bias.toExponential(3)})`; }
}
function markDirty() { needInit = true; setStatus('settings changed: press Initialise', 'warn'); updateGeo(); }

// ---------- rendering ----------
function arrFor(v) { return v === 'V' ? curV : v === 'g' ? curG : curE; }
function resizeStage() {
  if (!info) return;
  const [R, C] = info.Np, availW = Math.min($('viewgroup').clientWidth, 760), narrow = window.matchMedia('(max-width: 900px)').matches, maxH = narrow ? Math.max(180, Math.round(window.innerHeight * 0.36)) : Math.max(300, window.innerHeight - 385);
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
  if (arr) for (let k = 0; k < R * C; k++) {
    let u = (arr[k] - lo) / span; u = u < 0 ? 0 : u > 1 ? 1 : u;
    const i3 = Math.round(u * 255) * 3, o = 4 * k;
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
function drawClamp() {                                  // dashed outline around the pixels that are currently held
  const c = $('mask'), ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height);
  if (!info || !clampShown || clampLeft <= 0) return;
  const [R, C] = info.Np, sx = c.width / C, sy = c.height / R, m = clampShown, on = (i, j) => i >= 0 && j >= 0 && i < R && j < C && m[i * C + j];
  ctx.beginPath();
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (m[i * C + j]) {
    if (!on(i - 1, j)) { ctx.moveTo(j * sx, i * sy); ctx.lineTo((j + 1) * sx, i * sy); }
    if (!on(i + 1, j)) { ctx.moveTo(j * sx, (i + 1) * sy); ctx.lineTo((j + 1) * sx, (i + 1) * sy); }
    if (!on(i, j - 1)) { ctx.moveTo(j * sx, i * sy); ctx.lineTo(j * sx, (i + 1) * sy); }
    if (!on(i, j + 1)) { ctx.moveTo((j + 1) * sx, i * sy); ctx.lineTo((j + 1) * sx, (i + 1) * sy); }
  }
  ctx.setLineDash([6, 4]); ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.stroke(); ctx.lineWidth = 1.6; ctx.strokeStyle = '#ffffff'; ctx.stroke(); ctx.setLineDash([]);
}
function updateClampNote() {
  const N = Math.floor(num('clampN')), dt = info ? info.dt : 0.01;
  $('clampNote').textContent = N > 0 ? `${N} steps = ${(N * dt).toFixed(2)} s at dt = ${dt} s. The painted G_pol (and, on a fresh tissue, the matching Vmem) is held for that many steps from the moment the paint is applied (Run or +1 s), then released. Painting again replaces a clamp that is still running.` : 'Enter a number of steps (1 or more).';
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
}

async function initialise() {
  stopRun();
  const cfg = gatherCfg();
  try {
    setStatus('initialising…');
    const r = await api('/api/init', cfg);
    info = r.info; hist = { t: [], V: [], fh: [], g: [] }; effSpeed = null; $('r-speed').textContent = '–';
    $('dtnote').textContent = info.dt.toExponential(2) + ' s' + (info.dt < 0.01 ? ' (reduced for GJ stability)' : '');
    maskG = new Uint8Array(info.Np[0] * info.Np[1]); maskV = new Uint8Array(maskG.length); clampShown = null; clampLeft = 0; updateClampNote();
    needInit = false; resizeStage(); applyFrame(r.frame); updateRangeInputs(false);
    setStatus(`ready · grid ${info.Np[0]}×${info.Np[1]} px`);
  } catch (e) { setStatus(e.message, 'bad'); }
}
async function syncState() {
  if (!dirtyState) return;
  const N = Math.floor(num('clampN')), any = maskG.some(v => v) || maskV.some(v => v), hold = $('clampOn').checked && N > 0 && any;
  const enc = a => MODE === 'js' ? a.slice() : toB64u8(a);
  const body = MODE === 'js' ? { V: curV, g: curG, want_ebar: view === 'e' } : { V: toB64(curV), g: toB64(curG), want_ebar: view === 'e' };
  if (hold) body.clamp = { steps: N, maskV: enc(maskV), maskG: enc(maskG) };
  const r = await api('/api/state', body);
  clampShown = hold ? Uint8Array.from(maskG, (v, k) => v | maskV[k]) : null;                    // outline of what is held
  maskG.fill(0); maskV.fill(0);
  clampLeft = r.frame.clamp_left || 0; $('r-clamp').hidden = clampLeft <= 0; $('r-clampn').textContent = clampLeft; drawClamp();
  dirtyState = false;
}
async function advance(seconds, token) {
  const f = await api('/api/step', { steps: Math.max(1, Math.round(seconds / info.dt)), until: null, max_ms: 1e9, want_ebar: view === 'e' });
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
    setStatus('running…');
    let lastT = hist.t.length ? hist.t[hist.t.length - 1] : 0, lastW = performance.now();
    while (running && token === runToken) {
      const frameMs = 40, t0 = performance.now(), speed = Math.max(0.1, num('speedN') || 20);
      const steps = Math.max(1, Math.round(speed * frameMs / 1000 / info.dt));
      const r = await api('/api/step', { steps, until: T, max_ms: MODE === 'js' ? 34 : 120, want_ebar: view === 'e' });
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
  const follow = tgt === 'g' && stepNow === 0, vEq = follow ? PotentiaCoreLib.PotentiaCore.equilibriumV(v, $('branch').value, 1.5) * 1e3 : 0;   // fresh tissue: Vmem follows the painted G_pol
  const ci = [i0], cj = [j0];
  if ($('miry').checked) ci.push(R - 1 - i0); if ($('mirx').checked) cj.push(C - 1 - j0);
  const rr = Math.ceil(rad);
  for (const ic of ci) for (const jc of cj)
    for (let di = -rr; di <= rr; di++) for (let dj = -rr; dj <= rr; dj++) {
      if (di * di + dj * dj > rad * rad) continue;
      const i = ic + di, j = jc + dj; if (i < 0 || j < 0 || i >= R || j >= C) continue;
      const k = i * C + j; arr[k] = v; if (follow) curV[k] = vEq;
      if (tgt === 'g') { maskG[k] = 1; if (follow) maskV[k] = 1; } else maskV[k] = 1;                   // remembered for the clamp (a fresh tissue holds the matching Vmem too)
    }
  dirtyState = true;
}
function drawBrush(ev) {
  const c = $('cursor'), ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height);
  if (!ev || !info || selectedIC() !== 'paint' || running) return;
  const r = c.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * c.width, y = (ev.clientY - r.top) / r.height * c.height;
  const [R, C] = info.Np, rad = num('prad') / C * c.width;
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, Math.max(rad, 2), 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 0.5; ctx.stroke();
}
function selectedIC() { return document.querySelector('input[name=ic]:checked').value; }
let painting = false;
function bindPaint() {
  const c = $('cursor');
  c.addEventListener('pointerdown', ev => {
    if (selectedIC() !== 'paint' || !info) return;
    if (running) { setStatus('pause to paint', 'warn'); return; }
    painting = true; c.setPointerCapture(ev.pointerId); const [i, j] = pixelAt(ev); paintAt(i, j); draw(); ev.preventDefault();
  });
  c.addEventListener('pointermove', ev => { drawBrush(ev); if (painting && !running) { const [i, j] = pixelAt(ev); paintAt(i, j); draw(); } });
  c.addEventListener('pointerup', () => { painting = false; });
  c.addEventListener('pointerleave', () => drawBrush(null));
}

// ---------- view / ranges ----------
function updateRangeInputs(store) {
  if (store) { RANGES[view] = [num('vmin'), num('vmax')]; }
  [$('vmin').value, $('vmax').value] = RANGES[view];
}
async function setView(v) {
  RANGES[view] = [num('vmin'), num('vmax')]; view = v; [$('vmin').value, $('vmax').value] = RANGES[view];
  if (v === 'e' && info && !running) { try { await syncState(); const f = await api('/api/step', { steps: 0, want_ebar: true }); applyFrame(f.frame); } catch (e) { setStatus(e.message, 'bad'); } }
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
  fillParams(D.params);
  ['nx', 'ny'].forEach(id => $(id).addEventListener('input', syncRsFromPct));   // registered first: the tissue size changes the range in um, the percentage stays
  const dirtyIds = ['nx', 'ny', 'm', 'bias', 'fs', 'gain', 'weight', 'tau', 'vec', 'GJ', 'leakmode', 'leakv', 'g0', 'branch', 'sigG', 'seed', 'uniform'];
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
    const v = selectedIC(); $('ic-random').classList.toggle('on', v === 'random'); $('ic-paint').classList.toggle('on', v === 'paint');
    $('stage').style.touchAction = v === 'paint' ? 'none' : 'auto'; markDirty();
  }));
  document.querySelectorAll('input[name=view]').forEach(r => r.addEventListener('change', () => setView(document.querySelector('input[name=view]:checked').value)));
  ['vmin', 'vmax'].forEach(id => $(id).addEventListener('input', draw));
  $('showgrid').addEventListener('change', drawGrid); $('T').addEventListener('input', drawTS);
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
  const stateObject = () => ({ config: MODE === 'js' ? JS.cfg : null, t: parseFloat($('r-t').textContent), shape: info.Np, V_mV: Array.from(curV), g: Array.from(curG) });
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
  bindPaint(); updateRangeInputs(false); updateGeo();
  await initialise();
}
main();
