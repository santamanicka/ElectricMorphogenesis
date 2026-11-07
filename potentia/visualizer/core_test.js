// Holds the JavaScript core to the Python model: node core_test.js   (needs core_reference.json from make_js_reference.py)
const fs = require('fs'), path = require('path');
try { require('./src/field_simd.wasm.js'); } catch (e) { /* SIMD kernel not built: the JavaScript loop is tested alone */ }
const { PotentiaCore } = require('./core.js');
const refAll = JSON.parse(fs.readFileSync(path.join(__dirname, 'core_reference.json'), 'utf8')), ref = refAll.states;
let fails = 0;
for (const c of ref) for (const field of ['direct', 'fft']) {
  const core = new PotentiaCore({ sizeCells: c.sizeCells, m: c.m, periodic: c.periodic, mask: c.mask ? Uint8Array.from(c.mask) : null, field, params: c.params });
  const V = Float64Array.from(c.V0), g = Float64Array.from(c.g0);
  let w40 = null;
  for (let it = 0; it < c.steps; it++) {
    if (it === 40) w40 = Math.max(...Array.from(V, (v, k) => Math.abs(v - c.V40[k])), ...Array.from(g, (v, k) => Math.abs(v - c.g40[k])));
    core.step(V, g);
  }
  const dV = Math.max(...Array.from(V, (v, k) => Math.abs(v - c.Vend[k]))), dg = Math.max(...Array.from(g, (v, k) => Math.abs(v - c.gend[k])));
  const ok = dV < 1e-9 && dg < 1e-8 && core.Np0 === c.Np[0] && core.Np1 === c.Np[1] && Math.abs(core.dt - c.dt) < 1e-15 && w40 < 1e-9;
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${(field + ': ' + c.name).padEnd(66)} max|dV|=${dV.toExponential(1)} V  max|dg|=${dg.toExponential(1)}  dt=${core.dt}  kernel entries=${core.nEntries}`);
}
// optimised field sum (padded + symmetry-paired) vs the plain reference sum, on random states
for (const c of ref) for (const field of ['direct', 'fft']) {
  const core = new PotentiaCore({ sizeCells: c.sizeCells, m: c.m, periodic: c.periodic, mask: c.mask ? Uint8Array.from(c.mask) : null, field, params: c.params }), n = core.Np0 * core.Np1;
  let s = 12345; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const V = Float64Array.from({ length: n }, (_, k) => (core.mask && !core.mask[k]) ? 0 : -55e-3 + 50e-3 * rnd());
  core._fieldSimple(V); const Ea = Float64Array.from(core.E); core._field(V);
  let rel = 0, big = 0; for (let k = 0; k < Ea.length; k++) { big = Math.max(big, Math.abs(Ea[k])); }
  for (let k = 0; k < Ea.length; k++) rel = Math.max(rel, Math.abs(core.E[k] - Ea[k]) / big);
  const ok = rel < 1e-12; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${field} field == plain sum, ${c.name.padEnd(46)} max relative difference ${rel.toExponential(1)}`);
}
// WebAssembly SIMD field kernel vs the JavaScript pair loop: same IEEE operations in the same order, so the results must be bit-identical
for (const c of ref) {
  const mk = simd => new PotentiaCore({ sizeCells: c.sizeCells, m: c.m, periodic: c.periodic, mask: c.mask ? Uint8Array.from(c.mask) : null, field: 'direct', params: c.params, simd });
  const on = mk(true), off = mk(false), n = on.Np0 * on.Np1;
  if (!on.vector) { console.log(`SKIP  SIMD vs JS, ${c.name} (scalar potential runs on the JS path)`); continue; }
  let s = 777; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const V = Float64Array.from({ length: n }, (_, k) => (on.mask && !on.mask[k]) ? 0 : -55e-3 + 50e-3 * rnd());
  on._field(V); off._field(V);
  let diffE = 0; for (let k = 0; k < on.E.length; k++) diffE = Math.max(diffE, Math.abs(on.E[k] - off.E[k]));
  const V1 = Float64Array.from(c.V0), g1 = Float64Array.from(c.g0), V2 = Float64Array.from(c.V0), g2 = Float64Array.from(c.g0);
  for (let it = 0; it < c.steps; it++) { on.step(V1, g1); off.step(V2, g2); }
  let diffS = 0; for (let k = 0; k < n; k++) diffS = Math.max(diffS, Math.abs(V1[k] - V2[k]), Math.abs(g1[k] - g2[k]));
  const ok = !!on.simd && diffE === 0 && diffS === 0; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  SIMD == JS pair loop, ${c.name.padEnd(46)} simd active: ${!!on.simd}  field max|diff| ${diffE}  after ${c.steps} steps max|diff| ${diffS}`);
}
// the FFT itself against a plain DFT: sizes with the factors 2, 3, 5, composite sizes with 7, 11, 13 and primes (generic butterflies); transforms along the slow index of n x B arrays
{
  const { fftPlan, fftAlong } = PotentiaCore._fft; let worst = 0, count = 0;
  for (const n of [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 20, 22, 25, 30, 36, 49, 64, 77, 90, 97, 100, 120, 144, 150, 200, 216, 256, 270, 288, 300, 320]) for (const B of [1, 5]) {
    const re = new Float64Array(n * B), im = new Float64Array(n * B); let sd = 7 + n; const rnd = () => (sd = (sd * 1664525 + 1013904223) >>> 0) / 4294967296 - 0.5;
    for (let k = 0; k < re.length; k++) { re[k] = rnd(); im[k] = rnd(); }
    const R = new Float64Array(re), I = new Float64Array(im); fftAlong(fftPlan(n), B, R, I, 0, new Float64Array(re.length), new Float64Array(re.length));
    let err = 0, scale = 0;
    for (let q = 0; q < B; q++) for (let k = 0; k < n; k++) {
      let sr = 0, si = 0; for (let j = 0; j < n; j++) { const a = -2 * Math.PI * ((j * k) % n) / n, c = Math.cos(a), sn = Math.sin(a); sr += re[j * B + q] * c - im[j * B + q] * sn; si += re[j * B + q] * sn + im[j * B + q] * c; }
      err = Math.max(err, Math.abs(sr - R[k * B + q]), Math.abs(si - I[k * B + q])); scale = Math.max(scale, Math.abs(sr), Math.abs(si));
    }
    worst = Math.max(worst, err / scale); count++;
  }
  const ok = worst < 1e-13; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  FFT == plain DFT, ${count} transforms (34 sizes, 1 or 5 columns)   worst relative error ${worst.toExponential(1)}`);
}
// the FFT engine against the direct engine on larger tissues and longer ranges than the reference set (trajectories over 300 steps, finite and periodic)
for (const [name, size, m, per, extra] of [['30x30, m=2, R_s=14', [30, 30], 2, false, { fieldScreenSize: 14 }], ['30x20, m=3, R_s=9, periodic', [30, 20], 3, true, { fieldScreenSize: 9 }],
                                            ['24x24, m=2, R_s=30 (kernel wider than the tissue), periodic', [24, 24], 2, true, { fieldScreenSize: 30 }], ['17x13 (prime sizes), m=2, R_s=6, periodic', [17, 13], 2, true, { fieldScreenSize: 6 }]]) {
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null, ...extra };
  const a = new PotentiaCore({ sizeCells: size, m, periodic: per, field: 'direct', params: p }), b = new PotentiaCore({ sizeCells: size, m, periodic: per, field: 'fft', params: p }), n = a.Np0 * a.Np1;
  let s = 4242; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const V1 = Float64Array.from({ length: n }, () => -55e-3 + 50e-3 * rnd()), g1 = Float64Array.from({ length: n }, () => 0.2 + 1.6 * rnd()), V2 = Float64Array.from(V1), g2 = Float64Array.from(g1);
  for (let it = 0; it < 300; it++) { a.step(V1, g1); b.step(V2, g2); }
  let d = 0; for (let k = 0; k < n; k++) d = Math.max(d, Math.abs(V1[k] - V2[k]), Math.abs(g1[k] - g2[k]) * 1e-3);
  const ok = d < 1e-9 && b.useFFT && !a.useFFT; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  FFT == direct trajectories, ${name.padEnd(62)} max|dV| ${d.toExponential(1)} V after 300 steps (FFT grid ${b.fftM.join('x')})`);
}
// tissue shapes: the JavaScript generator reproduces shapes.py pixel for pixel
{
  let bad = 0, count = 0;
  for (const r of refAll.shapes) {
    const m = PotentiaCore.shapeMask(r.name, r.R, r.C, r.p); count++;
    let diff = 0; for (let k = 0; k < m.length; k++) if (String(m[k]) !== r.bits[k]) diff++;
    if (diff) { bad++; console.log(`  shape ${r.name} ${r.R}x${r.C} p=${r.p}: ${diff} pixels differ`); }
  }
  const ok = bad === 0; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  shape masks == shapes.py on ${count} cases (10 shapes x 6 grids, default and other parameter)`);
}
// a rectangular island in a bigger grid behaves as the smaller finite tissue; outside pixels stay exactly zero; a full mask is the plain rectangle
for (const [m, vector, off0, off1] of [[2, true, 3, 5], [3, true, 2, 4], [2, false, 4, 2]]) for (const field of ['direct', 'fft']) {
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: vector, leak: null };
  const sm = new PotentiaCore({ sizeCells: [5, 6], m, field, params: p }), R0 = sm.Np0 + 2 * off0, C0 = sm.Np1 + 2 * off1;
  const mask = new Uint8Array(R0 * C0); for (let i = 0; i < sm.Np0; i++) for (let j = 0; j < sm.Np1; j++) mask[(i + off0) * C0 + (j + off1)] = 1;
  const big = new PotentiaCore({ sizeCells: [R0 / m, C0 / m], m, field, mask, params: p });
  let s = 31 + m; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const Vs = Float64Array.from({ length: sm.Np0 * sm.Np1 }, () => -55e-3 + 50e-3 * rnd()), gs = Float64Array.from({ length: Vs.length }, () => 0.2 + 1.6 * rnd());
  const Vb = new Float64Array(R0 * C0), gb = new Float64Array(R0 * C0);
  for (let i = 0; i < sm.Np0; i++) for (let j = 0; j < sm.Np1; j++) { Vb[(i + off0) * C0 + j + off1] = Vs[i * sm.Np1 + j]; gb[(i + off0) * C0 + j + off1] = gs[i * sm.Np1 + j]; }
  for (let it = 0; it < 300; it++) { sm.step(Vs, gs); big.step(Vb, gb); }
  let d = 0, out = 0; for (let i = 0; i < sm.Np0; i++) for (let j = 0; j < sm.Np1; j++) d = Math.max(d, Math.abs(Vb[(i + off0) * C0 + j + off1] - Vs[i * sm.Np1 + j]), 1e-3 * Math.abs(gb[(i + off0) * C0 + j + off1] - gs[i * sm.Np1 + j]));
  for (let k = 0; k < mask.length; k++) if (!mask[k]) out = Math.max(out, Math.abs(Vb[k]), Math.abs(gb[k]));
  const ok = d < 1e-12 && out === 0; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${field}: rectangular island in a larger grid == the smaller finite tissue (m=${m}, ${vector ? 'vector' : 'scalar'}), 300 steps   max|dV| ${d.toExponential(1)} V, outside ${out}`);
}
{
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null };
  const a = new PotentiaCore({ sizeCells: [6, 6], m: 2, params: p }), b = new PotentiaCore({ sizeCells: [6, 6], m: 2, mask: new Uint8Array(144).fill(1), params: p });
  const ok = a.mask === null && b.mask === null; if (!ok) fails++;
  let threw = 0; for (const f of [() => new PotentiaCore({ sizeCells: [6, 6], m: 2, mask: new Uint8Array(143).fill(1), params: p }), () => new PotentiaCore({ sizeCells: [6, 6], m: 2, mask: new Uint8Array(144), params: p }),
                                  () => new PotentiaCore({ sizeCells: [6, 6], m: 2, periodic: true, mask: PotentiaCore.shapeMask('disc', 12, 12), params: p })]) { try { f(); } catch (e) { threw++; } }
  const ok2 = threw === 3; if (!ok2) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  an all-ones mask is the plain rectangle;  ${ok2 ? 'PASS' : 'FAIL'}  wrong-size, empty and periodic masks are rejected`);
}
// engine choice: the core and the page's estimate agree, and the choice follows the cost (direct for short ranges, FFT for long ones)
{
  const P0 = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null };
  let agree = true, short = null, long = null;
  for (const [n, m, Rs, per] of [[11, 2, 4, false], [25, 2, 4, false], [25, 2, 20, false], [60, 4, 4, false], [100, 2, 4, false], [100, 2, 36.34, false], [30, 2, 14, true], [96, 2, 36, true]]) {
    const c = new PotentiaCore({ sizeCells: [n, n], m, periodic: per, params: { ...P0, fieldScreenSize: Rs } }), e = PotentiaCore.estimate(n, n, m, Rs, true, per);
    if ((e.engine === 'fft') !== c.useFFT) { agree = false; console.log('  estimate and core disagree for', n, m, Rs, per, e.engine, c.useFFT); }
    if (n === 100 && m === 2 && Rs === 4) short = c.useFFT ? 'fft' : 'direct'; if (n === 100 && Rs === 36.34) long = c.useFFT ? 'fft' : 'direct';
  }
  const ok = agree && short === 'direct' && long === 'fft'; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  engine choice: the page's estimate and the core pick the same engine in 8 cases; 100x100 m=2: Stigmergic range -> ${short}, 257 um range -> ${long}`);
}
// G_pol -> Vmem equilibrium (initial conditions are set on G_pol)
{
  const e = refAll.eq; let d = 0;
  for (const br of ['dep', 'hyp']) for (let k = 0; k < e.g.length; k++) d = Math.max(d, Math.abs(PotentiaCore.equilibriumV(e.g[k], br) - e[br][k]));
  const w = PotentiaCore.bistableWindow(), ok = d < 1e-9 && Math.abs(w[0] - e.window[0]) < 1e-12 && Math.abs(w[1] - e.window[1]) < 1e-12;
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  equilibrium Vmem(G_pol), ${e.g.length} g values x 2 branches   max|dV|=${d.toExponential(1)} V   bistable window [${w[0].toFixed(3)}, ${w[1].toFixed(3)}]`);
}
// speed (Node/V8; browsers are similar)
function speed(label, size, m, params, field, simd) {
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null, ...params };
  const core = new PotentiaCore({ sizeCells: size, m, params: p, field, simd });
  const V = new Float64Array(core.Np0 * core.Np1).fill(-9.2e-3), g = new Float64Array(V.length).fill(1);
  for (let i = 0; i < 20; i++) core.step(V, g);
  const n = Math.max(5, Math.min(2000, Math.round(300 / Math.max(0.05, field === 'fft' ? core.estMs.fft : core.estMs.direct)))), t0 = process.hrtime.bigint();
  for (let i = 0; i < n; i++) core.step(V, g);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / n;
  console.log(`speed ${label.padEnd(30)} ${(field + (field === 'direct' && core.simd ? '+simd' : '')).padEnd(12)} ${ms.toFixed(3).padStart(8)} ms/step  ->  ${(core.dt * 1000 / ms).toFixed(1).padStart(6)} simulated s per wall s`);
}
for (const [label, size, m, params] of [['11x11, m=2, R_s=4', [11, 11], 2, {}], ['25x25, m=2, R_s=4', [25, 25], 2, {}], ['25x25, m=2, R_s=10', [25, 25], 2, { fieldScreenSize: 10 }],
                                         ['100x100, m=2, R_s=4 (28 um)', [100, 100], 2, {}], ['100x100, m=2, R_s=36 (257 um)', [100, 100], 2, { fieldScreenSize: 36.34 }]]) {
  speed(label, size, m, params, 'direct', false); speed(label, size, m, params, 'direct', true); speed(label, size, m, params, 'fft', true);
}
console.log(fails ? `\n${fails} FAILURES` : '\nJS CORE MATCHES PYTHON'); process.exit(fails ? 1 : 0);
