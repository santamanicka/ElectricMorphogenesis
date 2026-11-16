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
// position-dependent field range: the local thickness of tissue masks equals localrange.py (exact integers), and trajectories with a range map equal Potentia.set_range_map with both field engines
{
  let bad = 0, cnt = 0, worstW = 0, worstG = 0, worstR = 0;
  for (const r of refAll.thickness) {
    const T = PotentiaCore.localThickness(r.plain ? null : Uint8Array.from(r.mask), r.R, r.C); cnt++;
    let diff = 0; for (let k = 0; k < T.length; k++) if (T[k] !== r.T[k]) diff++;
    if (diff) { bad++; console.log(`  thickness ${r.name} ${r.R}x${r.C}: ${diff} pixels differ`); }
    const Wd = PotentiaCore.localWidthSmooth(r.plain ? null : Uint8Array.from(r.mask), r.R, r.C); let dW = 0; for (let k = 0; k < Wd.length; k++) dW = Math.max(dW, Math.abs(Wd[k] - r.W[k])); worstW = Math.max(worstW, dW);
    if (dW > 1e-10) { bad++; console.log(`  smooth width ${r.name} ${r.R}x${r.C}: max difference ${dW}`); }
    for (const [frac, ref] of [[0.10, r.G], [0.25, r.G25]]) { const Gd = PotentiaCore.localGaussian(r.plain ? null : Uint8Array.from(r.mask), r.R, r.C, frac); let dG = 0; for (let k = 0; k < Gd.length; k++) dG = Math.max(dG, Math.abs(Gd[k] - ref[k])); worstG = Math.max(worstG, dG); if (dG > 1e-10) { bad++; console.log(`  Gaussian thickness (${frac}) ${r.name} ${r.R}x${r.C}: max difference ${dG}`); } }
    for (const [frac, edge, ref] of [[0.10, 1.0, r.RG], [0.25, 0.5, r.RG05], [0.10, 0.0, r.RG0]]) { const Rd = PotentiaCore.localRangeMap(r.plain ? null : Uint8Array.from(r.mask), r.R, r.C, 30.0, 1.0, 'gaussian', frac, edge); let dR = 0; for (let k = 0; k < Rd.length; k++) dR = Math.max(dR, Math.abs(Rd[k] - ref[k])); worstR = Math.max(worstR, dR); if (dR > 1e-9) { bad++; console.log(`  Gaussian range map (${frac}, edge ${edge}) ${r.name} ${r.R}x${r.C}: max difference ${dR}`); } }
  }
  const ok = bad === 0; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  local thickness (exact) and smooth width (max difference ${worstW.toExponential(1)} pixels) and Gaussian thickness (${worstG.toExponential(1)} pixels, sigma 10% and 25%) and the Gaussian range map with its edge shoulder (${worstR.toExponential(1)} pixels, edge 0, 0.5 and 1) == localrange.py on ${cnt} masks (12 shapes, a random mask, odd and even sizes)`);
}
for (const c of refAll.local_range) for (const field of ['direct', 'fft', 'chain']) {
  const core = new PotentiaCore({ sizeCells: c.sizeCells, m: c.m, mask: c.mask ? Uint8Array.from(c.mask) : null, field: field === 'chain' ? 'auto' : field, params: c.params });
  core.setRangeMap(Float64Array.from(c.rangeMap), field === 'chain' ? { chain: true } : undefined);
  const V = Float64Array.from(c.V0), g = Float64Array.from(c.g0);
  for (let it = 0; it < c.steps; it++) core.step(V, g);
  const dV = Math.max(...Array.from(V, (v, k) => Math.abs(v - c.Vend[k]))), dg = Math.max(...Array.from(g, (v, k) => Math.abs(v - c.gend[k])));
  const ok = dV < 1e-9 && dg < 1e-8 && core.lv.n === c.levels && Math.abs(core.dt - c.dt) < 1e-15; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${(field + ': ' + c.name).padEnd(84)} ${core.lv.n} levels  max|dV|=${dV.toExponential(1)} V  max|dg|=${dg.toExponential(1)}`);
}
// the faster paths of the position-dependent range (each level computed only for the cells that use it: direct sum over their probes with the SIMD kernel or the JavaScript loop, a thin annulus below the
// FFT top level, or the whole-grid FFT) give the same perceived field and the same trajectory as one FFT per level
{
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null };
  const seen = new Set();
  for (const [shape, nx, ny, pct, vec] of [['tadpole', 50, 130, 25.7, true], ['planarian', 24, 80, 30, true], ['ellipse', 20, 60, 40, true], ['dumbbell', 20, 60, 50, false]]) {
    const m = 2, R = nx * m, C = ny * m, mk = PotentiaCore.shapeMask(shape, R, C), n = R * C, pp = { ...p, fieldVector: vec };
    const mkCore = (field, opt, mode, simd) => { const c = new PotentiaCore({ sizeCells: [nx, ny], m, params: pp, mask: mk, field, simd }); c.setRangeMap(PotentiaCore.localRangeMap(c.mask, R, C, pct, c.h, mode), opt); return c; };
    const auto = mkCore('auto', { chain: false }), full = mkCore('fft'), cnt = {}; for (const x of auto.lv.modes) cnt[x] = (cnt[x] || 0) + 1;
    let s = 99; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
    const V1 = Float64Array.from({ length: n }, (_, k) => mk[k] ? -55e-3 + 50e-3 * rnd() : 0), g1 = Float64Array.from({ length: n }, (_, k) => mk[k] ? 0.2 + 1.6 * rnd() : 0), V2 = Float64Array.from(V1), g2 = Float64Array.from(g1);
    auto._sense(V1); full._sense(V2); let rel = 0, big = 0; for (let k = 0; k < n; k++) big = Math.max(big, Math.abs(full.ebar[k]));
    for (let k = 0; k < n; k++) rel = Math.max(rel, Math.abs(auto.ebar[k] - full.ebar[k]) / big);
    for (let it = 0; it < 150; it++) { auto.step(V1, g1); full.step(V2, g2); }
    let d = 0; for (let k = 0; k < n; k++) d = Math.max(d, Math.abs(V1[k] - V2[k]), Math.abs(g1[k] - g2[k]) * 1e-3);
    for (const x of auto.lv.modes) seen.add(x);
    const ok = rel < 1e-11 && d < 1e-9 && cnt.sub > 0; if (!ok) fails++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  position-dependent range, mixed engines == one FFT per level: ${shape} ${nx}x${ny}${vec ? '' : ' scalar'} ${pct}%  ${auto.lv.n} levels ${JSON.stringify(cnt)}  field rel ${rel.toExponential(1)}, 150 steps max|d| ${d.toExponential(1)}`);
  }
  const okm = seen.has('sub') && seen.has('ann') && seen.has('fft'); if (!okm) fails++;
  console.log(`${okm ? 'PASS' : 'FAIL'}  the four configurations together used every path: ${[...seen].join(', ')} (direct sums over the cells of a level, an annulus below the FFT top level, the FFT of the grid)`);
}
// the chain engine of the position-dependent range (the cells of every pair of levels computed in one pass: the lower level's kernel, then the shell up to the next level's) gives the same perceived field and
// trajectory as one FFT per level, in both width modes, with the WebAssembly SIMD kernel and with the JavaScript loop (which must agree bit for bit), vector and scalar potential; the planner picks it where it is cheaper
{
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null };
  let chosen = 0;
  for (const [shape, nx, ny, pct, vec, mode] of [['tadpole', 40, 110, 25.7, true, 'smooth'], ['planarian', 24, 80, 30, true, 'thickness'], ['ellipse', 20, 60, 40, true, 'smooth'], ['dumbbell', 20, 60, 50, false, 'smooth'], ['tadpole', 24, 80, 40, true, 'thickness']]) {
    const m = 2, R = nx * m, C = ny * m, mk = PotentiaCore.shapeMask(shape, R, C), n = R * C, pp = { ...p, fieldVector: vec };
    const mkCore = (field, opt, simd) => { const c = new PotentiaCore({ sizeCells: [nx, ny], m, params: pp, mask: mk, field, simd }); c.setRangeMap(PotentiaCore.localRangeMap(c.mask, R, C, pct, c.h, mode), opt); return c; };
    const full = mkCore('fft'), chS = mkCore('auto', { chain: true }, true), chJ = mkCore('auto', { chain: true }, false), auto = mkCore('auto');
    let s = 99; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
    const V0 = Float64Array.from({ length: n }, (_, k) => mk[k] ? -55e-3 + 50e-3 * rnd() : 0), g0 = Float64Array.from({ length: n }, (_, k) => mk[k] ? 0.2 + 1.6 * rnd() : 0);
    const cs = [full, chS, chJ, auto], Vs = cs.map(() => V0.slice()), gs = cs.map(() => g0.slice());
    cs.forEach((c, i) => c._sense(Vs[i])); let big = 0, rel = 0, bit = 0; for (let k = 0; k < n; k++) big = Math.max(big, Math.abs(full.ebar[k]));
    for (let k = 0; k < n; k++) { rel = Math.max(rel, Math.abs(chS.ebar[k] - full.ebar[k]) / big, Math.abs(chJ.ebar[k] - full.ebar[k]) / big, Math.abs(auto.ebar[k] - full.ebar[k]) / big); bit = Math.max(bit, Math.abs(chS.ebar[k] - chJ.ebar[k])); }
    for (let it = 0; it < 150; it++) cs.forEach((c, i) => c.step(Vs[i], gs[i]));
    let d = 0; for (let i = 1; i < cs.length; i++) for (let k = 0; k < n; k++) d = Math.max(d, Math.abs(Vs[i][k] - Vs[0][k]), Math.abs(gs[i][k] - gs[0][k]) * 1e-3);
    const simdOn = !!(chS.lv.cores[0].simd && chS.lv.cores[0].simd.fn2), same = !simdOn || bit === 0; if (auto.lv.chain) chosen++;
    const ok = rel < 1e-11 && d < 1e-9 && same && chS.lv.chain && chJ.lv.chain; if (!ok) fails++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  chain engine == one FFT per level: ${shape} ${nx}x${ny}${vec ? '' : ' scalar'} ${mode} ${pct}%  ${chS.lv.n} levels, SIMD ${simdOn ? 'on' : 'off'}, field rel ${rel.toExponential(1)}, SIMD vs JS loop ${bit.toExponential(1)}, 150 steps max|d| ${d.toExponential(1)}${auto.lv.chain ? ', chosen by the planner' : ''}`);
  }
  const okc = chosen >= 2; if (!okc) fails++;
  console.log(`${okc ? 'PASS' : 'FAIL'}  the planner chooses the chain engine where it is cheaper (${chosen} of 5 tissues)`);
}
// the edge shoulder of the position-dependent range keeps the cells near the boundary of a broad region active.  Without it the cells within about one range of the edge of the tadpole's head are suppressed by the
// lopsided field of the half of their range that lies outside the tissue (in the page's screenshot every cell within 16 pixels of the edge of the head has g < 0.1); with it they are not.
{
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null };
  const nx = 30, ny = 78, m = 2, R = nx * m, C = ny * m, n = R * C, mk = PotentiaCore.shapeMask('tadpole', R, C), d = PotentiaCore.distToOutside(mk, R, C), res = {};
  for (const edge of [0, 1]) {
    const core = new PotentiaCore({ sizeCells: [nx, ny], m, params: p, mask: mk, field: 'auto' });
    core.setRangeMap(PotentiaCore.localRangeMap(core.mask, R, C, 30, core.h, 'gaussian', 0.10, edge));
    const g = new Float64Array(n).fill(1), V = PotentiaCore.equilibriumV(g, 'dep', 1.5); for (let k = 0; k < n; k++) if (!mk[k]) { V[k] = 0; g[k] = 0; }
    for (let it = 0; it < 4000; it++) core.step(V, g);
    const band = (a, b) => { let q = 0, c = 0; for (let i = 0; i < R; i++) for (let j = 0; j < Math.floor(0.35 * C); j++) { const k = i * C + j; if (mk[k] && d[k] > a && d[k] <= b) { c++; if (g[k] < 0.1) q++; } } return q / c; };
    res[edge] = { edgeBand: band(0, 4), inner: band(16, 1e9), mean: g.reduce((s, v, k) => s + (mk[k] ? v : 0), 0) / mk.reduce((s, v) => s + v, 0) };
  }
  const ok = res[0].edgeBand >= 0.9 && res[1].edgeBand <= 0.4 && res[1].mean > res[0].mean; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  edge shoulder: the cells within 4 pixels of the edge of the tadpole's head (30 x 78 cells, 30%, 40 s) are quiet (g < 0.1) without it (${(100 * res[0].edgeBand).toFixed(0)}% of them) and active with it (${(100 * res[1].edgeBand).toFixed(0)}% quiet; inner cells ${(100 * res[0].inner).toFixed(0)}% against ${(100 * res[1].inner).toFixed(0)}%; mean g ${res[0].mean.toFixed(2)} against ${res[1].mean.toFixed(2)})`);
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
    const m = PotentiaCore.shapeMask(r.name, r.R, r.C, r.p, r.p2); count++;
    let diff = 0; for (let k = 0; k < m.length; k++) if (String(m[k]) !== r.bits[k]) diff++;
    if (diff) { bad++; console.log(`  shape ${r.name} ${r.R}x${r.C} p=${r.p}: ${diff} pixels differ`); }
  }
  const ok = bad === 0; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  shape masks == shapes.py on ${count} cases (12 shapes x 6 grids, default and other parameter; the tadpole also on 8 grids x 8 settings of its two parameters)`);
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
// development: the schedule, state inheritance, range schedule and a complete run agree with development.py / Potentia
{
  const fromHex = (h, n) => { const out = new Uint8Array(n); for (let k = 0; k < n; k++) out[k] = (parseInt(h[k >> 2], 16) >> (3 - (k & 3))) & 1; return out; };
  const D = PotentiaCore.Development;
  let bad = 0, cnt = 0;
  for (const r of refAll.dev_masks) {
    const d = new D(r.R, r.C, 100);
    r.taus.forEach((t, i) => { cnt++; const m = d.maskAt(t), ref = fromHex(r.masks[i], r.R * r.C); let diff = 0; for (let k = 0; k < m.length; k++) if (m[k] !== ref[k]) diff++; if (diff || d.stageAt(t).name !== r.stages[i]) { bad++; console.log(`  development mask ${r.R}x${r.C} tau=${t}: ${diff} pixels differ`); } });
  }
  let ok = bad === 0; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  development masks and stage labels == development.py on ${cnt} (grid, time) cases (5 grids x 11 times, edge-touching grids included)`);
  // exact distance transform against a brute-force search
  let worst = 0; let sd = 5; const rnd = () => (sd = (sd * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (const [R, C] of [[9, 12], [14, 7], [16, 16]]) {
    const m = Uint8Array.from({ length: R * C }, () => rnd() > 0.5 ? 1 : 0); for (let j = 0; j < C; j++) m[j] = 1; for (let i = 0; i < R; i++) m[i * C] = 1;
    const sdf = PotentiaCore.signedDistance(m, R, C), at = (i, j) => (i >= 0 && i < R && j >= 0 && j < C) ? m[i * C + j] : 0;
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      let best = Infinity; for (let a = -1; a <= R; a++) for (let b = -1; b <= C; b++) if (at(a, b) !== m[i * C + j]) best = Math.min(best, (a - i) * (a - i) + (b - j) * (b - j));
      worst = Math.max(worst, Math.abs(Math.abs(sdf[i * C + j]) - Math.sqrt(best)));
    }
  }
  ok = worst === 0; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  signed distance == brute force on 3 random masks (grid edge touching)   max difference ${worst}`);
  // state inheritance
  let gd = 0;
  for (const c of refAll.grow) { const R = c.R, C = c.C, old = fromHex(c.old, R * C), nw = fromHex(c.new, R * C), r = PotentiaCore.growState(Float64Array.from(c.V), Float64Array.from(c.g), old, nw, R, C, c.fill[0], c.fill[1]);
    for (let k = 0; k < R * C; k++) gd = Math.max(gd, Math.abs(r.V[k] - c.Vo[k]), Math.abs(r.g[k] - c.go[k])); }
  ok = gd < 1e-15; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  state inheritance (3 cases incl. an island touching no tissue) == development.grow_state   max difference ${gd.toExponential(1)}`);
  // range schedule
  let rd = 0; const d0 = new D(100, 260, 100);
  for (const r of refAll.dev_range) { const m = fromHex(r.mask, 100 * 260); rd = Math.max(rd, Math.abs(d0.rangeUm(r.mode, r.tau, m, 5.0, 28.3, 25.0, [10, 20, 30, 40, 50, 60]) - r.um)); }
  ok = rd < 1e-12; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  field-range schedule (constant, proportional, per-stage; ${refAll.dev_range.length} cases) == development.py   max difference ${rd.toExponential(1)} um`);
  // set_range == a core built with that range, for both engines
  for (const field of ['direct', 'fft']) {
    const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null };
    const a = new PotentiaCore({ sizeCells: [8, 10], m: 2, field, params: p }); a.setRange(9.0);
    const b = new PotentiaCore({ sizeCells: [8, 10], m: 2, field, params: { ...p, fieldScreenSize: 9.0 } });
    let s2 = 9; const r2 = () => (s2 = (s2 * 1664525 + 1013904223) >>> 0) / 4294967296; const n = a.Np0 * a.Np1;
    const V1 = Float64Array.from({ length: n }, () => -55e-3 + 50e-3 * r2()), g1 = Float64Array.from({ length: n }, () => 0.2 + 1.6 * r2()), V2 = Float64Array.from(V1), g2 = Float64Array.from(g1);
    for (let it = 0; it < 200; it++) { a.step(V1, g1); b.step(V2, g2); }
    let dd = 0; for (let k = 0; k < n; k++) dd = Math.max(dd, Math.abs(V1[k] - V2[k]), 1e-3 * Math.abs(g1[k] - g2[k]));
    ok = dd === 0 && a.R === b.R; if (!ok) fails++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${field}: setRange(R_s) == a core built with that range (200 steps)   max difference ${dd}`);
  }
  // the complete run
  const rr = refAll.dev_run, cf = rr.cfg;
  for (const field of ['direct', 'fft']) {
    const R = cf.size[0] * cf.m, C = cf.size[1] * cf.m, dv = new D(R, C, cf.duration, cf.update), mask0 = fromHex(cf.mask0, R * C);
    const p = { fieldScreenSize: cf.Rs0, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null };
    const core = new PotentiaCore({ sizeCells: cf.size, m: cf.m, field, mask: mask0, params: p });
    const g = new Float64Array(R * C).fill(1), V = PotentiaCore.equilibriumV(g, 'dep'); for (let q = 0; q < core.outIdx.length; q++) { V[core.outIdx[q]] = 0; g[core.outIdx[q]] = 0; }
    const K = Math.round(cf.update / core.dt); let nUpd = 0, updBad = 0;
    for (let n = 1; n <= cf.steps; n++) {
      core.step(V, g);
      if (n % K === 0) { const st = PotentiaCore.devUpdate(core, dv, n, V, g, { mode: 'prop', pct: cf.pct }, cf.fill[0], cf.fill[1]), u = rr.updates[nUpd++]; if (st.tissue !== u[0] || Math.abs(st.rangeUm - u[1]) > 1e-9 * u[1] || st.stage !== u[2]) updBad++; }
    }
    let dV = 0; for (let k = 0; k < R * C; k++) dV = Math.max(dV, Math.abs(V[k] - rr.V[k]), 1e-3 * Math.abs(g[k] - rr.g[k]));
    let fm = 0; const fin = fromHex(rr.final_mask, R * C); for (let k = 0; k < R * C; k++) if ((core.mask ? core.mask[k] : 1) !== fin[k]) fm++;
    ok = dV < 1e-9 && updBad === 0 && fm === 0 && nUpd === rr.updates.length; if (!ok) fails++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${field}: a 20 s development run (mask grows egg -> tadpole, range proportional to the tissue, ${nUpd} updates) == Potentia   max|dV| ${dV.toExponential(1)} V, ${updBad} update mismatches, ${fm} final-mask pixels differ`);
  }
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
