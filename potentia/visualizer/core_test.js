// Holds the JavaScript core to the Python model: node core_test.js   (needs core_reference.json from make_js_reference.py)
const fs = require('fs'), path = require('path');
try { require('./src/field_simd.wasm.js'); } catch (e) { /* SIMD kernel not built: the JavaScript loop is tested alone */ }
const { PotentiaCore } = require('./core.js');
const refAll = JSON.parse(fs.readFileSync(path.join(__dirname, 'core_reference.json'), 'utf8')), ref = refAll.states;
let fails = 0;
for (const c of ref) {
  const core = new PotentiaCore({ sizeCells: c.sizeCells, m: c.m, params: c.params });
  const V = Float64Array.from(c.V0), g = Float64Array.from(c.g0);
  let w40 = null;
  for (let it = 0; it < c.steps; it++) {
    if (it === 40) w40 = Math.max(...Array.from(V, (v, k) => Math.abs(v - c.V40[k])), ...Array.from(g, (v, k) => Math.abs(v - c.g40[k])));
    core.step(V, g);
  }
  const dV = Math.max(...Array.from(V, (v, k) => Math.abs(v - c.Vend[k]))), dg = Math.max(...Array.from(g, (v, k) => Math.abs(v - c.gend[k])));
  const ok = dV < 1e-9 && dg < 1e-8 && core.Np0 === c.Np[0] && core.Np1 === c.Np[1] && Math.abs(core.dt - c.dt) < 1e-15 && w40 < 1e-9;
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${c.name.padEnd(58)} max|dV|=${dV.toExponential(1)} V  max|dg|=${dg.toExponential(1)}  dt=${core.dt}  kernel entries=${core.nEntries}`);
}
// optimised field sum (padded + symmetry-paired) vs the plain reference sum, on random states
for (const c of ref) {
  const core = new PotentiaCore({ sizeCells: c.sizeCells, m: c.m, params: c.params }), n = core.Np0 * core.Np1;
  let s = 12345; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const V = Float64Array.from({ length: n }, () => -55e-3 + 50e-3 * rnd());
  core._fieldSimple(V); const Ea = Float64Array.from(core.E); core._field(V);
  let rel = 0, big = 0; for (let k = 0; k < Ea.length; k++) { big = Math.max(big, Math.abs(Ea[k])); }
  for (let k = 0; k < Ea.length; k++) rel = Math.max(rel, Math.abs(core.E[k] - Ea[k]) / big);
  const ok = rel < 1e-12; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  fast field sum == plain sum, ${c.name.padEnd(46)} max relative difference ${rel.toExponential(1)}`);
}
// WebAssembly SIMD field kernel vs the JavaScript pair loop: same IEEE operations in the same order, so the results must be bit-identical
for (const c of ref) {
  const mk = simd => new PotentiaCore({ sizeCells: c.sizeCells, m: c.m, params: c.params, simd });
  const on = mk(true), off = mk(false), n = on.Np0 * on.Np1;
  if (!on.vector) { console.log(`SKIP  SIMD vs JS, ${c.name} (scalar potential runs on the JS path)`); continue; }
  let s = 777; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const V = Float64Array.from({ length: n }, () => -55e-3 + 50e-3 * rnd());
  on._field(V); off._field(V);
  let diffE = 0; for (let k = 0; k < on.E.length; k++) diffE = Math.max(diffE, Math.abs(on.E[k] - off.E[k]));
  const V1 = Float64Array.from(c.V0), g1 = Float64Array.from(c.g0), V2 = Float64Array.from(c.V0), g2 = Float64Array.from(c.g0);
  for (let it = 0; it < c.steps; it++) { on.step(V1, g1); off.step(V2, g2); }
  let diffS = 0; for (let k = 0; k < n; k++) diffS = Math.max(diffS, Math.abs(V1[k] - V2[k]), Math.abs(g1[k] - g2[k]));
  const ok = !!on.simd && diffE === 0 && diffS === 0; if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  SIMD == JS pair loop, ${c.name.padEnd(46)} simd active: ${!!on.simd}  field max|diff| ${diffE}  after ${c.steps} steps max|diff| ${diffS}`);
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
function speed(label, size, m, params, simd) {
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null, ...params };
  const core = new PotentiaCore({ sizeCells: size, m, params: p, simd });
  const V = new Float64Array(core.Np0 * core.Np1).fill(-9.2e-3), g = new Float64Array(V.length).fill(1);
  for (let i = 0; i < 20; i++) core.step(V, g);
  const n = Math.max(20, Math.min(2000, Math.round(3e8 / core.info().work_per_step))), t0 = process.hrtime.bigint();
  for (let i = 0; i < n; i++) core.step(V, g);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / n;
  console.log(`speed ${label.padEnd(34)} ${ms.toFixed(3)} ms/step  ->  ${(core.dt * 1000 / ms).toFixed(1)} simulated s per wall s  (simd ${!!core.simd}, ${core.nEntries} kernel entries)`);
}
for (const simd of [false, true]) {
  console.log(simd ? '-- WebAssembly SIMD field kernel' : '-- JavaScript field kernel');
  speed('11x11, m=2, R_s=4', [11, 11], 2, {}, simd);
  speed('25x25, m=2, R_s=4', [25, 25], 2, {}, simd);
  speed('11x11, m=4, R_s=4', [11, 11], 4, {}, simd);
  speed('25x25, m=2, R_s=10', [25, 25], 2, { fieldScreenSize: 10 }, simd);
}
console.log(fails ? `\n${fails} FAILURES` : '\nJS CORE MATCHES PYTHON'); process.exit(fails ? 1 : 0);
