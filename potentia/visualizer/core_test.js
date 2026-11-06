// Holds the JavaScript core to the Python model: node core_test.js   (needs core_reference.json from make_js_reference.py)
const fs = require('fs'), path = require('path');
const { PotentiaCore } = require('./core.js');
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'core_reference.json'), 'utf8'));
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
// speed (Node/V8; browsers are similar)
function speed(label, size, m, params) {
  const p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null, ...params };
  const core = new PotentiaCore({ sizeCells: size, m, params: p });
  const V = new Float64Array(core.Np0 * core.Np1).fill(-9.2e-3), g = new Float64Array(V.length).fill(1);
  for (let i = 0; i < 20; i++) core.step(V, g);
  const n = Math.max(20, Math.min(2000, Math.round(3e8 / core.info().work_per_step))), t0 = process.hrtime.bigint();
  for (let i = 0; i < n; i++) core.step(V, g);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / n;
  console.log(`speed ${label.padEnd(34)} ${ms.toFixed(3)} ms/step  ->  ${(core.dt * 1000 / ms).toFixed(1)} simulated s per wall s  (work ${(core.info().work_per_step / 1e6).toFixed(1)}M MAC/step)`);
}
speed('11x11, m=2, R_s=4', [11, 11], 2, {});
speed('25x25, m=2, R_s=4', [25, 25], 2, {});
speed('11x11, m=4, R_s=4', [11, 11], 4, {});
speed('25x25, m=2, R_s=10', [25, 25], 2, { fieldScreenSize: 10 });
console.log(fails ? `\n${fails} FAILURES` : '\nJS CORE MATCHES PYTHON'); process.exit(fails ? 1 : 0);
