// Headless-browser smoke test of the Potentia visualizer.  Usage: node ui_smoke_test.js <url> <screenshot-dir> [expected backend label text]
const { chromium } = require('playwright');
const url = process.argv[2] || 'http://127.0.0.1:8765/', shots = process.argv[3] || '.';
const expectBackend = process.argv[4] || '';
let fails = 0;
const check = (name, cond, extra = '') => { if (!cond) fails++; console.log((cond ? 'PASS ' : 'FAIL ') + name + (extra ? '   ' + extra : '')); };
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  const txt = id => page.locator('#' + id).innerText();
  const waitStatus = async (re, ms = 20000) => { await page.waitForFunction(r => new RegExp(r).test(document.getElementById('status').textContent), re.source, { timeout: ms }); };

  await page.goto(url);
  await waitStatus(/ready/);
  if (expectBackend) check('backend label', (await txt('backend')).includes(expectBackend), await txt('backend'));
  check('loads, initialises with the Stigmergic defaults', (await page.inputValue('#Rs')) === '4' && (await page.inputValue('#gain')) === '-1' && (await page.inputValue('#weight')) === '1000');
  const simd = await page.evaluate(() => {
    const L = PotentiaCoreLib.PotentiaCore, p = { fieldScreenSize: 4, fieldStrength: 1, gain: -1, weight: 1000, bias: 5.308665553936392e-4, tau: 10, GJStrength: 0.05, fieldVector: true, leak: null };
    const a = new L({ sizeCells: [7, 7], m: 2, field: 'direct', params: p, simd: true }), b = new L({ sizeCells: [7, 7], m: 2, field: 'direct', params: p, simd: false }), n = a.Np0 * a.Np1;
    let s = 99; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
    const V1 = Float64Array.from({ length: n }, () => -55e-3 + 50e-3 * rnd()), V2 = Float64Array.from(V1), g1 = new Float64Array(n).fill(1), g2 = new Float64Array(n).fill(1);
    for (let i = 0; i < 200; i++) { a.step(V1, g1); b.step(V2, g2); }
    let d = 0; for (let k = 0; k < n; k++) d = Math.max(d, Math.abs(V1[k] - V2[k]), Math.abs(g1[k] - g2[k]));
    return { available: L.simdAvailable(), active: !!a.simd, diff: d };
  });
  check('WebAssembly SIMD field kernel: active in this browser and bit-identical to the JS loop over 200 steps', !simd.available || (simd.active && simd.diff === 0), JSON.stringify(simd));
  const note = await txt('eqnote');
  check('initial-condition note states the solver\'s bistable window', /0\.80/.test(note) && /1\.44/.test(note), note.slice(0, 120));
  check('geometry readout', (await txt('geo')).includes('110 × 110 µm') && (await txt('geo')).includes('22 × 22 px'), await txt('geo'));
  const rngRead = await page.evaluate(() => ({ text: document.getElementById('r-range').textContent, pct: document.getElementById('r-pct').textContent, title: document.getElementById('r-rangespan').title, vis: document.getElementById('r-rangespan').offsetParent !== null, R: info.R_um }));
  check('the field range is on the dashboard (readout line under the field): 28.3 µm and 25.7% of the shorter side at the Stigmergic defaults, with the pixels in its tooltip', rngRead.vis && rngRead.text === '28.3' && rngRead.pct === '25.7' && Math.abs(rngRead.R - 28.28) < 0.05 && /5\.7 pixels/.test(rngRead.title) && /25\.7% of the 110/.test(rngRead.title), JSON.stringify(rngRead));
  // the field range is controlled as a percentage of the tissue's shorter side; R_s (the model parameter) is derived and stays exact
  const st = () => page.evaluate(() => ({ pct: document.getElementById('Rpct').value, Rs: gatherCfg().params.fieldScreenSize, geo: document.getElementById('geo').textContent, note: document.getElementById('range-note').textContent }));
  const um = Rs => 5 * Math.SQRT2 * (Rs + 0.001);
  let r = await st();
  check('default range: R_s = 4 exactly, shown as 25.7% of the 110 um tissue', r.Rs === 4 && r.pct === '25.7' && /28\.3 µm/.test(r.note), `${r.pct}% | ${r.note}`);
  await page.fill('#Rpct', '50'); r = await st();
  check('typing 50% gives a 55 um range (R_s = 55/(5*sqrt2) - 0.001)', Math.abs(um(r.Rs) - 55) < 1e-9 && /55\.0 µm/.test(r.geo), `R_s=${r.Rs.toFixed(4)} | ${r.geo.slice(0, 100)}`);
  await page.fill('#nx', '25'); await page.fill('#ny', '25'); r = await st();
  check('default: changing the tissue size keeps the range in um (55 um stays, now 22% of the 250 um tissue)', !(await page.isChecked('#holdpct')) && Math.abs(um(r.Rs) - 55) < 1e-9 && r.pct === '22.0', `${r.pct}% -> ${um(r.Rs).toFixed(1)} um`);
  await page.check('#holdpct'); await page.fill('#Rpct', '50'); await page.fill('#nx', '30'); await page.fill('#ny', '30'); r = await st();
  check('"keep the same %": changing the tissue size holds the percentage, so the physical range scales (50% of 300 um = 150 um)', Math.abs(parseFloat(r.pct) - 50) < 1e-9 && Math.abs(um(r.Rs) - 150) < 1e-9, `${r.pct}% -> ${um(r.Rs).toFixed(1)} um`);
  await page.fill('#nx', '9'); await page.fill('#ny', '15'); r = await st();
  check('non-square tissue: the percentage refers to the shorter side (50% of 90 um = 45 um)', Math.abs(um(r.Rs) - 45) < 1e-9 && /90 µm shorter side/.test(r.note), r.note);
  const slider = await page.evaluate(() => { const s = document.getElementById('Rpct_sl'); s.value = '30'; s.dispatchEvent(new Event('input', { bubbles: true })); return { pct: document.getElementById('Rpct').value, Rs: gatherCfg().params.fieldScreenSize }; });
  check('the slider drives the same value (30% of 90 um = 27 um)', slider.pct === '30.0' && Math.abs(um(slider.Rs) - 27) < 1e-9, JSON.stringify(slider));
  await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.click('#b-defaults'); r = await st();
  check('Stigmergic defaults restore R_s = 4 exactly (25.7% of 110 um)', r.Rs === 4 && r.pct === '25.7', `${r.pct}% R_s=${r.Rs}`);
  await page.uncheck('#holdpct');
  await page.click('#b-init'); await waitStatus(/ready/);
  await page.screenshot({ path: `${shots}/01_initial.png` });

  // run (slow enough that there is a real window to pause in)
  await page.fill('#T', '40'); await page.fill('#speedN', '8');
  await page.click('#b-run');
  await page.waitForTimeout(1800);
  const t1 = parseFloat(await txt('r-t'));
  check('running advances simulated time', t1 > 3 && t1 < 40, `t=${t1}`);
  await page.screenshot({ path: `${shots}/02_running.png` });
  await page.click('#b-run');                         // pause
  await page.waitForTimeout(400);
  const tp = parseFloat(await txt('r-t')); await page.waitForTimeout(700);
  check('pause stops time', parseFloat(await txt('r-t')) === tp && tp < 40, `t=${tp}`);
  await page.fill('#speedN', '400'); await page.click('#b-run');
  await waitStatus(/finished at t = 40/, 60000);
  check('stops exactly at the run time T=40 s', parseFloat(await txt('r-t')) === 40, `t=${await txt('r-t')}`);
  await page.screenshot({ path: `${shots}/03_finished_T40.png` });
  await page.click('#b-run'); await page.waitForTimeout(400); await page.click('#b-run');
  check('Run after finishing extends T', parseFloat(await page.inputValue('#T')) > 40, `T=${await page.inputValue('#T')}`);

  // views
  await page.check('input[name=view][value=g]'); await page.waitForTimeout(300); await page.screenshot({ path: `${shots}/04_view_g.png` });
  await page.check('input[name=view][value=e]'); await page.waitForTimeout(600); await page.screenshot({ path: `${shots}/05_view_sensed.png` });
  await page.check('input[name=view][value=V]');

  // the field stays visible while the settings panel scrolls
  await page.evaluate(() => { document.querySelector('aside').scrollTop = 99999; });
  const sb = await page.locator('#stage').boundingBox();
  check('field view stays on screen while the settings panel scrolls', sb.y >= 0 && sb.y + sb.height <= 900, `stage y=${Math.round(sb.y)}..${Math.round(sb.y + sb.height)}`);
  await page.evaluate(() => { document.querySelector('aside').scrollTop = 0; });

  // manual painting
  await page.check('input[name=ic][value=paint]');
  await page.click('#b-init'); await waitStatus(/ready/);
  const v0 = await page.evaluate(() => Array.from(curV).slice(0, 5));
  await page.fill('#prad', '2.5'); await page.check('#mirx'); await page.check('#miry');
  const box = await page.locator('#cursor').boundingBox();
  const at = (fx, fy) => [box.x + fx * box.width, box.y + fy * box.height];
  await page.mouse.move(...at(0.30, 0.30)); await page.mouse.down();
  for (let k = 1; k <= 10; k++) await page.mouse.move(...at(0.30 + 0.02 * k, 0.30 + 0.01 * k));
  await page.mouse.up();
  const painted = await page.evaluate(() => { let n = 0; for (const v of curV) if (v < -50) n++; return n; });
  check('painting sets pixels to the brush value (mirrored x4)', painted > 20, `${painted} pixels at -55 mV`);
  const [pr, pc] = await page.evaluate(() => { const C = info.Np[1], R = info.Np[0]; let a = 0, b = 0; for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (curV[i*C+j] < -50) { if (j < C/2) a++; else b++; } return [a, b]; });
  check('left-right mirror is symmetric', pr === pc, `left ${pr}, right ${pc}`);
  await page.screenshot({ path: `${shots}/06_painted.png` });
  await page.click('#b-step'); await page.waitForTimeout(700);                       // +1 s: sends the painted state to the server first
  const [pv, hv] = await page.evaluate(() => { const C = info.Np[1]; return [curV[6 * C + 6], curV[10 * C + 18]]; });
  check('painted values reach the server (patch still hyperpolarised after +1 s, background not)', pv < -40 && hv > -20, `painted pixel ${pv.toFixed(1)} mV, background pixel ${hv.toFixed(1)} mV`);
  await page.fill('#T', '25'); await page.fill('#speedN', '400'); await page.click('#b-run');
  await waitStatus(/finished at t = 25/, 60000);
  await page.screenshot({ path: `${shots}/07_painted_after_run.png` });

  // painted G_pol: Vmem of painted pixels follows it straight away on a fresh tissue; the dynamics keep it
  await page.check('input[name=ic][value=paint]'); await page.fill('#g0', '1'); await page.click('#b-init'); await waitStatus(/ready/);
  await page.uncheck('#mirx'); await page.uncheck('#miry'); await page.fill('#prad', '3');
  const bx = await page.locator('#cursor').boundingBox(); const ctr = [bx.x + bx.width * 0.5, bx.y + bx.height * 0.5];
  await page.mouse.move(...ctr); await page.mouse.down(); await page.mouse.up();
  const vc = await page.evaluate(() => { const C = info.Np[1]; return [curV[11 * C + 11], curG[11 * C + 11], curV[2 * C + 2]]; });
  check('painting G_pol sets Vmem to the matching equilibrium immediately (g=2 -> -53.0 mV; background -9.24 mV)', Math.abs(vc[0] + 53.0) < 0.05 && vc[1] === 2 && Math.abs(vc[2] + 9.24) < 0.05, `painted V=${vc[0].toFixed(2)} mV, g=${vc[1]}, background V=${vc[2].toFixed(2)} mV`);

  // Vmem painted over monostable G_pol is overwritten; over bistable G_pol it persists (why G_pol is the initial condition)
  const patchAfter = async (g0, fs) => {
    await page.fill('#g0', String(g0)); await page.fill('#fs', String(fs)); await page.selectOption('#ptarget', 'V'); await page.click('#b-init'); await waitStatus(/ready/);
    await page.fill('#prad', '4'); await page.mouse.move(...ctr); await page.mouse.down(); await page.mouse.up();
    const before = await page.evaluate(() => curV[11 * info.Np[1] + 11]);
    await page.fill('#T', '3'); await page.fill('#speedN', '400'); await page.click('#b-run'); await waitStatus(/finished at t = 3\b/, 60000);
    return [before, await page.evaluate(() => curV[11 * info.Np[1] + 11])];
  };
  const [b03, a03] = await patchAfter(0.3, 1), [b10, a10] = await patchAfter(1.0, 0), [b11, a11] = await patchAfter(1.0, 1);
  check('painted Vmem (-55 mV) is overwritten within 3 s where G_pol is monostable (g0=0.3)', Math.abs(b03 + 55) < 0.01 && a03 > -12, `${b03.toFixed(1)} -> ${a03.toFixed(1)} mV`);
  check('painted Vmem persists where G_pol is bistable (g0=1.0), field off (isolates the Vmem dynamics)', Math.abs(b10 + 55) < 0.01 && a10 < -45, `${b10.toFixed(1)} -> ${a10.toFixed(1)} mV`);
  check('... but with the field on, the stigmergic feedback lowers g through the window edge and erases a small patch', Math.abs(b11 + 55) < 0.01 && a11 > -12, `${b11.toFixed(1)} -> ${a11.toFixed(1)} mV`);
  await page.fill('#fs', '1');
  // on a tissue that has run (paused): painted G_pol must be visible. Vmem follows at once where g is monostable, keeps its branch inside the bistable window,
  // and painted pixels are outlined and announced in every view
  const fieldSig = () => page.evaluate(() => { const c = document.getElementById('view'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let s = 0; for (let k = 0; k < d.length; k++) s = (s * 31 + d[k]) >>> 0; return s; });
  const outlinePx = () => page.evaluate(() => { const c = document.getElementById('mask'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let k = 3; k < d.length; k += 4) if (d[k]) n++; return n; });
  const centreVG = () => page.evaluate(() => { const i = 11 * info.Np[1] + 11; return [curV[i], curG[i]]; });
  await page.selectOption('#ptarget', 'g'); await page.check('input[name=view][value=V]'); await page.fill('#prad', '3');
  let [vBefore] = await centreVG(), sigBefore = await fieldSig();
  await page.fill('#pval', '1'); await page.mouse.move(...ctr); await page.mouse.down(); await page.mouse.up();
  let [vAfter, gAfter] = await centreVG(), hint = await txt('status');
  check('paused after a run, painting G_pol inside the bistable window (g=1): Vmem keeps its branch, but the paint is outlined and the status says where to see it',
        vAfter === vBefore && gAfter === 1 && (await outlinePx()) > 50 && /G_pol view/.test(hint) && /yellow outline/.test(hint), `V ${vBefore.toFixed(2)} -> ${vAfter.toFixed(2)} mV; outline ${await outlinePx()} px; "${hint.slice(0, 110)}"`);
  sigBefore = await fieldSig(); [vBefore] = await centreVG();
  await page.fill('#pval', '0'); await page.mouse.move(...ctr); await page.mouse.down(); await page.mouse.up();
  [vAfter, gAfter] = await centreVG(); const sigAfter = await fieldSig();
  check('paused after a run, painting G_pol where it is monostable (g=0): Vmem follows at once (-5.0 mV) and the field view changes', Math.abs(vAfter + 5.0) < 1e-3 && gAfter === 0 && sigAfter !== sigBefore, `V ${vBefore.toFixed(2)} -> ${vAfter.toFixed(2)} mV; field view changed: ${sigAfter !== sigBefore}`);
  await page.check('input[name=view][value=g]'); const sigG0 = await fieldSig(); await page.fill('#pval', '2'); await page.mouse.move(...ctr); await page.mouse.down(); await page.mouse.up();
  check('the G_pol view shows the painted values', (await fieldSig()) !== sigG0, `g at the centre now ${(await centreVG())[1]}`);
  await page.check('input[name=view][value=V]');
  await page.click('#b-step'); await page.waitForTimeout(300);
  check('painted-but-not-applied outline disappears once the paint is applied (+1 s)', (await outlinePx()) === 0, `${await outlinePx()} px`);
  await page.fill('#g0', '1'); await page.selectOption('#ptarget', 'g'); await page.fill('#pval', '2'); await page.fill('#prad', '2.5');
  await page.click('#b-init'); await waitStatus(/ready/);

  // clamp: painted pixels are held for N steps, then released
  const cell = () => page.evaluate(() => { const i = 11 * info.Np[1] + 11; return { g: curG[i], V: curV[i], left: document.getElementById('r-clampn').textContent, shown: !document.getElementById('r-clamp').hidden,
    outline: (() => { const c = document.getElementById('mask'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let k = 3; k < d.length; k += 4) if (d[k]) n++; return n; })() }; });
  const paintCentre = async () => { const b2 = await page.locator('#cursor').boundingBox(); await page.mouse.move(b2.x + b2.width / 2, b2.y + b2.height / 2); await page.mouse.down(); await page.mouse.up(); };
  await page.check('input[name=ic][value=paint]'); await page.fill('#g0', '1'); await page.selectOption('#ptarget', 'g'); await page.fill('#pval', '0'); await page.fill('#prad', '3');
  await page.click('#b-init'); await waitStatus(/ready/);
  await page.check('#clampOn'); await page.fill('#clampN', '300');
  const cnote = await txt('clampNote');
  check('clamp length is shown in seconds (300 steps = 3.00 s at dt = 0.01 s)', /300 steps = 3\.00 s at dt = 0\.01 s/.test(cnote), cnote.slice(0, 70));
  await paintCentre(); await page.click('#b-step'); await page.waitForTimeout(300); let cs = await cell();
  check('clamp: after 100 of 300 steps the painted pixel is held exactly (g = 0, Vmem = -5.00 mV) and 200 steps are left', cs.g === 0 && Math.abs(cs.V + 5.0) < 1e-3 && cs.left === '200' && cs.shown, JSON.stringify({ g: cs.g, V: cs.V, left: cs.left }));
  check('clamp: the held pixels are outlined on the field view', cs.outline > 50, `${cs.outline} outline pixels`);
  await page.click('#b-step'); await page.waitForTimeout(200); cs = await cell();
  check('clamp: still held after 200 steps (100 left)', cs.g === 0 && cs.left === '100', JSON.stringify({ g: cs.g, left: cs.left }));
  await page.click('#b-step'); await page.waitForTimeout(300); cs = await cell();
  check('clamp: released after 300 steps (g drifts free of 0), readout and outline gone', cs.g > 0 && !cs.shown && cs.outline === 0, JSON.stringify({ g: cs.g, shown: cs.shown, outline: cs.outline }));
  await page.click('#b-init'); await waitStatus(/ready/); await page.uncheck('#clampOn'); await paintCentre(); await page.click('#b-step'); await page.waitForTimeout(300); cs = await cell();
  check('control: without the clamp the painted G_pol drifts from the first step', cs.g > 0 && !cs.shown && cs.outline === 0, JSON.stringify({ g: cs.g, shown: cs.shown }));
  await page.fill('#pval', '2'); await page.click('#b-init'); await waitStatus(/ready/);

  // painting is refused while running
  await page.fill('#T', '600'); await page.fill('#speedN', '5'); await page.click('#b-run'); await page.waitForTimeout(500);
  const before = await page.evaluate(() => curV.slice(0).filter(v => v < -50).length);
  await page.mouse.move(...at(0.5, 0.5)); await page.mouse.down(); await page.mouse.move(...at(0.55, 0.55)); await page.mouse.up();
  check('painting while running is refused', (await txt('status')).length > 0 && (await page.evaluate(() => document.getElementById('status').textContent)).includes('pause to paint'));
  await page.click('#b-run');

  // bigger tissue + random start + m
  await page.check('input[name=ic][value=random]'); await page.fill('#nx', '25'); await page.fill('#ny', '25'); await page.selectOption('#m', '2');
  await page.fill('#sigG', '0.3'); await page.fill('#seed', '5');
  check('changing settings flags "press Initialise"', (await txt('status')).includes('Initialise'));
  await page.click('#b-init'); await waitStatus(/ready · grid 50×50/);
  const rs = await page.evaluate(() => { let s = 0, s2 = 0; for (const v of curG) { s += v; s2 += v * v; } const n = curG.length, mu = s / n;
    const eq = PotentiaCoreLib.PotentiaCore.equilibriumV(Array.from(curG), 'dep'); let d = 0; for (let k = 0; k < n; k++) d = Math.max(d, Math.abs(curV[k] - eq[k] * 1e3));
    return { mu, sd: Math.sqrt(s2 / n - mu * mu), d }; });
  check('random start acts on G_pol (mean ~1, sd ~0.3) and Vmem follows it', Math.abs(rs.mu - 1) < 0.05 && rs.sd > 0.25 && rs.sd < 0.33 && rs.d < 0.01, `g ${rs.mu.toFixed(3)} +- ${rs.sd.toFixed(3)}, max|V - eq(g)| = ${rs.d.toExponential(1)} mV`);
  await page.screenshot({ path: `${shots}/08_random_25x25.png` });
  await page.fill('#T', '60'); await page.fill('#speedN', '400'); await page.click('#b-run'); await waitStatus(/finished at t = 60/, 120000);
  await page.check('#showgrid'); await page.waitForTimeout(200);
  await page.screenshot({ path: `${shots}/09_random_25x25_t60.png` });

  // periodic boundaries
  {
    const bcTag = () => txt('bc');
    check('periodic option: off by default, header says finite tissue', !(await page.isChecked('#periodic')) && (await bcTag()) === 'finite tissue');
    await page.fill('#nx', '15'); await page.fill('#ny', '15'); await page.selectOption('#m', '2');
    await page.check('input[name=ic][value=homogeneous]'); await page.check('#periodic');
    const bn = await page.evaluate(() => ({ t: document.getElementById('bc-note').textContent, w: document.getElementById('bc-note').classList.contains('warn') }));
    check('periodic + homogeneous start: the note warns that it stays uniform', bn.w && /stays uniform/.test(bn.t) && /wrap around/.test(bn.t), bn.t.slice(0, 110));
    await page.click('#b-init'); await waitStatus(/ready · grid 30×30 px · periodic/);
    check('after Initialise the header says periodic tissue', (await bcTag()) === 'periodic tissue (torus)');
    await page.fill('#T', '40'); await page.fill('#speedN', '400'); await page.click('#b-run'); await waitStatus(/finished at t = 40/, 120000);
    const u = await page.evaluate(() => { let lo = Infinity, hi = -Infinity, glo = Infinity, ghi = -Infinity; for (let k = 0; k < curV.length; k++) { lo = Math.min(lo, curV[k]); hi = Math.max(hi, curV[k]); glo = Math.min(glo, curG[k]); ghi = Math.max(ghi, curG[k]); } return { lo, hi, glo, ghi }; });
    check('periodic homogeneous tissue stays exactly uniform (no edge to nucleate patterns)', u.hi - u.lo < 1e-4 && u.ghi - u.glo < 1e-6, `Vmem ${u.lo.toFixed(3)}..${u.hi.toFixed(3)} mV, g ${u.glo.toFixed(4)}..${u.ghi.toFixed(4)} at t = 40 s`);
    await page.uncheck('#periodic'); await page.click('#b-init'); await waitStatus(/ready · grid 30×30 px$/); await page.click('#b-run'); await waitStatus(/finished at t = 40/, 120000);
    const f = await page.evaluate(() => { let lo = Infinity, hi = -Infinity; for (let k = 0; k < curV.length; k++) { lo = Math.min(lo, curV[k]); hi = Math.max(hi, curV[k]); } return { lo, hi }; });
    check('control: the same finite tissue is not uniform at t = 40 s (the edges act)', f.hi - f.lo > 1, `Vmem ${f.lo.toFixed(2)}..${f.hi.toFixed(2)} mV`);
    // random start in a periodic tissue evolves
    await page.check('#periodic'); await page.check('input[name=ic][value=random]'); await page.fill('#seed', '2');
    await page.click('#b-init'); await waitStatus(/ready · grid 30×30 px · periodic/); await page.fill('#T', '60'); await page.click('#b-run'); await waitStatus(/finished at t = 60/, 120000);
    await page.screenshot({ path: `${shots}/10_periodic_random_t60.png` });
    const r60 = await page.evaluate(() => { let lo = Infinity, hi = -Infinity; for (let k = 0; k < curV.length; k++) { lo = Math.min(lo, curV[k]); hi = Math.max(hi, curV[k]); } return { lo, hi }; });
    check('periodic tissue with a random start develops structure', r60.hi - r60.lo > 1, `Vmem ${r60.lo.toFixed(2)}..${r60.hi.toFixed(2)} mV at t = 60 s`);
    // painting wraps across the edge when periodic
    await page.check('input[name=ic][value=paint]'); await page.selectOption('#ptarget', 'g'); await page.fill('#pval', '0'); await page.fill('#prad', '3');
    const paintCorner = async () => { await page.click('#b-init'); await waitStatus(/ready/);
      const box = await page.locator('#cursor').boundingBox(), [R, C] = await page.evaluate(() => info.Np);
      await page.mouse.move(box.x + 0.5 * box.width / C, box.y + 0.5 * box.height / R); await page.mouse.down(); await page.mouse.up();
      return page.evaluate(() => { const [R, C] = info.Np; let n = 0; for (let k = 0; k < curG.length; k++) if (curG[k] === 0) n++; return { n, opposite: curG[(R - 1) * C + C - 1] === 0, rowWrap: curG[(R - 1) * C] === 0 }; }); };
    const pp = await paintCorner(); await page.uncheck('#periodic'); const pf = await paintCorner();
    check('painting at the corner wraps to the opposite edges in a periodic tissue (full 29-pixel disc)', pp.n === 29 && pp.opposite && pp.rowWrap, JSON.stringify(pp));
    check('control: in a finite tissue the same brush is cut off at the corner (quarter disc, no wrap)', pf.n < 29 && !pf.opposite && !pf.rowWrap, JSON.stringify(pf));
    // field range beyond half the tissue
    await page.check('#periodic'); await page.check('input[name=ic][value=random]'); await page.fill('#Rpct', '80');
    const rw = await page.evaluate(() => ({ t: document.getElementById('bc-note').textContent, w: document.getElementById('bc-note').classList.contains('warn') }));
    check('range above half the tissue (80%): the note says cells sense wrapped images', rw.w && /wrapped images/.test(rw.t), rw.t.slice(0, 160));
    await page.uncheck('#periodic'); await page.click('#b-defaults');
  }

  // painting is a tool on the current tissue, independent of the initial-condition choice: start from a homogeneous or a random tissue, run, pause, paint, continue (no re-initialisation)
  for (const mode of ['homogeneous', 'random']) {
    await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.selectOption('#m', '2'); await page.uncheck('#periodic');
    await page.check(`input[name=ic][value=${mode}]`); await page.uncheck('#paintOn'); if (mode === 'random') await page.fill('#seed', '4');
    await page.selectOption('#ptarget', 'g'); await page.fill('#pval', '2'); await page.fill('#prad', '3'); await page.uncheck('#mirx'); await page.uncheck('#miry'); await page.uncheck('#clampOn');
    await page.click('#b-init'); await waitStatus(/ready/);
    const centre = async () => { const box = await page.locator('#cursor').boundingBox(); return [box.x + box.width / 2, box.y + box.height / 2]; };
    const [cx, cy] = await centre(); await page.mouse.move(cx, cy); await page.mouse.down(); await page.mouse.up();
    const offG = await page.evaluate(() => { const [R, C] = info.Np; return curG[Math.floor(R / 2) * C + Math.floor(C / 2)]; });
    check(`${mode} start, painting off: a click on the field paints nothing`, offG !== 2, `g at the centre = ${offG}`);
    await page.evaluate(() => { window.__inits = 0; const o = initialise; initialise = async function () { window.__inits++; return o.apply(this, arguments); }; });
    await page.fill('#T', '60'); await page.fill('#speedN', '20'); await page.click('#b-run'); await page.waitForTimeout(900); await page.click('#b-run');
    const tp = parseFloat(await txt('r-t')), stepP = parseInt(await txt('r-step'));
    await page.check('#paintOn');
    const st = await page.evaluate(() => ({ needInit, status: document.getElementById('status').textContent, ic: document.querySelector('input[name=ic]:checked').value }));
    check(`${mode} start, paused at t = ${tp} s: switching painting on keeps the initial condition and does not ask for Initialise`, !st.needInit && st.ic === mode && !/Initialise/.test(st.status), JSON.stringify(st));
    await page.mouse.move(cx, cy); await page.mouse.down(); await page.mouse.up();
    const pg = await page.evaluate(() => { const [R, C] = info.Np, k = Math.floor(R / 2) * C + Math.floor(C / 2); return { g: curG[k], v: curV[k] }; });
    check('the paint shows on the paused tissue (G_pol = 2 at the centre)', pg.g === 2, `g = ${pg.g}, Vmem = ${pg.v.toFixed(1)} mV`);
    await page.click('#b-run'); await page.waitForFunction(t => parseFloat(document.getElementById('r-t').textContent) >= t + 1, tp, { timeout: 60000 }); await page.click('#b-run');
    const t2 = parseFloat(await txt('r-t')), inits = await page.evaluate(() => window.__inits), gAfter = await page.evaluate(() => { const [R, C] = info.Np; return curG[Math.floor(R / 2) * C + Math.floor(C / 2)]; });
    check(`${mode} start: Run continues from the painted tissue (t ${tp} -> ${t2} s, not restarted, no Initialise)`, t2 >= tp + 1 && inits === 0 && parseInt(await txt('r-step')) > stepP && gAfter > 1.5, `initialise calls: ${inits}, centre g after ${(t2 - tp).toFixed(1)} s = ${gAfter.toFixed(2)}`);
    await page.uncheck('#paintOn');
  }
  {
    await page.check('input[name=ic][value=homogeneous]'); await page.check('#paintOn');
    check('choosing the "paint" start still switches painting on (homogeneous background)', await (async () => { await page.uncheck('#paintOn'); await page.check('input[name=ic][value=paint]'); return page.isChecked('#paintOn'); })());
    await page.uncheck('#paintOn'); await page.check('input[name=ic][value=homogeneous]');
  }

  // tissue shapes: disc, annulus, a drawn shape, and every preset (no tissue outside: nothing there, no gap junctions, statistics over the tissue only)
  {
    await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.selectOption('#m', '2'); await page.uncheck('#periodic'); await page.uncheck('#paintOn');
    await page.check('input[name=ic][value=homogeneous]'); await page.selectOption('#shape', 'rectangle'); await page.click('#b-init'); await waitStatus(/ready/);
    const opts = await page.evaluate(() => Array.from(document.getElementById('shape').options).map(o => o.value));
    check('shape selector: rectangle, 11 shapes and custom', opts.length === 13 && opts[0] === 'rectangle' && ['disc', 'ellipse', 'annulus', 'stadium', 'tapered', 'hexagon', 'triangle', 'cross', 'dumbbell', 'tadpole', 'planarian', 'custom'].every(v => opts.includes(v)), opts.join(','));
    await page.selectOption('#shape', 'disc');
    const prev = await page.evaluate(() => { const c = document.getElementById('mask'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let cyan = 0, dim = 0;
      for (let k = 0; k < d.length; k += 4) { if (d[k] < 60 && d[k + 1] > 150 && d[k + 2] > 200 && d[k + 3] > 200) cyan++; if (d[k] < 40 && d[k + 2] < 60 && d[k + 3] > 150 && d[k + 3] < 200) dim++; }
      return { cyan, dim, periodicDisabled: document.getElementById('periodic').disabled, status: document.getElementById('status').textContent, note: document.getElementById('bc-note').textContent }; });
    check('choosing a shape previews it on the field (cyan outline, dimmed outside) and asks for Initialise', prev.cyan > 100 && prev.dim > 1000 && /Initialise/.test(prev.status), JSON.stringify({ cyan: prev.cyan, dim: prev.dim }));
    check('periodic boundaries are switched off and disabled for a shaped tissue, and the note says why', prev.periodicDisabled && /no-flux/.test(prev.note), prev.note.slice(0, 90));
    await page.click('#b-init'); await waitStatus(/ready/);
    const tissue = () => page.evaluate(() => { let inside = 0, bad = 0; for (let k = 0; k < curV.length; k++) { if (curMask[k]) inside++; else if (curV[k] !== 0 || curG[k] !== 0) bad++; } return { inside, n: curV.length, bad }; });
    let ti = await tissue();
    check('disc: about pi/4 of the box is tissue', Math.abs(ti.inside / ti.n - Math.PI / 4) < 0.02, `${ti.inside} of ${ti.n} pixels`);
    await page.fill('#T', '20'); await page.fill('#speedN', '400'); await page.click('#b-run'); await waitStatus(/finished at t = 20/, 120000);
    ti = await tissue();
    const dsk = await page.evaluate(() => { const [R, C] = info.Np; let rot = 0, flip = 0, s = 0, nt = 0; for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { const k = i * C + j; rot = Math.max(rot, Math.abs(curV[k] - curV[j * C + (R - 1 - i)])); flip = Math.max(flip, Math.abs(curV[k] - curV[(R - 1 - i) * C + j])); if (curMask[k]) { s += curV[k]; nt++; } }
      const px = document.getElementById('view').getContext('2d').getImageData(0, 0, 1, 1).data[3], ctr = document.getElementById('view').getContext('2d').getImageData(C >> 1, R >> 1, 1, 1).data[3];
      return { rot, flip, mean: s / nt, shown: parseFloat(document.getElementById('r-v').textContent), cornerAlpha: px, centreAlpha: ctr }; });
    check('disc after 20 s: pixels outside the tissue are exactly zero (V and G_pol)', ti.bad === 0, `${ti.bad} outside pixels non-zero`);
    check('disc after 20 s: the homogeneous start keeps the symmetry of the shape (rotation and flip)', dsk.rot < 2e-3 && dsk.flip < 2e-3, `max asymmetry ${Math.max(dsk.rot, dsk.flip).toExponential(1)} mV`);
    check('readout statistics are over the tissue only', Math.abs(dsk.mean - dsk.shown) < 0.06, `tissue mean ${dsk.mean.toFixed(2)} mV, readout ${dsk.shown}`);
    check('pixels outside the shape are transparent on the field view (checkerboard shows through)', dsk.cornerAlpha === 0 && dsk.centreAlpha === 255, `corner alpha ${dsk.cornerAlpha}, centre alpha ${dsk.centreAlpha}`);
    // painting is confined to the tissue
    await page.check('#paintOn'); await page.selectOption('#ptarget', 'g'); await page.fill('#pval', '2'); await page.fill('#prad', '2');
    const box = await page.locator('#cursor').boundingBox(), [R1, C1] = await page.evaluate(() => info.Np);
    await page.mouse.move(box.x + 0.5 * box.width / C1, box.y + 0.5 * box.height / R1); await page.mouse.down(); await page.mouse.up();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down(); await page.mouse.up();
    const pn = await page.evaluate(() => { const [R, C] = info.Np; return { corner: curG[0], centre: curG[(R >> 1) * C + (C >> 1)], outsideG: (() => { let b = 0; for (let k = 0; k < curG.length; k++) if (!curMask[k] && curG[k] !== 0) b++; return b; })() }; });
    check('painting does nothing outside the tissue (corner stays 0) and works inside (centre = 2)', pn.corner === 0 && pn.centre === 2 && pn.outsideG === 0, JSON.stringify(pn));
    await page.uncheck('#paintOn');
    // annulus
    await page.selectOption('#shape', 'annulus'); await page.click('#b-init'); await waitStatus(/ready/);
    ti = await tissue(); const hole = await page.evaluate(() => { const [R, C] = info.Np; return curMask[(R >> 1) * C + (C >> 1)]; });
    check('annulus: about 3/4 of the disc is tissue (22x22 pixels: coarse, so a 4% tolerance) and the centre is a hole', Math.abs(ti.inside / ti.n - Math.PI / 4 * 0.75) < 0.04 && hole === 0, `${(100 * ti.inside / ti.n).toFixed(1)}% of the box`);
    // a drawn shape starts as a copy of the previous one; draw with the brush, then build it
    await page.selectOption('#shape', 'custom');
    const cp = await page.evaluate(() => { const a = customMask, b = PotentiaCoreLib.PotentiaCore.shapeMask('annulus', info.Np[0], info.Np[1], 0.5); let d = 0; for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) d++; return { d, vis: document.getElementById('shape-draw').classList.contains('on') }; });
    check('custom shape starts as a copy of the previous shape and shows the drawing controls', cp.d === 0 && cp.vis);
    await page.check('#shapeEdit'); await page.selectOption('#shapeBrush', 'add'); await page.fill('#prad', '4');
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down(); await page.mouse.up();
    await page.selectOption('#shapeBrush', 'erase'); await page.fill('#prad', '2');
    await page.mouse.move(box.x + 0.12 * box.width, box.y + 0.5 * box.height); await page.mouse.down(); await page.mouse.move(box.x + 0.12 * box.width, box.y + 0.3 * box.height); await page.mouse.up();
    const dr = await page.evaluate(() => { const [R, C] = info.Np; return { centre: customMask[(R >> 1) * C + (C >> 1)], status: document.getElementById('status').textContent, need: needInit }; });
    check('drawing: the brush adds tissue in the hole and erases tissue on the ring; the settings are flagged for Initialise', dr.centre === 1 && dr.need && /Initialise/.test(dr.status), JSON.stringify(dr));
    await page.uncheck('#shapeEdit'); await page.click('#b-init'); await waitStatus(/ready/);
    const built = await page.evaluate(() => { const [R, C] = info.Np; let erased = 0; for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (j < 0.2 * C && j > 0.04 * C && i > 0.2 * R && i < 0.6 * R && !curMask[i * C + j]) erased++; return { centre: curMask[(R >> 1) * C + (C >> 1)], erased }; });
    ti = await tissue(); await page.click('#b-step'); await page.waitForTimeout(300); const ti2 = await tissue();
    check('the drawn shape is built (centre now tissue, erased strip is empty), runs, and nothing leaks outside it', built.centre === 1 && built.erased > 5 && ti.bad === 0 && ti2.bad === 0, JSON.stringify(built));
    // the planarian chosen on the default square box: the box becomes 11 x 48 cells; the outline of the target: a pointed head with two auricle spikes, a notch behind them, a spindle-shaped body, a tail tapering to a point
    await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.selectOption('#shape', 'planarian');
    const plNote = await page.evaluate(() => ({ ny: document.getElementById('ny').value, note: document.getElementById('shape-note').textContent }));
    await page.click('#b-init'); await waitStatus(/ready/);
    const plW = await page.evaluate(() => { const [R, C] = info.Np, wd = col => { let n = 0; for (let i = 0; i < R; i++) n += curMask[i * C + col]; return n; };
      let c0 = 0; while (c0 < C && wd(c0) === 0) c0++; let c1 = C - 1; while (c1 > 0 && wd(c1) === 0) c1--; const L = c1 - c0 + 1, at = s => wd(Math.floor(c0 + s * (L - 1)));
      let spike = 0, iSp = c0; for (let j = c0; j < c0 + Math.floor(0.3 * L); j++) if (wd(j) > spike) { spike = wd(j); iSp = j; }
      let edge = 0; for (let i = 0; i < R; i++) edge += curMask[i * C] + curMask[i * C + C - 1]; for (let j = 0; j < C; j++) edge += curMask[j] + curMask[(R - 1) * C + j];
      let sym = true; for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (curMask[i * C + j] !== curMask[(R - 1 - i) * C + j]) sym = false;
      return { R, C, nose: wd(c0), spike, spikeAt: +((iSp - c0) / L).toFixed(3), notch: at(0.2), body: at(0.5), tip: wd(c1), edge, sym }; });
    check('planarian chosen on the default 11x11 box: the box becomes 11 x 48 cells; pointed head, auricle spikes at about 14% of the length with a notch behind them, a body as deep as the spikes, a pointed tail, symmetric, nothing touching the edge of the box',
      plNote.ny === '48' && /box was changed to 11 × 48/.test(plNote.note) && plW.nose <= 3 && plW.spikeAt >= 0.11 && plW.spikeAt <= 0.17 && plW.spike - plW.notch >= 0.12 * (plW.R - 2) && plW.body >= 0.9 * plW.spike && plW.tip > 0 && plW.tip <= 3 && plW.edge === 0 && plW.sym, JSON.stringify(plW));
    // a fast drag gives only a few pointer events: the brush is stamped along the whole stroke (no gaps)
    await page.selectOption('#shape', 'custom'); await page.evaluate(() => { customMask.fill(1); }); await page.check('#shapeEdit'); await page.selectOption('#shapeBrush', 'erase'); await page.fill('#prad', '1');
    const bx = await page.locator('#cursor').boundingBox();                          // measured again: the page may have scrolled since the last drag
    await page.mouse.move(bx.x + 0.05 * bx.width, bx.y + 0.5 * bx.height); await page.mouse.down(); await page.mouse.move(bx.x + 0.95 * bx.width, bx.y + 0.5 * bx.height); await page.mouse.up();
    const strk = await page.evaluate(() => { const [R, C] = info.Np, i = R >> 1; let gaps = 0; for (let j = Math.ceil(0.07 * C); j <= Math.floor(0.93 * C); j++) if (customMask[i * C + j]) gaps++; let row = ''; for (let j = 0; j < C; j++) row += customMask[i * C + j]; return { gaps, above: customMask[(i - 4) * C + (C >> 1)], R, C, row }; });
    check('a two-event drag across the field erases a continuous line (the brush is interpolated between pointer events)', strk.gaps === 0 && strk.above === 1, JSON.stringify(strk));
    await page.uncheck('#shapeEdit'); await page.selectOption('#shape', 'annulus');
    // an elongated shape chosen on a square box must not come out as a disc: the box is made elongated, and a warning appears if it is made square again
    await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.selectOption('#shape', 'tadpole');
    const tp = await page.evaluate(() => ({ ny: document.getElementById('ny').value, note: document.getElementById('shape-note').textContent }));
    await page.click('#b-init'); await waitStatus(/ready/);
    const tadMeasure = () => page.evaluate(() => { const [R, C] = info.Np, wd = col => { let n = 0; for (let i = 0; i < R; i++) n += curMask[i * C + col]; return n; };
      let c0 = 0; while (c0 < C && wd(c0) === 0) c0++; let c1 = C - 1; while (c1 > 0 && wd(c1) === 0) c1--; const L = c1 - c0 + 1, at = s => wd(Math.floor(c0 + s * (L - 1)));
      let edge = 0; for (let i = 0; i < R; i++) edge += curMask[i * C] + curMask[i * C + C - 1]; for (let j = 0; j < C; j++) edge += curMask[j] + curMask[(R - 1) * C + j];
      let sym = true; for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (curMask[i * C + j] !== curMask[(R - 1 - i) * C + j]) sym = false;
      return { R, C, snout: wd(c0), body: at(0.2), neck: at(0.45), tail: at(0.75), tip: wd(c1), edge, sym }; });
    const tl = await tadMeasure();
    check('tadpole chosen on the default 11x11 box: the box becomes 11 x 50 cells (stated in the note); large oval body, a neck narrowing to a thin tail ending in a point, a rounded front (the first column under 40% of the height), symmetric about the long axis, margin all round',
      tp.ny === '50' && /box was changed to 11 × 50/.test(tp.note) && tl.body >= 0.9 * (tl.R - 2) && tl.neck <= 0.5 * (tl.R - 2) + 1 && tl.tail >= 1 && tl.tail <= 6 && tl.tail < tl.neck && tl.tip > 0 && tl.tip <= 3 && tl.snout <= 0.4 * tl.R && tl.edge === 0 && tl.sym, JSON.stringify(tl));
    // the second parameter of the tadpole: head size / tail size; limited so that the head fits in the box (0.97 / tail depth)
    const p2a = await page.evaluate(() => ({ on: document.getElementById('shapeP2-wrap').classList.contains('on'), label: document.getElementById('shapeP2-label').textContent, val: document.getElementById('shapeP2').value, max: document.getElementById('shapeP2').max, maxSl: document.getElementById('shapeP2_sl').max, p: document.getElementById('shapeP').value }));
    await page.fill('#shapeP', '0.2'); const p2b = await page.evaluate(() => document.getElementById('shapeP2').max); await page.fill('#shapeP', '0.41');
    check('tadpole: a second slider "head size / tail size" (default 2.354), limited to 2.36 at the default tail depth of 0.41 and to 4.85 at 0.2',
      p2a.on && /head size \/ tail size/.test(p2a.label) && p2a.val === '2.354' && p2a.max === '2.36' && p2a.maxSl === '2.36' && p2b === '4.85' && p2a.p === '0.41', JSON.stringify({ ...p2a, max02: p2b }));
    await page.fill('#shapeP2', '1.9'); await page.click('#b-init'); await waitStatus(/ready/);
    const tl2 = await tadMeasure();
    check('tadpole: a smaller head size / tail size (1.9 against 2.354) gives a smaller head and leaves the tail as deep as before', tl2.body < 0.9 * tl.body && tl2.tail === tl.tail && tl2.sym && tl2.edge === 0, JSON.stringify({ before: tl.body, after: tl2.body, tail: [tl.tail, tl2.tail] }));
    await page.selectOption('#shape', 'disc'); const p2off = await page.evaluate(() => document.getElementById('shapeP2-wrap').classList.contains('on')); await page.selectOption('#shape', 'tadpole');
    const p2back = await page.evaluate(() => document.getElementById('shapeP2').value);
    check('the second slider appears only for the tadpole and returns to its default when the shape is chosen again', !p2off && p2back === '2.354', p2back);
    // tapered capsule: a tapering-ratio slider (default 0.6) narrows the posterior end; 0 is the plain capsule
    await page.selectOption('#shape', 'tapered');
    const tprof = () => page.evaluate(() => { const pm = pendingMask(), [R, C] = pxDims(), dep = c => { let n = 0; for (let i = 0; i < R; i++) n += pm[i * C + c]; return n; }; let head = 0; for (let c = 0; c < C / 2; c++) head = Math.max(head, dep(c));
      let area = 0; for (let k = 0; k < pm.length; k++) area += pm[k]; return { head, tail: dep(Math.floor(0.9 * C)), area, R, C, val: document.getElementById('shapeP').value, label: document.getElementById('shapeP-label').textContent, on: document.getElementById('shapeP-wrap').classList.contains('on'), max: document.getElementById('shapeP_sl').max, p2: document.getElementById('shapeP2-wrap').classList.contains('on') }; });
    const tp6 = await tprof(); await page.fill('#shapeP', '0'); const tp0 = await tprof(); await page.fill('#shapeP', '0.9'); const tp9 = await tprof(); await page.fill('#shapeP', '0.6');
    await page.selectOption('#shape', 'stadium'); const tpSt = await tprof(); await page.selectOption('#shape', 'tapered');
    check('tapered capsule: a "tapering ratio" slider (default 0.6, up to 0.95; no second slider); the depth of the tail at 90% of the length is 20-60% of the head depth at 0.6, the full depth at 0 (where the shape is exactly the stadium: same tissue pixels) and under 30% at 0.9',
      tp6.on && /tapering ratio/.test(tp6.label) && tp6.val === '0.6' && tp6.max === '0.95' && !tp6.p2 && tp6.tail / tp6.head > 0.2 && tp6.tail / tp6.head < 0.6 && tp0.tail / tp0.head > 0.95 && tp0.area === tpSt.area && tp9.tail / tp9.head < 0.3 && tp9.area < tp6.area && tp6.area < tp0.area && tp6.C >= 2.5 * tp6.R * 0.9,
      JSON.stringify({ r6: +(tp6.tail / tp6.head).toFixed(2), r0: +(tp0.tail / tp0.head).toFixed(2), r9: +(tp9.tail / tp9.head).toFixed(2), areas: [tp0.area, tp6.area, tp9.area, tpSt.area], box: [tp6.R, tp6.C] }));
    await page.selectOption('#shape', 'tadpole');
    // pattern metrics on the dashboard: positional information (bits) and the number of persistent features of G_pol, updated at most four times a second
    await page.click('#b-init'); await waitStatus(/ready/); await page.waitForTimeout(400);
    const pmRead = () => page.evaluate(() => ({ pi: document.getElementById('r-pi').textContent, ft: document.getElementById('r-feat').textContent, tPi: document.getElementById('r-pi-span').title, tFt: document.getElementById('r-feat-span').title }));
    const pm0 = await pmRead();
    check('dashboard: pattern metrics of a homogeneous start are 0.00 bits and 0 features, with tooltips', pm0.pi === '0.00' && pm0.ft === '0' && /positional information/.test(pm0.tPi) && /peaks/.test(pm0.tFt), JSON.stringify(pm0).slice(0, 200));
    const pmSet = await page.evaluate(() => { const [R, C] = info.Np, m = info.m, g = new Float32Array(R * C); for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) g[i * C + j] = curMask && !curMask[i * C + j] ? 0 : 1 + Math.sign(Math.sin(2 * Math.PI * (j + 0.5) * (10 / m) / 80));
      curG = g; updatePatternMetrics(true); const r = PotentiaCoreLib.PotentiaCore.patternMetrics(curG, curMask, R, C, m); return { pi: document.getElementById('r-pi').textContent, ft: document.getElementById('r-feat').textContent, bits: r.bits, features: r.features, peaks: r.peaks, basins: r.basins, ms: patMs }; });
    check('dashboard: bands of 80 µm in the tadpole give 0.8 to 1 bit and a feature count that is the peaks plus the basins (one pair per band), the page shows what the core computes', +pmSet.pi >= 0.8 && +pmSet.pi <= 1.0 && pmSet.pi === pmSet.bits.toFixed(2) && +pmSet.ft === pmSet.features && pmSet.features === pmSet.peaks + pmSet.basins && pmSet.features >= 10, JSON.stringify(pmSet));
    await page.fill('#T', '8'); await page.fill('#speedN', '400'); await page.click('#b-init'); await waitStatus(/ready/); const calls0 = await page.evaluate(() => patCalls); const t00 = Date.now(); await page.click('#b-run'); await waitStatus(/finished at t = 8/, 60000); const wall = (Date.now() - t00) / 1000; await page.waitForTimeout(500);
    const pmRun = await page.evaluate(() => { const [R, C] = info.Np, r = PotentiaCoreLib.PotentiaCore.patternMetrics(curG, curMask, R, C, info.m); return { pi: document.getElementById('r-pi').textContent, ft: document.getElementById('r-feat').textContent, bits: r.bits, features: r.features, calls: patCalls, ms: patMs }; });
    check('dashboard: after a run the numbers are those of the final G_pol (also when the last frames came faster than the update), at most 4 updates per second of wall time, each under 5 ms', pmRun.pi === pmRun.bits.toFixed(2) && +pmRun.ft === pmRun.features && pmRun.calls - calls0 <= 4 * wall + 3 && pmRun.ms < 5, JSON.stringify({ ...pmRun, calls: pmRun.calls - calls0, wall: +wall.toFixed(1) }));
    // "smooth" view option: a display-only Gaussian smoothing (noise removed), smooth interpolation, raw values kept elsewhere
    await page.check('input[name=view][value=g]');
    const canvasStats = () => page.evaluate(() => { const c = document.getElementById('view'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let tv = 0, n = 0, lum = 0, m = 0;
      for (let k = 0; k + 4 < d.length; k += 4) { if (d[k + 3] > 200 && d[k + 7] > 200) { tv += Math.abs(d[k] - d[k + 4]) + Math.abs(d[k + 1] - d[k + 5]) + Math.abs(d[k + 2] - d[k + 6]); n++; } if (d[k + 3] > 200) { lum += d[k] + d[k + 1] + d[k + 2]; m++; } }
      return { tv: tv / Math.max(n, 1), lum: lum / Math.max(m, 1), rendering: getComputedStyle(c).imageRendering, wrap: document.getElementById('smooth-wrap').classList.contains('on'), sum: Array.from(d).reduce((a, b, i) => (a + b * ((i % 977) + 1)) % 1000003, 0) }; });
    await page.evaluate(() => { const [R, C] = info.Np, m = info.m, g = new Float32Array(R * C); for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { const band = 1 + Math.sign(Math.sin(2 * Math.PI * (j + 0.5) * (10 / m) / 80)) * 0.7, noise = (((i + j) & 1) ? 0.4 : -0.4); g[i * C + j] = curMask && !curMask[i * C + j] ? 0 : Math.min(2, Math.max(0, band + noise)); }
      curG = g; updatePatternMetrics(true); draw(); });
    const smRaw = await canvasStats(); const piRaw = await page.evaluate(() => document.getElementById('r-pi').textContent + '|' + document.getElementById('r-feat').textContent);
    const smOffVisible = await page.evaluate(() => !document.getElementById('smooth-wrap').classList.contains('on'));
    await page.check('#smoothOn'); const smOn = await canvasStats(); const smLen = await page.inputValue('#smoothUm');
    await page.fill('#smoothUm', '20'); const smWide = await canvasStats(); await page.fill('#smoothUm', '7.5');
    const piOn = await page.evaluate(() => { updatePatternMetrics(true); return document.getElementById('r-pi').textContent + '|' + document.getElementById('r-feat').textContent; });
    await page.uncheck('#smoothOn'); const smBack = await canvasStats();
    check('smooth view option (G_pol, a checkerboard noise of ±0.4 on 80 µm bands): off by default with pixelated rendering; on, the length is 7.5 µm and the picture is much smoother (neighbouring-pixel differences below 40% of the raw ones, average colour within 12%) with interpolated rendering, more smoothing is smoother still, the dashboard numbers do not change, and switching it off gives back exactly the raw picture',
      smOffVisible && smRaw.rendering === 'pixelated' && smOn.wrap && smLen === '7.5' && smOn.tv < 0.4 * smRaw.tv && Math.abs(smOn.lum / smRaw.lum - 1) < 0.12 && smOn.rendering !== 'pixelated' && smWide.tv < smOn.tv && piOn === piRaw && smBack.sum === smRaw.sum && smBack.rendering === 'pixelated',
      JSON.stringify({ tvRaw: +smRaw.tv.toFixed(1), tvOn: +smOn.tv.toFixed(1), tvWide: +smWide.tv.toFixed(1), lumRatio: +(smOn.lum / smRaw.lum).toFixed(3), piRaw, piOn, rendering: [smRaw.rendering, smOn.rendering, smBack.rendering], same: smBack.sum === smRaw.sum }));
    await page.check('input[name=view][value=r]'); const rRaw = await canvasStats(); await page.check('#smoothOn'); const rOn = await canvasStats(); await page.uncheck('#smoothOn'); await page.check('input[name=view][value=V]');
    check('the field range view is not smoothed (the range map is not noise): the same picture with the option on', rRaw.sum === rOn.sum, JSON.stringify({ rRaw: rRaw.sum, rOn: rOn.sum }));
    // position-dependent field range: cells in narrow regions sense over a shorter range (tadpole: thick body, thin tail)
    const lrOff = await page.evaluate(() => ({ exists: !!document.getElementById('localRange'), checked: document.getElementById('localRange').checked, disabled: document.getElementById('localRange').disabled }));
    await page.check('#localRange'); const perLocked = await page.evaluate(() => document.getElementById('periodic').disabled && !document.getElementById('periodic').checked);
    await page.click('#b-init'); await waitStatus(/ready/);
    const lr1 = await page.evaluate(() => { const [R, C] = info.Np; return { local: info.local_range, lo: info.R_um_range && +info.R_um_range[0].toFixed(1), hi: info.R_um_range && +info.R_um_range[1].toFixed(1), levels: info.levels, text: document.getElementById('r-range').textContent, pct: document.getElementById('r-pct').textContent, title: document.getElementById('r-rangespan').title, body: curRng[(R >> 1) * C + Math.floor(0.2 * C)], tail: curRng[(R >> 1) * C + Math.floor(0.85 * C)], note: document.getElementById('local-note').textContent, label: document.getElementById('Rpct-label').textContent }; });
    check('position-dependent range (tadpole, 25.7% of the local width): the readout shows the range between 15.0 µm (3 px, the thin tail) and 25.7 µm (the thick body), several levels, a tooltip; the range map has the body at 25.7 and the tail at 15.0',
      !lrOff.checked && lrOff.exists && perLocked && lr1.local && lr1.lo === 15 && lr1.hi === 25.7 && lr1.levels >= 3 && lr1.text === '15.0–25.7' && lr1.pct === '25.7' && /local (thickness|width)/.test(lr1.title) && Math.abs(lr1.body - 25.7) < 0.1 && Math.abs(lr1.tail - 15) < 0.1 && /local width/.test(lr1.label), JSON.stringify(lr1));
    // smooth (the default) against the stepped thickness: smooth also lowers the range towards every boundary
    const edgeProbe = () => page.evaluate(() => { const [R, C] = info.Np; const col = Math.floor(0.2 * C); let top = -1; for (let i = 0; i < R; i++) if (curMask[i * C + col]) { top = i; break; } const mid = R >> 1; return { top, mode: info.local_mode, edge: curRng[top * C + col], inner: curRng[(top + 2) * C + col], mid: curRng[mid * C + col], sel: document.getElementById('localMode').value, wrapOn: document.getElementById('local-mode-wrap').classList.contains('on'), note: document.getElementById('local-note').textContent, rd: document.getElementById('r-range').textContent, sig: document.getElementById('local-sigma-wrap').classList.contains('on'), sigVal: document.getElementById('localSigma').value, edWrap: document.getElementById('local-edge-wrap').classList.contains('on'), edVal: document.getElementById('localEdge').value }; });
    const eG = await edgeProbe();
    check('Gaussian-smoothed thickness with an edge shoulder (the default; a selector next to the checkbox, a smoothing length of 10% of the thickest part, a shoulder of 1 range): the cell at the edge of the tadpole body has a range well below that of the middle (below 65% of it: the 3 px floor), 2 px in 90% of it, and the range is the plateau in the middle',
      eG.sel === 'gaussian' && eG.wrapOn && eG.sig && eG.sigVal === '10' && eG.edWrap && eG.edVal === '1' && eG.mode === 'gaussian' && eG.top >= 0 && eG.edge < 0.65 * eG.mid && eG.inner > 0.85 * eG.mid && eG.inner < eG.mid + 1e-9 && /Gaussian/.test(eG.note) && /half-chord/.test(eG.note), JSON.stringify(eG));
    await page.fill('#localEdge', '0'); await page.click('#b-init'); await waitStatus(/ready/);
    const eG0 = await edgeProbe();
    check('edge shoulder 0 (off): the cell at the edge has the range of the middle again (the plateau of the thickness to the rim, within 5%)', eG0.mode === 'gaussian' && eG0.edge > 0.95 * eG0.mid && eG0.edVal === '0', JSON.stringify(eG0));
    await page.fill('#localEdge', '1');
    await page.selectOption('#localMode', 'smooth'); const sigHid = await page.evaluate(() => !document.getElementById('local-sigma-wrap').classList.contains('on')); await page.click('#b-init'); await waitStatus(/ready/);
    const eS = await edgeProbe();
    check('smooth width (the chord width; no smoothing length): the range of a cell at the edge of the tadpole body is well below that of the middle (below 70% of it; 2 px in below 85%), rising inwards',
      sigHid && !eS.edWrap && eS.sel === 'smooth' && eS.mode === 'smooth' && eS.top >= 0 && eS.edge < 0.7 * eS.mid && eS.inner < 0.85 * eS.mid && eS.edge < eS.inner && eS.inner < eS.mid + 1e-9 && /boundary/.test(eS.note), JSON.stringify(eS));
    await page.selectOption('#localMode', 'thickness'); await page.click('#b-init'); await waitStatus(/ready/);
    const eT = await edgeProbe();
    check('thickness mode (stepped): the edge cell has the range of the middle, and the readout is still 15.0–25.7 µm',
      eT.mode === 'thickness' && Math.abs(eT.edge - eT.mid) < 0.1 && eT.rd === '15.0–25.7' && !/boundary/.test(eT.note) && !eT.sig && !eT.edWrap, JSON.stringify(eT));
    await page.selectOption('#localMode', 'gaussian'); await page.fill('#localSigma', '25'); await page.click('#b-init'); await waitStatus(/ready/);
    const eW = await edgeProbe(); const rgW = await page.evaluate(() => { const [R, C] = info.Np; let lo = Infinity; for (let k = 0; k < curRng.length; k++) if (curMask[k] && curRng[k] < lo) lo = curRng[k]; return { lo, title: document.getElementById('r-rangespan').title }; });
    check('a smoothing length of 25% is used (reported in the tooltip) and still gives 15.0 to 25.7 µm at the thin tail and the thick body', eW.mode === 'gaussian' && /25% of the thickest/.test(rgW.title) && eW.rd === '15.0–25.7' && Math.abs(rgW.lo - 15) < 0.1, JSON.stringify({ rd: eW.rd, lo: rgW.lo, title: rgW.title.slice(0, 120) }));
    await page.fill('#localSigma', '10'); await page.click('#b-init'); await waitStatus(/ready/);
    await page.check('input[name=view][value=r]'); await page.waitForTimeout(200);
    const lrView = await page.evaluate(() => ({ unit: document.getElementById('cb-unit').textContent, hi: document.getElementById('cb-hi').textContent }));
    check('the "field range" view shows the range map (unit µm, scale up to the largest range)', /µm/.test(lrView.unit) && parseFloat(lrView.hi) >= 25.7, JSON.stringify(lrView));
    await page.check('input[name=view][value=V]');
    await page.click('#b-step'); await page.waitForTimeout(400); const tiL = await tissue();
    check('a run with the position-dependent range keeps nothing outside the tadpole', tiL.bad === 0 && tiL.inside > 100, JSON.stringify(tiL));
    await page.uncheck('#localRange'); await page.selectOption('#shape', 'rectangle'); await page.check('#periodic');
    const lrExc = await page.evaluate(() => ({ dis: document.getElementById('localRange').disabled, checked: document.getElementById('localRange').checked }));
    await page.uncheck('#periodic'); await page.check('#localRange');
    await page.click('#b-init'); await waitStatus(/ready/);
    const lrRect = await page.evaluate(() => ({ local: info.local_range, hi: +info.R_um_range[1].toFixed(1), lo: +info.R_um_range[0].toFixed(1), note: document.getElementById('local-note').textContent }));
    check('periodic boundaries disable the position-dependent range, and it disables them (a shaped or periodic tissue excludes it)', lrExc.dis && !lrExc.checked && perLocked, JSON.stringify(lrExc));
    check('a plain rectangle: the range is the single range (28.3 µm = 25.7% of 110) except at the corners, and the note says so', lrRect.local && Math.abs(lrRect.hi - 28.3) < 0.1 && lrRect.lo < lrRect.hi && /rectangle/.test(lrRect.note), JSON.stringify(lrRect));
    await page.uncheck('#localRange'); await page.selectOption('#shape', 'tadpole');
    await page.fill('#ny', '11'); const warnTxt = await page.evaluate(() => ({ t: document.getElementById('bc-note').textContent, w: document.getElementById('bc-note').classList.contains('warn') }));
    check('making the box square again with the tadpole selected warns that it is almost a disc', warnTxt.w && /almost a disc/.test(warnTxt.t), warnTxt.t.slice(0, 100));
    // every preset builds and runs (an elongated box so that the elongated shapes make sense)
    await page.fill('#nx', '10'); await page.fill('#ny', '18');
    const bad = [];
    for (const name of opts.filter(o => o !== 'rectangle' && o !== 'custom')) {
      await page.selectOption('#shape', name); await page.click('#b-init'); await waitStatus(/ready/); await page.click('#b-step'); await page.waitForTimeout(200);
      const r = await page.evaluate(() => { let finite = true, inside = 0, outBad = 0; for (let k = 0; k < curV.length; k++) { if (!Number.isFinite(curV[k]) || !Number.isFinite(curG[k])) finite = false; if (curMask[k]) inside++; else if (curV[k] !== 0 || curG[k] !== 0) outBad++; } return { finite, inside, outBad, n: curV.length }; });
      if (!(r.finite && r.inside > 0 && r.inside < r.n && r.outBad === 0)) bad.push(name + ':' + JSON.stringify(r));
    }
    check('all 11 presets build, run 1 s on a 10x18-cell box (made elongated where the shape needs it) and keep the outside empty', bad.length === 0, bad.join(' | '));
    await page.selectOption('#shape', 'rectangle'); const rc = await page.evaluate(() => ({ dis: document.getElementById('periodic').disabled })); await page.click('#b-init'); await waitStatus(/ready/);
    check('back to the rectangle: periodic boundaries are available again and the tissue has no mask', !rc.dis && (await page.evaluate(() => curMask === null)));
    await page.fill('#nx', '11'); await page.fill('#ny', '11');
  }

  // development: the tissue grows from an egg to a tadpole on a schedule (prescribed shape; state inherited by new tissue); the field range can follow the tissue
  {
    await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.selectOption('#m', '2'); await page.selectOption('#shape', 'rectangle'); await page.uncheck('#periodic');
    await page.check('input[name=ic][value=homogeneous]'); await page.click('#b-init'); await waitStatus(/ready/);
    check('development is off by default and the shape selector is enabled', !(await page.isChecked('#devOn')) && !(await page.evaluate(() => document.getElementById('shape').disabled)));
    await page.check('#devOn'); await waitStatus(/development on: initialised at the egg/, 60000);
    const on = await page.evaluate(() => ({ nx: document.getElementById('nx').value, ny: document.getElementById('ny').value, shapeDisabled: document.getElementById('shape').disabled, perDisabled: document.getElementById('periodic').disabled,
      note: document.getElementById('dev-note').textContent, strip: (() => { const c = document.getElementById('devStrip'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let k = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) k++; return k; })() }));
    check('switching development on sets an elongated grid (50 x 130 cells), locks the shape and periodic boundaries, draws the stage strip and describes the sizes',
      on.nx === '50' && on.ny === '130' && on.shapeDisabled && on.perDisabled && on.strip > 1000 && /about 17 cells across/.test(on.note) && /tadpole 126 cells/.test(on.note), `${on.nx}x${on.ny}; ${on.note.slice(0, 80)}`);
    await page.evaluate(() => { const s = document.getElementById('devScrub'); s.value = '1'; s.dispatchEvent(new Event('input', { bubbles: true })); });
    const pv = await page.evaluate(() => { const c = document.getElementById('mask'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let cyan = 0; for (let k = 0; k < d.length; k += 4) if (d[k] < 60 && d[k + 1] > 150 && d[k + 2] > 200 && d[k + 3] > 200) cyan++; return { cyan, label: document.getElementById('devScrubLabel').textContent }; });
    check('the stage slider previews the shape on the field (outline) and names the stage', pv.cyan > 100 && /Tadpole/.test(pv.label), `${pv.cyan} outline pixels, ${pv.label}`);
    await page.fill('#devD', '24'); await page.fill('#T', '30'); await page.fill('#speedN', '400');
    await page.click('#b-init'); await waitStatus(/ready · grid 100×260 px/, 60000);
    const e0 = await page.evaluate(() => { let n = 0, bad = 0; for (let k = 0; k < curMask.length; k++) { if (curMask[k]) n++; else if (curV[k] !== 0 || curG[k] !== 0) bad++; } return { n, bad, readout: document.getElementById('r-dev').textContent, hidden: document.getElementById('r-dev').hidden }; });
    const rdev = await page.evaluate(() => ({ text: document.getElementById('r-range').textContent, pct: document.getElementById('r-pct').textContent, dev: lastDevStatus.range_um }));
    check('development mode: the dashboard shows the range in effect at the egg stage (the same number as the stage status) and its share of the egg (170 um extent): 28.3 / 170 = 16.6%', Math.abs(parseFloat(rdev.text) - rdev.dev) < 0.06 && Math.abs(parseFloat(rdev.pct) - 16.6) < 0.2, JSON.stringify(rdev));
    check('Initialise starts at the egg (912 px, stage readout) with nothing outside it', e0.n === 912 && e0.bad === 0 && !e0.hidden && /Egg \(NF 1\)/.test(e0.readout) && /912 px/.test(e0.readout), e0.readout);
    // painting in development: only the tissue that exists can be painted, and a click outside it says so instead of doing nothing silently
    await page.check('#paintOn'); await page.selectOption('#ptarget', 'g'); await page.fill('#pval', '2'); await page.fill('#prad', '3');
    const pbx = await page.locator('#cursor').boundingBox();
    await page.mouse.move(pbx.x + 0.8 * pbx.width, pbx.y + 0.5 * pbx.height); await page.mouse.down(); await page.mouse.up();
    const po = await page.evaluate(() => ({ status: document.getElementById('status').textContent, dirty: dirtyState, painted: Array.from(curG).filter(v => v === 2).length }));
    await page.mouse.move(pbx.x + (0.5 / 2.6) * pbx.width, pbx.y + 0.5 * pbx.height); await page.mouse.down(); await page.mouse.up();
    const pi_ = await page.evaluate(() => ({ status: document.getElementById('status').textContent, painted: Array.from(curG).filter(v => v === 2).length }));
    check('development: a click outside the egg paints nothing, leaves the state clean and says so; a click on the egg paints 29 pixels',
      /nothing painted/.test(po.status) && /development only the tissue that exists/.test(po.status) && !po.dirty && po.painted === 0 && pi_.painted === 29 && /painted G_pol = 2 on 29 pixels/.test(pi_.status), `${po.status.slice(0, 60)} | ${pi_.status.slice(0, 50)}`);
    await page.uncheck('#paintOn');
    await page.click('#b-run'); const seen = [];
    for (let k = 0; k < 40; k++) { await page.waitForTimeout(250); seen.push(parseInt(await page.evaluate(() => document.getElementById('r-tissue').textContent))); if (/finished/.test(await txt('status'))) break; }
    await waitStatus(/finished at t = 30/, 120000);
    const fin = await page.evaluate(() => { const [R, C] = info.Np; let n = 0, bad = 0, diff = 0, ref = PotentiaCoreLib.PotentiaCore.shapeMask('tadpole', R, C); for (let k = 0; k < curMask.length; k++) { if (curMask[k]) n++; else if (curV[k] !== 0 || curG[k] !== 0) bad++; if (curMask[k] !== ref[k]) diff++; }
      let finite = true; for (let k = 0; k < curV.length; k++) if (!Number.isFinite(curV[k])) finite = false; return { n, bad, diff, finite, readout: document.getElementById('r-dev').textContent }; });
    check('the tissue grew monotonically while running (the tissue count never fell) and passed through several sizes', seen.length > 3 && seen.every((v, i) => i === 0 || v >= seen[i - 1]) && new Set(seen).size > 3, seen.join(','));
    check('after the development duration the tissue is exactly the tadpole preset (10426 px), nothing outside it, state finite, readout says Tadpole', fin.n === 10426 && fin.diff === 0 && fin.bad === 0 && fin.finite && /Tadpole \(NF 45\)/.test(fin.readout), JSON.stringify({ n: fin.n, diff: fin.diff, bad: fin.bad }));
    await page.screenshot({ path: `${shots}/11_development_tadpole.png` });
    // the field range follows the tissue: 25% of the shorter extent (egg 170 um -> 42.5 um, tadpole 460 um high -> 115 um)
    await page.selectOption('#devRange', 'prop'); await page.fill('#devPct', '25'); await page.click('#b-init'); await waitStatus(/ready/);
    const r0 = await page.evaluate(() => lastDevStatus.range_um), d0 = await txt('r-range'), p0 = await txt('r-pct'); await page.click('#b-run'); await waitStatus(/finished at t = 30/, 120000);
    const r1 = await page.evaluate(() => ({ r: lastDevStatus.range_um, core: (typeof JS !== 'undefined' && JS.core) ? JS.core.R * 1e6 : null }));
    check('proportional field range: 42.5 um at the egg, 115 um at the tadpole', Math.abs(r0 - 42.5) < 1 && Math.abs(r1.r - 115) < 1 && (r1.core === null || Math.abs(r1.core - 115) < 1) && d0 === '42.5' && (await txt('r-range')) === '115' && Math.abs(parseFloat(p0) - 25) < 0.3 && Math.abs(parseFloat(await txt('r-pct')) - 25) < 0.3, `${r0.toFixed(1)} -> ${r1.r.toFixed(1)} um; dashboard ${d0} (${p0}%) -> ${await txt('r-range')} (${await txt('r-pct')}%)`);
    await page.selectOption('#devRange', 'stage');
    await page.evaluate(() => document.querySelectorAll('#devStageInputs input').forEach((i, k) => { i.value = String(20 + 5 * k); i.dispatchEvent(new Event('input', { bubbles: true })); }));
    await page.click('#b-init'); await waitStatus(/ready/); const q0 = await page.evaluate(() => lastDevStatus.range_um); await page.click('#b-run'); await waitStatus(/finished at t = 30/, 120000); const q1 = await page.evaluate(() => lastDevStatus.range_um);
    check('per-stage range: 20 um at the egg, 45 um at the tadpole (values typed per stage)', Math.abs(q0 - 20) < 0.5 && Math.abs(q1 - 45) < 0.5, `${q0.toFixed(1)} -> ${q1.toFixed(1)} um`);
    await page.evaluate(() => { document.getElementById('devRange').value = 'stage'; document.querySelectorAll('#devStageInputs input').forEach(i => { i.value = '28.3'; }); }); const sIn = await page.evaluate(() => Array.from(document.querySelectorAll('#devStageInputs input')).map(i => i.value));
    check('per-stage range: six inputs, filled with the current range', sIn.length === 6 && sIn.every(v => v === sIn[0]), sIn.join(','));
    await page.selectOption('#devRange', 'const');
    // switching off restores the shape selector and a plain rectangle
    await page.uncheck('#devOn'); await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.click('#b-init'); await waitStatus(/ready/);
    const off = await page.evaluate(() => ({ shapeDisabled: document.getElementById('shape').disabled, perDisabled: document.getElementById('periodic').disabled, devHidden: document.getElementById('r-dev').hidden, mask: curMask === null }));
    check('development off: shape selector and periodic boundaries available again, no stage readout, plain rectangle', !off.shapeDisabled && !off.perDisabled && off.devHidden && off.mask, JSON.stringify(off));
  }

  // layout on a short laptop window: the field, readouts and time series fit the column (nothing cut off), with and without the development readout line
  {
    await page.setViewportSize({ width: 1366, height: 600 });
    const lay = async () => page.evaluate(() => { const L = document.getElementById('left'), s = document.getElementById('stage').getBoundingClientRect(), ts = document.getElementById('ts').getBoundingClientRect(); return { overflow: L.scrollHeight - L.clientHeight, tsBottom: Math.round(ts.bottom), innerH: innerHeight, stage: Math.round(s.width) + 'x' + Math.round(s.height) }; });
    await page.fill('#nx', '100'); await page.fill('#ny', '100'); await page.click('#b-init'); await waitStatus(/ready/, 60000); const plain = await lay();
    await page.check('#devOn'); await waitStatus(/development on: initialised at the egg/, 60000); await page.fill('#nx', '100'); await page.fill('#ny', '100'); await page.click('#b-init'); await waitStatus(/ready/, 60000); const dv = await lay();
    check('1366 x 600 window, 100 x 100 cells: nothing is cut off, plain and with the development readout (stage sized to the room left)', plain.overflow <= 1 && dv.overflow <= 1 && plain.tsBottom <= plain.innerH && dv.tsBottom <= dv.innerH && parseInt(dv.stage) >= 200, `plain ${JSON.stringify(plain)} | development ${JSON.stringify(dv)}`);
    await page.setViewportSize({ width: 1366, height: 768 }); await page.waitForTimeout(200); const tall = await lay();
    check('1366 x 768 window: the field uses the room (at least 380 px for a square grid) and nothing is cut off', tall.overflow <= 1 && parseInt(tall.stage) >= 380, JSON.stringify(tall));
    await page.uncheck('#devOn'); await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.click('#b-init'); await waitStatus(/ready/, 60000); await page.setViewportSize({ width: 1280, height: 900 });
  }

  // a large tissue: with the default (range in um kept) 100x100 cells at m=2 runs; holding 25.7% there makes a 257 um range, which the in-browser version refuses with advice
  if ((await txt('backend')).includes('in this browser')) {
    await page.check('input[name=ic][value=random]'); await page.fill('#nx', '100'); await page.fill('#ny', '100'); await page.selectOption('#m', '2');
    const g100 = await page.evaluate(() => ({ Rs: gatherCfg().params.fieldScreenSize, pct: document.getElementById('Rpct').value, geo: document.getElementById('geo').textContent }));
    check('100x100 cells, m=2: the range stays 28.3 um (R_s = 4, now 2.8% of the tissue) and the page estimates it is runnable', g100.Rs === 4 && g100.pct === '2.8' && !/too heavy/.test(g100.geo) && /est\. speed/.test(g100.geo), g100.geo.slice(-130));
    await page.click('#b-init'); await waitStatus(/ready · grid 200×200 px/, 60000);
    await page.fill('#T', '5'); await page.fill('#speedN', '50'); await page.click('#b-run'); await waitStatus(/finished at t = 5/, 120000);
    check('100x100 cells at m=2 runs in the browser', true, await txt('r-speed') + ' sim-s/s');
    // holding 25.7% on 100x100 (a 257 um range, 51 px) used to be refused as too heavy for the direct field sum; the FFT field makes it affordable
    await page.check('#holdpct'); await page.fill('#Rpct', '25.7');
    const geoLong = await txt('geo');
    check('holding 25.7% on 100x100 (a 257 um range) is no longer flagged too heavy: the FFT field handles the long range', !/too heavy/.test(geoLong) && /257\.\d µm = 51\.\d px/.test(geoLong) && /est\. speed/.test(geoLong), geoLong.slice(-150));
    await page.click('#b-init'); await waitStatus(/ready · grid 200×200 px/, 60000);
    check('... and the header says the FFT field is in use', /FFT field/.test(await txt('backend')), await txt('backend'));
    await page.fill('#T', '3'); await page.fill('#speedN', '50'); await page.click('#b-run'); await waitStatus(/finished at t = 3/, 180000);
    const sLong = parseFloat(await txt('r-speed')); check('... and runs at a usable speed (more than 0.3 simulated s per wall s; the direct sum managed 0.1)', sLong > 0.3, `${sLong} sim-s/s`);
    // a setting that is still too heavy: a periodic tissue whose size has a large prime factor (127 cells x 2 = 254 = 2 x 127) cannot use the FFT cheaply, and the direct sum at 326 um is slow
    await page.check('#periodic'); await page.fill('#nx', '127'); await page.fill('#ny', '127');
    const geoHeavy = await txt('geo');
    check('periodic 127x127 at 25.7%: the geometry note says too heavy and which range fits', /too heavy/.test(geoHeavy) && /up to about \d+ µm \(\d+\.\d% of the shorter side\) runs at roughly 1 simulated s per wall s/.test(geoHeavy), geoHeavy.slice(-230));
    await page.click('#b-init'); await page.waitForTimeout(800);
    const heavy = await txt('status');
    check('Initialise refuses it with the range, the estimate and what fits', /too heavy for the in-browser version/.test(heavy) && /3\d\d µm = \d+ px/.test(heavy) && /up to about \d+ µm/.test(heavy) && /128×128 cells/.test(heavy) && /Python server/.test(heavy), heavy);
    await page.uncheck('#periodic'); await page.uncheck('#holdpct'); await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.click('#b-defaults'); await page.click('#b-init'); await waitStatus(/ready/);
  }

  // oversize request surfaces the server's message
  await page.fill('#nx', '200'); await page.fill('#ny', '200'); await page.selectOption('#m', '8'); await page.click('#b-init'); await page.waitForTimeout(800);
  check('oversize tissue gives a readable error', (await txt('status')).includes('outside the allowed range'), await txt('status'));

  const unexpected = errors.filter(e => !/status of 400/.test(e));       // the one deliberate oversize request logs a 400
  check('no JavaScript or console errors (apart from the deliberate 400)', unexpected.length === 0 && errors.length <= 1, errors.join(' | '));
  await browser.close();
  console.log(fails ? `\n${fails} FAILURES` : '\nALL PASS'); process.exit(fails ? 1 : 0);
})().catch(e => { console.error('TEST CRASH:', e); process.exit(2); });
