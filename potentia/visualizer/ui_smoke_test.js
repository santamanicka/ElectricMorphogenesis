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
    check('shape selector: rectangle, 10 shapes and custom', opts.length === 12 && opts[0] === 'rectangle' && ['disc', 'ellipse', 'annulus', 'stadium', 'hexagon', 'triangle', 'cross', 'dumbbell', 'tadpole', 'planarian', 'custom'].every(v => opts.includes(v)), opts.join(','));
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
    // the planarian chosen on the default square box: the box becomes 11 x 51 cells; the head is as wide as the body (no neck), the tail tapers to a point and a margin is left all round
    await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.selectOption('#shape', 'planarian');
    const plNote = await page.evaluate(() => ({ ny: document.getElementById('ny').value, note: document.getElementById('shape-note').textContent }));
    await page.click('#b-init'); await waitStatus(/ready/);
    const plW = await page.evaluate(() => { const [R, C] = info.Np, wd = col => { let n = 0; for (let i = 0; i < R; i++) n += curMask[i * C + col]; return n; }; let neckMin = 99; for (let col = Math.ceil(0.18 * C); col < Math.floor(0.5 * C); col++) neckMin = Math.min(neckMin, wd(col)); let edge = 0; for (let i = 0; i < R; i++) edge += curMask[i * C] + curMask[i * C + C - 1]; for (let j = 0; j < C; j++) edge += curMask[j] + curMask[(R - 1) * C + j]; return { R, C, nose: wd(1), head: wd(Math.ceil(0.18 * C)), neckMin, body: wd(Math.floor(0.45 * C)), tail: wd(Math.floor(0.97 * C)), edge }; });
    check('planarian chosen on the default 11x11 box: the box becomes 11 x 51 cells; the head is as wide as the body (no bottleneck between them), the tail is narrow and nothing touches the edge of the box',
      plNote.ny === '51' && /box was changed to 11 × 51/.test(plNote.note) && plW.head === plW.R - 2 && plW.neckMin === plW.R - 2 && plW.body === plW.R - 2 && plW.tail > 0 && plW.tail <= 4 && plW.nose < plW.head && plW.edge === 0, JSON.stringify(plW));
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
    const tl = await page.evaluate(() => { const [R, C] = info.Np, col = j => { let n = 0, top = -1, bot = -1; for (let i = 0; i < R; i++) if (curMask[i * C + j]) { n++; if (top < 0) top = i; bot = i; } return { n, top, bot }; };
      let edge = 0; for (let i = 0; i < R; i++) edge += curMask[i * C] + curMask[i * C + C - 1]; for (let j = 0; j < C; j++) edge += curMask[j] + curMask[(R - 1) * C + j];
      return { R, C, body: col(Math.floor(0.25 * C)), head: col(Math.floor(0.15 * C)), fin: col(Math.floor(0.55 * C)), tip: col(C - 4), snout: col(1), edge }; });
    check('tadpole chosen on the default 11x11 box: the box becomes 11 x 41 cells (stated in the note); oval body, tail with fins as deep as the body, narrow tail tip, symmetric about the long axis, margin all round',
      tp.ny === '41' && /box was changed to 11 × 41/.test(tp.note) && tl.body.n >= 0.7 * tl.R && tl.fin.n >= tl.body.n && [tl.body, tl.head, tl.fin].every(c => c.top + c.bot === tl.R - 1) && tl.tip.n > 0 && tl.tip.n <= 4 && tl.snout.n < tl.body.n / 2 && tl.edge === 0, JSON.stringify(tl));
    // the second parameter of the tadpole: head size / tail size; limited so that the head fits in the box (0.998 / tail depth)
    const p2a = await page.evaluate(() => ({ on: document.getElementById('shapeP2-wrap').classList.contains('on'), label: document.getElementById('shapeP2-label').textContent, val: document.getElementById('shapeP2').value, max: document.getElementById('shapeP2').max, maxSl: document.getElementById('shapeP2_sl').max, p: document.getElementById('shapeP').value }));
    await page.fill('#shapeP', '0.5'); const p2b = await page.evaluate(() => document.getElementById('shapeP2').max); await page.fill('#shapeP', '0.9');
    check('tadpole: a second slider "head size / tail size" (default 0.924), limited to 1.1 at a tail depth of 0.9 and to 1.99 at 0.5',
      p2a.on && /head size \/ tail size/.test(p2a.label) && p2a.val === '0.924' && p2a.max === '1.1' && p2a.maxSl === '1.1' && p2b === '1.99', JSON.stringify({ ...p2a, max05: p2b }));
    await page.fill('#shapeP2', '0.6'); await page.click('#b-init'); await waitStatus(/ready/);
    const tl2 = await page.evaluate(() => { const [R, C] = info.Np, col = j => { let n = 0; for (let i = 0; i < R; i++) n += curMask[i * C + j]; return n; }; return { body: col(Math.floor(0.25 * C)), tail: col(Math.floor(0.75 * C)) }; });
    check('tadpole: a smaller head size / tail size (0.6) gives a smaller head and leaves the tail as deep as before', tl2.body < tl.body.n * 0.8 && tl2.tail === (await page.evaluate(() => { const [R, C] = info.Np; let n = 0; for (let i = 0; i < R; i++) n += PotentiaCoreLib.PotentiaCore.shapeMask('tadpole', R, C)[i * C + Math.floor(0.75 * C)]; return n; })), JSON.stringify({ before: tl.body.n, after: tl2.body, tail: tl2.tail }));
    await page.selectOption('#shape', 'disc'); const p2off = await page.evaluate(() => document.getElementById('shapeP2-wrap').classList.contains('on')); await page.selectOption('#shape', 'tadpole');
    const p2back = await page.evaluate(() => document.getElementById('shapeP2').value);
    check('the second slider appears only for the tadpole and returns to its default when the shape is chosen again', !p2off && p2back === '0.924', p2back);
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
    check('all 10 presets build, run 1 s on a 10x18-cell box (made elongated where the shape needs it) and keep the outside empty', bad.length === 0, bad.join(' | '));
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
    check('after the development duration the tissue is exactly the tadpole preset (16526 px), nothing outside it, state finite, readout says Tadpole', fin.n === 16526 && fin.diff === 0 && fin.bad === 0 && fin.finite && /Tadpole \(NF 45\)/.test(fin.readout), JSON.stringify({ n: fin.n, diff: fin.diff, bad: fin.bad }));
    await page.screenshot({ path: `${shots}/11_development_tadpole.png` });
    // the field range follows the tissue: 25% of the shorter extent (egg 170 um -> 42.5 um, tadpole 420 um high -> 105 um)
    await page.selectOption('#devRange', 'prop'); await page.fill('#devPct', '25'); await page.click('#b-init'); await waitStatus(/ready/);
    const r0 = await page.evaluate(() => lastDevStatus.range_um), d0 = await txt('r-range'), p0 = await txt('r-pct'); await page.click('#b-run'); await waitStatus(/finished at t = 30/, 120000);
    const r1 = await page.evaluate(() => ({ r: lastDevStatus.range_um, core: (typeof JS !== 'undefined' && JS.core) ? JS.core.R * 1e6 : null }));
    check('proportional field range: 42.5 um at the egg, 105 um at the tadpole', Math.abs(r0 - 42.5) < 1 && Math.abs(r1.r - 105) < 1 && (r1.core === null || Math.abs(r1.core - 105) < 1) && d0 === '42.5' && (await txt('r-range')) === '105' && Math.abs(parseFloat(p0) - 25) < 0.3 && Math.abs(parseFloat(await txt('r-pct')) - 25) < 0.3, `${r0.toFixed(1)} -> ${r1.r.toFixed(1)} um; dashboard ${d0} (${p0}%) -> ${await txt('r-range')} (${await txt('r-pct')}%)`);
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
