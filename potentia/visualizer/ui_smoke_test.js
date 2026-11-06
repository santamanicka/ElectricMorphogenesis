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
    const a = new L({ sizeCells: [7, 7], m: 2, params: p, simd: true }), b = new L({ sizeCells: [7, 7], m: 2, params: p, simd: false }), n = a.Np0 * a.Np1;
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
  // the field range is controlled as a percentage of the tissue's shorter side; R_s (the model parameter) is derived and stays exact
  const st = () => page.evaluate(() => ({ pct: document.getElementById('Rpct').value, Rs: gatherCfg().params.fieldScreenSize, geo: document.getElementById('geo').textContent, note: document.getElementById('range-note').textContent }));
  const um = Rs => 5 * Math.SQRT2 * (Rs + 0.001);
  let r = await st();
  check('default range: R_s = 4 exactly, shown as 25.7% of the 110 um tissue', r.Rs === 4 && r.pct === '25.7' && /28\.3 µm/.test(r.note), `${r.pct}% | ${r.note}`);
  await page.fill('#Rpct', '50'); r = await st();
  check('typing 50% gives a 55 um range (R_s = 55/(5*sqrt2) - 0.001)', Math.abs(um(r.Rs) - 55) < 1e-9 && /55\.0 µm/.test(r.geo), `R_s=${r.Rs.toFixed(4)} | ${r.geo.slice(0, 100)}`);
  await page.fill('#nx', '25'); await page.fill('#ny', '25'); r = await st();
  check('changing the tissue size holds the percentage, so the physical range scales (50% of 250 um = 125 um)', Math.abs(parseFloat(r.pct) - 50) < 1e-9 && Math.abs(um(r.Rs) - 125) < 1e-9, `${r.pct}% -> ${um(r.Rs).toFixed(1)} um`);
  await page.fill('#nx', '9'); await page.fill('#ny', '15'); r = await st();
  check('non-square tissue: the percentage refers to the shorter side (50% of 90 um = 45 um)', Math.abs(um(r.Rs) - 45) < 1e-9 && /90 µm shorter side/.test(r.note), r.note);
  const slider = await page.evaluate(() => { const s = document.getElementById('Rpct_sl'); s.value = '30'; s.dispatchEvent(new Event('input', { bubbles: true })); return { pct: document.getElementById('Rpct').value, Rs: gatherCfg().params.fieldScreenSize }; });
  check('the slider drives the same value (30% of 90 um = 27 um)', slider.pct === '30.0' && Math.abs(um(slider.Rs) - 27) < 1e-9, JSON.stringify(slider));
  await page.fill('#nx', '11'); await page.fill('#ny', '11'); await page.click('#b-defaults'); r = await st();
  check('Stigmergic defaults restore R_s = 4 exactly (25.7% of 110 um)', r.Rs === 4 && r.pct === '25.7', `${r.pct}% R_s=${r.Rs}`);
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

  // oversize request surfaces the server's message
  await page.fill('#nx', '200'); await page.fill('#ny', '200'); await page.selectOption('#m', '8'); await page.click('#b-init'); await page.waitForTimeout(800);
  check('oversize tissue gives a readable error', (await txt('status')).includes('outside the allowed range'), await txt('status'));

  const unexpected = errors.filter(e => !/status of 400/.test(e));       // the one deliberate oversize request logs a 400
  check('no JavaScript or console errors (apart from the deliberate 400)', unexpected.length === 0 && errors.length <= 1, errors.join(' | '));
  await browser.close();
  console.log(fails ? `\n${fails} FAILURES` : '\nALL PASS'); process.exit(fails ? 1 : 0);
})().catch(e => { console.error('TEST CRASH:', e); process.exit(2); });
