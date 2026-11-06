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
  // after the tissue has run, painting G_pol leaves Vmem alone (it relaxes by itself)
  const pre = await page.evaluate(() => curV[11 * info.Np[1] + 11]);
  await page.selectOption('#ptarget', 'g'); await page.fill('#pval', '0');
  await page.mouse.move(...ctr); await page.mouse.down(); await page.mouse.up();
  const post = await page.evaluate(() => [curV[11 * info.Np[1] + 11], curG[11 * info.Np[1] + 11]]);
  check('on a running tissue painting G_pol leaves Vmem unchanged', post[0] === pre && post[1] === 0, `V ${pre.toFixed(2)} -> ${post[0].toFixed(2)} mV, g -> ${post[1]}`);
  await page.fill('#g0', '1'); await page.selectOption('#ptarget', 'g'); await page.fill('#pval', '2'); await page.fill('#prad', '2.5');
  await page.click('#b-init'); await waitStatus(/ready/);

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
