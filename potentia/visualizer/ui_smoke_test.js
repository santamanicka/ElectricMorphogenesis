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

  // painting is refused while running
  await page.fill('#T', '600'); await page.fill('#speedN', '5'); await page.click('#b-run'); await page.waitForTimeout(500);
  const before = await page.evaluate(() => curV.slice(0).filter(v => v < -50).length);
  await page.mouse.move(...at(0.5, 0.5)); await page.mouse.down(); await page.mouse.move(...at(0.55, 0.55)); await page.mouse.up();
  check('painting while running is refused', (await txt('status')).length > 0 && (await page.evaluate(() => document.getElementById('status').textContent)).includes('pause to paint'));
  await page.click('#b-run');

  // bigger tissue + random start + m
  await page.check('input[name=ic][value=random]'); await page.fill('#nx', '25'); await page.fill('#ny', '25'); await page.selectOption('#m', '2');
  await page.fill('#sigV', '8'); await page.fill('#seed', '5');
  check('changing settings flags "press Initialise"', (await txt('status')).includes('Initialise'));
  await page.click('#b-init'); await waitStatus(/ready · grid 50×50/);
  const sd = await page.evaluate(() => { let s = 0, s2 = 0; for (const v of curV) { s += v; s2 += v*v; } const n = curV.length, mu = s/n; return Math.sqrt(s2/n - mu*mu); });
  check('random start has the requested spread (~8 mV)', sd > 6 && sd < 10, `sd=${sd.toFixed(2)} mV`);
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
