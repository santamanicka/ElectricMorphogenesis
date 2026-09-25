// Checks the relay movie's controls and its bookkeeping against a stub DOM:
//   the slider positions (initial state, windows of movement, final state), the phase labels, that arrows are cleared
//   each frame, the two chart views, that play advances, scrubbing stops it and it halts at the end, and that the
//   running-total line on every frame adds up, chains into the next frame and sums from 0 to the final gap.
//   node checkRelayMovie.js figures/boundaryHarmonicSwitchRule.html
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const script = page.match(/<script>([\s\S]*?)<\/script>/g).pop().replace(/^<script>|<\/script>$/g, '');
const registry = {}, created = [];
const makeElement = (tag) => {
  const e = {tag, children: [], style: {}, dataset: {}, textContent: '', innerHTML: '', listeners: {}, attrs: {},
    setAttribute(k, v){ this.attrs[k] = v; }, getAttribute: () => '0 0 900 300', insertBefore(){},
    removeChild(c){ const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); },
    replaceChildren(){}, addEventListener(t, f){ (this.listeners[t] = this.listeners[t] || []).push(f); },
    querySelectorAll: () => [], querySelector: () => makeElement(),
    getBoundingClientRect: () => ({width:900,height:300,top:0,left:0}), classList:{add(){},remove(){},toggle(){}},
    appendChild(c){ this.children.push(c); return c; }, get firstChild(){ return this.children[0]; }};
  e.parentNode = {setAttribute(){}, appendChild(){}, insertBefore(){}};
  return e;
};
global.window = {addEventListener(){}, matchMedia: () => ({matches:false, addEventListener(){}}), getComputedStyle: () => ({getPropertyValue: () => ''})};
global.getComputedStyle = global.window.getComputedStyle; global.matchMedia = global.window.matchMedia;
global.document = {getElementById: id => (registry[id] = registry[id] || makeElement('byId:' + id)),
  createElementNS: (ns, t) => { const e = makeElement(t); created.push(e); return e; }, createElement: t => makeElement(t),
  documentElement: makeElement('html'), body: makeElement('body'), addEventListener(){}, querySelectorAll: () => [], querySelector: () => makeElement()};
try { new Function(script)(); } catch (e) { console.log('script error:', e.message); process.exit(1); }
const S = registry.movieSlider, B = registry.movieButton, R = registry.movieRescale, SH = registry.movieShares;
const groups = registry.relayMovieFigure.children.filter(c => c.tag === 'g');
const arrowGroup = groups[0], chartGroup = groups[1];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fire = (el, type) => el.listeners[type].forEach(f => f());
const at = p => { S.value = p; fire(S, 'input'); };
const texts = () => created.filter(e => e.tag === 'text').map(e => e.textContent);
const runningLine = () => texts().find(s => /^running total/.test(s));
const num = s => parseFloat(s.replace('−', '-'));
const ok = (name, cond, extra='') => { console.log((cond ? 'PASS' : 'FAIL') + '  ' + name + (extra ? '  (' + extra + ')' : '')); if (!cond) process.exitCode = 1; };
(async () => {
  const LAST = +S.max, NF = LAST - 1;
  ok('slider runs from the initial state through the windows to the final state', LAST === 37 && +S.value === 0, `positions 0..${LAST}`);
  at(0);
  ok('position 0 is the initial state: nothing moved, running total 0', registry.movieRange.textContent === '0' && /initial state/.test(registry.moviePhase.textContent)
     && arrowGroup.children.length === 0 && /\+0\.000/.test(runningLine()) && /identical/.test(runningLine()), runningLine());
  at(1);
  ok('position 1 is the first window, in the hold, with the clamp injecting', registry.movieRange.textContent === '0–49' && /hold: the ring is written/.test(registry.moviePhase.textContent)
     && arrowGroup.children.length > 0 && texts().some(s => /the clamp adds \+/.test(s)), registry.moviePhase.textContent);
  at(7);
  ok('position 7 holds the hold’s last step', registry.movieRange.textContent === '300–349' && /the hold ends/.test(registry.moviePhase.textContent) && texts().some(s => /hold’s last step, iteration 300/.test(s)));
  at(20); ok('a post-hold window has no injection note', !texts().some(s => /the clamp adds/.test(s)));
  const around = [];
  for (let p = 1; p <= NF; p++) { at(p); if (/around the trough/.test(registry.moviePhase.textContent)) around.push(p + ' = ' + registry.movieRange.textContent); }
  ok('exactly one window is labelled as around the trough, the one holding iteration 585', around.length === 1 && /550–599/.test(around[0]), around.join('; '));
  at(NF); ok('the last window ends at the readout', registry.movieRange.textContent === '1750–1765', registry.movieRange.textContent);
  at(LAST);
  ok('the last position is the final state: no arrows, running total = the final gap', registry.movieRange.textContent === '1765' && /final state/.test(registry.moviePhase.textContent)
     && arrowGroup.children.length === 0 && /= the final gap/.test(runningLine()) && /\+0\.356/.test(runningLine()), runningLine());

  // the bookkeeping as displayed: then + change = now on every window, each window starts where the last ended, and 0 + all changes = final
  const rows = [];
  for (let p = 1; p <= NF; p++) { at(p); const m = runningLine().match(/running total ([+−]\d\.\d+) ([+−]) (\d\.\d+) = ([+−]\d\.\d+)/); rows.push(m && { then: num(m[1]), change: (m[2] === '+' ? 1 : -1) * num(m[3]), now: num(m[4]) }); }
  ok('every window’s running-total line parses', rows.every(Boolean));
  ok('then + change = now on every window (to display rounding)', rows.every(r => Math.abs(r.then + r.change - r.now) < 0.0011), `worst ${Math.max(...rows.map(r => Math.abs(r.then + r.change - r.now))).toFixed(4)}`);
  ok('each window starts where the previous one ended', rows.every((r, i) => i === 0 || Math.abs(r.then - rows[i - 1].now) < 1e-9));
  ok('the first window starts at 0 and the last ends at the final gap 0.356', Math.abs(rows[0].then) < 1e-9 && Math.abs(rows[NF - 1].now - 0.356) < 0.0011, `${rows[0].then} -> ${rows[NF - 1].now}`);
  ok('initial 0 + all the changes = final', Math.abs(rows.reduce((s, r) => s + r.change, 0) - 0.356) < 0.005, `sum of displayed changes ${rows.reduce((s, r) => s + r.change, 0).toFixed(3)}`);

  const counts = new Set(); for (const p of [3, 11, 12, 20, 36, 0, 12]) { at(p); counts.add(arrowGroup.children.length); }
  ok('arrows are cleared each frame, not accumulated', Math.max(...counts) <= 2 * (45 + 30), `part counts seen ${[...counts].sort((a, b) => a - b).join(', ')}`);
  ok('the chart starts in the running-total view', chartGroup.children.some(c => c.tag === 'text' && /what each group adds to the readout/.test(c.textContent)));
  at(LAST); ok('running-total view: the chart readout ends at total +0.356', texts().some(s => /total \+0\.356/.test(s)));
  SH.checked = true; fire(SH, 'change');
  ok('the checkbox switches the chart to shares of the final gap', chartGroup.children.some(c => c.tag === 'text' && /what each group holds of the final gap/.test(c.textContent)) && chartGroup.children.length > 0);
  at(12); ok('shares view: the three readouts still add up (face + background + ring)', texts().some(s => /^face .*background .*ring /.test(s)));
  SH.checked = false; fire(SH, 'change'); ok('and switches back', chartGroup.children.some(c => c.tag === 'text' && /what each group adds to the readout/.test(c.textContent)));
  R.checked = true; fire(R, 'change'); R.checked = false; fire(R, 'change'); ok('rescale toggle redraws without error', true);

  at(5); fire(B, 'click');
  ok('play switches the button to pause', /pause|10074/.test(B.innerHTML), B.innerHTML);
  await sleep(1100); const advanced = +S.value;
  ok('play advances the slider', advanced > 5, `5 -> ${advanced}`);
  fire(S, 'input'); const held = +S.value; await sleep(700);
  ok('scrubbing stops playback', +S.value === held && /play/.test(B.innerHTML), `held at ${held}`);
  at(LAST - 2); fire(B, 'click'); await sleep(1300);
  ok('playback stops at the final state and resets the button', +S.value === LAST && /play/.test(B.innerHTML), `stopped at ${S.value}`);
  fire(B, 'click'); await sleep(300);
  ok('pressing play at the end restarts from the initial state', +S.value < LAST, `restarted at ${S.value}`); fire(S, 'input');
  process.exit(process.exitCode || 0);
})();
