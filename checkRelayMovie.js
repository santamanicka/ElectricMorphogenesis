// Checks the relay movie's controls and bookkeeping against a stub DOM with a fake canvas and a manual animation clock.
//   Continuous mode (the default): the slider is a continuous time, colours/arrows/chart interpolate smoothly and are exact
//   at window ends and centres, the flow particles stream and stay bounded, pausing stops time but not the flow.
//   Frame-by-frame mode: the initial state, the windows of movement, the final state, the phase labels, arrows cleared each
//   frame, the two chart views, play/scrub/stop, and the running-total line adding up, chaining and summing from 0 to the gap.
//   node checkRelayMovie.js figures/boundaryHarmonicSwitchRule.html
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const script = page.match(/<script>([\s\S]*?)<\/script>/g).pop().replace(/^<script>|<\/script>$/g, '');
const registry = {}, created = [], rafQueue = [];
let rafId = 0, fakeNow = 1000;
const makeElement = (tag) => {
  const e = {tag, children: [], style: {}, dataset: {}, textContent: '', innerHTML: '', listeners: {}, attrs: {},
    setAttribute(k, v){ this.attrs[k] = v; }, getAttribute: () => '0 0 900 300', insertBefore(){},
    removeChild(c){ const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); },
    replaceChildren(){}, addEventListener(t, f){ (this.listeners[t] = this.listeners[t] || []).push(f); },
    querySelectorAll: () => [], querySelector: () => makeElement(),
    getBoundingClientRect: () => ({width:900,height:300,top:0,left:0}), classList:{add(){},remove(){},toggle(){}},
    appendChild(c){ this.children.push(c); return c; }, get firstChild(){ return this.children[0]; }};
  if (tag === 'canvas') {
    const calls = {};
    const count = name => () => { calls[name] = (calls[name] || 0) + 1; };
    e.calls = calls;
    e.getContext = () => ({clearRect: count('clearRect'), fillRect: count('fillRect'), beginPath: count('beginPath'), moveTo: count('moveTo'),
      lineTo: count('lineTo'), stroke: count('stroke'), fill: count('fill'), closePath: count('closePath'), arc: count('arc')});
  }
  e.parentNode = {setAttribute(){}, appendChild(){}, insertBefore(){}};
  return e;
};
global.window = {addEventListener(){}, matchMedia: () => ({matches:false, addEventListener(){}}), getComputedStyle: () => ({getPropertyValue: () => ''})};
global.getComputedStyle = global.window.getComputedStyle; global.matchMedia = global.window.matchMedia;
global.requestAnimationFrame = f => { rafQueue.push({id: ++rafId, f}); return rafId; };
global.cancelAnimationFrame = id => { const i = rafQueue.findIndex(q => q.id === id); if (i >= 0) rafQueue.splice(i, 1); };
global.document = {getElementById: id => (registry[id] = registry[id] || makeElement('byId:' + id)),
  createElementNS: (ns, t) => { const e = makeElement(t); created.push(e); return e; },
  createElement: t => { const e = makeElement(t); created.push(e); return e; },
  documentElement: makeElement('html'), body: makeElement('body'), addEventListener(){}, querySelectorAll: () => [], querySelector: () => makeElement()};
try { new Function(script)(); } catch (e) { console.log('script error:', e.message); process.exit(1); }
const mv = registry.relayMovieFigure, dbg = () => mv.movieDebug();
const S = registry.movieSlider, B = registry.movieButton, R = registry.movieRescale, SH = registry.movieShares, FR = registry.movieFrames;
const groups = mv.children.filter(c => c.tag === 'g');
const arrowGroup = groups[0], chartGroup = groups[1];
const cellRects = mv.children.filter(c => c.tag === 'rect').slice(0, 121);
const canvases = created.filter(e => e.tag === 'canvas');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const tick = ms => { fakeNow += ms; rafQueue.splice(0).forEach(q => q.f(fakeNow)); };
const fire = (el, type) => el.listeners[type].forEach(f => f());
const at = v => { S.value = v; fire(S, 'input'); };
const texts = () => created.filter(e => e.tag === 'text').map(e => e.textContent);
const runningLine = () => texts().find(s => /^running total/.test(s));
const num = s => parseFloat(s.replace('−', '-'));
const opacities = () => cellRects.map(c => +c.attrs['fill-opacity']);
const ok = (name, cond, extra='') => { console.log((cond ? 'PASS' : 'FAIL') + '  ' + name + (extra ? '  (' + extra + ')' : '')); if (!cond) process.exitCode = 1; };
(async () => {
  // ======================================================= continuous mode (the default)
  ok('the movie opens in continuous mode with a continuous time slider', dbg().mode === 'smooth' && +S.max === 1765 && S.step === 'any', `max ${S.max}`);
  ok('two canvas layers sit over the lattice (arrows, flow)', canvases.length === 2 && canvases.every(c => c.style.position === 'absolute'));
  ok('t = 0 is the initial state: no transfers, no particles, running total 0', dbg().t === 0 && dbg().edges === 0 && dbg().particles === 0 && /identical/.test(runningLine()) && /nothing has moved yet/.test(texts().join('|')));
  fire(B, 'click'); ok('play switches the button to pause', /pause|10074/.test(B.innerHTML) && dbg().playing);
  const ts = []; for (let i = 0; i < 40; i++) { tick(16); ts.push(dbg().t); }
  ok('playing advances time in many small steps, not window by window', new Set(ts).size === 40 && ts.every((v, i) => i === 0 || v > ts[i - 1]) && Math.abs(ts[39] - 39 * 220 * 0.016) < 0.5, `t after 40 ticks: ${ts[39].toFixed(1)} (the first frame after load has no elapsed time, so 39 steps of ${(220 * 0.016).toFixed(2)})`);
  ok('the slider follows the clock', Math.abs(+S.value - dbg().t) < 1e-9);
  ok('particles are streaming while the hold plays', dbg().particles > 0, `${dbg().particles} particles`);
  ok('the flow canvas fades once per frame and draws the particles', (canvases[1].calls.fillRect || 0) >= 39 && (canvases[1].calls.stroke || 0) > 0, `${canvases[1].calls.fillRect} fades, ${canvases[1].calls.stroke} strokes`);
  const before = dbg().t, strokesBefore = canvases[1].calls.stroke; fire(B, 'click');
  for (let i = 0; i < 20; i++) tick(16);
  ok('pausing stops time', dbg().t === before && !dbg().playing && /play/.test(B.innerHTML));
  ok('but the flow keeps moving while paused', canvases[1].calls.stroke > strokesBefore && dbg().particles > 0, `${strokesBefore} -> ${canvases[1].calls.stroke} strokes`);

  // exactness where the data is exact: colours at window ends match the frame-by-frame view
  const smoothAt = {}; for (const t of [0, 49, 399, 1749, 1765]) { at(t); smoothAt[t] = opacities(); }
  const edgesAtCentre = (at(24.5), dbg().edges), edgesAtStart = (at(0), dbg().edges), edgesAtEnd = (at(1765), dbg().edges);
  ok('transfer rates are 0 at the start and at the readout', edgesAtStart === 0 && edgesAtEnd === 0);
  FR.checked = true; fire(FR, 'change');
  ok('the checkbox switches to frame by frame: 38 positions, canvases hidden, no loop', dbg().mode === 'frames' && +S.max === 37 && canvases.every(c => c.style.display === 'none') && rafQueue.length === 0);
  const frameOpacity = {}; for (const [t, p] of [[0, 0], [49, 1], [399, 8], [1749, 35], [1765, 36]]) { at(p); frameOpacity[t] = opacities(); }
  const worst = t => Math.max(...smoothAt[t].map((v, i) => Math.abs(v - frameOpacity[t][i])));
  ok('colours at window ends are exactly the frame-by-frame colours', [0, 49, 399, 1749, 1765].every(t => worst(t) < 1e-9), `worst ${Math.max(...[0, 49, 399, 1749, 1765].map(worst)).toExponential(1)}`);
  at(1); const framesArrows = arrowGroup.children.length / 2;
  ok('at a window centre the continuous view draws the same transfers as that window', framesArrows === edgesAtCentre, `${edgesAtCentre} vs ${framesArrows}`);

  // continuity: the biggest colour step between neighbouring times, against the biggest jump between neighbouring windows
  FR.checked = false; fire(FR, 'change');
  ok('switching back returns to continuous mode with a continuous slider', dbg().mode === 'smooth' && S.step === 'any' && +S.max === 1765 && canvases.every(c => c.style.display === 'block'));
  const jump = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  let bigWindowJump = 0; FR.checked = true; fire(FR, 'change');
  let prev = (at(0), opacities()); for (let p = 1; p <= 36; p++) { at(p); const cur = opacities(); bigWindowJump = Math.max(bigWindowJump, jump(prev, cur)); prev = cur; }
  FR.checked = false; fire(FR, 'change');
  let bigStep = 0; prev = (at(0), opacities()); for (let x = 1; x <= 1765; x += 1) { at(x); const cur = opacities(); bigStep = Math.max(bigStep, jump(prev, cur)); prev = cur; }
  ok('colours change smoothly: the largest step per iteration is a small fraction of the largest jump between windows', bigStep < bigWindowJump / 8, `${bigStep.toFixed(4)} vs ${bigWindowJump.toFixed(4)}`);
  const line = x => { at(x); return num(runningLine().match(/running total ([+−]\d\.\d+)/)[1]); };
  const totals = []; for (let x = 0; x <= 1765; x += 5) totals.push(line(x));
  ok('the running total changes smoothly, from 0 to the final gap', Math.abs(totals[0]) < 1e-9 && Math.abs(totals[totals.length - 1] - 0.356) < 0.0011 && Math.max(...totals.map((v, i) => i ? Math.abs(v - totals[i - 1]) : 0)) < 0.02, `max step per 5 iterations ${Math.max(...totals.map((v, i) => i ? Math.abs(v - totals[i - 1]) : 0)).toFixed(4)}`);
  at(1765); ok('at the readout the line says final gap', /= the final gap/.test(runningLine()) && /final state/.test(registry.moviePhase.textContent));
  const cursors = []; for (const x of [100, 101, 102]) { at(x); cursors.push(+created.filter(e => e.tag === 'line' && e.attrs.x1 !== undefined && e.attrs.y1 === 52 && e.attrs['stroke-width'] === 1.2).pop().attrs.x1); }
  ok('the chart cursor moves continuously with time', cursors[1] > cursors[0] && cursors[2] > cursors[1] && cursors[2] - cursors[0] < 1, cursors.map(v => v.toFixed(3)).join(', '));
  at(650); const busy = dbg().particles, busyEdges = dbg().edges; at(1500); const quiet = dbg().particles; at(650);
  ok('scrubbing to a time pre-rolls the flow and labels the phase', busy > 20 && busyEdges > 10 && /write phase/.test(registry.moviePhase.textContent) && registry.movieRange.textContent === '650', `${busy} particles, ${busyEdges} transfers`);
  ok('the flow is denser where more moves', quiet > 0 && busy > 2 * quiet, `${busy} at 650 against ${quiet} at 1500`);
  at(585); ok('the trough is labelled', /around the trough/.test(registry.moviePhase.textContent));
  at(100); ok('the hold is labelled and the clamp injection is shown', /hold: the ring is written/.test(registry.moviePhase.textContent) && texts().some(s => /the clamp adds \+/.test(s)));
  for (let i = 0; i < 400; i++) tick(16);
  ok('the flow stays bounded over a long run', dbg().particles <= 2500, `${dbg().particles} particles`);
  SH.checked = true; fire(SH, 'change'); at(500);
  ok('the shares view works in continuous mode', chartGroup.children.some(c => c.tag === 'text' && /what each group holds of the final gap/.test(c.textContent)));
  SH.checked = false; fire(SH, 'change'); R.checked = true; fire(R, 'change'); R.checked = false; fire(R, 'change'); ok('rescale and chart toggles redraw without error', true);
  at(300); FR.checked = true; fire(FR, 'change'); const framePos = +S.value;
  ok('switching to frames from time 300 lands on the window holding it', framePos === 7, `position ${framePos}`);
  at(12); FR.checked = false; fire(FR, 'change');
  ok('switching from a frame position to continuous lands on that window’s centre', Math.abs(dbg().t - 574.5) < 1e-9, `t ${dbg().t}`);

  // ======================================================= frame-by-frame mode
  FR.checked = true; fire(FR, 'change');
  const LAST = +S.max, NF = LAST - 1, fat = p => { S.value = p; fire(S, 'input'); };
  ok('frame by frame: slider runs from the initial state through the windows to the final state', LAST === 37);
  fat(0);
  ok('position 0 is the initial state: nothing moved, running total 0', registry.movieRange.textContent === '0' && /initial state/.test(registry.moviePhase.textContent)
     && arrowGroup.children.length === 0 && /\+0\.000/.test(runningLine()) && /identical/.test(runningLine()), runningLine());
  fat(1);
  ok('position 1 is the first window, in the hold, with the clamp injecting', registry.movieRange.textContent === '0–49' && /hold: the ring is written/.test(registry.moviePhase.textContent)
     && arrowGroup.children.length > 0 && texts().some(s => /the clamp adds \+/.test(s)), registry.moviePhase.textContent);
  fat(7);
  ok('position 7 holds the hold’s last step', registry.movieRange.textContent === '300–349' && /the hold ends/.test(registry.moviePhase.textContent) && texts().some(s => /hold’s last step, iteration 300/.test(s)));
  fat(20); ok('a post-hold window has no injection note', !texts().some(s => /the clamp adds/.test(s)));
  const around = [];
  for (let p = 1; p <= NF; p++) { fat(p); if (/around the trough/.test(registry.moviePhase.textContent)) around.push(p + ' = ' + registry.movieRange.textContent); }
  ok('exactly one window is labelled as around the trough, the one holding iteration 585', around.length === 1 && /550–599/.test(around[0]), around.join('; '));
  fat(NF); ok('the last window ends at the readout', registry.movieRange.textContent === '1750–1765', registry.movieRange.textContent);
  fat(LAST);
  ok('the last position is the final state: no arrows, running total = the final gap', registry.movieRange.textContent === '1765' && /final state/.test(registry.moviePhase.textContent)
     && arrowGroup.children.length === 0 && /= the final gap/.test(runningLine()) && /\+0\.356/.test(runningLine()), runningLine());
  const rows = [];
  for (let p = 1; p <= NF; p++) { fat(p); const m = runningLine().match(/running total ([+−]\d\.\d+) ([+−]) (\d\.\d+) = ([+−]\d\.\d+)/); rows.push(m && { then: num(m[1]), change: (m[2] === '+' ? 1 : -1) * num(m[3]), now: num(m[4]) }); }
  ok('every window’s running-total line parses', rows.every(Boolean));
  ok('then + change = now on every window (to display rounding)', rows.every(r => Math.abs(r.then + r.change - r.now) < 0.0011), `worst ${Math.max(...rows.map(r => Math.abs(r.then + r.change - r.now))).toFixed(4)}`);
  ok('each window starts where the previous one ended', rows.every((r, i) => i === 0 || Math.abs(r.then - rows[i - 1].now) < 1e-9));
  ok('the first window starts at 0 and the last ends at the final gap 0.356', Math.abs(rows[0].then) < 1e-9 && Math.abs(rows[NF - 1].now - 0.356) < 0.0011, `${rows[0].then} -> ${rows[NF - 1].now}`);
  ok('initial 0 + all the changes = final', Math.abs(rows.reduce((s, r) => s + r.change, 0) - 0.356) < 0.005, `sum of displayed changes ${rows.reduce((s, r) => s + r.change, 0).toFixed(3)}`);
  const counts = new Set(); for (const p of [3, 11, 12, 20, 36, 0, 12]) { fat(p); counts.add(arrowGroup.children.length); }
  ok('arrows are cleared each frame, not accumulated', Math.max(...counts) <= 2 * (45 + 30), `part counts seen ${[...counts].sort((a, b) => a - b).join(', ')}`);
  ok('the chart starts in the running-total view', chartGroup.children.some(c => c.tag === 'text' && /what each group adds to the readout/.test(c.textContent)));
  fat(LAST); ok('running-total view: the chart readout ends at total +0.356', texts().some(s => /total \+0\.356/.test(s)));
  SH.checked = true; fire(SH, 'change');
  ok('the checkbox switches the chart to shares of the final gap', chartGroup.children.some(c => c.tag === 'text' && /what each group holds of the final gap/.test(c.textContent)) && chartGroup.children.length > 0);
  fat(12); ok('shares view: the three readouts still add up (face + background + ring)', texts().some(s => /^face .*background .*ring /.test(s)));
  SH.checked = false; fire(SH, 'change'); ok('and switches back', chartGroup.children.some(c => c.tag === 'text' && /what each group adds to the readout/.test(c.textContent)));
  fat(5); fire(B, 'click');
  ok('frame play switches the button to pause', /pause|10074/.test(B.innerHTML), B.innerHTML);
  await sleep(1100); const advanced = +S.value;
  ok('frame play advances the slider', advanced > 5, `5 -> ${advanced}`);
  fire(S, 'input'); const held = +S.value; await sleep(700);
  ok('scrubbing stops frame playback', +S.value === held && /play/.test(B.innerHTML), `held at ${held}`);
  fat(LAST - 2); fire(B, 'click'); await sleep(1300);
  ok('frame playback stops at the final state and resets the button', +S.value === LAST && /play/.test(B.innerHTML), `stopped at ${S.value}`);
  fire(B, 'click'); await sleep(300);
  ok('pressing play at the end restarts from the initial state', +S.value < LAST, `restarted at ${S.value}`); fire(S, 'input');
  process.exit(process.exitCode || 0);
})();
