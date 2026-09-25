// Checks the relay movie's controls against a stub DOM: the slider range, the phase labels, that arrows are cleared each frame,
// and that play advances, scrubbing stops it, and it halts at the last frame.   node checkRelayMovie.js figures/boundaryHarmonicSwitchRule.html
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
const S = registry.movieSlider, B = registry.movieButton, R = registry.movieRescale;
const arrowGroup = created.filter(e => e.tag === 'g').find(g => g.children.length > 0);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fire = (el, type) => el.listeners[type].forEach(f => f());
const ok = (name, cond, extra='') => { console.log((cond ? 'PASS' : 'FAIL') + '  ' + name + (extra ? '  (' + extra + ')' : '')); if (!cond) process.exitCode = 1; };
(async () => {
  ok('slider range set from the data', S.max === 29 && S.value === 0, `max ${S.max}`);
  ok('frame 0 drawn', registry.movieRange.textContent === '300–349' && arrowGroup.children.length > 0, `${registry.movieRange.textContent}, ${arrowGroup.children.length} svg parts`);
  const counts = new Set();
  for (const k of [3, 11, 12, 20, 29, 0, 11]) { S.value = k; fire(S, 'input'); counts.add(arrowGroup.children.length); }
  ok('arrows are cleared each frame, not accumulated', Math.max(...counts) <= 2 * (45 + 30), `part counts seen ${[...counts].sort((a,b)=>a-b).join(', ')}`);
  const around = [];
  for (let k = 0; k < 30; k++) { S.value = k; fire(S, 'input'); if (/around the trough/.test(registry.moviePhase.textContent)) around.push(k + ' = ' + registry.movieRange.textContent); }
  ok('exactly one frame is labelled as around the trough, the one holding iteration 585', around.length === 1 && /550–599/.test(around[0]), around.join('; '));
  S.value = 3; fire(S, 'input'); ok('frame 3 is the clear phase', /clear phase/.test(registry.moviePhase.textContent));
  S.value = 20; fire(S, 'input'); ok('frame 20 is the write phase', /write phase/.test(registry.moviePhase.textContent));
  S.value = 29; fire(S, 'input'); ok('last frame ends at the readout', registry.movieRange.textContent === '1750–1765', registry.movieRange.textContent);
  R.checked = true; fire(R, 'change'); R.checked = false; fire(R, 'change'); ok('rescale toggle redraws without error', true);
  S.value = 5; fire(S, 'input'); fire(B, 'click');
  ok('play switches the button to pause', /pause|10074/.test(B.innerHTML), B.innerHTML);
  await sleep(1100); const advanced = S.value;
  ok('play advances the slider', advanced > 5, `5 -> ${advanced}`);
  fire(S, 'input'); const held = S.value; await sleep(700);
  ok('scrubbing stops playback', S.value === held && /play/.test(B.innerHTML), `held at ${held}`);
  S.value = 27; fire(S, 'input'); fire(B, 'click'); await sleep(1300);
  ok('playback stops at the last frame and resets the button', S.value === 29 && /play/.test(B.innerHTML), `stopped at ${S.value}`);
  fire(B, 'click'); await sleep(300);
  ok('pressing play at the end restarts from the beginning', S.value < 29, `restarted at ${S.value}`); fire(S, 'input');
  process.exit(process.exitCode || 0);
})();
