// Measures every element a report's figures draw against that figure's viewBox, so content that would be
// clipped (or would bleed into the caption below) is caught here rather than by eye.
//   node checkFigureBounds.js figures/boundaryHarmonicSwitchRule.html
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const script = page.match(/<script>([\s\S]*?)<\/script>/g).pop().replace(/^<script>|<\/script>$/g, '');

// viewBox per svg id, read from the markup
const viewBoxes = {};
// the id may sit on the <svg> itself, or on an inner <g> that the drawing code targets
for (const m of page.matchAll(/<svg id="([A-Za-z0-9_]+)"[^>]*viewBox="0 0 ([\d.]+) ([\d.]+)"/g)) {
  viewBoxes[m[1]] = {width: +m[2], height: +m[3]};
}
for (const m of page.matchAll(/<svg[^>]*viewBox="0 0 ([\d.]+) ([\d.]+)"[^>]*>\s*<g id="([A-Za-z0-9_]+)"/g)) {
  viewBoxes[m[3]] = {width: +m[1], height: +m[2]};
}

const roots = {};
let current = null;
const widthOf = (text, size, family) => String(text).length * size * (/Mono/.test(family || '') ? 0.62 : 0.55);
// text widths are estimated from character counts, so a few px of reported overflow on a text node is noise;
// rects, lines and paths are exact geometry and are held to a tight tolerance
const TEXT_TOLERANCE = node => (node.tag === 'text' ? 4 : 1.5);

const make = (tag = 'div') => {
  const e = {tag, attrs: {}, children: [], listeners: {}, style: {}, textContent: '', innerHTML: '',
    setAttribute(k, v) { e.attrs[k] = v; },
    getAttribute(k) { return e.attrs[k] !== undefined ? e.attrs[k] : (viewBoxes[e.id] ? `0 0 ${viewBoxes[e.id].width} ${viewBoxes[e.id].height}` : '0 0 900 300'); },
    appendChild(c) { e.children.push(c); c.parent = e; return c; },
    insertBefore(c) { e.children.push(c); return c; },
    removeChild() {}, replaceChildren() { e.children = []; },
    addEventListener(t, f) { (e.listeners[t] = e.listeners[t] || []).push(f); },
    querySelectorAll: () => [], querySelector: () => make(),
    classList: {add() {}, remove() {}, toggle() {}},
    getBoundingClientRect: () => ({width: 900, height: 300, top: 0, left: 0}),
    childNodes: [], firstChild: null, dataset: {},
    // canvas-backed figures draw through a 2D context; stub enough of it to let them run
    width: 300, height: 150,
    getContext: () => ({
      createImageData: (w, h) => ({width: w, height: h, data: new Uint8ClampedArray(Math.max(1, w * h * 4))}),
      putImageData() {}, fillRect() {}, clearRect() {}, drawImage() {}, beginPath() {}, moveTo() {}, lineTo() {},
      stroke() {}, fill() {}, save() {}, restore() {}, translate() {}, scale() {}, setTransform() {},
      fillText() {}, measureText: () => ({width: 0}), arc() {}, closePath() {},
      set fillStyle(v) {}, set strokeStyle(v) {}, set lineWidth(v) {}, set font(v) {}}),
  };
  e.parent = null;
  // a real recorder, so a figure that resets its own viewBox at the end is measured against the new one
  e.parentNode = {attrs: {}, setAttribute(k, v) { this.attrs[k] = v; }, appendChild() {}, insertBefore() {}};
  return e;
};
const byId = {};
const getEl = id => {
  if (!byId[id]) { byId[id] = Object.assign(make(), {id, value: '0', checked: true}); if (viewBoxes[id]) roots[id] = byId[id]; }
  return byId[id];
};
global.window = {addEventListener() {}, matchMedia: () => ({matches: false, addEventListener() {}}),
  getComputedStyle: () => ({getPropertyValue: () => '#000'}), devicePixelRatio: 1, innerWidth: 1000};
global.document = {getElementById: getEl, createElementNS: (ns, t) => make(t), createElement: t => make(t),
  querySelectorAll: () => [], querySelector: () => make(), body: make(), documentElement: make(), addEventListener() {}};
global.setInterval = () => 1; global.clearInterval = () => {}; global.requestAnimationFrame = cb => cb(0);
// some pages call these bare rather than through window
global.getComputedStyle = () => ({getPropertyValue: () => '#000'});
global.matchMedia = () => ({matches: false, addEventListener() {}});
eval(script);

const num = v => { const n = parseFloat(v); return Number.isFinite(n) ? n : null; };
function boxOf(node) {
  const a = node.attrs, t = node.tag;
  if (t === 'rect') {
    const x = num(a.x), y = num(a.y), w = num(a.width), h = num(a.height);
    if (x === null || y === null) return null;
    return [x, y, x + (w || 0), y + (h || 0)];
  }
  if (t === 'line') {
    const x1 = num(a.x1), x2 = num(a.x2), y1 = num(a.y1), y2 = num(a.y2);
    if ([x1, x2, y1, y2].some(v => v === null)) return null;
    return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
  }
  if (t === 'circle') {
    const cx = num(a.cx), cy = num(a.cy), r = num(a.r) || 0;
    if (cx === null || cy === null) return null;
    return [cx - r, cy - r, cx + r, cy + r];
  }
  if (t === 'text') {
    if (a.transform && /rotate/.test(a.transform)) return null;   // rotated labels measured separately
    const x = num(a.x), y = num(a.y);
    if (x === null || y === null) return null;
    const size = num(a['font-size']) || 12;
    const w = widthOf(node.textContent, size, a['font-family']);
    const anchor = a['text-anchor'] || 'start';
    const left = anchor === 'end' ? x - w : anchor === 'middle' ? x - w / 2 : x;
    return [left, y - size * 0.82, left + w, y + size * 0.24];
  }
  if (t === 'path') {
    const d = a.d || ''; const xs = [], ys = [];
    for (const m of d.matchAll(/[ML]\s*(-?[\d.]+)\s+(-?[\d.]+)/g)) { xs.push(+m[1]); ys.push(+m[2]); }
    if (!xs.length) return null;
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }
  return null;
}
let problems = 0;
for (const [id, root] of Object.entries(roots)) {
  let vb = viewBoxes[id];
  const set = root.parentNode && root.parentNode.attrs && root.parentNode.attrs.viewBox;
  if (set) {
    const parts = String(set).trim().split(/\s+/).map(Number);
    if (parts.length === 4 && parts.every(Number.isFinite)) vb = {width: parts[2], height: parts[3], dynamic: true};
  }
  let worst = {l: 0, t: 0, r: 0, b: 0}, offenders = [];
  const walk = node => {
    const b = boxOf(node);
    if (b) {
      const [x0, y0, x1, y1] = b;
      const over = {l: -x0, t: -y0, r: x1 - vb.width, b: y1 - vb.height};
      const worstSide = Math.max(over.l, over.t, over.r, over.b);
      if (worstSide > TEXT_TOLERANCE(node)) {
        offenders.push({tag: node.tag, text: (node.textContent || '').slice(0, 34), over, box: b.map(v => Math.round(v))});
      }
      for (const k of ['l', 't', 'r', 'b']) worst[k] = Math.max(worst[k], over[k]);
    }
    node.children.forEach(walk);
  };
  walk(root);
  const clean = offenders.length === 0;
  console.log(`${id.padEnd(15)} viewBox ${vb.width}x${vb.height}${vb.dynamic ? '*' : ' '} overflow L${worst.l.toFixed(0)} T${worst.t.toFixed(0)} R${worst.r.toFixed(0)} B${worst.b.toFixed(0)}  ${clean ? 'OK' : 'CLIPPED'}`);
  if (!clean) {
    problems++;
    const seen = new Set();
    offenders.slice(0, 6).forEach(o => {
      const key = o.tag + o.text;
      if (seen.has(key)) return; seen.add(key);
      const sides = Object.entries(o.over).filter(([, v]) => v > 1.5).map(([k, v]) => `${k}+${v.toFixed(0)}`).join(' ');
      console.log(`   ${o.tag.padEnd(7)} ${sides.padEnd(12)} [${o.box}] ${o.text ? '"' + o.text + '"' : ''}`);
    });
  }
}
console.log(problems ? `\n${problems} figure(s) still clipped` : '\nall figures fit their viewBox');
