const fs = require('fs');
// usage: node checkRelayLoop.js [page.html]   (default figures/relayLoopOrder4.html, built by buildRelayLoopOrder4Artifact11x11.py)
const html = fs.readFileSync(process.argv[2] || 'figures/relayLoopOrder4.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const registry = {}, created = [];
const makeElement = (tag) => {
  const e = {tag, children: [], style: {}, dataset: {}, textContent: '', innerHTML: '', listeners: {}, attrs: {}, hidden: false,
    setAttribute(k,v){this.attrs[k]=v;}, getAttribute(k){return this.attrs[k];}, removeAttribute(k){delete this.attrs[k];},
    addEventListener(t,f){(this.listeners[t]=this.listeners[t]||[]).push(f);},
    appendChild(c){this.children.push(c); c.parentNode=this; return c;},
    removeChild(c){const i=this.children.indexOf(c); if(i>=0) this.children.splice(i,1); return c;},
    get firstChild(){return this.children[0];},
    querySelectorAll(sel){ if(sel==='.toggle') return created.filter(x=>x.className==='toggle');
      if(sel==='.gridCell') return this.children.filter(x=>x.className==='gridCell');
      if(sel==='path') return this.children.filter(x=>x.tag==='path'); return []; },
    classList:{add(){},remove(){}},
    getBoundingClientRect(){return {width:640,height:660,left:0,top:0};},
    closest(){return this;}};
  return e;
};
global.window = {addEventListener(){}, matchMedia: () => ({matches:false, addEventListener(){}, addListener(){}}),
  getComputedStyle: () => ({getPropertyValue: (n) => ({'--flood':'#B8791C','--clear':'#B84052','--write':'#33548F'}[n]||'')}),
  devicePixelRatio:1, ResizeObserver: undefined};
global.getComputedStyle = global.window.getComputedStyle; global.matchMedia = global.window.matchMedia;
global.requestAnimationFrame = () => 1; global.cancelAnimationFrame = () => {};
global.setInterval = () => 1;
global.document = {
  getElementById: id => (registry[id] = registry[id] || makeElement('byId:'+id)),
  createElementNS: (ns,t) => { const e = makeElement(t); created.push(e); return e; },
  createElement: t => { const e = makeElement(t); created.push(e); return e; },
  documentElement: makeElement('html'),
};
registry['net'] = makeElement('svg'); registry['net'].parentNode = makeElement('div');
registry['net'].getBoundingClientRect = () => ({width:640,height:660,left:0,top:0});
try { new Function(script)(); } catch(e) { console.log('SCRIPT ERROR:', e.message, e.stack); process.exit(1); }
const ok = (name,cond,extra='') => { console.log((cond?'PASS':'FAIL')+'  '+name+(extra?'  ('+extra+')':'')); if(!cond) process.exitCode=1; };
const fire = (el,type) => (el.listeners[type]||[]).forEach(f=>f());
const allDescendants = root => root.children.flatMap(c => [c, ...allDescendants(c)]);
const modeBtn = key => created.find(x => x.className === 'modeBtn' && x.dataset.key === key);
const lens = registry['lens'];
const ribbonPaths = () => allDescendants(registry['ribbon']).filter(c => c.tag === 'path');
try {
  // single codes under both lenses
  const select = registry['variant'];
  ok('variant select has the trained code and ten curated codes', select.children.length === 11, select.children.length);
  for (const lensValue of ['top3', 'tracked']) {
    lens.value = lensValue; fire(lens, 'change');
    fire(modeBtn('single'), 'click');
    for (const opt of select.children) { select.value = opt.value; fire(select, 'change'); if (ribbonPaths().length < 4) throw new Error('ribbon empty for ' + opt.value); }
  }
  ok('every single code draws under both lenses, ribbon included', true);
  // sliders
  fire(modeBtn('slider'), 'click');
  const order = registry['sliderOrder'], t = registry['sliderT'];
  ok('slider has five orders', order.children.length === 5, order.children.length);
  for (const o of order.children) { order.value = o.value; fire(order, 'change'); for (let i = 0; i <= 6; i++) { t.value = String(i); fire(t, 'input'); if (ribbonPaths().length < 4) throw new Error('ribbon empty ' + o.value + ' ' + i); } }
  ok('every slider stop of every order draws', true);
  t.value = '4'; fire(t, 'input');
  ok('slider stop 4 is the trained code', /trained/.test(registry['sliderReadout'].textContent) && registry['sliderNote'].textContent === 'the trained code', registry['sliderReadout'].textContent);
  // grids
  fire(modeBtn('grid'), 'click');
  const pair = registry['gridPair'];
  ok('there are ten order pairs', pair.children.length === 10, pair.children.length);
  for (const opt of pair.children) {
    pair.value = opt.value; fire(pair, 'change');
    const cells = registry['gridWrap'].children.filter(x => x.className === 'gridCell');
    if (cells.length !== 16) throw new Error('grid ' + opt.value + ' has ' + cells.length + ' cells');
    for (const cell of cells) { fire(cell, 'click'); if (ribbonPaths().length < 4) throw new Error('ribbon empty for grid cell ' + opt.value); }
  }
  ok('every grid has 4 x 4 panels and each opens', true);
  ok('grid axis labels name four levels', registry['gridColLabels'].children.length === 4 && /trained/.test(registry['gridColLabels'].children[2].textContent), registry['gridColLabels'].children.map(c => c.textContent).join(' | '));
  // lab
  const sliders = registry['labSliders'].children;
  ok('the steering lab has five order sliders', sliders.length === 5, sliders.length);
  ok('the lab story and standfirst are written', registry['labStory'].innerHTML.length > 400 && /quatrefoil/.test(registry['labStandfirst'].innerHTML), registry['labStandfirst'].innerHTML.slice(0, 160));
  ok('the confirmation paragraph is written', /8 of 9/.test(registry['labConfirmation'].innerHTML), registry['labConfirmation'].innerHTML.slice(0, 120));
  const preset = registry['labPreset'];
  for (const opt of preset.children) { preset.value = opt.value; fire(preset, 'change'); }
  ok('every lab preset draws', true, preset.children.length + ' presets');
} catch (e) { console.log('FAIL  ' + e.message); process.exitCode = 1; }
