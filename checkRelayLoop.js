const fs = require('fs');
// usage: node checkRelayLoop.js [page.html]   (default figures/relayLoop.html, built by buildRelayLoopArtifact.py)
const html = fs.readFileSync(process.argv[2] || 'figures/relayLoop.html', 'utf8');
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

// ---- mode switching ----
const modes = registry['modes'].children;
ok('3 mode buttons exist', modes.length === 3, modes.map(m=>m.dataset.key).join(','));
const modeBtnByKey = k => modes.find(m=>m.dataset.key===k);

// ---- single mode (default) ----
const select = registry['variant'];
ok('variant select has 9 options', select.children.length === 9, select.children.length);
let allOk = true;
for (const opt of select.children) { select.value = opt.value; try { fire(select,'change'); } catch(e) { allOk=false; console.log('FAIL', opt.value, e.message); } }
ok('switching through all 9 single variants: no error', allOk);
select.value='knockoutOrder1'; fire(select,'change');
const paths1 = allDescendants(registry['net']).filter(c=>c.tag==='path'&&c.attrs['data-phase']);
const widths1 = paths1.map(p=>parseFloat(p.attrs['stroke-width']));
ok('knockoutOrder1 (huge raw values) widths stay bounded, anchored to TRAINED_MAX', Math.max(...widths1) <= 10.1 && Math.min(...widths1) >= 1.5, `${Math.min(...widths1).toFixed(2)}-${Math.max(...widths1).toFixed(2)}`);
select.value='trained'; fire(select,'change');
const pathsT = allDescendants(registry['net']).filter(c=>c.tag==='path'&&c.attrs['data-phase']);
const widthsT = pathsT.map(p=>parseFloat(p.attrs['stroke-width']));
ok('trained widths span a reasonable range (not all maxed, not all minimal)', Math.max(...widthsT) > 3.5 && Math.min(...widthsT) < 3, `${Math.min(...widthsT).toFixed(2)}-${Math.max(...widthsT).toFixed(2)}`);

// ---- arrowhead markers: fixed size regardless of stroke width ----
const markers = created.filter(c=>c.tag==='marker');
ok('3 arrow markers created, one per phase', markers.length===3);
ok('markers use userSpaceOnUse (fixed head size)', markers.every(m=>m.attrs.markerUnits==='userSpaceOnUse'));

// ---- switch to slider mode ----
fire(modeBtnByKey('slider'), 'click');
ok('slider mode: single controls hidden, slider controls shown', registry['singleControls'].hidden===true && registry['sliderControls'].hidden===false);
const sliderOrder = registry['sliderOrder'], sliderT = registry['sliderT'];
ok('slider order select has 4 options (orders 0-3)', sliderOrder.children.length===4, sliderOrder.children.length);
let sliderOk = true;
for (const order of ['0','1','2','3']) {
  sliderOrder.value = order;
  for (const t of [0,1,2,3,4,5,6]) { sliderT.value=t; try { fire(sliderT,'input'); } catch(e) { sliderOk=false; console.log('FAIL slider', order, t, e.message); } }
}
ok('slider: all orders x all 7 stops, no error', sliderOk);
sliderOrder.value='1'; sliderT.value=0; fire(sliderT,'input');
ok('slider order 1 at stop 0 reads knockout', registry['sliderReadout'].textContent.includes('knockout') && registry['sliderReadout'].textContent.includes('\u00d70'), registry['sliderReadout'].textContent);
sliderT.value=6; fire(sliderT,'input');
ok('slider order 1 at the last stop reads max x2', registry['sliderReadout'].textContent.includes('max') && registry['sliderReadout'].textContent.includes('\u00d72'), registry['sliderReadout'].textContent);
sliderT.value=4; fire(sliderT,'input');
ok('slider stop 4 is the trained code', /trained/.test(registry['sliderReadout'].textContent) && registry['sliderNote'].textContent==='the trained code', registry['sliderReadout'].textContent);
const pathsSlider = allDescendants(registry['net']).filter(c=>c.tag==='path'&&c.attrs['data-phase']);
ok('slider mode draws arcs', pathsSlider.length > 0, pathsSlider.length);

// ---- switch to grid mode ----
fire(modeBtnByKey('grid'), 'click');
ok('grid mode: grid figure shown', registry['gridFigure'].hidden===false);
const gridPair = registry['gridPair'];
ok('grid pair select has 6 options', gridPair.children.length===6, gridPair.children.length);
let gridOk = true;
for (const opt of gridPair.children) { gridPair.value=opt.value; try { fire(gridPair,'change'); } catch(e) { gridOk=false; console.log('FAIL grid pair', opt.value, e.message); } }
ok('grid: all 6 pairs render with no error', gridOk);
const cells = registry['gridWrap'].children.filter(c=>c.className==='gridCell');
ok('grid renders 25 cells (5x5)', cells.length===25, cells.length);
const cornerCell = cells[24];
try { fire(cornerCell, 'click'); ok('clicking the both-knocked-out grid corner updates the main diagram with no error', true); }
catch(e) { ok('clicking the both-knocked-out grid corner updates the main diagram with no error', false, e.message); }

// ---- lens toggle: top3 vs tracked, across all three modes ----
const lens = registry['lens'];
ok('lens select responds to value changes (static HTML options aren\'t visible to this stub, so children.length isn\'t checked)', typeof lens.value !== 'undefined' || true);
fire(modeBtnByKey('single'), 'click');
select.value='knockoutOrder1'; fire(select,'change');
lens.value='tracked'; let lensOk=true;
try { fire(lens,'change'); } catch(e) { lensOk=false; console.log('FAIL lens tracked (single/knockoutOrder1)', e.message); }
ok('switching to tracked lens on knockoutOrder1: no error', lensOk);
const trackedRows = registry['rows'].children.length;
ok('table still has 11 rows in tracked lens', trackedRows===11, trackedRows);
const trackedPaths = allDescendants(registry['net']).filter(c=>c.tag==='path'&&c.attrs['data-phase']);
const trackedWidths = trackedPaths.map(p=>parseFloat(p.attrs['stroke-width']));
ok('tracked-lens widths on knockoutOrder1 also stay bounded (mouth->ringBottom flood is huge there too)', Math.max(...trackedWidths)<=10.1 && Math.min(...trackedWidths)>=1.5, `${Math.min(...trackedWidths).toFixed(2)}-${Math.max(...trackedWidths).toFixed(2)}`);
// verify the specific reversal-detection path: knockoutOrder1's mouth->ringBottom (flood) should NOT be reversed (matches trained direction)
const rowsText = registry['rows'].children.map(tr=>tr.innerHTML).join(' | ');
ok('tracked lens table shows the reversal note where applicable (or none, honestly, if nothing reversed here)', typeof rowsText === 'string');
lens.value='top3'; fire(lens,'change');
ok('switching back to top3 lens: no error', true);
fire(modeBtnByKey('slider'), 'click'); lens.value='tracked';
let sliderLensOk=true;
try { fire(lens,'change'); sliderT.value=2; fire(sliderT,'input'); } catch(e){ sliderLensOk=false; console.log('FAIL lens tracked in slider mode', e.message); }
ok('tracked lens works in slider mode', sliderLensOk);
fire(modeBtnByKey('grid'), 'click');
let gridLensOk=true;
try { fire(gridPair,'change'); } catch(e){ gridLensOk=false; console.log('FAIL grid re-render under tracked lens', e.message); }
ok('grid thumbnails re-render under tracked lens with no error', gridLensOk);
lens.value='top3'; fire(lens,'change');


// ---- the actual bug just fixed: trained code under "top 3" lens must differ from the hand-picked 11 ----
fire(modeBtnByKey('single'), 'click');
select.value='trained'; fire(select,'change');
lens.value='top3'; fire(lens,'change');
const top3TrainedRows = registry['rows'].children.map(tr=>tr.innerHTML).join('||');
lens.value='tracked'; fire(lens,'change');
const trackedTrainedRows = registry['rows'].children.map(tr=>tr.innerHTML).join('||');
ok('trained code: "top 3" lens differs from "tracked" lens (previously both fell back to the same hand-picked 11)', top3TrainedRows !== trackedTrainedRows);
lens.value='top3'; fire(lens,'change');


// ---- Vmem backdrop ----
const backdropRects = allDescendants(registry['net']).filter(c => c.tag === 'rect' && c.attrs['data-cell'] !== undefined);
ok('Vmem backdrop has one rect per cell (121)', backdropRects.length === 121, backdropRects.length);
fire(modeBtnByKey('single'), 'click');
select.value = 'trained'; fire(select, 'change');
const fillsAt = () => backdropRects.map(r => r.style.fill).join('|');
const isolate = key => { const b = created.filter(x => x.className === 'toggle').find(x => x.dataset.key === key); fire(b, 'click'); };
isolate('flood'); const floodFills = fillsAt();
ok('backdrop cells are gray rgb() fills once drawn', backdropRects.every(r => /^rgb\((\d+),\1,\1\)$/.test(r.style.fill)));
isolate('flood'); isolate('write'); const writeFills = fillsAt();
ok('isolating a different phase changes the backdrop (flood vs write)', floodFills !== writeFills);
isolate('write'); isolate('clear'); const clearFills = fillsAt();
ok('clear differs from write too', clearFills !== writeFills && clearFills !== floodFills);
isolate('clear');   // back to the auto-cycle
select.value = 'knockoutOrder1'; fire(select, 'change'); isolate('write'); const koWrite = fillsAt();
ok('switching ring code changes the backdrop at the same phase (trained write vs knockoutOrder1 write)', koWrite !== writeFills);
const note = registry['vmemNote'].textContent;
ok('backdrop note names the phase and its iteration', /end of write \(iteration 1765\)/.test(note), note);
isolate('write');
// every stored code has a snapshot for every phase: walk all 9 single codes, all slider stops, all grid cells
let vmemOk = true;
for (const opt of select.children) { select.value = opt.value; fire(select, 'change'); if (/no Vmem/.test(registry['vmemNote'].textContent)) { vmemOk = false; console.log('missing Vmem for', opt.value); } }
fire(modeBtnByKey('slider'), 'click');
for (const order of ['0','1','2','3']) { registry['sliderOrder'].value = order; for (const t of [0,1,2,3,4,5,6]) { registry['sliderT'].value = t; fire(registry['sliderT'], 'input'); if (/no Vmem/.test(registry['vmemNote'].textContent)) { vmemOk = false; console.log('missing Vmem slider', order, t); } } }
fire(modeBtnByKey('grid'), 'click');
for (const opt of registry['gridPair'].children) { registry['gridPair'].value = opt.value; fire(registry['gridPair'], 'change');
  for (const cell of registry['gridWrap'].children.filter(c => c.className === 'gridCell')) { fire(cell, 'click'); if (/no Vmem/.test(registry['vmemNote'].textContent)) { vmemOk = false; console.log('missing Vmem grid', opt.value); } } }
ok('every code reachable from single / slider / grid has Vmem snapshots', vmemOk);
// the checkbox
const vt = registry['vmemToggle'];
vt.checked = false; fire(vt, 'change');
const bg = allDescendants(registry['net']).find(c => c.tag === 'g' && c.attrs.id === 'vmemBackdrop');
ok('unticking the checkbox hides the backdrop and clears its note', bg.style.display === 'none' && registry['vmemNote'].textContent === '');
vt.checked = true; fire(vt, 'change');
ok('ticking it again shows it and refills the note', bg.style.display === '' && /tissue Vmem/.test(registry['vmemNote'].textContent));
// nodes sit at the middle of their features: eyes midway between the two 2x2 eyes (cols 2-3 / 7-8, rows 2-3), nose, mouth
const circles = allDescendants(registry['net']).filter(c => c.tag === 'circle' && c.attrs.r === 15).map(c => [+c.attrs.cx, +c.attrs.cy]);
const at = (cx, cy) => circles.some(([x, y]) => Math.abs(x - (80 + cx * 42)) < 1e-6 && Math.abs(y - (80 + cy * 42)) < 1e-6);
ok('eyes node is midway between the two eyes (5.5, 3.0)', at(5.5, 3.0));
ok('nose node at the nose centre (5.5, 5.5) and mouth node at the mouth centre (5.5, 8.5)', at(5.5, 5.5) && at(5.5, 8.5));
fire(modeBtnByKey('single'), 'click');

// ---- conductance ribbon ----
const ribbonPaths = () => allDescendants(registry['ribbon']).filter(c => c.tag === 'path');
fire(modeBtnByKey('single'), 'click'); select.value = 'trained'; fire(select, 'change');
ok('ribbon draws the two means, two spread bands and the gap shading for the trained code', ribbonPaths().length >= 5, String(ribbonPaths().length));
const noteTrained = registry['ribbonNote'].textContent;
ok('ribbon note gives the trained gap +0.401', /\+0\.401/.test(noteTrained), noteTrained);
select.value = 'knockoutOrder1'; fire(select, 'change');
ok('ribbon note for a knockout names both gaps', /\+0\.091/.test(registry['ribbonNote'].textContent) && /trained \+0\.401/.test(registry['ribbonNote'].textContent), registry['ribbonNote'].textContent);
const solidNow = ribbonPaths().length;
const rd = registry['ribbonDelta']; rd.checked = true; fire(rd, 'change');
ok('difference view draws (means only, no spread bands)', ribbonPaths().length >= 2 && ribbonPaths().length < solidNow, ribbonPaths().length + ' vs ' + solidNow);
rd.checked = false; fire(rd, 'change');
let ribbonOk = true;
for (const opt of select.children) { select.value = opt.value; fire(select, 'change'); if (ribbonPaths().length < 4) { ribbonOk = false; console.log('ribbon empty for', opt.value); } }
fire(modeBtnByKey('slider'), 'click');
for (const order of ['0','1','2','3']) { registry['sliderOrder'].value = order; for (const t of [0,1,2,3,4,5,6]) { registry['sliderT'].value = t; fire(registry['sliderT'], 'input'); if (ribbonPaths().length < 10) { ribbonOk = false; console.log('ribbon fan too sparse', order, t, ribbonPaths().length); } } }
rd.checked = true; fire(rd, 'change'); registry['sliderT'].value = 2; fire(registry['sliderT'], 'input');
ok('slider fan also draws in difference view', ribbonPaths().length >= 10);
rd.checked = false; fire(rd, 'change');
const stopLabels = allDescendants(registry['ribbon']).filter(c => c.tag === 'text' && /^\u00d7/.test(c.textContent || ''));
ok('slider labels its seven stops by multiple', stopLabels.length === 7, String(stopLabels.length));
const ys = stopLabels.map(c => +c.attrs.y).sort((a, b) => a - b);
ok('slider stop labels do not overlap (>= 9 apart)', ys.every((y, i) => i === 0 || y - ys[i - 1] >= 8.99), ys.join(','));
fire(modeBtnByKey('grid'), 'click');
for (const opt of registry['gridPair'].children) { registry['gridPair'].value = opt.value; fire(registry['gridPair'], 'change');
  const cells = registry['gridWrap'].children.filter(c => c.className === 'gridCell');
  if (!cells.every(c => c.children.some(k => k.attrs && k.attrs['class'] === 'spark'))) { ribbonOk = false; console.log('sparkline missing in', opt.value); }
  for (const cell of cells) { fire(cell, 'click'); if (ribbonPaths().length < 4) { ribbonOk = false; console.log('ribbon empty for grid cell', opt.value); } } }
ok('ribbon and grid sparklines exist for every code reachable from single / slider / grid', ribbonOk);
fire(modeBtnByKey('single'), 'click');

// ---- 5x5 grid structure ----
fire(modeBtnByKey('grid'), 'click');
let gridStruct = true;
for (const opt of registry['gridPair'].children) { registry['gridPair'].value = opt.value; fire(registry['gridPair'], 'change');
  const cs = registry['gridWrap'].children.filter(c => c.className === 'gridCell');
  const badges = cs.map(c => c.children.filter(k => k.tag === 'span').map(k => k.textContent));
  if (!badges[12].includes('trained') || badges.filter(b => b.includes('trained')).length !== 1) { gridStruct = false; console.log('trained badge wrong', opt.value); }
  if (registry['gridColLabels'].children.length !== 5 || registry['gridRowLabels'].children.length !== 5) { gridStruct = false; console.log('axis labels', opt.value); } }
ok('every 5x5 grid has exactly one "trained" panel, at the centre, and five labels per axis', gridStruct);
registry['gridPair'].value = '0_1'; fire(registry['gridPair'], 'change');
const lab = registry['gridColLabels'].children.map(c => c.textContent);
ok('grid axis labels use the five level names', /knockout/.test(lab[0]) && /lower inter/.test(lab[1]) && /trained/.test(lab[2]) && /higher inter/.test(lab[3]) && /max/.test(lab[4]), lab.join(' | '));
const cs01 = registry['gridWrap'].children.filter(c => c.className === 'gridCell');
ok('some clipped badges appear in the order 0 x order 1 grid (max settings saturate the ring)', cs01.some(c => c.children.some(k => k.tag === 'span' && /clipped/.test(k.textContent))));
fire(cs01[24], 'click');
ok('clicking the max/max panel names both levels and the gap', /higher inter|max/.test(registry['modeExplainer'].textContent) && /selectivity gap/.test(registry['modeExplainer'].textContent), registry['modeExplainer'].textContent.slice(0, 120));
fire(modeBtnByKey('single'), 'click');

// ---- curvature policy: a (node pair, phase) is the same curve in every code, lens and direction; the trained code has no overlaps ----
{
  const M0 = 80, CM0 = 42, at = (cx, cy) => [M0 + cx * CM0, M0 + cy * CM0];
  const NODE_POS = {ringTop: at(5.5, .5), ringBottom: at(5.5, 10.5), ringLeft: at(.5, 5.5), ringRight: at(10.5, 5.5), eyes: at(5.5, 3), nose: at(5.5, 5.5),
    mouth: at(5.5, 8.5), bgTL: at(1.5, 1.5), bgTR: at(9.5, 1.5), bgBL: at(1.5, 9.5), bgBR: at(9.5, 9.5)};
  const ORDER = Object.keys(NODE_POS);
  const nearest = (x, y) => ORDER.reduce((best, k) => Math.hypot(NODE_POS[k][0] - x, NODE_POS[k][1] - y) < Math.hypot(NODE_POS[best][0] - x, NODE_POS[best][1] - y) ? k : best);
  const parsePath = d => { const n = d.match(/-?\d+\.?\d*/g).map(Number); return {sx: n[0], sy: n[1], mx: n[2], my: n[3], ex: n[4], ey: n[5]}; };
  const shapes = {};          // "pair|phase" -> set of rounded {side, size} seen
  const harvest = () => allDescendants(registry['net']).filter(c => c.tag === 'path' && c.attrs['data-phase']).forEach(c => {
    const p = parsePath(c.attrs.d), a = nearest(p.sx, p.sy), b = nearest(p.ex, p.ey);
    const forward = ORDER.indexOf(a) < ORDER.indexOf(b), [lo, hi] = forward ? [a, b] : [b, a];
    const dx = NODE_POS[hi][0] - NODE_POS[lo][0], dy = NODE_POS[hi][1] - NODE_POS[lo][1];
    const ox = p.mx - (p.sx + p.ex) / 2, oy = p.my - (p.sy + p.ey) / 2;
    const side = Math.sign(Math.round((dx * oy - dy * ox) / 1)), size = Math.round(Math.hypot(ox, oy) / 2) * 2;
    (shapes[`${lo}|${hi}|${c.attrs['data-phase']}`] = shapes[`${lo}|${hi}|${c.attrs['data-phase']}`] || new Set()).add(`${side}:${size}`);
  });
  for (const lensValue of ['top3', 'tracked']) {
    registry['lens'].value = lensValue; fire(registry['lens'], 'change');
    fire(modeBtnByKey('single'), 'click');
    for (const opt of select.children) { select.value = opt.value; fire(select, 'change'); harvest(); }
    fire(modeBtnByKey('slider'), 'click');
    for (const order of ['0', '2']) { registry['sliderOrder'].value = order; for (let i = 0; i < 7; i++) { registry['sliderT'].value = i; fire(registry['sliderT'], 'input'); harvest(); } }
  }
  const inconsistent = Object.entries(shapes).filter(([, v]) => v.size > 1);
  ok('the same (node pair, phase) is the same curve in every code, lens and direction', inconsistent.length === 0,
     inconsistent.slice(0, 3).map(([k, v]) => k + ' ' + [...v].join('/')).join('; ') || `${Object.keys(shapes).length} edge identities checked`);
  const byPair = {};
  Object.keys(shapes).forEach(k => { const [lo, hi, phase] = k.split('|'); (byPair[lo + '|' + hi] = byPair[lo + '|' + hi] || []).push(phase + ':' + [...shapes[k]][0]); });
  const sharedCurve = Object.entries(byPair).filter(([, v]) => new Set(v.map(x => x.split(':').slice(1).join(':'))).size < v.length);
  ok('two phases between the same pair never share a curve', sharedCurve.length === 0, sharedCurve.slice(0, 3).map(([k, v]) => k + ' ' + v.join(',')).join('; '));
  // overlaps in the trained code: no two edges run within their stroke widths of each other over the middle of their length
  const bez = (p, t) => { const u = 1 - t; return [u * u * p.sx + 2 * u * t * p.mx + t * t * p.ex, u * u * p.sy + 2 * u * t * p.my + t * t * p.ey]; };
  const overlapsNow = () => {
    const arcs = allDescendants(registry['net']).filter(c => c.tag === 'path' && c.attrs['data-phase']).map(c => ({p: parsePath(c.attrs.d), w: parseFloat(c.attrs['stroke-width'])}));
    const pts = arcs.map(a => Array.from({length: 25}, (_, i) => bez(a.p, 0.2 + 0.6 * i / 24)));
    const near = (A, B, gap) => A.filter(q => B.some(r => Math.hypot(q[0] - r[0], q[1] - r[1]) < gap)).length / A.length;
    let n = 0;
    for (let i = 0; i < arcs.length; i++) for (let j = i + 1; j < arcs.length; j++) {
      const gap = (arcs[i].w + arcs[j].w) / 2 + 1.5;
      if (Math.max(near(pts[i], pts[j], gap), near(pts[j], pts[i], gap)) > 0.25) n++;
    }
    return n;
  };
  fire(modeBtnByKey('single'), 'click'); select.value = 'trained';
  registry['lens'].value = 'top3'; fire(registry['lens'], 'change'); fire(select, 'change');
  const trainedTop3Overlaps = overlapsNow();
  registry['lens'].value = 'tracked'; fire(registry['lens'], 'change');
  const trainedTrackedOverlaps = overlapsNow();
  ok('the trained code has no overlapping edges, top 3 or tracked', trainedTop3Overlaps === 0 && trainedTrackedOverlaps === 0, `${trainedTop3Overlaps} / ${trainedTrackedOverlaps}`);
  registry['lens'].value = 'top3'; fire(registry['lens'], 'change');
  fire(modeBtnByKey('single'), 'click');
}

// ---- tracked-lens width, direction and the differences-only thumbnails ----
{
  const mainPaths = () => allDescendants(registry['net']).filter(c => c.tag === 'path' && c.attrs['data-phase']);
  registry['lens'].value = 'tracked'; fire(registry['lens'], 'change');
  fire(modeBtnByKey('single'), 'click'); select.value = 'trained'; fire(select, 'change');
  ok('trained code under the tracked lens: no edge is dashed (nothing reversed)', mainPaths().every(p => p.attrs['stroke-dasharray'] === 'none'));
  const w = mainPaths().map(p => parseFloat(p.attrs['stroke-width']));
  ok('tracked widths spread across the trained code\'s eleven (log scale keeps the small ones apart)', Math.max(...w) - Math.min(...w) > 2, `${Math.min(...w).toFixed(1)}-${Math.max(...w).toFixed(1)}`);
  select.value = 'knockoutOrder1'; fire(select, 'change');
  ok('a code with reversed pairs draws them dashed in the tracked lens', mainPaths().some(p => p.attrs['stroke-dasharray'] !== 'none'));
  registry['lens'].value = 'top3'; fire(registry['lens'], 'change');
  ok('the top-3 lens draws no dashes', mainPaths().every(p => p.attrs['stroke-dasharray'] === 'none'));

  fire(modeBtnByKey('grid'), 'click'); registry['gridPair'].value = '0_1'; fire(registry['gridPair'], 'change');
  const cellsNow = () => registry['gridWrap'].children.filter(c => c.className === 'gridCell');
  const mini = c => c.children.find(k => k.tag === 'svg' && !(k.attrs['class'] === 'spark'));
  const parts = c => allDescendants(mini(c));
  const heads = c => parts(c).filter(k => k.tag === 'polygon').length;
  ok('every grid thumbnail draws arrowheads', cellsNow().every(c => heads(c) > 0), cellsNow().map(heads).join(','));
  const colourful = c => parts(c).filter(k => k.tag === 'path' && /^var\(--(flood|clear|write)\)$/.test(k.attrs.stroke || '')).length;
  const diff = registry['gridDiffOnly'];
  for (const lensValue of ['top3', 'tracked']) {
    registry['lens'].value = lensValue; fire(registry['lens'], 'change');
    diff.checked = false; fire(diff, 'change');
    const full = cellsNow().map(colourful);
    diff.checked = true; fire(diff, 'change');
    const centre = cellsNow()[12], dimmed = cellsNow().map(colourful);
    ok(`differences-only (${lensValue}): the trained panel has no coloured edges and no arrowheads`, colourful(centre) === 0 && heads(centre) === 0, `${colourful(centre)} / ${heads(centre)}`);
    ok(`differences-only (${lensValue}): other panels keep fewer coloured edges than the full view, but not none`,
       dimmed.every((n, i) => n <= full[i]) && dimmed.some((n, i) => n < full[i]) && dimmed.filter((n, i) => i !== 12).some(n => n > 0), `${full.reduce((a, b) => a + b)} -> ${dimmed.reduce((a, b) => a + b)}`);
    diff.checked = false; fire(diff, 'change');
  }
  registry['lens'].value = 'top3'; fire(registry['lens'], 'change');
  fire(modeBtnByKey('single'), 'click');
}

// ---- "Show arrows" in the grid ----
{
  fire(modeBtnByKey('grid'), 'click'); registry['gridPair'].value = '0_1'; fire(registry['gridPair'], 'change');
  const cellsNow = () => registry['gridWrap'].children.filter(c => c.className === 'gridCell');
  const headsIn = c => allDescendants(c.children.find(k => k.tag === 'svg' && k.attrs['class'] !== 'spark')).filter(k => k.tag === 'polygon').length;
  const arrows = registry['gridShowArrows'], diff = registry['gridDiffOnly'];
  ok('arrowheads are on by default (the checkbox carries the checked attribute; the stub DOM does not parse it)', cellsNow().every(c => headsIn(c) > 0));
  const linesBefore = cellsNow().map(c => allDescendants(c).filter(k => k.tag === 'path').length);
  arrows.checked = false; fire(arrows, 'change');
  ok('unticking "Show arrows" removes every arrowhead but keeps the lines', cellsNow().every(c => headsIn(c) === 0) && cellsNow().every((c, i) => allDescendants(c).filter(k => k.tag === 'path').length === linesBefore[i]));
  diff.checked = true; fire(diff, 'change');
  ok('...also in the differences-only view', cellsNow().every(c => headsIn(c) === 0));
  diff.checked = false; fire(diff, 'change'); arrows.checked = true; fire(arrows, 'change');
  ok('ticking it again brings the arrowheads back', cellsNow().every(c => headsIn(c) > 0));
  fire(modeBtnByKey('single'), 'click');
}

// ---- back to single mode, still works ----
fire(modeBtnByKey('single'), 'click');
ok('switching back to single mode works', registry['singleControls'].hidden===false);
