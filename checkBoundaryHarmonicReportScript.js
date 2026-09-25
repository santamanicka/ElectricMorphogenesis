// Runs the built report's drawing code against a stub DOM, so a figure that throws is caught here
// rather than appearing as a blank panel on the page. One thrown error stops every figure after it.
//   node checkBoundaryHarmonicReportScript.js figures/boundaryHarmonicTraining.html
const fs = require('fs');

const page = fs.readFileSync(process.argv[2] || 'figures/boundaryHarmonicTraining.html', 'utf8');
const script = page.match(/<script>([\s\S]*?)<\/script>/g).pop().replace(/^<script>|<\/script>$/g, '');

const makeElement = () => {
  const element = {
    setAttribute() {}, getAttribute: () => '0 0 900 300', appendChild() {}, insertBefore() {}, removeChild() {},
    replaceChildren() {}, addEventListener() {}, querySelectorAll: () => [], querySelector: () => makeElement(),
    getBoundingClientRect: () => ({width: 900, height: 300, top: 0, left: 0}),
    textContent: '', innerHTML: '', style: {}, dataset: {}, children: [], childNodes: [], firstChild: null,
    classList: {add() {}, remove() {}, toggle() {}},
  };
  element.parentNode = {setAttribute() {}, appendChild() {}, insertBefore() {}};
  return element;
};
global.window = {addEventListener() {}, matchMedia: () => ({matches: false, addEventListener() {}}),
                 getComputedStyle: () => ({getPropertyValue: () => '#000'}), devicePixelRatio: 1, innerWidth: 1000};
global.document = {getElementById: () => makeElement(), createElementNS: () => makeElement(),
                   createElement: () => makeElement(), querySelectorAll: () => [], querySelector: () => makeElement(),
                   body: makeElement(), documentElement: makeElement(), addEventListener() {}};
global.requestAnimationFrame = () => 0;                 // animation loops are not run here; the figures they sit in still draw

try {
  eval(script);
  console.log('every figure ran without error');
} catch (error) {
  const where = (error.stack || '').split('\n')[1] || '';
  console.error(`a figure threw: ${error.message}\n${where}`);
  process.exit(1);
}
