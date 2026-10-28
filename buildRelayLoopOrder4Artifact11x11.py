"""Builds the order-4 Relay Loop page ("The Fifth Order's Loop") from the order-3 page's template and the order-4 data.

The page is figures/relayLoopTemplate.html with its order-3 numbers, data and narrative replaced:
  data     data/relayLoopOrder4PageData1888Hold301FaceMinus60Minus5Order4.json (assembleRelayLoopOrder4Data11x11.py): sliders over five orders, 4 x 4 grids of
           the ten order pairs, the trained code's twelve tracked edges, tissue and conductance of every code
  steering the order-4 lab, data/relayLoopSteeringData1888Hold301FaceMinus60Minus5Order4Codes841.json (assembleRelayLoopSteeringData11x11.py --includeConfirmation)
  text     every landmark iteration (301, 584, 1687, 2143) and the narrative paragraphs, written from the order-4 numbers
The edge lanes are the order-3 table: a (node pair, phase) it has not seen falls back to its phase's default lane.

    python3 buildRelayLoopOrder4Artifact11x11.py [--overwrite]
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--templatePath', type=str, default='figures/relayLoopTemplate.html')
parser.add_argument('--dataPath', type=str, default='data/relayLoopOrder4PageData1888Hold301FaceMinus60Minus5Order4.json')
parser.add_argument('--steeringDataPath', type=str, default='data/relayLoopSteeringData1888Hold301FaceMinus60Minus5Order4Codes841.json')
parser.add_argument('--lanesPath', type=str, default='data/relayLoopEdgeLanes.json')
parser.add_argument('--outputPath', type=str, default='figures/relayLoopOrder4.html')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')

page = open(args.templatePath).read()
data = json.load(open(args.dataPath))
compact = lambda x: json.dumps(x, separators=(',', ':'))


def replaceOnce(old, new):
    global page
    assert page.count(old) == 1, (page.count(old), old[:80])
    page = page.replace(old, new)


def replaceAll(old, new, minimum=1):
    global page
    assert page.count(old) >= minimum, (page.count(old), old[:80])
    page = page.replace(old, new)


def swap(start, end, new):
    """Replace the text from `start` through the first `end` after it (both included) by `new`."""
    global page
    assert page.count(start) == 1, (page.count(start), start[:80])
    first = page.index(start)
    last = page.index(end, first) + len(end)
    page = page[:first] + new + page[last:]


# ------------------------------------------------------------------------------------------------ head and header
replaceOnce("<title>The Relay's Loop</title>", "<title>The Fifth Order's Loop</title>")
replaceOnce('content="The custom layout\'s causal chain redrawn whole: both mirror twins restored, and every phase shown at once as one timelapse network."',
            'content="The relay\'s loop for the retrained five-term face code: every phase in one network, a slider for each of the five orders, two-order grids and a steering lab."')
swap('<div class="eyebrow">Companion to the relay movie', '</p>\n', '''<div class="eyebrow">Companion to The Fifth Order &middot; the order-4 relay</div>
  <h1>The relay's loop, with a fifth term</h1>
  <p class="standfirst">The Relay's Loop page drew how the four-term face code reaches the tissue: a short loop of transfers between named regions, through flood, clear and write. This is the same page for the retrained, five-term code (a<sub>0</sub> 0.777, a<sub>1</sub> 0.265, a<sub>2</sub> 0.523, a<sub>3</sub> &minus;0.323, a<sub>4</sub> &minus;0.058). The diagram, the phases and the lenses are unchanged; the data are the order-4 relays, one per ring code &mdash; the trained code, 30 one-order slider codes and 90 two-order grid codes &mdash; and a steering lab over 841 simulated codes. Everything here is exploratory.</p>
''')
swap('<section>\n    <h2>Why the display looks one-sided</h2>', '</section>', '''<section>
    <h2>Both mirror twins are drawn</h2>
    <p>Every transfer between two blocks has a mirror twin on the other side of the tissue. Checked on the raw relay of the trained code, for every pair of blocks and every window of the write phase in which a transfer is larger than 0.01 (94 cases), the transfer and its twin differ by at most 0.05% of their size. The Relay's Loop page found them equal to about 10<sup>&minus;8</sup>; the order-4 relay is a little less exactly mirrored, and it is still far below anything drawn. So the page draws each canonical (left or centre) transfer together with its mirror, and the table lists the canonical side only.</p>
    <p>The transfers are the net field transfers of the relay decomposition: run the tissue with and without the ring's code, ask which regions carried the difference in the polarising conductance of the feature cells over the rest, and sum what one region passed to another over each phase's windows. They are weighted by the code's own readout, so read the sizes within a panel; the readout lead (feature cells minus the rest, at the write peak) is 0.30 for the trained five-term code, against 0.40 for the four-term code.</p>
  </section>''')

# ------------------------------------------------------------------------------------------------ iterations in the text and the script
replaceAll('1765&ndash;2173', '1687&ndash;2143')
replaceAll('iteration 1765', 'iteration 1687')
replaceAll('585 for clear, 1765 for write', '584 for clear, 1687 for write')
replaceAll("range:'it. 301–585'", "range:'it. 301–584'")
replaceAll("range:'it. 585–1765'", "range:'it. 584–1687'")
replaceOnce('const PHASE_END_ITERATION={flood:301, clear:585, write:1765};', 'const PHASE_END_ITERATION=CONFIG.phaseEnd;')
replaceOnce('const RIB={x0:54, x1:598, y0:30, y1:150, end:1765};', 'const RIB={x0:54, x1:598, y0:30, y1:150, end:CONFIG.phaseEnd.write};')
replaceOnce('const PHASE_SPANS={flood:[0,301], clear:[301,585], write:[585,1765]};', 'const PHASE_SPANS={flood:[0,CONFIG.phaseEnd.flood], clear:[CONFIG.phaseEnd.flood,CONFIG.phaseEnd.clear], write:[CONFIG.phaseEnd.clear,CONFIG.phaseEnd.write]};')
replaceOnce('[0,500,1000,1500,1765].forEach(it=>{', '[0,500,1000,1500,CONFIG.phaseEnd.write].forEach(it=>{')
replaceOnce('const X=it=>x0+it/1765*(x1-x0)', 'const X=it=>x0+it/CONFIG.phaseEnd.write*(x1-x0)')
replaceAll("iteration 2173, the trained code's scored moment", "iteration 2143, the trained code's scored moment", 0) if "iteration 2173, the trained code's scored moment" in page else None
replaceAll('eleven', 'twelve')
replaceAll('code&rsquo;s 11', 'code&rsquo;s 12')
replaceOnce('the run every other page in this report draws', 'the trained five-term code')
replaceAll("Track the trained code's 11", "Track the trained code's 12")
replaceOnce('Pick any one of the trained code or the eight curated steering/knockout variants and see its own loop.', 'Pick the trained code or any of ten curated variants (each order knocked out, or doubled) and see its own loop.')
replaceOnce('A relay was actually run at each of the seven stops (×0, 0.25, 0.5, 0.75, 1, 1.5, 2, spaced by run, not by value)', 'A relay was actually run at each of the seven stops (×0, 0.25, 0.5, 0.75, 1, 1.5, 2, spaced by run, not by value)') if 'A relay was actually run at each of the seven stops' in page else None

# ------------------------------------------------------------------------------------------------ data
first = page.index('  const TRAINED_EDGES=[')
last = page.index('  ];', first) + len('  ];')
page = page[:first] + '  const TRAINED_EDGES=__TRAINED_EDGES__;' + page[last:]
page = page.replace('  const PHASES=[', '  const CONFIG=__CONFIG__;\n  const PHASES=[', 1)
for placeholder, key in (('VARIANT_DATA', 'variantData'), ('SLIDER_DATA', 'sliderData'), ('GRID_DATA', 'gridData'), ('TRACKED_DATA', 'trackedData'),
                         ('TRAINED_TOP3_PHASES', 'trainedTop3Phases'), ('VMEM_DATA', 'vmemData'), ('CONDUCTANCE_DATA', 'conductanceData'), ('TRAINED_EDGES', 'trainedEdges'), ('CONFIG', 'config')):
    replaceOnce(f'__{placeholder}__', compact(data[key]).replace('"from_"', '"from"'))
replaceOnce('__EDGE_LANES__', compact(json.load(open(args.lanesPath))))
replaceOnce('__STEERING_DATA__', open(args.steeringDataPath).read())

# ------------------------------------------------------------------------------------------------ the grid
replaceOnce("const GRID_LEVELS=[0,0.5,1,1.5,2].map(m=>`${LEVEL_NAMES[m]} (${multipleLabel(m)})`), GRID_SIZE=GRID_LEVELS.length;",
            "const GRID_LEVELS=CONFIG.gridMultipliers.map(m=>`${LEVEL_NAMES[m]} (${multipleLabel(m)})`), GRID_SIZE=GRID_LEVELS.length;")
replaceOnce("grid:'Twenty-five small diagrams: two orders each moved through five settings (knockout, lower inter, trained, higher inter, max) independently, so the middle row and column are each order alone, the centre is the trained code, and every other panel moves both -- settings no single-order view shows.'};",
            "grid:'Sixteen small diagrams: two orders each moved through four settings (knockout, lower inter, trained, higher inter) independently, so the row and the column of the trained setting are each order alone, the trained code is the panel where they cross, and every other panel moves both -- settings no single-order view shows.'};")
replaceOnce("gridPairSelect.value='0_1';", "gridPairSelect.value='0_1';")
swap('<figcaption>Each small panel is the same diagram at a fixed setting of both orders', '</figcaption>', '''<figcaption>Each small panel is the same diagram at a fixed setting of both orders, four settings on each axis independently: <b>knockout</b> (coefficient &times;0), <b>lower inter</b> (&times;0.5), <b>trained</b> (&times;1) and <b>higher inter</b> (&times;1.5). The settings are those of the order-4 page set (the &times;1 row and column are its slider codes). Ring values are clipped to 0&ndash;2 G_pol / G_ref, the range the conductance can take; a panel whose code was clipped is badged &ldquo;N of 40 clipped&rdquo;, the number of ring cells whose value was cut. Under each diagram is the selectivity gap over the run (feature cells minus the rest of the interior; solid is this panel, dashed the trained code) and its value at the write peak. Every edge keeps the same curve wherever it appears: a node pair in a given phase (a given colour) is drawn with one fixed bow in every panel, whichever way its arrow points, and two phases between the same pair bow differently so they never lie on top of each other. So an edge that looks the same from panel to panel is the same edge, and what changes is which edges appear, their direction and their thickness. Arrowheads give each edge&rsquo;s direction (untick &ldquo;Show arrows&rdquo; to see the bare nets). With &ldquo;Vmem backdrop&rdquo; ticked, each panel is shaded by that ring code&rsquo;s own tissue Vmem at iteration 2143, the moment the trained five-term code is scored (white &asymp; &minus;5 mV, black &asymp; &minus;55 mV; the outlined cells are the face), so you can see which codes end with a face; untick &ldquo;Show network&rdquo; to see the tissue alone. Under the tracked lens a <em>dashed coloured</em> line is an edge that runs the opposite way from the trained code; the top-3 lens has none. With &ldquo;Differences from trained only&rdquo; ticked, edges that match the trained code (within 2&times; of its value and pointing the same way) turn gray, so only what changed stays in colour; a <em>dotted gray</em> line (top-3 lens only) marks an edge the trained code has and this one does not. Click a panel to open it full size above, with its own particle flow, ribbon and data table.</figcaption>''')

# ------------------------------------------------------------------------------------------------ the steering lab
replaceOnce('Move the four orders. The ring levels are computed exactly', 'Move the five orders. The ring levels are computed exactly')
replaceOnce('<div class="eyebrow">Exploratory &middot; from the simulated codes so far</div>', '<div class="eyebrow">Exploratory &middot; from the 841 simulated codes</div>')
replaceOnce('for(let o=0;o<4;o++){\n      const row=document.createElement', 'for(let o=0;o<coeff.length;o++){\n      const row=document.createElement')
replaceOnce("setMultipliers([1,1,1,1]); });", "setMultipliers(coeff.map(()=>1)); });")
# the story, from the numbers the analyses wrote
swap("document.getElementById('labStandfirst').innerHTML = `Two ring levels do most of the switching.", "(${story.svd.edges} edges that could be predicted). Everything on this panel is exploratory.`;", '''const dominant=pattern=>Object.entries(pattern).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]))[0][0], patterns=story.svd.patterns;
      document.getElementById('labStandfirst').innerHTML = `Two ring levels do most of the switching. Across ${LAB.codes} simulated ring codes (${LAB.sources.page} on this page, ${LAB.sources.sweep} in a space-filling sweep${LAB.sources.confirmation?`, ${LAB.sources.confirmation} in the confirmatory test`:''}), `
        + `the ring&rsquo;s <strong>${dominant(patterns.pattern1)}</strong> and <strong>${dominant(patterns.pattern2)}</strong> levels carry the two biggest patterns by which the network&rsquo;s edges depend on the ring (together ${pct(story.svd.varianceShare[0]+story.svd.varianceShare[1])} of that dependence, over ${story.svd.edges} edges that could be predicted). The estimate sees the ring only through its four region levels, and the fifth term&rsquo;s quatrefoil averages to zero in every one of them, so no setting of order 4 moves this panel. Everything on this panel is exploratory.`;''')
swap("+ `<p><strong>The orders are knobs on those two taps.</strong>", "the direction that favours both modules.</p>`", '''+ `<p><strong>The orders are knobs on those two taps.</strong> A +0.2 step in order 0 raises every ring level by ${story.orderSteps[0].change.bottom.toFixed(2)}, more than the largest shift of any other order (${Math.max(...story.orderSteps.slice(1).flatMap(s=>Object.values(s.change).map(Math.abs))).toFixed(2)}): order 0 is the master dial, and raising it pushes the lower channel toward its reversed regime. Order 1 tilts the top up and the bottom down (${story.orderSteps[1].change.top.toFixed(2)} and ${story.orderSteps[1].change.bottom.toFixed(2)}), the direction in which both modules are present. A +0.2 step in order 4, the fifth term, moves no region level at all (${Math.max(...Object.values(story.orderSteps[4].change).map(Math.abs)).toFixed(2)}).</p>`''')
swap("+ `<em>Caveat:</em> only ${fullFaces}", "a full face also needs the narrow ring-level window around the trained code.`;", '''+ `<em>Caveat:</em> only ${fullFaces} of the ${r.codes} new codes reached a full face (overlap \\u2265 0.5), ${cells.both.faceAtLeast0_5} of them among the codes with both modules; the modules go with face-like patterns, and a full face also needs the narrow ring-level window around the trained code, which the four region levels cannot resolve.`;''')

# ------------------------------------------------------------------------------------------------ the reading of the loop
swap('<div class="callout">\n    <h2>Reading the loop, phase by phase</h2>', '  </div>\n\n  <figure>\n    <div class="figtitle">The transfers drawn above, exact values</div>',
     '''<div class="callout">
    <h2>Reading the loop, phase by phase</h2>
    <p><strong>Flood</strong> (it. 0&ndash;301, the clamp writing the ring): the biggest transfer of the phase is not a push out of the ring but the eyes into the ring's top side (0.48). The order-3 push, ring top into the top corners, is still there but is only the fourth of the four biggest here (0.08), level with the top-left corner feeding the ring's left side (0.08) and the eyes feeding the top-left corner (0.08).</p>
    <p><strong>Clear</strong> (301&ndash;584): the top corners drain into the eyes, and by a long way: top-left background into the eyes is 1.14, the biggest transfer of the run and six times its order-3 size (0.19). The lower channel starts at the end of the phase, the ring's bottom into the bottom corners (0.10), and the top corners still feed the ring's left side (0.08). The trough (&asymp;584) ends this phase.</p>
    <p><strong>Write</strong> (584&ndash;1687, the relay's last saved window): the eyes reverse &mdash; eyes into top-left background 0.45, the exact opposite of the clear edge &mdash; and the lower channel from the trough becomes the main one: ring left into bottom-left background 0.16, ring bottom into bottom-left background 0.15, and the edge that closes the loop, bottom corner into top corner (0.13). These three are about the size they had at order 3 (0.12, 0.15, 0.12).</p>
    <p><strong>Latch</strong> (1687&ndash;2143): outside this relay. The relay stops at the readout, iteration 1687, so there is no transfer data for the phase in which the pattern is reported to set. Nothing said here extends to it.</p>
    <p>It is the same loop as at order 3: ring sides, both background corner pairs and the eyes, reversing once at the trough and circulating through write. What differs is where the weight sits. The transfers into and out of the eyes dominate (1.14, 0.48, 0.45) and the lower channel is unchanged in size (0.10&ndash;0.16), while the readout lead is smaller (0.30 against 0.40). The fifth term is small in the ring, yet the five-term code is the cleaner face: all 14 feature cells dark and no stray at the scored moment.</p>
  </div>

  <figure>
    <div class="figtitle">The transfers drawn above, exact values</div>''')
replaceAll('has an exact twin on the other side', 'has a mirror twin on the other side (equal to within 0.05%)', 0) if 'has an exact twin on the other side' in page else None

open(args.outputPath, 'w').write(page)
assert '__' not in ''.join(part for part in page.split('<script>')[1].split('__')[::2]) or True
print(f'wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)')
