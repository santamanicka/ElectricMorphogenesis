"""Render the relay section of the report to a PNG with headless Firefox, so the figures can be looked at.

Every other section is hidden. The movie can be set to a time (the default continuous view) or to a slider position
(with --frames, the exact windows), to the shares view of its chart and to per-frame arrow scaling before the
screenshot, and the page can be forced into the dark theme. The continuous view pre-rolls its particles so the
streaks are already established in a still image.

    python3 renderRelaySection.py figures/boundaryHarmonicSwitchRule.html out.png [--height 6400] [--dark]
                                  [--time 650 | --frames --position 12] [--shares] [--rescale]

The relay section sits well down the page, so use a tall window (the default) and crop the result. Needs firefox
on the PATH; the page's web fonts load only if the machine is online, otherwise fallback fonts are used.
"""
import argparse
import os
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('page', help='the built report, figures/boundaryHarmonicSwitchRule.html')
parser.add_argument('output', help='where to write the PNG')
parser.add_argument('--height', type=int, default=6400, help='window height in pixels (width is 1000)')
parser.add_argument('--dark', action='store_true', help='force the dark theme')
parser.add_argument('--time', type=float, default=None, help='continuous view: the iteration to show, 0 to 1765')
parser.add_argument('--frames', action='store_true', help='switch the movie to frame by frame (exact windows)')
parser.add_argument('--position', type=int, default=None, help='with --frames: slider position, 0 is the initial state, the last is the final state')
parser.add_argument('--shares', action='store_true', help='switch the movie chart to shares of the final gap')
parser.add_argument('--rescale', action='store_true', help='rescale the movie arrows to each frame')
args = parser.parse_args()

html = open(args.page, encoding='utf-8').read()
setup = '<style>section:not(:has(#relayNetworkFigure)){display:none !important} body{margin:0}</style>'
if args.dark:
    setup += '<script>document.documentElement.setAttribute("data-theme","dark")</script>'
html = html.replace('<title>', setup + '<title>', 1)
if args.position is not None or args.time is not None or args.frames or args.shares or args.rescale:
    steps = ''
    if args.frames:
        steps += 'const fr=document.getElementById("movieFrames");fr.checked=true;fr.dispatchEvent(new Event("change"));'
    if args.rescale:
        steps += 'document.getElementById("movieRescale").checked=true;'
    if args.shares:
        steps += 'const sh=document.getElementById("movieShares");sh.checked=true;sh.dispatchEvent(new Event("change"));'
    shown = args.position if args.frames else args.time
    if shown is not None:
        steps += 'const s=document.getElementById("movieSlider");s.value=%s;s.dispatchEvent(new Event("input"));' % shown
    html += '<script>window.addEventListener("load",()=>{%s});</script>' % steps

with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, 'relayOnly.html')
    open(path, 'w', encoding='utf-8').write(html)
    if os.path.exists(args.output):
        os.remove(args.output)
    subprocess.run(['timeout', '150', 'firefox', '--headless', '--no-remote', '--screenshot', args.output,
                    '--window-size=1000,%d' % args.height, 'file://' + path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(args.output, os.path.getsize(args.output) if os.path.exists(args.output) else 'NOT WRITTEN')
