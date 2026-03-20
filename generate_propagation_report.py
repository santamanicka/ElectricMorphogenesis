"""
Generate PDF report: "Electric Talk Between Embryos"
Synthesizes two-embryo poke experiment observations and their mechanistic interpretation.
"""

import textwrap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

OUTPUT = "data/embryo_propagation_report.pdf"

# ── colour palette ──────────────────────────────────────────────────────────
C_HEAD  = "#2C3E50"
C_BODY  = "#2C2C2C"
C_ACC1  = "#2980B9"
C_ACC2  = "#E74C3C"
C_ACC3  = "#27AE60"
C_LIGHT = "#ECF0F1"
C_MID   = "#BDC3C7"

# Page dimensions (inches): 8.5 × 11
PW, PH = 8.5, 11.0

def new_page():
    fig = plt.figure(figsize=(PW, PH))
    fig.patch.set_facecolor('white')
    return fig

def rule(fig, y, x0=0.08, x1=0.92, lw=0.6, color=C_MID):
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axhline(y=y, xmin=x0, xmax=x1, linewidth=lw, color=color)
    ax.axis('off')

def heading(fig, y, text, size=13, color=C_HEAD, x=0.10, ha='left'):
    fig.text(x, y, text, ha=ha, va='top', fontsize=size,
             color=color, fontweight='bold')

def footer(fig, page_num):
    fig.text(0.50, 0.025, f"— {page_num} —", ha='center',
             fontsize=8, color=C_MID)
    fig.text(0.10, 0.025,
             "Inter-Embryo Propagation Analysis  ·  Confidential Draft",
             ha='left', fontsize=7.5, color=C_MID)

# ── reliable body-text renderer using textwrap ───────────────────────────────
def body(fig, y, text, size=9.5, color=C_BODY, x=0.10, chars=88,
         indent_cont="     "):
    """
    Render body text with proper line wrapping.
    Each \n starts a new paragraph (blank \n adds vertical gap).
    Bullet lines (starting with spaces + bullet) preserve indentation.
    Returns the y coordinate just below the last line rendered.
    """
    # Line height in normalised figure units
    lh = (size / 72.0) / PH * 1.55   # pt → inches → normalised, ×linespacing

    cur = y
    for para in text.split('\n'):
        if not para.strip():
            cur -= lh * 0.55          # blank-line gap (slightly less than full line)
            continue
        # detect bullet / numbered prefix so continuation lines align
        stripped = para.lstrip()
        prefix_len = len(para) - len(stripped)
        if stripped.startswith(('•', '-', '1.', '2.', '3.', '4.',
                                 '[No]', '[Yes]')):
            subsequent = ' ' * (prefix_len + 4)
        else:
            subsequent = ' ' * prefix_len

        lines = textwrap.wrap(para, width=chars,
                              subsequent_indent=subsequent) or ['']
        for line in lines:
            fig.text(x, cur, line, ha='left', va='top',
                     fontsize=size, color=color)
            cur -= lh

    return cur   # caller can use this to position next element

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1  –  Title
# ════════════════════════════════════════════════════════════════════════════
def page_title(pdf):
    fig = new_page()

    # Top decorative bar
    ax = fig.add_axes([0.08, 0.89, 0.84, 0.016])
    ax.set_facecolor(C_ACC1); ax.axis('off')

    fig.text(0.50, 0.855,
             "Electric Talk Between Embryos",
             ha='center', va='top', fontsize=22, fontweight='bold',
             color=C_HEAD)
    fig.text(0.50, 0.795,
             "Interpreting Two-Embryo Poke Experiments:\n"
             "Propagation Speeds, Mechanisms, and the Ca\u00b2\u207a Wave Hypothesis",
             ha='center', va='top', fontsize=13, color=C_ACC1,
             linespacing=1.6)

    rule(fig, 0.745)

    fig.text(0.50, 0.715,
             "A mechanistic synthesis for the experimental biologist",
             ha='center', va='top', fontsize=10.5, color=C_BODY,
             style='italic')

    # Summary box  [left, bottom, width, height]
    ax2 = fig.add_axes([0.11, 0.35, 0.78, 0.31])
    ax2.set_facecolor(C_LIGHT); ax2.axis('off')
    ax2.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                  boxstyle="round,pad=0.015",
                                  fc=C_LIGHT, ec=C_MID, lw=1.2,
                                  transform=ax2.transAxes))

    fig.text(0.50, 0.645, "Summary", ha='center', va='top',
             fontsize=11.5, fontweight='bold', color=C_HEAD)

    # Summary text — manually wrapped to fit the box (chars ≈ 72 for box width)
    summary_lines = [
        "When one embryo is poked, the neighboring embryo responds —",
        "sometimes within seconds, sometimes only after minutes.",
        "",
        "Three distinct types of response have been observed, spanning",
        "speed ranges from ~10 to more than 500 \u03bcm/s. This document",
        "shows that all responses are consistent with a single biological",
        "mechanism: a Ca\u00b2\u207a wave propagating through embryonic tissue.",
        "",
        "The apparent speed differences arise from the geometry of the",
        "stimulus and the anisotropic architecture of the embryo — not",
        "from multiple independent signaling pathways.",
    ]

    lh = (9.5 / 72.0) / PH * 1.6
    cy = 0.611
    for line in summary_lines:
        if not line:
            cy -= lh * 0.5
        else:
            fig.text(0.50, cy, line, ha='center', va='top',
                     fontsize=9.5, color=C_BODY)
            cy -= lh

    # Bottom decorative bar
    ax3 = fig.add_axes([0.08, 0.095, 0.84, 0.016])
    ax3.set_facecolor(C_ACC1); ax3.axis('off')

    footer(fig, 1)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2  –  The Experiment & Observations
# ════════════════════════════════════════════════════════════════════════════
def page_observations(pdf):
    fig = new_page()

    heading(fig, 0.955, "1.  The Experiment")
    rule(fig, 0.928)

    y = body(fig, 0.910,
        "A glass needle is used to mechanically poke one embryo (the sender) "
        "while a second embryo (the neighbor) rests nearby in shared medium. "
        "Ca\u00b2\u207a imaging (fluorescent reporter) records the response in both embryos. "
        "Because the recording begins 0\u20135 seconds after the poke \u2014 not "
        "simultaneously with it \u2014 any response visible at the very start of the "
        "movie had already completed its travel by the time the camera was rolling. "
        "This timing offset matters for calculating propagation speeds, as we show below.")

    heading(fig, y - 0.015, "2.  The Three Observations")
    rule(fig, y - 0.042)

    y2 = body(fig, y - 0.060,
        "Three qualitatively different responses were recorded. "
        "The raw measurements are shown in the table below.")

    # ── data table ──────────────────────────────────────────────────────────
    tbl_top = y2 - 0.015
    ax = fig.add_axes([0.06, 0.37, 0.88, tbl_top - 0.37])
    ax.axis('off')

    col_labels = ["Observation", "Distance\n(\u03bcm)", "Time to\nresponse",
                  "Calculated\nspeed (\u03bcm/s)", "Notes"]
    col_w = [0.29, 0.12, 0.20, 0.18, 0.21]
    rows = [
        ["Tail of poked embryo\nlights up",
         "2\u202f439",
         "\u2264 5 s\n(done before\nrecording start)",
         "\u2265 488",
         "Response complete\nbefore camera rolling"],
        ["Tail of neighbor\nlights up",
         "3\u202f174",
         "\u2264 5 s\n(done before\nrecording start)",
         "\u2265 635",
         "Response complete\nbefore camera rolling"],
        ["Mid of neighbor lights up\n(poke at mid of sender)",
         "1\u202f953",
         "15 s in movie\n+ 0\u20135 s delay\n= 15\u201320 s total",
         "97\u2013130",
         "Propagation observed\ndirectly in recording"],
    ]

    hdr_h = 0.11
    row_h = 0.165
    y0 = 0.98
    x = 0
    for lab, w in zip(col_labels, col_w):
        ax.add_patch(plt.Rectangle((x, y0 - hdr_h), w, hdr_h,
                                    fc=C_ACC1, ec='white', lw=1))
        ax.text(x + w/2, y0 - hdr_h/2, lab,
                ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', multialignment='center')
        x += w

    for r, row in enumerate(rows):
        bg = C_LIGHT if r % 2 == 0 else 'white'
        x = 0
        yr = y0 - hdr_h - (r + 1) * row_h
        for cell, w in zip(row, col_w):
            ax.add_patch(plt.Rectangle((x, yr), w, row_h,
                                        fc=bg, ec=C_MID, lw=0.5))
            ax.text(x + w/2, yr + row_h/2, cell,
                    ha='center', va='center', fontsize=8,
                    color=C_BODY, multialignment='center')
            x += w
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Formula box
    body(fig, 0.355,
         "The key formula for speed calculation, accounting for the recording start delay:")

    ax2 = fig.add_axes([0.10, 0.22, 0.80, 0.105])
    ax2.set_facecolor("#EBF5FB"); ax2.axis('off')
    ax2.add_patch(FancyBboxPatch((0.01, 0.05), 0.98, 0.90,
                                  boxstyle="round,pad=0.02",
                                  fc="#EBF5FB", ec=C_ACC1, lw=1.2,
                                  transform=ax2.transAxes))
    fig.text(0.50, 0.307,
             "speed  =  distance  /  (time in recording  +  recording start delay)",
             ha='center', va='center', fontsize=10.5,
             color=C_ACC1, fontweight='bold', family='monospace')
    fig.text(0.50, 0.254,
             "recording start delay = 0\u20135 s  (unknown within this range)",
             ha='center', va='center', fontsize=9, color=C_BODY, style='italic')

    body(fig, 0.205,
         "For Observations 1 and 2, no '+ X s in movie' term was recorded, "
         "indicating the response was visible at the very start of the movie (X \u2248 0). "
         "The calculated speeds are therefore lower bounds: the actual response "
         "may have traveled even faster, completing before the camera turned on.")

    footer(fig, 2)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3  –  Speed scale + the paradox
# ════════════════════════════════════════════════════════════════════════════
def page_speed_scale(pdf):
    fig = new_page()

    heading(fig, 0.955, "3.  Putting the Speeds in Context")
    rule(fig, 0.928)

    y = body(fig, 0.910,
        "To decide which biological mechanism could explain each response, "
        "we compare the observed speeds against the known ranges for candidate "
        "processes in embryonic tissue.")

    # Log speed diagram — placed below intro text with clear gap
    fig_bottom = 0.455
    fig_height = 0.38
    ax = fig.add_axes([0.22, fig_bottom, 0.70, fig_height])

    mechanisms = [
        ("Pure Ca\u00b2\u207a diffusion",       0.7,  1.7,  "#8E44AD", 0.7),
        ("CICR / IP3 Ca\u00b2\u207a wave",      0.7,  2.0,  C_ACC3,    0.7),
        ("eATP paracrine wave",                  1.3,  2.0,  "#16A085", 0.7),
        ("Bulk perivitelline flow",              2.0,  4.0,  "#F39C12", 0.7),
        ("Bioelectric / ephaptic",               8.5,  9.5,  C_ACC1,    0.5),
        ("Acoustic / pressure wave",             8.9, 10.0,  "#7F8C8D", 0.5),
    ]

    bar_h = 0.10
    for i, (label, lo, hi, col, alp) in enumerate(mechanisms):
        y_bar = i * 0.145 + 0.04
        ax.barh(y_bar, hi - lo, left=lo, height=bar_h,
                color=col, alpha=alp, edgecolor='none')
        ax.text(-0.12, y_bar + bar_h/2, label,
                ha='right', va='center', fontsize=8, color=C_BODY,
                transform=ax.transData)

    obs_ranges = [
        ("Obs 1  \u2265488 \u03bcm/s",  np.log10(488),  np.log10(5000), C_ACC2, 0.90),
        ("Obs 2  \u2265635 \u03bcm/s",  np.log10(635),  np.log10(5000), C_ACC2, 0.76),
        ("Obs 3  97\u2013130 \u03bcm/s", np.log10(97),  np.log10(130),  C_ACC3, 0.62),
    ]
    for label, lo, hi, col, yp in obs_ranges:
        ax.plot([lo, hi], [yp, yp], color=col, lw=4, solid_capstyle='round')
        ax.plot([lo, hi], [yp, yp], '|', color=col, ms=9, mew=2)
        ax.text(hi + 0.08, yp, label, va='center', fontsize=8,
                color=col, fontweight='bold')

    ax.set_xlim(0.3, 10.5)
    ax.set_ylim(-0.05, 0.98)
    ax.set_xlabel("log\u2081\u2080 (speed / \u03bcm s\u207b\u00b9)", fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(
        ["10\u2070","10\u00b9","10\u00b2","10\u00b3","10\u2074",
         "10\u2075","10\u2076","10\u2077","10\u2078","10\u2079"],
        fontsize=8)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Known mechanism speeds vs observed responses",
                 fontsize=9.5, color=C_HEAD, pad=6)

    # Caption just below figure
    y3 = body(fig, fig_bottom - 0.025,
        "Key findings:\n"
        "  \u2022  Observations 1 and 2 (tail responses, \u2265488\u2013635 \u03bcm/s) are too fast "
        "for Ca\u00b2\u207a diffusion or CICR waves, and far too slow for acoustic pressure "
        "or electromagnetic fields. They fall in the range of bulk perivitelline flow.\n"
        "  \u2022  Observation 3 (mid response, 97\u2013130 \u03bcm/s) sits precisely in the "
        "CICR / IP3 Ca\u00b2\u207a wave range \u2014 the only mechanism consistent with these speeds "
        "at millimetre distances on second-to-minute timescales.")

    heading(fig, y3 - 0.012, "4.  The Apparent Paradox")
    rule(fig, y3 - 0.040)

    body(fig, y3 - 0.058,
        "At first glance, Observations 1 and 2 present a puzzle: the neighbor's "
        "tail (Obs 2, distance 3\u202f174 \u03bcm) appears to respond at least as fast as "
        "the poked embryo's own tail (Obs 1, distance 2\u202f439 \u03bcm), even though "
        "the signal seemingly has further to travel. If propagation were limited by "
        "tissue traversal, the neighbor's response should be slower \u2014 not equal or faster.\n\n"
        "This paradox dissolves once we recognise that both tail responses were "
        "complete before recording began. We have no direct measurement of how "
        "they propagated \u2014 only that they finished within 5 seconds. The apparent "
        "speed asymmetry likely reflects nothing more than measurement uncertainty "
        "in the recording start time.")

    footer(fig, 3)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4  –  One mechanism, not two
# ════════════════════════════════════════════════════════════════════════════
def page_one_mechanism(pdf):
    fig = new_page()

    heading(fig, 0.955, "5.  One Mechanism, Not Two")
    rule(fig, 0.928)

    y = body(fig, 0.910,
        "An earlier interpretation suggested that two separate mechanisms might "
        "be at work: a fast physical one (bulk flow / pressure) for the tail "
        "responses, and a slow biological one (Ca\u00b2\u207a wave) for the mid response. "
        "This would be biologically awkward \u2014 why would two unrelated mechanisms "
        "coexist for the same inter-embryo communication task?\n\n"
        "The resolution is that the fast tail responses are not a separate signaling "
        "mechanism at all. The poke is a mechanical perturbation, and the pressure "
        "wave it creates is simply physics \u2014 it is the stimulus, not the signal. "
        "The biological response to that stimulus is, in both cases, a Ca\u00b2\u207a wave.")

    # Cascade diagram — positioned just below body text with clear gap
    diag_top    = y - 0.025
    diag_height = 0.20
    diag_bottom = diag_top - diag_height

    ax = fig.add_axes([0.05, diag_bottom, 0.90, diag_height])
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.2)

    steps = [
        ("Poke",                           C_ACC2),
        ("Pressure wave\n(~instant)",      "#F39C12"),
        ("Mechano-\nsensitive\nchannels",  C_ACC3),
        ("Local\nCa\u00b2\u207a\nentry",           C_ACC3),
        ("CICR threshold\ncrossed",        C_ACC1),
        ("Ca\u00b2\u207a wave\n10\u2013130 \u03bcm/s", C_ACC3),
    ]
    bw, bh = 1.35, 0.88
    gap = 0.19
    total = len(steps) * bw + (len(steps) - 1) * gap
    x0 = (10 - total) / 2
    y0 = 0.80

    for i, (label, col) in enumerate(steps):
        bx = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((bx, y0), bw, bh,
                                    boxstyle="round,pad=0.04",
                                    fc=col, ec='white', lw=1.4, alpha=0.88))
        ax.text(bx + bw/2, y0 + bh/2, label,
                ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', multialignment='center')
        if i < len(steps) - 1:
            ax.annotate("", xy=(bx + bw + gap, y0 + bh/2),
                        xytext=(bx + bw, y0 + bh/2),
                        arrowprops=dict(arrowstyle="-|>", color=C_MID, lw=1.4))

    ax.text(5, 0.25,
            "The pressure wave is the perturbation.   The Ca\u00b2\u207a wave is the signal.",
            ha='center', va='center', fontsize=9, color=C_BODY, style='italic')

    # Body text below diagram
    y2 = body(fig, diag_bottom - 0.025,
        "Under this unified view:\n"
        "  \u2022  The fast tail responses (Obs 1 & 2) represent mechanosensitive Ca\u00b2\u207a "
        "entry triggered almost instantly at the tail (the pressure wave arrives "
        "with millisecond latency). They complete before recording begins.\n"
        "  \u2022  The slow mid response (Obs 3) is a regenerative CICR wave that "
        "nucleates at the stimulus entry point and propagates across the embryo's "
        "tissue \u2014 the wave we actually observe propagating.\n"
        "  \u2022  One mechanism \u2014 Ca\u00b2\u207a \u2014 accounts for all three observations.")

    heading(fig, y2 - 0.015, "6.  Calculating the Observed Ca\u00b2\u207a Wave Speeds")
    rule(fig, y2 - 0.042)

    y3 = body(fig, y2 - 0.060,
        "For Observation 3 (mid response), the timing is well-constrained because "
        "the wave is actively propagating within the recording window:")

    # Calculation box — placed just below that text
    box_h  = 0.115
    box_bottom = y3 - 0.015 - box_h
    ax2 = fig.add_axes([0.10, box_bottom, 0.80, box_h])
    ax2.set_facecolor("#EBF5FB"); ax2.axis('off')
    ax2.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.95,
                                  boxstyle="round,pad=0.02",
                                  fc="#EBF5FB", ec=C_ACC1, lw=1,
                                  transform=ax2.transAxes))

    lines_calc = [
        ("Distance:                 1\u202f952.8 \u03bcm", False),
        ("Time in recording:        15 s", False),
        ("Recording start delay:    0\u20135 s", False),
        ("Total time from poke:     15 + 0  to  15 + 5  =  15\u201320 s", False),
        ("", False),
        ("Speed = 1\u202f952.8 / 20  to  1\u202f952.8 / 15  =  97.6  to  130.2 \u03bcm/s", True),
    ]
    lh_calc = (9.0 / 72.0) / PH * 1.5
    cy = box_bottom + box_h - 0.018
    for txt, bold in lines_calc:
        if txt:
            fig.text(0.14, cy, txt,
                     ha='left', va='top', fontsize=9,
                     color=C_ACC1 if bold else C_BODY,
                     fontweight='bold' if bold else 'normal',
                     family='monospace')
        cy -= lh_calc

    footer(fig, 4)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5  –  Anisotropy
# ════════════════════════════════════════════════════════════════════════════
def page_anisotropy(pdf):
    fig = new_page()

    heading(fig, 0.955,
            "7.  The Anisotropy: A Window into the Signal's Nature")
    rule(fig, 0.928)

    y = body(fig, 0.910,
        "The most mechanistically revealing observation comes from comparing "
        "how the Ca\u00b2\u207a wave propagates in two different embryo-pair orientations. "
        "When the poked embryo and neighbor are arranged head-to-head, the wave "
        "is roughly 2.3\u00d7 faster than in the tail-to-head arrangement \u2014 "
        "despite covering a longer distance.")

    # Anisotropy comparison diagram — placed just below intro text
    diag_top    = y - 0.018
    diag_height = 0.27
    diag_bottom = diag_top - diag_height

    ax = fig.add_axes([0.04, diag_bottom, 0.92, diag_height])
    ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 3.1)

    def draw_embryo(ax, cx, cy, angle_deg, col_head, col_body,
                    label="", lyo=-0.38):
        theta = np.radians(angle_deg)
        t = np.linspace(0, 2*np.pi, 120)
        rx, ry = 1.10, 0.42
        ex = cx + rx*np.cos(t)*np.cos(theta) - ry*np.sin(t)*np.sin(theta)
        ey = cy + rx*np.cos(t)*np.sin(theta) + ry*np.sin(t)*np.cos(theta)
        ax.fill(ex, ey, color=col_body, alpha=0.55, zorder=2)
        ax.plot(ex, ey, color='white', lw=0.8, zorder=3)
        hx = cx + 1.02*np.cos(theta)
        hy = cy + 1.02*np.sin(theta)
        ax.add_patch(plt.Circle((hx, hy), 0.27,
                                color=col_head, alpha=0.9, zorder=4))
        if label:
            ax.text(cx, cy + lyo, label, ha='center', va='top',
                    fontsize=7.5, color=C_BODY, style='italic',
                    multialignment='center')

    # Tail → Head row
    row1_y = 2.30
    ax.text(0.3, 2.95, "Tail \u2192 Head  (slow: 10.1\u201310.4 \u03bcm/s)",
            ha='left', fontsize=9, color=C_ACC2, fontweight='bold')
    draw_embryo(ax, 2.3, row1_y, 0,   "#E74C3C", "#E59866",
                label="poked\n(tail right)")
    draw_embryo(ax, 5.2, row1_y, 0,   "#2980B9", "#85C1E9",
                label="neighbor\n(head left)")
    ax.annotate("", xy=(4.15, row1_y), xytext=(3.45, row1_y),
                arrowprops=dict(arrowstyle="-|>", color=C_MID, lw=1.2))
    ax.text(3.80, row1_y + 0.22, "signal", ha='center',
            fontsize=7, color=C_MID)
    ax.text(6.55, row1_y,
            "10.1\u201310.4 \u03bcm/s\n1\u202f657 \u03bcm in ~159 s",
            ha='left', va='center', fontsize=8.5,
            color=C_ACC2, fontweight='bold')

    # Head → Head row
    row2_y = 0.88
    ax.text(0.3, 1.5, "Head \u2192 Head  (fast: 22.5\u201323.7 \u03bcm/s)",
            ha='left', fontsize=9, color=C_ACC3, fontweight='bold')
    draw_embryo(ax, 2.3, row2_y, 180, "#E74C3C", "#E59866",
                label="poked\n(head right)", lyo=-0.40)
    draw_embryo(ax, 5.2, row2_y, 0,   "#2980B9", "#85C1E9",
                label="neighbor\n(head left)", lyo=-0.40)
    ax.annotate("", xy=(4.15, row2_y), xytext=(3.45, row2_y),
                arrowprops=dict(arrowstyle="-|>", color=C_MID, lw=1.2))
    ax.text(3.80, row2_y + 0.22, "signal", ha='center',
            fontsize=7, color=C_MID)
    ax.text(6.55, row2_y,
            "22.5\u201323.7 \u03bcm/s\n2\u202f203 \u03bcm in ~93 s",
            ha='left', va='center', fontsize=8.5,
            color=C_ACC3, fontweight='bold')

    ax.set_xlim(0, 10); ax.set_ylim(0, 3.1)

    # Speed-ratio calculation box — placed just below diagram
    y2 = body(fig, diag_bottom - 0.020, "Calculating the speed ratio:")

    box_h  = 0.083
    box_bt = y2 - 0.010 - box_h
    ax3 = fig.add_axes([0.10, box_bt, 0.80, box_h])
    ax3.set_facecolor("#EAFAF1"); ax3.axis('off')
    ax3.add_patch(FancyBboxPatch((0.01, 0.05), 0.98, 0.90,
                                  boxstyle="round,pad=0.02",
                                  fc="#EAFAF1", ec=C_ACC3, lw=1,
                                  transform=ax3.transAxes))
    fig.text(0.14, box_bt + box_h - 0.014,
             "Speed ratio  =  22.5\u201323.7 \u03bcm/s  \u00f7  10.1\u201310.4 \u03bcm/s  =  2.2\u20132.3\u00d7",
             ha='left', va='top', fontsize=10,
             color=C_ACC3, fontweight='bold', family='monospace')
    fig.text(0.14, box_bt + box_h - 0.034,
             "Head-head is 2.3\u00d7 faster despite the distance being 33% longer.",
             ha='left', va='top', fontsize=9, color=C_BODY, style='italic')

    y3 = body(fig, box_bt - 0.020,
        "This directional dependence \u2014 called anisotropy \u2014 is a key diagnostic "
        "tool. It immediately rules out several candidate mechanisms:\n"
        "  [No]  Bulk flow and pressure waves are isotropic: they propagate equally "
        "in all directions through a uniform medium.\n"
        "  [No]  Free chemical diffusion (eATP, Ca\u00b2\u207a) is also isotropic in solution.\n"
        "  [Yes] Ca\u00b2\u207a waves propagating through tissue are anisotropic, because "
        "tissue has structure: cell size, ER density, IP3 receptor distribution, "
        "and gap junction coupling all vary along the embryo's head-to-tail axis.")

    body(fig, y3 - 0.012,
        "The anisotropy is therefore positive evidence that the slow Ca\u00b2\u207a wave "
        "(97\u2013130 \u03bcm/s, Obs 3; 10\u201324 \u03bcm/s in the orientation experiment) "
        "is propagating through the embryo's own tissue architecture \u2014 not "
        "through the surrounding medium.")

    footer(fig, 5)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6  –  Biological interpretation
# ════════════════════════════════════════════════════════════════════════════
def page_biology(pdf):
    fig = new_page()

    heading(fig, 0.955, "8.  Why the Head-Head Direction Is Faster")
    rule(fig, 0.928)

    y = body(fig, 0.910,
        "In Xenopus and other amphibian embryos, the animal-vegetal (A-V) and "
        "anterior-posterior (A-P) axes are established from the earliest cleavages, "
        "creating systematic differences in cell biology along the head-to-tail axis:\n\n"
        "  1.  IP3 receptor density:  Anterior (head) blastomeres express more IP3Rs "
        "and carry more endoplasmic reticulum \u2014 the Ca\u00b2\u207a store from which CICR draws. "
        "More ER means more Ca\u00b2\u207a available per unit volume, raising the CICR rate "
        "and therefore wave speed.\n\n"
        "  2.  Cell size:  Head-region cells are smaller in post-cleavage stages. "
        "Smaller cells mean shorter distances between ER cisternae, shortening the "
        "relay time between successive CICR events.\n\n"
        "  3.  Gap junction coupling:  Connexin expression is non-uniform along the "
        "A-P axis. Tighter electrical and chemical coupling anteriorly allows IP3 and "
        "Ca\u00b2\u207a to pass more readily between cells, accelerating wave propagation.\n\n"
        "  4.  Bioelectric polarity:  The embryo maintains a head-positive, "
        "tail-negative membrane-potential gradient (Vmem). Ca\u00b2\u207a flux driven by this "
        "gradient is facilitated in the head-to-head direction and retarded in the "
        "tail-to-head direction \u2014 a direct electrochemical contribution to the "
        "observed anisotropy.")

    heading(fig, y - 0.018, "9.  What This Tells Us About How Embryos Communicate")
    rule(fig, y - 0.045)

    body(fig, y - 0.063,
        "Taken together, the three observations paint a coherent picture:\n\n"
        "  \u2022  A poke creates an instantaneous physical perturbation (pressure wave) "
        "that arrives at both embryos in milliseconds. This triggers rapid, local "
        "Ca\u00b2\u207a entry through mechanosensitive channels \u2014 the fast tail responses "
        "(Obs 1 & 2). These are a direct response to the mechanical stimulus, "
        "not to a biological inter-embryo signal.\n\n"
        "  \u2022  The true inter-embryo biological signal is the Ca\u00b2\u207a wave (Obs 3). "
        "It nucleates at the point of mechanical stimulation, propagates through "
        "the sender embryo's tissue (via CICR), exits as eATP or Ca\u00b2\u207a flux at the "
        "embryo surface, crosses the inter-embryo gap by diffusion over ~10\u201320 "
        "seconds, then nucleates a fresh Ca\u00b2\u207a wave in the neighbor that propagates "
        "at 10\u2013130 \u03bcm/s according to tissue architecture.\n\n"
        "  \u2022  The spatial specificity (mid poke \u2192 mid response in neighbor) "
        "follows naturally from diffusion geometry: the chemical signal is most "
        "concentrated directly across the gap from the poke site.")

    footer(fig, 6)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 7  –  CEMA connection + summary
# ════════════════════════════════════════════════════════════════════════════
def page_cema(pdf):
    fig = new_page()

    heading(fig, 0.955,
            "10.  Implications for Collective Morphogenetic Rescue (CEMA)")
    rule(fig, 0.928)

    y = body(fig, 0.910,
        "The CEMA (Collective Embryonic Morphogenetic Assistance) phenomenon "
        "describes the finding that perturbed embryos recover better when grouped "
        "together \u2014 and that adding healthy, unperturbed embryos to the group does "
        "not help (Tung et al.\u00a02024). The poke experiments are relevant for two reasons:\n\n"
        "  1.  They reveal the inter-embryo signaling pathway experimentally. "
        "The slow, spatially specific Ca\u00b2\u207a wave (Obs 3, 15\u201320 s delay, "
        "97\u2013130 \u03bcm/s) is mechanistically consistent with the eATP \u2192 P2Y "
        "receptor \u2192 Ca\u00b2\u207a \u2192 AMPK \u2192 ATP recovery pathway: stressed embryos "
        "release extracellular ATP (eATP) at their surface; eATP diffuses across "
        "the inter-embryo gap in ~10\u201320 s for gaps of ~100\u2013150 \u03bcm "
        "(D_ATP \u2248 400 \u03bcm\u00b2/s:  t \u2248 d\u00b2 / 2D \u2248 100\u00b2 / 800 \u2248 12.5 s); "
        "it then binds P2 receptors on the neighbor's surface, elevating Ca\u00b2\u207a "
        "and triggering a tissue-wide Ca\u00b2\u207a wave.\n\n"
        "  2.  The anisotropy shows that wave propagation in the neighbor is faster "
        "along the head-to-head axis. This suggests that CEMA rescue may be spatially "
        "non-uniform: embryos oriented head-to-head might communicate more efficiently, "
        "and recovery might begin anteriorly before spreading posteriorly.")

    heading(fig, y - 0.018, "11.  Summary Table")
    rule(fig, y - 0.045)

    # Summary table
    tbl_top    = y - 0.060
    tbl_height = 0.33
    tbl_bottom = tbl_top - tbl_height

    ax = fig.add_axes([0.05, tbl_bottom, 0.90, tbl_height])
    ax.axis('off')

    col_labels = ["Observation", "Speed", "Timescale", "Mechanism", "CEMA relevance"]
    col_w = [0.22, 0.14, 0.13, 0.27, 0.24]

    rows_s = [
        ["Obs 1: poked\ntail lights up",
         "\u2265\u202f488 \u03bcm/s", "< 5 s",
         "Pressure wave \u2192\nmechanosensitive\nCa\u00b2\u207a entry",
         "Artifact of poke\n(not a CEMA signal)"],
        ["Obs 2: neighbor\ntail lights up",
         "\u2265\u202f635 \u03bcm/s", "< 5 s",
         "Same pressure wave\narrives at neighbor\nsimultaneously",
         "Artifact of poke\n(not a CEMA signal)"],
        ["Obs 3: neighbor\nmid lights up",
         "97\u2013130 \u03bcm/s", "15\u201320 s",
         "eATP diffusion \u2192\nP2Y \u2192 CICR\nCa\u00b2\u207a wave",
         "Primary inter-embryo\nbiological signal"],
        ["Orientation\nexperiment",
         "10\u201324 \u03bcm/s", "30\u2013165 s",
         "Same CICR wave;\nanisotropic due to\nA-P tissue structure",
         "Reveals spatial\npattern of recovery"],
    ]

    hdr_h = 0.11
    row_h = 0.195
    y0 = 0.98
    x = 0
    for lab, w in zip(col_labels, col_w):
        ax.add_patch(plt.Rectangle((x, y0 - hdr_h), w, hdr_h,
                                    fc=C_ACC1, ec='white', lw=1))
        ax.text(x + w/2, y0 - hdr_h/2, lab,
                ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', multialignment='center')
        x += w

    for r, row in enumerate(rows_s):
        bg = C_LIGHT if r % 2 == 0 else 'white'
        x = 0
        yr = y0 - hdr_h - (r + 1) * row_h
        for cell, w in zip(row, col_w):
            ax.add_patch(plt.Rectangle((x, yr), w, row_h,
                                        fc=bg, ec=C_MID, lw=0.4))
            ax.text(x + w/2, yr + row_h/2, cell,
                    ha='center', va='center', fontsize=8,
                    color=C_BODY, multialignment='center')
            x += w
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    body(fig, tbl_bottom - 0.022,
        "Conclusion:  All four observations are explained by a single biological "
        "mechanism \u2014 the Ca\u00b2\u207a wave \u2014 operating at two timescales: a fast, "
        "mechanically triggered local response (< 5 s) and a slower, chemically "
        "relayed inter-embryo wave (15\u2013165 s) that carries genuine developmental "
        "state information between embryos.")

    # Bottom decorative bar
    axb = fig.add_axes([0.08, 0.095, 0.84, 0.012])
    axb.set_facecolor(C_ACC1); axb.axis('off')

    footer(fig, 7)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    with PdfPages(OUTPUT) as pdf:
        page_title(pdf)
        page_observations(pdf)
        page_speed_scale(pdf)
        page_one_mechanism(pdf)
        page_anisotropy(pdf)
        page_biology(pdf)
        page_cema(pdf)

        d = pdf.infodict()
        d['Title']   = 'Electric Talk Between Embryos'
        d['Author']  = 'Computational analysis'
        d['Subject'] = 'Inter-embryo Ca\u00b2\u207a wave propagation, CEMA signaling'

    print(f"PDF written to {OUTPUT}")