"""Blind rating sheets: the user's manually found frame among 11 of mine, numbered and shuffled, drawn with the same renderer (magma, G_pol 0 to 2) on the same size.
The key (which number is which file, and its feature match) is written to anatomy/rating/key_<shape>.json and is not shown.   python anatomy_rating_sheet.py"""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anatomy_score_image as SI, anatomy_features as AF

IMG = '/tmp/claude-0/-home-user/25532e67-af17-5cb7-bd9b-a39d2880860b/images/'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anatomy', 'rating')


def pool(shape):
    pats = ['anatomy/feat/best/planarian_*.npz', 'anatomy/local/best/planarian_*.npz'] if shape == 'planarian' else ['anatomy/feat/best/tadpole_*.npz']
    rows = []
    for p in pats:
        for f in glob.glob(p):
            d = np.load(f); g, m = d['g'].astype(float), d['mask']
            if m.shape[0] < 30 or '_m4' in f or any(f'_{o}' in os.path.basename(f) for o in ('stadium', 'ellipse', 'rectangle')):
                continue
            I, v = SI.rectify_mask(g, m); s = AF.match(AF.features(I), AF.target_features(shape))['score']
            rows.append((s, f, g, m))
    return sorted(rows, key=lambda r: -r[0])


def pick(rows, n=11):
    """Spread over the range of the current score: the top 4, 4 around the middle, 3 near the bottom quartile; one frame per run family (no near-duplicates)."""
    idx = list(range(4)) + list(np.linspace(len(rows) * 0.4, len(rows) * 0.6, 4).astype(int)) + list(np.linspace(len(rows) * 0.75, len(rows) * 0.9, 3).astype(int))
    return [rows[i] for i in idx][:n]


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    for shape, f in (('planarian', '4.png'), ('tadpole', '5.png')):
        rows = pick(pool(shape))
        gu, mu = SI.load(IMG + f); su = AF.match(AF.features(SI.rectify_mask(gu, mu)[0]), AF.target_features(shape))['score']
        items = [dict(src='USER', score=su, g=gu, m=mu)] + [dict(src=os.path.basename(r[1]), score=r[0], g=r[2], m=r[3]) for r in rows]
        order = rng.permutation(len(items))
        fig, ax = plt.subplots(6, 2, figsize=(19.2, 12.6), gridspec_kw=dict(wspace=0.03, hspace=0.25))
        key = {}
        for k, i in enumerate(order):
            it = items[i]; a = ax[k % 6, k // 6]
            a.imshow(np.where(it['m'], it['g'], np.nan), cmap='magma', vmin=0, vmax=2, interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_visible(False)
            a.set_title(f'{k + 1}', fontsize=22, loc='left', color='k', fontweight='bold')
            key[k + 1] = dict(src=it['src'], current_score=round(float(it['score']), 3))
        fig.suptitle(f'{shape}: G_pol, magma 0-2 (numbers only; no scores shown)', fontsize=16)
        fig.savefig(os.path.join(OUT, f'sheet_{shape}.png'), dpi=60, bbox_inches='tight', facecolor='white')
        json.dump(key, open(os.path.join(OUT, f'key_{shape}.json'), 'w'), indent=1)
        print(shape, 'user frame is number', [k for k, v in key.items() if v['src'] == 'USER'][0], 'current scores', sorted([v['current_score'] for v in key.values()], reverse=True))
