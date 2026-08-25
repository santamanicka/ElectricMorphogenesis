"""Show the best face obtained so far in every tracked 30x30 class -- both clamp families
(boundary-only fieldDomeTwoFoldSymmetry vs. full-tissue single-shot tissueGpolTwoFoldSymmetry)
crossed with screen size, correlation-loss and globalsum-loss side by side.

Each class pools every file-number range ever used for it, including reruns after a timeout or
node failure, so this always reflects the best checkpoint found across all attempts, not just the
most recent submission.

Reads bestModelParameters_fieldVector_30x30_*.dat checkpoints directly -- each stores its own
targetVmem/actualVmem, so nothing is re-simulated here.
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Every (clampMode, fieldScreenSize, fieldTransductionWeight) combination actually trained, found by
# scanning each checkpoint's own stored fieldParameters/clampParameters rather than assumed -- the two
# weights were not run on the same set of screens (weight 1000 has no single-shot runs at all, and
# its boundary runs used different screens than weight 700's), so coverage is uneven by design.
boundaryClassesW700 = [
    ('screen2', [1601,1602,1603,1604,1605,1606], [1701,1702,1703,1704,1705,1706]),
    ('screen3', [1801,1802,1803,1804,1805,1806], [1901,1902,1903,1904,1905,1906]),
    ('screen8', [1401,1402,1403,1404,1405,1406,1201,1202,1203,1204,1205,1206],
                [1501,1502,1503,1504,1505,1506,1301,1302,1303,1304,1305,1306]),
]

boundaryClassesW1000 = [
    ('screen4', [901,902,903,904,905,906],
                [501,502,503,504,505,506,507,508,509,510,511,512,513,514,
                 601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,
                 701,702,703,704,705,706,707,708,9002]),
    ('screen10', [951,952,953,954,955,956], []),
    ('screen24', [1001,1002,1003,1004,1005,1006], [1101,1102,1103,1104,1105,1106]),
]

singleShotSymClassesW700 = [
    ('screen2', [1931,1932,1933,1934,1935,1936], [1937,1938,1939,1940,1941,1942]),
    ('screen4', [1907,1908,1909,1910,1911,1912,1943,1944,1945,1946,1947,1948],
                [1913,1914,1915,1916,1917,1918,1949,1950,1951,1952,1953,1954]),
    ('screen8', [1919,1920,1921,1922,1923,1924], [1925,1926,1927,1928,1929,1930]),
]
# No single-shot (tissueGpolTwoFoldSymmetry) runs exist at weight 1000 -- only boundary was tried there.

def bestInGroup(nums):
    best = None
    for n in nums:
        f = f"data/bestModelParameters_fieldVector_30x30_{n}.dat"
        try:
            p = torch.load(f, map_location='cpu', weights_only=False)
        except Exception:
            continue
        L = float(p['trainParameters']['bestLoss'])
        if best is None or L < best[0]:
            best = (L, n, p)
    return best

rows, cols = 30, 30
sections = [('boundary-clamp (fieldDome, 100 iters), weight 700', boundaryClassesW700, 'figures/bestFaceBoundaryClampW700.png'),
            ('boundary-clamp (fieldDome, 100 iters), weight 1000', boundaryClassesW1000, 'figures/bestFaceBoundaryClampW1000.png'),
            ('full-tissue single-shot (tissueGpol, 1 iter), weight 700', singleShotSymClassesW700, 'figures/bestFaceFullControlW700.png')]

for sectionLabel, classes, outputPath in sections:
    fig, axes = plt.subplots(len(classes), 3, figsize=(9.5, 3.2 * len(classes)))
    for row, (screenLabel, corrNums, globNums) in enumerate(classes):
        c = bestInGroup(corrNums)
        g = bestInGroup(globNums)

        axTarget = axes[row, 0]
        if c is not None:
            target = c[2]['trainParameters']['targetVmem'].reshape(rows, cols).numpy() * 1000
            axTarget.imshow(target, cmap='gray')
        axTarget.set_ylabel(screenLabel, fontsize=11)
        axTarget.set_title('target' if row == 0 else '', fontsize=10)
        axTarget.set_xticks([]); axTarget.set_yticks([])

        axCorr = axes[row, 1]
        if c is not None:
            L, n, p = c
            actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
            axCorr.imshow(actual, cmap='gray')
            axCorr.set_title(f'correlation\nfile {n}, loss {L:.3f}' if row == 0 else f'file {n}, loss {L:.3f}', fontsize=9)
        else:
            axCorr.set_title('no checkpoint', fontsize=9)
        axCorr.set_xticks([]); axCorr.set_yticks([])

        axGlob = axes[row, 2]
        if g is not None:
            L, n, p = g
            actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
            axGlob.imshow(actual, cmap='gray')
            axGlob.set_title(f'globalsum\nfile {n}, loss {L:.3f}' if row == 0 else f'file {n}, loss {L:.3f}', fontsize=9)
        else:
            axGlob.set_title('no checkpoint', fontsize=9)
        axGlob.set_xticks([]); axGlob.set_yticks([])

    fig.suptitle(f'Best face so far -- {sectionLabel} (30x30, pooled across all attempts)', fontsize=12)
    fig.tight_layout()
    fig.savefig(outputPath, dpi=140, bbox_inches='tight')
    print(f'wrote {outputPath}')
