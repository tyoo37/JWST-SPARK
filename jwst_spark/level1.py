"""MIRI level-1 correction. A pixel with one good group (group 0) gets the pipeline rate lin0 / TGROUP, which lacks the group-0
pedestal: in pixels with several good groups lin0 = a X + b (X = rate x TGROUP, a ~ 0.78 of a frame, b ~ 4400 DN in W51).
The corrected rate is (lin0 - b) / (a TGROUP). No training: (a, b) are measured on each exposure's own pixels, separately for
each integration (after the first, the detector starts from a different state: in the brick F2550W data a = 0.71-0.73,
b = 4,700-4,900 DN in the first integration and a = 0.78, b = 5,800 DN in the second).
Fitting (a, b) per detector row parity was tried (2026-10-30) and removed: it moved level-1 values by only ~0.0025 dex, was
unstable in small samples (brick), and left the even/odd pattern of level-1 pixels (+-0.01-0.02 dex, growing with brightness
near saturation, against +-0.004 for bright unsaturated pixels) in place.
Pixels flagged only because a neighbour saturated keep the correction: neighbours flagged from group 3 or 4 agree with their own
increments before the neighbour saturates to 0.00-0.04 dex (2026-10-31, satrestore_tests/diag_l1spill.py, W51 and four test sets).
Isolated faint pixels drift from the relation (cloudef +0.04 dex at 4,000-6,000 DN, W51 -0.05), neighbours of the same
brightness do not (-0.02 to +0.01), and 85-100% of the pixels the correction serves are brighter than 6,000 DN, so no drift
correction is applied."""
import numpy as np
from scipy import ndimage

LEVELS = (2, 3, 4)       # pixels whose first saturated group is 2-4: bright, with 2-4 good groups, like the level-1 pixels
FLUX_ESTIMATED = 2**28   # PIXELDQ: jwst 1.21 marks the pixels flagged only because a neighbour saturated
NB = np.ones((3, 3), bool)
NB[1, 1] = False         # the 8 neighbours


def samples(e, i, rng, nmax=50_000):
    """(X, lin0) of integration i's pixels with first saturated group in LEVELS, no jump, PIXELDQ 0, positive rate."""
    T = e['ro'].tgroup
    s = np.isin(e['lev'][i], LEVELS) & ~e['jump'][i] & (e['pdq'] == 0) & np.isfinite(e['rates'][i]) & (e['rates'][i] > 0)
    X, y = e['rates'][i][s] * T, e['lin0'][i][s]
    if len(X) > nmax:
        k = rng.choice(len(X), nmax, replace=False)
        X, y = X[k], y[k]
    return X.astype(np.float64), y.astype(np.float64)


def fit(X, y):
    a, b = np.polyfit(X, y, 1)
    return float(a), float(b)


def selftest(X, y, rng):
    """Half the pixels fit (a, b); the other half predict their rate from lin0 alone, as a level-1 pixel would. Returns the
    median and 16-84% half-width (dex) of predicted / true rate, over all test pixels and over the brightest 10% (the level-1
    pixels are brighter still, so this checks the trend toward them)."""
    k = rng.permutation(len(X))
    tr, te = k[:len(X) // 2], k[len(X) // 2:]
    a, b = fit(X[tr], y[tr])
    with np.errstate(invalid='ignore'):
        d = np.log10((y[te] - b) / a / X[te])
    ok = np.isfinite(d)                           # a few faint pixels have lin0 below b
    d, Xte = d[ok], X[te][ok]
    top = Xte > np.percentile(Xte, 90)
    hw = lambda v: float((np.percentile(v, 84) - np.percentile(v, 16)) / 2)
    return dict(n=len(d), median=float(np.median(d)), half_width=hw(d), median_top10=float(np.median(d[top])), half_width_top10=hw(d[top]))


def pixels(e):
    """Pixel-integrations the correction serves: level 1, and levels 2-4 left without a pipeline rate (jwst 1.15 drops MIRI's
    first group, so a 2-group ramp keeps one group and gets no rate; sgrb2). Needs no pipeline rate, only lin0."""
    return ((e['lev'] == 1) | (np.isin(e['lev'], LEVELS) & ~np.isfinite(e['rates']))) & np.isfinite(e['lin0'])


def neighbour_gap(e, i, ab):
    """Level-1 pixels of integration i flagged only because a neighbour saturated (raw group 1 below saturation): mask (ny, nx),
    and log10 of their level-1 value over their own first increment g1 - g0, calibrated to rate x TGROUP on pixels of the same
    raw group-0 level (2,000-DN bins) with no saturation in themselves or their neighbours up to group 4. The increment also holds
    charge from the neighbour that saturates during it (later increments hold more: 0.1-0.2 dex), so the gap is an upper bound of
    these pixels' error, carried in their ERR. ab: (a, b) of the integration."""
    G, lev, raw = e['ro'].ngroups, e['lev'][i], e['raw'][i]
    nb = (lev == 1) & (raw[1] < e['satdn']) & np.isfinite(e['lin0'][i])
    if G < 2 or not nb.any():
        return nb, np.array([])
    g5 = min(G, 5)
    X = e['rates'][i] * e['ro'].tgroup
    clean = ((e['pdq'] & ~np.uint32(FLUX_ESTIMATED)) == 0) & ~e['jump'][i]
    inc = raw[1] - raw[0]
    ctl = clean & (lev >= g5) & (ndimage.minimum_filter(lev, footprint=NB, mode='nearest') >= g5) & np.isfinite(X) & (X > 0) & (inc > 0)
    edges = np.arange(0, 66_000, 2000)
    k = np.digitize(raw[0], edges)
    fac = np.full(len(edges) + 1, np.nan)
    for j in np.unique(k[ctl]):
        s = ctl & (k == j)
        if s.sum() >= 30:
            fac[j] = np.median(X[s] / inc[s])
    x0 = inc * fac[k]
    xl1 = (e['lin0'][i] - ab[1]) / ab[0]
    s = nb & clean & (x0 > 0) & (xl1 > 0)
    return nb, np.log10(xl1[s] / x0[s])


def pipeline_ab(ro):
    """(a, b) that reproduce the pipeline's own one-group rate: NIRCam lin0 / t0, t0 = (NFRAMES + 1) / 2 x TFRAME (group 0 is
    the mean of its frames). Used for NIRCam ramps of fewer than 3 groups, where a fit on 2-group pixels is unreliable (their
    rate is g1 - g0, so X and lin0 share the group-0 noise: a = 0.272 against 0.30 expected in a SHALLOW2 2-group set)."""
    return (ro.nframes + 1) / 2 * ro.tframe / ro.tgroup, 0.0


def correct(e, ab):
    """Corrected rates (nints, ny, nx) of the pixels() above, NaN elsewhere; ab (nints, 2) per integration."""
    T = e['ro'].tgroup
    l1 = pixels(e)
    a, b = ab[:, 0, None, None], ab[:, 1, None, None]
    return np.where(l1, (e['lin0'] - b) / (a * T), np.nan).astype(np.float32)
