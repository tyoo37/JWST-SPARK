"""MIRI level-1 correction. A pixel with one good group (group 0) gets the pipeline rate lin0 / TGROUP, which lacks the group-0
pedestal: in pixels with several good groups lin0 = a X + b (X = rate x TGROUP, a ~ 0.78 of a frame, b ~ 4400 DN in W51).
The corrected rate is (lin0 - b) / (a TGROUP). No training: (a, b) are measured on each exposure's own pixels, separately for
each integration (after the first, the detector starts from a different state: in the brick F2550W data a = 0.71-0.73,
b = 4,700-4,900 DN in the first integration and a = 0.78, b = 5,800 DN in the second).
Fitting (a, b) per detector row parity was tried (2026-10-30) and removed: it moved level-1 values by only ~0.0025 dex, was
unstable in small samples (brick), and left the even/odd pattern of level-1 pixels (+-0.01-0.02 dex, growing with brightness
near saturation, against +-0.004 for bright unsaturated pixels) in place."""
import numpy as np

LEVELS = (2, 3, 4)       # pixels whose first saturated group is 2-4: bright, with 2-4 good groups, like the level-1 pixels


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


def correct(e, ab):
    """Corrected rates (nints, ny, nx) of the level-1 pixels, NaN elsewhere; ab (nints, 2) per integration."""
    T = e['ro'].tgroup
    l1 = (e['lev'] == 1) & np.isfinite(e['lin0'])     # no pipeline rate needed: jwst 1.15 leaves one-group ramps without one (sgrb2)
    a, b = ab[:, 0, None, None], ab[:, 1, None, None]
    return np.where(l1, (e['lin0'] - b) / (a * T), np.nan).astype(np.float32)
