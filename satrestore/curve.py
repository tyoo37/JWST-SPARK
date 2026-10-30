"""MIRI response curve: the raw signal of a saturated group as a function of the charge that reached the pixel, read backwards.

Charge of group j: Q_j = (j + a) X + b (X = rate x TGROUP; a, b from the level-1 fit). Raw DN of a saturated group
= R(q_j) + B(pixel, exposure) + delta_j, q = Q / Q_thr. R is one monotone curve per dataset, measured on calibrators (pixels whose
rate the pipeline measured from good groups and that saturated in a later group); B is a detector offset (column + row + an
8-px smooth of the residuals, plus a shift per exposure); delta_j are offsets of group 1 and of the last group. Inversion: the
x = X / Q_thr that best predicts a pixel's saturated groups below the top of the curve. Ported from production/miri_ramp
(W51), for any number of groups and integrations. No training: everything is measured on the dataset itself.
"""
import numpy as np
from scipy import ndimage

LQ = np.arange(-1.0, 1.6, 0.002)       # log10 q grid
MARGIN = 50.0                          # DN: groups within this of the curve's top give only a lower limit
SAT_DN = 56660.0                       # raw DN that defines Q_thr (CRDS MIRIMAGE saturation, uniform); only sets the unit of q
CURVE_GROUPS = (2, 3, 4)               # groups that set the curve (W51, 5 groups: 2-3); every other group gets its own offset
NORESTORE = 1 | 2**9 | 2**10 | 2**11 | 2**26 | 2**27 | 2**31   # PIXELDQ: DO_NOT_USE, NON_SCIENCE, DEAD, HOT, OPEN, ADJ_OPEN, REFERENCE_PIXEL
NOFLAT = 2**18                         # NO_FLAT_FIELD in the _cal DQ: outside the calibrated area, the pipeline gives no value


def charge(X, j, a, b):
    return (j + a) * X + b


def instances(e, k, sel, rates, rng, nmax):
    """Rows (pixel x integration) of exposure index k selected by sel (nints, ny, nx): x, y, exposure k, unit e = k x nints + i
    (an integration of an exposure: the level-1 constants and offsets differ between integrations), rate, level, raw groups."""
    i, y, x = np.nonzero(sel)
    if len(x) > nmax:
        s = np.sort(rng.choice(len(x), nmax, replace=False))
        i, y, x = i[s], y[s], x[s]
    return dict(x=x.astype(np.int32), y=y.astype(np.int32), k=np.full(len(x), k, np.int32), e=(k * e['ro'].nints + i).astype(np.int32),
                F=rates[i, y, x].astype(np.float32),
                lev=e['lev'][i, y, x].astype(np.int16), dn=e['raw'][i, :, y, x].astype(np.float32), int_=i.astype(np.int16),
                lin0=e['lin0'][i, y, x].astype(np.float32))


def calibrators(e, k, rates, rng, nmax=40_000):
    """Pixels with a known rate (rates: level-1 corrected) that saturated in a later group."""
    G = e['ro'].ngroups
    sel = (e['lev'] >= 1) & (e['lev'] <= G - 1) & ~e['jump'] & ((e['pdq'] & NORESTORE) == 0)[None] & np.isfinite(rates) & (rates > 0)
    return instances(e, k, sel, rates, rng, nmax)


def targets(e, k):
    """Pixels saturated in their first group (no pipeline rate)."""
    sel = (e['lev'] == 0) & ~e['jump'] & ((e['pdq'] & NORESTORE) == 0)[None] & ((e['dq'] & NOFLAT) == 0)[None]
    return instances(e, k, sel, np.full(e['lev'].shape, np.nan, np.float32), None, np.inf)


def cat(parts):
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def qthr_of(c, AB, T):
    """Charge at which the median raw DN of the calibrators' groups crosses SAT_DN: the unit of q."""
    G = c['dn'].shape[1]
    j = np.arange(1, G)[None, :]
    lq = np.log10(charge(c['F'][:, None] * T, j, AB[c['e'], 0][:, None], AB[c['e'], 1][:, None])).ravel()
    y = c['dn'][:, 1:].ravel()
    edges = np.arange(4.3, 5.2, 0.01)
    idx = np.digitize(lq, edges) - 1
    m = np.array([np.median(y[idx == i]) if (idx == i).sum() > 50 else np.nan for i in range(len(edges))])
    ok = np.isfinite(m)
    return float(10 ** np.interp(SAT_DN, np.maximum.accumulate(m[ok]), edges[ok] + 0.005))


def sat_points(c, rows, AB, T, qthr):
    """(log q, DN, group, row) for the saturated groups (j >= level, j >= 1) of the calibrator rows."""
    G = c['dn'].shape[1]
    out = []
    for j in range(1, G):
        r = rows[c['lev'][rows] <= j]
        lq = np.log10(charge(c['F'][r] * T, j, AB[c['e'][r], 0], AB[c['e'][r], 1]) / qthr)
        out.append((lq, c['dn'][r, j], np.full(len(r), j, np.int16), r))
    return [np.concatenate(v) for v in zip(*out)]


def fit_curve(lq, y, nmin=300):
    """Monotone binned-median curve on LQ, flat beyond the last well-sampled bin: (R on LQ, log q of that bin)."""
    o = np.argsort(lq)
    lq, y = lq[o], y[o]
    lo, hi = np.searchsorted(lq, LQ - 0.005), np.searchsorted(lq, LQ + 0.005)
    R = np.array([np.median(y[a:b]) if b - a >= nmin else np.nan for a, b in zip(lo, hi)])
    good = np.where(np.isfinite(R))[0]
    i0, i1 = good[0], good[-1]
    R[i0:i1 + 1] = np.maximum.accumulate(np.interp(LQ[i0:i1 + 1], LQ[good], R[good]))
    R[:i0], R[i1 + 1:] = R[i0], R[i1]
    return R, float(LQ[i1])


def build(c, rows, bias, AB, T, qthr, nmin=300):
    """Curve from the early saturated groups CURVE_GROUPS (up to G-2) of the calibrator rows, bias subtracted: a saturated
    pixel can use only its early groups, and the raw signal at a given charge drifts with the group (ngc6334, 20 groups: early
    groups 150-190 DN above later ones). Every other group gets its own offset (median residual about the curve)."""
    G = c['dn'].shape[1]
    lq, y, j, r = sat_points(c, rows, AB, T, qthr)
    y = y - bias[r]
    cg = [g for g in CURVE_GROUPS if g <= G - 2] or list(range(1, G))
    R, lqmax = fit_curve(lq[np.isin(j, cg)], y[np.isin(j, cg)], nmin)
    res = y - np.interp(lq, LQ, R)
    delta = np.zeros(G)
    for g in set(range(1, G)) - set(cg):
        s = (j == g) & (lq > np.log10(1.1)) & (lq < lqmax - 0.05)
        if s.sum() > 100:
            delta[g] = np.median(res[s])
    return dict(R=R, lqmax=lqmax, delta=delta, qthr=qthr)


def offsets(x, y, r, shape, sigma=8, it=20):
    """Detector offset map from residuals r at pixels (x, y): column + row terms plus the sigma-px smooth of what is left."""
    ny, nx = shape
    c, rr = np.zeros(nx), np.zeros(ny)
    cx, cy = np.maximum(np.bincount(x, minlength=nx), 1), np.maximum(np.bincount(y, minlength=ny), 1)
    for _ in range(it):
        c = np.bincount(x, r - rr[y], nx) / cx
        rr = np.bincount(y, r - c[x], ny) / cy
    pix = y * nx + x
    n = np.bincount(pix, minlength=nx * ny).astype(float).reshape(shape)
    s = np.bincount(pix, r - c[x] - rr[y], nx * ny).reshape(shape)
    sm = ndimage.gaussian_filter(s, sigma) / np.maximum(ndimage.gaussian_filter(n, sigma), 1e-3)
    return c[None, :] + rr[:, None] + sm


def residuals(c, rows, cv, AB, T):
    """Mean raw residual (bias not subtracted) of each calibrator row's saturated groups about the curve, q in 1.05 .. q_max."""
    lq, y, j, r = sat_points(c, rows, AB, T, cv['qthr'])
    s = (lq > np.log10(1.05)) & (lq < cv['lqmax'] - 0.02)
    res = y[s] - np.interp(lq[s], LQ, cv['R']) - cv['delta'][j[s]]
    u, inv = np.unique(r[s], return_inverse=True)
    return u, np.bincount(inv, res) / np.bincount(inv)


def calibrate(c, rows, AB, T, shape, n_exp, iterations=5):
    """Curve and offsets from calibrator rows, iterated: curve -> per-row residuals -> offset map and exposure shifts -> curve."""
    bias = np.zeros(len(c['x']), np.float32)
    de = np.zeros(n_exp)
    qthr = qthr_of({k: v[rows] for k, v in c.items()}, AB, T)
    for _ in range(iterations):
        cv = build(c, rows, bias, AB, T, qthr)
        u, r = residuals(c, rows, cv, AB, T)
        off = offsets(c['x'][u], c['y'][u], r - de[c['e'][u]], shape)
        z = r - off[c['y'][u], c['x'][u]]
        for k in range(n_exp):
            m = c['e'][u] == k
            de[k] = np.median(z[m]) if m.sum() > 200 else 0.0
        de -= np.median(de)
        bias = (off[c['y'], c['x']] + de[c['e']]).astype(np.float32)
    cv['bias_map'], cv['shift'] = off, de
    return cv, bias


def invert(dn, use, e, AB, cv, chunk=20_000):
    """Best log10 x for each row of dn (n, G; bias subtracted) from its groups `use` below the curve's top - MARGIN.
    Returns (log10 x, NaN where no group is usable; number of groups used; first group used)."""
    R, lqmax, delta = cv['R'], cv['lqmax'], cv['delta']
    G = dn.shape[1]
    top = np.interp(lqmax, LQ, R) - MARGIN + delta
    use = use & (dn < top[None, :])
    grid = LQ[(LQ >= -0.9) & (LQ <= lqmax + 0.15)]
    out = np.full(len(dn), np.nan)
    for k in np.unique(e):
        a, beta = AB[k, 0], AB[k, 1] / cv['qthr']
        pred = np.stack([np.interp(np.log10((j + a) * 10 ** grid + beta), LQ, R) + delta[j] for j in range(G)], 1)
        rows = np.where(e == k)[0]
        for c0 in range(0, len(rows), chunk):
            r_ = rows[c0:c0 + chunk]
            chi = np.zeros((len(r_), len(grid)))
            for j in range(G):
                d = np.where(use[r_, j, None], dn[r_, j, None] - pred[None, :, j], 0.0)
                chi += d * d
            out[r_] = grid[np.argmin(chi, 1)]
    nuse = use.sum(1)
    out[nuse == 0] = np.nan
    return out, nuse, np.where(nuse > 0, use.argmax(1), -1)


def selftest(c, AB, T, shape, n_exp, folds=4, max_level=3):
    """Leave-one-group-of-exposures-out (folds by exposure, so both integrations of an exposure are held out together; n_exp
    counts units): curve and offsets from the other exposures, unit shifts from the held-out unit's own calibrators (as in production); the held-out calibrators with first saturated group 1..max_level are
    restored from their saturated groups at the saturation level alone and compared with their measured rate. Returns rows (log q of the first used
    group, error in dex, first saturated group, groups used, first group used, unit)."""
    fold = c['k'] % folds
    out = []
    G = c['dn'].shape[1]
    jj = np.arange(G)[None, :]
    for f in range(folds):
        train = np.where(fold != f)[0]
        cv, _ = calibrate(c, train, AB, T, shape, n_exp)
        test = np.where((fold == f) & (c['lev'] <= max_level))[0]
        u, r = residuals(c, np.where(fold == f)[0], cv, AB, T)
        z = r - cv['bias_map'][c['y'][u], c['x'][u]]
        de = np.array([np.median(z[c['e'][u] == k]) if (c['e'][u] == k).sum() > 200 else 0.0 for k in range(n_exp)])
        bias = cv['bias_map'][c['y'][test], c['x'][test]] + de[c['e'][test]]
        use = (jj >= c['lev'][test][:, None]) & (c['dn'][test] >= SAT_DN)     # as for the targets: groups at the saturation level
        lx, nuse, first = invert(c['dn'][test] - bias[:, None], use, c['e'][test], AB, cv)
        truth = np.log10(c['F'][test] * T / cv['qthr'])
        a, b = AB[c['e'][test], 0], AB[c['e'][test], 1]
        q_first = np.where(first >= 0, (first + a) * 10 ** truth + b / cv['qthr'], np.nan)
        ok = np.isfinite(lx)
        out.append(np.stack([np.log10(q_first[ok]), (lx - truth)[ok], c['lev'][test][ok], nuse[ok], first[ok], c['e'][test][ok]], 1))
    return np.concatenate(out)
