"""Direct PSF fit of saturated stars (no training). The STPSF model of the instrument and filter, with the telescope's optical
state on the observation date when a MAST token is available, is fitted (flux, position, constant background) to the measured
pixels around a compact saturated region; the fitted model gives the region's pixels.

Self-test: bright unsaturated stars of the same exposures with their brightest pixels hidden, as if saturated, and predicted.
The hidden pixels of such stars sit closer to the measured ones than the cores of real saturated stars, so the test is a
lower bound on the error."""
import os
import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import least_squares
from photutils.psf import FittableImageModel

HALF = 12            # half-size of the fitting box beyond the region (px)
AMAX = 400           # largest region fitted (px); larger ones are extended emission, not a star
REL_ERR = 0.05       # relative error floor of a measured pixel (PSF model mismatch)
HIDE = (1, 3, 7, 15, 31, 63)   # self-test: number of brightest pixels hidden
RCHI2_MAX = 5.0      # a fit with a larger reduced chi^2 (5% error floor) is not a point source; in the F2550W tests such fits
                     # disagreed with the response curve by 0.1-0.3 dex, those below agreed to 0.002-0.02 dex
MIN_RING = 0.5       # a region is fitted only if this share of the 3-px ring around it is measured (not cut by an edge or no-data area)
MIN_MEASURED = 0.9   # self-test stars: share of the fitting box with measured pixels
SHARP = 1.5          # self-test stars: peak / mean of its 8 neighbours at most SHARP x the PSF's own (rejects hot pixels, cosmic rays)


def psf_model(h, cache_dir, oversample=4, fov=81, log=print):
    """FittableImageModel of the exposure's instrument and filter (primary header h), cached in cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    fn = f'{cache_dir}/' + f'psf_{h["INSTRUME"]}_{h["FILTER"]}_{h["DATE-OBS"]}_o{oversample}_f{fov}.fits'.lower()
    if not os.path.exists(fn):
        import stpsf
        inst = stpsf.MIRI() if h['INSTRUME'] == 'MIRI' else stpsf.NIRCam()
        inst.filter = h['FILTER']
        try:
            inst.load_wss_opd_by_date(h['DATE-OBS'], plot=False)
        except (OSError, RuntimeError, ValueError) as ex:   # no network or no token: the default optical state
            log(f'psf: no telescope state for {h["DATE-OBS"]} ({ex}); default', flush=True)
        inst.calc_psf(fov_pixels=fov, oversample=oversample).writeto(fn)
    with fits.open(fn) as p:
        data = p['OVERDIST'].data if 'OVERDIST' in p else p[0].data
    return FittableImageModel(data, oversampling=oversample)


def fit(sci, err, usable, cy, cx, model, half):
    """Fit flux, centre and background to the usable pixels of a box around (cy, cx). Returns (params, box slices, n used) or None."""
    ny, nx = sci.shape
    y0, y1, x0, x1 = max(int(cy) - half, 0), min(int(cy) + half + 1, ny), max(int(cx) - half, 0), min(int(cx) + half + 1, nx)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d, w = sci[y0:y1, x0:x1], usable[y0:y1, x0:x1] & np.isfinite(sci[y0:y1, x0:x1])
    if w.sum() < 20:
        return None
    bkg = np.nanpercentile(d[w], 10)
    sig = np.hypot(np.nan_to_num(err[y0:y1, x0:x1], nan=0., posinf=0., neginf=0.), REL_ERR * np.abs(d))[w]
    if not (sig > 0).any():
        return None
    sig = np.where(sig > 0, sig, np.median(sig[sig > 0]))
    frac = model.evaluate(xx[w], yy[w], 1., cx, cy).sum()
    f0 = max(np.sum(d[w] - bkg) / max(frac, 1e-3), 1.)
    res = lambda p: (model.evaluate(xx[w], yy[w], p[0], p[1], p[2]) + p[3] - d[w]) / sig
    r = least_squares(res, [f0, cx, cy, bkg], x_scale=[f0, 1, 1, max(abs(bkg), 1.)], bounds=([0, cx - 3, cy - 3, -np.inf], [np.inf, cx + 3, cy + 3, np.inf]))
    return r.x, (slice(y0, y1), slice(x0, x1)), int(w.sum()), float(np.sum(r.fun ** 2) / max(w.sum() - 4, 1))


def predict(p, ys, xs, model):
    return model.evaluate(xs.astype(float), ys.astype(float), p[0], p[1], p[2]) + p[3]


def regions(need, measured):
    """Compact connected regions of `need` pixels (8-connected) with measured pixels around them: list of (ys, xs)."""
    lab, n = ndimage.label(need, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = np.nonzero(lab[sl] == i)
        if len(ys) > AMAX:
            continue
        ys, xs = ys + sl[0].start, xs + sl[1].start
        reg = np.zeros(need.shape, bool)
        reg[ys, xs] = True
        y0, y1, x0, x1 = max(ys.min() - 4, 0), ys.max() + 5, max(xs.min() - 4, 0), xs.max() + 5
        ring = ndimage.binary_dilation(reg[y0:y1, x0:x1], iterations=3) & ~reg[y0:y1, x0:x1]
        if measured[y0:y1, x0:x1][ring].mean() >= MIN_RING:
            out.append((ys, xs))
    return out


def restore(sci, err, measured, need, model):
    """PSF values for the pixels of each compact region of `need` (True: no value or a value to compare with), fitted to the
    `measured` pixels around it, where the fit passes RCHI2_MAX. Returns (values image with NaN elsewhere, per-region dicts)."""
    out = np.full(sci.shape, np.nan, np.float32)
    info = []
    for ys, xs in regions(need, measured):
        cy, cx = ys.mean(), xs.mean()
        half = HALF + int(np.ceil(max(np.ptp(ys), np.ptp(xs)) / 2))
        f = fit(sci, err, measured, cy, cx, model, half)
        if f is None:
            continue
        p, _, nused, rchi2 = f
        if rchi2 <= RCHI2_MAX:
            out[ys, xs] = predict(p, ys, xs, model)
        info.append(dict(y=float(p[2]), x=float(p[1]), flux=float(p[0]), bkg=float(p[3]), area=len(ys), n_fit=nused, rchi2=rchi2, ys=ys, xs=xs))
    return out, info


def selftest(sci, err, measured, model, rng, nmax=60):
    """Bright isolated unsaturated stars: hide their HIDE brightest pixels, fit the rest, compare where the fit passes RCHI2_MAX.
    Rows (star, hidden count, error dex = log predicted / true, and the features of starmodel.FEATURES)."""
    good = np.isfinite(sci) & measured
    img = np.where(good, sci, -np.inf)
    peak = (img == ndimage.maximum_filter(img, size=2 * HALF + 1)) & good & (sci > np.nanpercentile(sci[good], 99.5))
    # isolated: most of the box measured
    full = ndimage.uniform_filter(good.astype(float), size=2 * HALF + 1) >= MIN_MEASURED
    full[:HALF], full[-HALF:], full[:, :HALF], full[:, -HALF:] = False, False, False, False
    ys, xs = np.nonzero(peak & full)
    ring = (ndimage.uniform_filter(np.where(good, sci, 0.), 3) * 9 - np.where(good, sci, 0.)) / 8
    o = model.oversampling[0] if np.ndim(model.oversampling) else model.oversampling
    c0 = model.evaluate(np.array([0., 1, 1, 0, -1, -1, -1, 0, 1]), np.array([0., 0, 1, 1, 1, 0, -1, -1, -1]), 1., 0., 0.)
    sharp_psf = c0[0] / c0[1:].mean()
    keep = sci[ys, xs] / ring[ys, xs] <= SHARP * sharp_psf
    ys, xs = ys[keep], xs[keep]
    if len(ys) > nmax:
        k = rng.choice(len(ys), nmax, replace=False)
        ys, xs = ys[k], xs[k]
    rows = []
    for sid, (cy, cx) in enumerate(zip(ys, xs)):
        box = (slice(cy - 5, cy + 6), slice(cx - 5, cx + 6))
        order = np.argsort(sci[box].ravel())[::-1]
        for nh in HIDE:
            hid = np.zeros(sci.shape, bool)
            hy, hx = np.unravel_index(order[:nh], (11, 11))
            hy, hx = hy + cy - 5, hx + cx - 5
            hid[hy, hx] = True
            f = fit(sci, err, good & ~hid, cy, cx, model, HALF)
            if f is None or f[3] > RCHI2_MAX:
                continue
            pred = predict(f[0], hy, hx, model)
            true = sci[hy, hx]
            ok = (pred > 0) & (true > 0)
            ft = features(f[0], hy[ok], hx[ok], model, nh, f[3])
            rows += [(sid, nh, float(v), *fr) for v, fr in zip(np.log10(pred[ok] / true[ok]), ft)]
    return rows


def features(p, ys, xs, model, n_hidden, rchi2):
    """Per-pixel features of a fitted star (starmodel.FEATURES): log predicted value, distance to the fitted centre (px),
    pixels hidden or saturated, log flux, background / predicted value, reduced chi^2 of the fit."""
    pred = predict(p, ys, xs, model)
    return np.stack([np.log10(np.maximum(pred, 1e-6)), np.hypot(xs - p[1], ys - p[2]), np.full(len(ys), n_hidden),
                     np.full(len(ys), np.log10(max(p[0], 1e-6))), p[3] / np.maximum(pred, 1e-6), np.full(len(ys), rchi2)], 1)
