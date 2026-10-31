"""Synthetic checks: the level-1 fit recovers (a, b), and the curve inversion recovers a known charge."""
import numpy as np
from jwst_spark import level1, curve


def test_level1_fit_and_selftest():
    rng = np.random.default_rng(0)
    X = rng.uniform(5000, 20000, 5000)
    y = 0.78 * X + 4400 + rng.normal(0, 30, len(X))
    a, b = level1.fit(X, y)
    assert abs(a - 0.78) < 0.01 and abs(b - 4400) < 60
    t = level1.selftest(X, y, rng)
    assert abs(t['median']) < 0.002 and t['half_width'] < 0.01


def test_invert_recovers_charge():
    R = 40000 + 20000 * (1 - np.exp(-(10 ** curve.LQ) / 2))          # a monotone curve that flattens towards its top
    cv = dict(R=R, lqmax=0.8, delta=np.zeros(5), qthr=60000.)
    AB = np.array([[0.78, 4400.]])
    lx_true = np.log10(np.array([0.9, 1.3, 2.0, 3.5]))
    j = np.arange(5)[None, :]
    q = (j + 0.78) * 10 ** lx_true[:, None] + 4400 / 60000.
    dn = np.interp(np.log10(q), curve.LQ, R)
    lx, nuse, first = curve.invert(dn, np.ones(dn.shape, bool), np.zeros(4, int), AB, cv)
    assert np.all(nuse > 0) and np.allclose(lx, lx_true, atol=0.003), (lx, lx_true)


def test_psf_fit_recovers_hidden_core():
    from photutils.psf import FittableImageModel
    from jwst_spark import psf
    o = 4
    yy, xx = np.mgrid[:81, :81] / o - 10
    model = FittableImageModel(np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2)), oversampling=o)
    y, x = np.mgrid[:50, :50]
    img = model.evaluate(x.astype(float), y.astype(float), 1e5, 24.4, 25.3) + 20
    hid = img > np.percentile(img, 99)
    p, _, _, rchi2 = psf.fit(img, np.ones(img.shape), ~hid, 25, 25, model, 12)
    assert rchi2 < 1e-3, rchi2
    assert np.allclose(psf.predict(p, *np.nonzero(hid), model), img[hid], rtol=1e-3)


def test_star_predictor_learns_psf_error():
    from jwst_spark import starmodel
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(60):
        for nh in (1, 7, 31):
            f = np.column_stack([rng.uniform(2, 5, nh), rng.uniform(0, 3, nh), np.full(nh, nh), np.full(nh, rng.uniform(6, 8)),
                                 rng.uniform(0, .2, nh), np.full(nh, rng.uniform(.5, 3))])
            err = 0.05 + 0.02 * f[:, 1] + rng.normal(0, 0.005, nh)          # PSF too bright, more so away from the centre
            rows += [(sid, nh, e_, *fr) for e_, fr in zip(err, f)]
    _, info = starmodel.train(rows)
    assert all(abs(r['trained_median']) < 0.01 and r['trained_half_width'] < 0.01 for r in info['selftest']), info


def test_neighbour_gap_measures_extra_charge():
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    X = rng.uniform(3000, 15000, (80, 80))
    lev = np.full((1, 80, 80), 5, np.int16)
    lev[0, :20, :20] = 1                                  # flagged from group 1 by a saturated neighbour
    raw = np.zeros((1, 5, 80, 80), np.float32)
    raw[0, 0] = 4400 + 0.78 * X
    for g in range(1, 5):
        raw[0, g] = raw[0, g - 1] + np.where((lev[0] == 1) & (g == 1), 1.2 * X, X)    # 20% extra charge in their first increment
    e = dict(ro=SimpleNamespace(ngroups=5, tgroup=1.0), lev=lev, raw=raw, rates=X[None], pdq=np.zeros((80, 80), np.uint32),
             jump=np.zeros((1, 80, 80), bool), lin0=(4400 + 0.78 * X)[None], satdn=np.full((80, 80), 56660.))
    nb, gap = level1.neighbour_gap(e, 0, (0.78, 4400.))
    assert nb.sum() == 400 and abs(np.median(gap) + np.log10(1.2)) < 0.005, (nb.sum(), np.median(gap))
