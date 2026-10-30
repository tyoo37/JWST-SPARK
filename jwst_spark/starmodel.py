"""Star predictor (the training version). It learns the systematic error of the direct PSF fit from the program's own unsaturated
stars with their brightest pixels hidden (true values known), and corrects the PSF values of the real saturated stars with it:
gradient-boosted trees on per-pixel features (psf.features). Self-test: cross-validation by star, against the direct fit.
Real saturated stars are brighter and hide more pixels than the training stars; trees are flat beyond their training range, so
the correction there is the one learned at the edge of that range."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

FEATURES = ['log_pred', 'r', 'n_hidden', 'log_flux', 'bkg_ratio', 'rchi2']
MIN_STARS = 30
hw = lambda v: float((np.percentile(v, 84) - np.percentile(v, 16)) / 2)
new_model = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20)


def train(rows):
    """rows (n, 3 + len(FEATURES)): star, pixels hidden, error (log predicted / true), features. Returns (model or None, info):
    the self-test compares, per hidden count, the direct fit's error with the error after the out-of-fold correction."""
    rows = np.asarray(rows, float).reshape(-1, 3 + len(FEATURES))
    stars = np.unique(rows[:, 0])
    if len(stars) < MIN_STARS:
        return None, dict(n_stars=int(len(stars)), note=f'fewer than {MIN_STARS} stars: the direct PSF fit is used')
    X, y, g = rows[:, 3:], rows[:, 2], rows[:, 0]
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        oof[te] = new_model().fit(X[tr], y[tr]).predict(X[te])
    test = []
    for nh in np.unique(rows[:, 1]):
        s = rows[:, 1] == nh
        if s.sum() >= 20:
            test.append(dict(hidden=int(nh), n=int(s.sum()), direct_median=float(np.median(y[s])), direct_half_width=hw(y[s]),
                             trained_median=float(np.median(y[s] - oof[s])), trained_half_width=hw(y[s] - oof[s])))
    return new_model().fit(X, y), dict(n_stars=int(len(stars)), selftest=test)


def correct(model, feats, values):
    """PSF values corrected by the learned error (error = log predicted / true)."""
    return values / 10 ** model.predict(feats)
