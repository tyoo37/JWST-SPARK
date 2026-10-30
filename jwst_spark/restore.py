"""Run the restoration on the exposures of one filter and readout, and write a sidecar file per exposure.

Mode 'notrain': level-1 correction, first group linearised (pixels flagged only as neighbours of saturated ones: the raw ->
linearised map of group 0, measured on the dataset and extended by a MIRI reference map), response curve (pixels saturated in their first group, and the other flagged pixels from their
own groups at the saturation level), and a direct PSF fit for compact regions the curve leaves empty. Everything is measured on the dataset itself, and each method's
self-test is reported; where the curve and the PSF fit both give a value, their agreement is reported too.
Mode 'train': the same, with the PSF values corrected by the star predictor (starmodel) trained on the program's own unsaturated
stars with hidden cores; with too few such stars it reports so and keeps the direct fit. Mode 'train' (the star predictor trained
on the program's own PSF fits) is not written yet.

Sidecar <out>/<exposure>_spark.fits: SCI (MJy/sr: the pipeline value, or the restored one), METHOD (index into the header
table METHOD<k>), ERR (dex, restored pixels), and the run's self-test summary in summary.json.
"""
import json
import os
import warnings
import numpy as np
from astropy.io import fits
from scipy import ndimage
from . import exposure, level1, curve, psf, starmodel

METHODS = ['pipeline', 'level-1 corrected', 'first group linearised', 'response curve', 'not restored', 'PSF fit', 'PSF fit, trained correction',
           'not restored: detector defect']
DEFECT = 0.9                   # saturated in its first group in at least this share of all integrations: a detector defect (hot
                               # pixel clusters at fixed detector positions in brick, cloudc and ngc6334), not saturated sky
NPSF_TEST = 10                 # exposures used for the PSF self-test
MAX_ERR = 0.1                  # the curve restores only up to the last q bin whose self-test half-width is at most this (dex):
                               # beyond it the curve is flat, and in ngc6334 the values near the ceiling came out noisy and striped
                               # (which rows fell just under the top depended on the even/odd row offset)
QBINS = np.array([0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.6])     # log10 q of the first used group, for the error table
MIN_ROWS = 20_000              # calibrator rows each row parity needs for a curve of its own; fewer: one curve for both (cloudc,
                               # 15,626 rows in all, ran so); fewer than MIN_ROWS // 2 in all: no curve (untested below 15,626)
REF_DIR = os.path.join(os.path.dirname(__file__), 'data')   # raw -> linearised group 0 maps, MIRI FASTR1 full frame, one per CRDS
                               # linearity reference (miri_group0_map_<reference>.csv); used beyond a dataset's own range
REF_AGREE = 0.003              # dex: the reference map is used only if it matches the dataset's own map this well where both
                               # are sampled (brick, cloudc against ngc6334: 0.0015 or better)
NMIN_L1 = 2000                 # pixels with 2-4 good groups needed for an exposure's own level-1 fit; fewer: the pooled fit
hw = lambda v: float((np.percentile(v, 84) - np.percentile(v, 16)) / 2)


def lin0_map(raw0, lin0):
    """Raw group-0 DN -> linearised group 0, binned medians over pixels whose group 0 is good (the pipeline does not linearise a
    flagged group). Use it only inside its sampled range: near saturation the linearity correction grows steeply (+12,600 DN at
    raw 40,000, +22,500 at 50,000, +30,800 at 56,000), and holding the last sampled value there left those pixels 0.3-0.6 dex too low."""
    edges = np.arange(0, curve.SAT_DN + 500, 500.)
    k = np.digitize(raw0, edges) - 1
    med = np.array([np.median(lin0[k == i]) if (k == i).sum() > 30 else np.nan for i in range(len(edges))])
    ok = np.isfinite(med)
    return edges[ok] + 250, med[ok]


def lin0_selftest(raw0, lin0, b, rng):
    """Half the sampled pixels build the map, the other half predict their linearised group 0 from the raw one: error (dex) of
    the charge lin0 - b that the first-group rate uses, over all test pixels and over raw group 0 >= 40,000 DN (where the
    flagged neighbours sit and the linearity correction is steep)."""
    k = rng.permutation(len(raw0))
    te = k[1::2]
    pred = np.interp(raw0[te], *lin0_map(raw0[k[::2]], lin0[k[::2]]), left=np.nan, right=np.nan)
    with np.errstate(all='ignore'):
        d = np.log10((pred - b) / (lin0[te] - b))
    ok = np.isfinite(d) & (lin0[te] - b > 1000)
    top = ok & (raw0[te] >= 40_000)
    return dict(n=int(ok.sum()), median=float(np.median(d[ok])), half_width=hw(d[ok]), n_bright=int(top.sum()),
                median_bright=float(np.median(d[top])) if top.sum() >= 20 else np.nan,
                half_width_bright=hw(d[top]) if top.sum() >= 20 else np.nan,
                raw0_max=float(lin0_map(raw0, lin0)[0].max()))


def error_table(st):
    """Bias and half-width of the curve self-test per q bin, and the error used for restored pixels (both in quadrature)."""
    st = st[:, :2]
    rows = []
    for lo, hi in zip(QBINS[:-1], QBINS[1:]):
        d = st[(st[:, 0] >= lo) & (st[:, 0] < hi), 1]
        if len(d) >= 100:
            rows.append(dict(log_q_lo=lo, log_q_hi=hi, n=len(d), median=float(np.median(d)), half_width=hw(d),
                             err=float(np.hypot(np.median(d), hw(d)))))
    return rows


def run(cal_paths, out_dir, mode='notrain', seed=0, psf_cache=None, log=print):
    assert mode in ('notrain', 'train'), mode
    os.makedirs(out_dir, exist_ok=True)
    warnings.simplefilter('ignore', RuntimeWarning)       # empty slices in nan-reductions over integrations
    rng = np.random.default_rng(seed)
    cal_paths = sorted(cal_paths)
    n = len(cal_paths)
    census, l1data, cal_parts, tgt_parts, ratios, m_raw, m_lin, lin_refs = [], [], [], [], [], [], [], set()
    lev0 = 0
    ro0 = None
    # pass 1: per exposure, the level-1 fit, the calibrators and targets of the curve, and the conversion to MJy/sr
    for k, p in enumerate(cal_paths):
        e = exposure.load(p)
        ro = e['ro']
        ro0 = ro0 or ro
        assert (ro.instrument, ro.filter, ro.ngroups, ro.nints, ro.readpatt) == (ro0.instrument, ro0.filter, ro0.ngroups, ro0.nints, ro0.readpatt), \
            f'{e["name"]}: one filter and readout per run ({ro} vs {ro0})'
        assert ro.instrument == 'MIRI', 'the response curve and level-1 correction are MIRI methods'
        lin_refs.add(str(e['header'].get('R_LINEAR', 'unknown')).replace('crds://', '').replace('.fits', ''))
        l1data += [level1.samples(e, i, rng) for i in range(ro.nints)]      # one entry per unit (exposure x integration)
        cal_parts.append(curve.calibrators(e, k, e['rates'], rng))   # level-1 rows corrected below, once (a, b) are known
        tgt_parts.append(curve.targets(e, k))
        ratios.append(exposure.flat_ratio(e))
        lev0 = lev0 + (e['lev'] == 0).sum(0)
        g = (e['lev'] >= 1) & ((e['pdq'] & curve.NORESTORE) == 0)[None] & ~e['jump']
        m_raw.append([]); m_lin.append([])
        for i in range(ro.nints):
            r0 = e['raw'][i, 0][g[i]]
            # a random sample, plus every bright pixel: the map must reach as close to saturation as the data allow
            s = np.union1d(rng.choice(len(r0), min(len(r0), 20_000), replace=False), np.nonzero(r0 >= 30_000)[0])
            m_raw[-1].append(r0[s]); m_lin[-1].append(e['lin0'][i][g[i]][s])
        census.append(dict(exposure=e['name'], nints=ro.nints, level0=int((e['lev'] == 0).sum()),
                           level0_flagged_neighbour=int(((e['lev'] == 0) & (e['raw'][:, 0] < curve.SAT_DN)).sum()),
                           level1=int((e['lev'] == 1).sum()), no_pipeline_value=int(np.isnan(e['sci']).sum())))
        log(f'{e["name"]}: level 0 {census[-1]["level0"]}, level 1 {census[-1]["level1"]}, level-1 fit pixels '
            f'{[len(d[0]) for d in l1data[-ro.nints:]]}', flush=True)
    T, shape, NI = ro0.tgroup, ratios[0].shape, ro0.nints
    nu = n * NI                                       # units: unit k x NI + i is integration i of exposure k
    # plus a 2-px margin: the rim of a defect cluster saturates in fewer integrations and is no sky either
    defect = ndimage.binary_dilation(lev0 >= DEFECT * nu, iterations=2) if n >= 4 else np.zeros(lev0.shape, bool)
    log(f'detector defects (saturated in their first group in >= {DEFECT:.0%} of {nu} integrations): {int(defect.sum())} pixels', flush=True)

    # level-1 correction: (a, b) per unit, pooled over the exposures (per integration) where a unit has too few pixels
    AB, l1tests = np.zeros((nu, 2)), []
    for i in range(NI):
        Xp, yp = np.concatenate([l1data[u][0] for u in range(i, nu, NI)]), np.concatenate([l1data[u][1] for u in range(i, nu, NI)])
        pooled = level1.fit(Xp, yp)
        for u in range(i, nu, NI):
            X, y = l1data[u]
            own = len(X) >= NMIN_L1
            AB[u] = level1.fit(X, y) if own else pooled
            l1tests.append(dict(exposure=os.path.basename(cal_paths[u // NI])[:-9], integration=i, a=AB[u, 0], b=AB[u, 1],
                                pooled=not own, **(level1.selftest(X, y, rng) if own else {})))
        if not any(not r['pooled'] for r in l1tests if r['integration'] == i):
            l1tests.append(dict(exposure='all (pooled)', integration=i, a=pooled[0], b=pooled[1], pooled=True, **level1.selftest(Xp, yp, rng)))
    del l1data
    inv_flat = exposure.inverse_flat(ratios)
    del ratios

    # the response curve, its offsets and its self-test, separately for even and odd detector rows (each with MIN_ROWS calibrator
    # rows; else one curve): MIRI's even rows saturate
    # differently (ngc6334: about a common curve, an even-row pixel's residual spreads over -2,600..+650 DN and does not repeat
    # between exposures, an odd-row one stays within +-100 DN; a common curve left even-row values 0.025 dex high, odd-row ones
    # 0.027 dex low). Targets are restored from their groups at the saturation level: a pixel flagged only as a neighbour of a
    # saturated one has a valid group 0 that the pipeline did not linearise, and its later groups saturate (almost always from
    # group 1), like the level-1 calibrators of the self-test.
    c, t = curve.cat(cal_parts), curve.cat(tgt_parts)
    del cal_parts, tgt_parts
    l1 = c['lev'] == 1
    c['F'][l1] = (c['lin0'][l1] - AB[c['e'][l1], 1]) / (AB[c['e'][l1], 0] * T)
    keep = np.isfinite(c['F']) & (c['F'] > 0)
    c = {key: v[keep] for key, v in c.items()}
    sat = t['dn'] >= curve.SAT_DN
    a, b = AB[t['e'], 0], AB[t['e'], 1]
    lx, q_first, err_curve, q_valid, qthr = (np.full(len(t['x']), np.nan) for _ in range(5))
    curves, npz = {}, {}
    n_par = np.bincount(c['y'] % 2, minlength=2)
    classes = [(0,), (1,)] if n_par.min() >= MIN_ROWS else [(0, 1)] if n_par.sum() >= MIN_ROWS // 2 else []
    log(f'calibrator rows, even / odd detector rows: {n_par[0]} / {n_par[1]}; curves for rows {classes}', flush=True)
    for cls in classes:
        p = ''.join(map(str, cls))
        cp = {key: v[np.isin(c['y'] % 2, cls)] for key, v in c.items()}
        cv, _ = curve.calibrate(cp, np.arange(len(cp['x'])), AB, T, shape, nu)
        st = curve.selftest(cp, AB, T, shape, nu)
        errs = error_table(st)
        qv = max([r['log_q_hi'] for r in errs if r['half_width'] <= MAX_ERR], default=0.0)
        log(f'curve, rows {p}: {len(cp["x"])} calibrator rows, q_thr {cv["qthr"]:.0f} DN, top at q {10 ** cv["lqmax"]:.2f}, restores up to '
            f'q {10 ** qv:.2f}; self-test (log q of first used group, n, median, half-width dex):',
            [(r['log_q_lo'], r['n'], round(r['median'], 4), round(r['half_width'], 4)) for r in errs], flush=True)
        tp = np.isin(t['y'] % 2, cls)
        bias = cv['bias_map'][t['y'][tp], t['x'][tp]] + cv['shift'][t['e'][tp]]
        lx[tp], _, first = curve.invert(t['dn'][tp] - bias[:, None], sat[tp], t['e'][tp], AB, cv)
        qthr[tp], q_valid[tp] = cv['qthr'], qv
        q_first[tp] = np.log10(np.where(first >= 0, (first + a[tp]) * 10 ** lx[tp] + b[tp] / cv['qthr'], np.nan))
        err_curve[tp] = np.interp(q_first[tp], [0.5 * (r['log_q_lo'] + r['log_q_hi']) for r in errs], [r['err'] for r in errs]) if errs else np.nan
        curves[f'rows {p}'] = dict(qthr=cv['qthr'], q_top=float(10 ** cv['lqmax']), delta=cv['delta'].tolist(), selftest=errs,
                                   n_calibrator_rows=int(len(cp['x'])), q_valid=float(10 ** qv))
        npz.update({f'{key}_{p}': v for key, v in dict(R=cv['R'], lqmax=cv['lqmax'], delta=cv['delta'], qthr=cv['qthr'],
                                                         bias_map=cv['bias_map'], shift=cv['shift'], selftest=st).items()})
    np.savez(f'{out_dir}/curve.npz', LQ=curve.LQ, AB=AB, **npz)

    # flagged as a neighbour (group 0 below the saturation level): the first group, linearised by the map inside its sampled range
    # (the level-1 correction then applies); outside it, the curve from the groups at the saturation level
    # (the relation belongs to the detector and the CRDS linearity reference: data reduced with the same reference agree to
    # 0.0015 dex (brick, cloudc, ngc6334: 0040) or +-150 DN (W51 filters: 0032), while 0032 and 0040 differ by up to 21,600 DN;
    # so beyond the dataset's own range the reference map of the same linearity file takes over, if the package has one)
    lin0 = np.full(len(t['x']), np.nan)
    l0tests = []
    lin_ref = lin_refs.pop() if len(lin_refs) == 1 else None
    ref_path = os.path.join(REF_DIR, f'miri_group0_map_{lin_ref}.csv')
    ref = np.loadtxt(ref_path, delimiter=',') if ((ro0.instrument, ro0.readpatt, ro0.subarray) == ('MIRI', 'FASTR1', 'FULL')
                                                  and os.path.exists(ref_path)) else None
    log(f'linearity reference {lin_ref}: reference map {"used beyond the own range" if ref is not None else "none"}', flush=True)
    for i in range(NI):
        r0, l0 = np.concatenate([m[i] for m in m_raw]), np.concatenate([m[i] for m in m_lin])
        xm, ym = lin0_map(r0, l0)
        bi = float(np.median(AB[i::NI, 1]))
        own_to, agree, used = float(xm.max()), np.nan, False
        if ref is not None:
            # the reference must match the dataset's own map where both are sampled (raw >= 20,000 DN), in the charge lin0 - b
            ov = np.interp(xm, ref[:, 0], ref[:, 1], left=np.nan, right=np.nan)
            ok = np.isfinite(ov) & (xm >= 20_000) & (ym - bi > 1000) & (ov - bi > 1000)
            agree = float(np.median(np.log10((ym[ok] - bi) / (ov[ok] - bi)))) if ok.sum() >= 5 else np.nan
            used = bool(abs(agree) <= REF_AGREE)
            if used:
                far = ref[:, 0] > xm.max()
                xm, ym = np.append(xm, ref[far, 0]), np.append(ym, ref[far, 1])
        s = t['int_'] == i
        lin0[s] = np.interp(t['dn'][s, 0], xm, ym, left=np.nan, right=np.nan)
        l0tests.append(dict(integration=i, **lin0_selftest(r0, l0, bi, rng), own_map_to=own_to, map_to=float(xm.max()),
                            reference_agreement_dex=agree, reference_used=used))
    log('first group linearised, map self-test:', l0tests, flush=True)
    g0 = ~sat[:, 0] & np.isfinite(lin0)
    rate = np.where(g0, (lin0 - b) / a / T, 10 ** lx * qthr / T)
    method = np.where(g0, 2, np.where(np.isfinite(lx), 3, 4))
    bad = defect[t['y'], t['x']]
    rate[bad], method[bad] = np.nan, 7
    beyond = (method == 3) & ~(q_first <= q_valid)
    rate[beyond], method[beyond] = np.nan, 4
    log(f'curve: {int(beyond.sum())} target rows beyond the validated range (self-test half-width <= {MAX_ERR} dex)', flush=True)
    l1err = float(np.median([np.hypot(r['median'], r['half_width']) for r in l1tests if 'median' in r]))
    l0err = float(np.hypot(l1err, np.nanmax([np.hypot(r[f'median{s}'], r[f'half_width{s}']) for r in l0tests for s in ('', '_bright')])))
    err = np.where(method == 3, err_curve, np.where(method == 2, l0err, np.nan))

    # PSF: one model per observation date; self-test on unsaturated stars of up to NPSF_TEST exposures
    psf_cache = psf_cache or f'{out_dir}/psf_cache'
    models = {}
    model_of = lambda h: models.setdefault(h['DATE-OBS'], psf.psf_model(h, psf_cache, log=log))
    prow = []
    for kk, p in enumerate(cal_paths if mode == 'train' else cal_paths[::max(1, n // NPSF_TEST)][:NPSF_TEST]):
        with fits.open(p) as h:
            sci0, err0, m = h['SCI'].data.astype(float), h['ERR'].data.astype(float), model_of(h[0].header)
        prow += [(r[0] + 1000 * kk, *r[1:]) for r in psf.selftest(sci0, err0, np.isfinite(sci0), m, rng)]   # star ids unique per run
    prow = np.array(prow).reshape(-1, 3 + len(starmodel.FEATURES))
    psf_test = [dict(hidden=int(nh), n=int((prow[:, 1] == nh).sum()), median=float(np.median(prow[prow[:, 1] == nh, 2])),
                     half_width=hw(prow[prow[:, 1] == nh, 2])) for nh in psf.HIDE if (prow[:, 1] == nh).sum() >= 20]
    psf_err = max([np.hypot(r['median'], r['half_width']) for r in psf_test], default=np.nan)
    log('PSF self-test (pixels hidden, n, median, half-width dex):', [(r['hidden'], r['n'], round(r['median'], 3), round(r['half_width'], 3)) for r in psf_test], flush=True)
    star_model, star_info = starmodel.train(prow) if mode == 'train' else (None, {})
    if mode == 'train':
        star_err = max([np.hypot(r['trained_median'], r['trained_half_width']) for r in star_info.get('selftest', [])], default=psf_err)
        log('star predictor:', star_info, flush=True)

    # pass 2: one sidecar per exposure
    counts, both, both_direct = [], [], []
    for k, p in enumerate(cal_paths):
        e = exposure.load(p)
        conv = e['photmjsr'] * inv_flat
        R = np.where(e['lev'] >= 2, e['rates'], np.nan)
        M = np.zeros(e['lev'].shape, np.int8)
        E = np.full(e['lev'].shape, np.nan, np.float32)
        l1 = e['lev'] == 1
        R[l1], M[l1], E[l1] = level1.correct(e, AB[k * NI:(k + 1) * NI])[l1], 1, l1err
        s = t['k'] == k
        R[t['int_'][s], t['y'][s], t['x'][s]] = rate[s]
        M[t['int_'][s], t['y'][s], t['x'][s]] = method[s]
        E[t['int_'][s], t['y'][s], t['x'][s]] = err[s]
        touched = (e['lev'] <= 1).any(0) & ((e['dq'] & curve.NOFLAT) == 0)
        with np.errstate(all='ignore'):
            new = np.nanmean(np.where(np.isfinite(R) & (R > 0), R, np.nan), 0) * conv
            # the conversion must reproduce the pipeline where it calibrated an untouched pixel
            ok = ~touched & np.isfinite(e['sci']) & (e['sci'] > 0) & (e['rates'] > 0).all(0)
            conv_check = float(np.median(np.abs(np.log10(e['rates'].mean(0)[ok] * conv[ok] / e['sci'][ok]))))
        sci = np.where(touched, new, e['sci']).astype(np.float32)
        used = (e['lev'] <= 1) & np.isfinite(R) & (R > 0)
        meth = np.where(touched, np.where(np.isfinite(new), np.where(used, M, 0).max(0), 4), 0).astype(np.uint8)
        errd = np.where(touched & np.isfinite(new), np.fmax.reduce(np.where(used, E, np.nan), axis=0), np.nan).astype(np.float32)
        # PSF fit to the measured pixels around each compact saturated region: fills what the curve left empty, checks the rest
        pm = model_of(e['header'])
        meth[defect & touched] = 7
        pv, info = psf.restore(sci, e['err'], np.isin(meth, [0, 1]) & np.isfinite(sci) & ~defect, touched & (meth >= 2) & (meth != 7), pm)
        code, perr = 5, psf_err
        if star_model is not None:
            pt = pv.copy()
            for r in info:
                ok = np.isfinite(pt[r['ys'], r['xs']])
                if ok.any():
                    ys, xs = r['ys'][ok], r['xs'][ok]
                    f = psf.features(np.array([r['flux'], r['x'], r['y'], r['bkg']]), ys, xs, pm, r['area'], r['rchi2'])
                    pt[ys, xs] = starmodel.correct(star_model, f, pt[ys, xs])
            cmp = (meth == 3) & (pv > 0) & (sci > 0)
            both_direct.append(np.log10(pv[cmp] / sci[cmp]))
            pv, code, perr = pt, 6, star_err
        cmp = (meth == 3) & (pv > 0) & (sci > 0)
        both.append(np.log10(pv[cmp] / sci[cmp]))
        fill = (meth == 4) & (pv > 0)
        sci[fill], meth[fill], errd[fill] = pv[fill], code, perr
        sci[meth == 7], errd[meth == 7] = np.nan, np.nan
        hdr = fits.Header()
        hdr['EXPOSURE'], hdr['MODE'], hdr['CONVCHK'] = e['name'], mode, (conv_check, 'median |dex| of rate x conv vs pipeline SCI')
        for i, m in enumerate(METHODS):
            hdr[f'METHOD{i}'] = m
        fits.HDUList([fits.PrimaryHDU(header=hdr), fits.ImageHDU(sci, name='SCI', header=fits.Header([('BUNIT', 'MJy/sr')])),
                      fits.ImageHDU(meth, name='METHOD'), fits.ImageHDU(errd, name='ERR', header=fits.Header([('BUNIT', 'dex')]))]
                     ).writeto(f'{out_dir}/{e["name"]}_spark.fits', overwrite=True)
        counts.append(dict(exposure=e['name'], conv_check_dex=conv_check, no_value_before=int(np.isnan(e['sci']).sum()),
                           no_value_after=int(np.isnan(sci).sum()), **{m: int((meth == i).sum()) for i, m in enumerate(METHODS) if i}))
        log(f'{e["name"]}: conversion check {conv_check:.4f} dex; ' + ', '.join(f'{m} {counts[-1][m]}' for m in METHODS[1:]), flush=True)

    agree_of = lambda v: dict(n=int(len(v)), median=float(np.median(v)), half_width=hw(v)) if len(v) >= 20 else {}
    agree = agree_of(np.concatenate(both))
    log('PSF fit against the response curve where both give a value:', agree, flush=True)
    if both_direct:
        star_info['direct_psf_minus_curve'] = agree_of(np.concatenate(both_direct))
        log('  (direct PSF fit, before the trained correction:', star_info['direct_psf_minus_curve'], ')', flush=True)
    summary = dict(mode=mode, readout=vars(ro0), n_exposures=n, census=census, detector_defects=int(defect.sum()), level1=l1tests,
                   curve=dict(curves, beyond_valid=int(beyond.sum())), first_group=dict(map_selftest=l0tests, err=l0err),
                   psf=dict(selftest=psf_test, err=psf_err, psf_minus_curve=agree, rchi2_max=psf.RCHI2_MAX), star_model=star_info, counts=counts)
    json.dump(summary, open(f'{out_dir}/summary.json', 'w'), indent=1)
    return summary
