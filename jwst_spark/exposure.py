"""One exposure: the readout setup from its header and the arrays the methods need, from the pipeline's _uncal, _ramp,
_rateints/_rate and _cal files next to each other."""
import os
from dataclasses import dataclass
import numpy as np
from astropy.io import fits
from .curve import SAT_DN

SATURATED, JUMP = 2, 4          # GROUPDQ bits (jwst.datamodels.dqflags)
DO_NOT_USE = 1                  # _cal DQ bit


@dataclass
class Readout:
    instrument: str
    detector: str
    filter: str
    readpatt: str
    ngroups: int
    nints: int
    tgroup: float
    subarray: str
    nframes: int = 1
    groupgap: int = 0
    tframe: float = 0.0

    @classmethod
    def from_header(cls, h):
        return cls(h['INSTRUME'], h['DETECTOR'], h['FILTER'], h['READPATT'], int(h['NGROUPS']), int(h['NINTS']),
                   float(h['TGROUP']), h['SUBARRAY'], int(h.get('NFRAMES', 1)), int(h.get('GROUPGAP', 0)), float(h.get('TFRAME', 0.0)))


_SAT = {}


def saturation_map(header, shape):
    """Raw DN from which the pipeline flags a group as saturated, per pixel. MIRI imager: uniform (curve.SAT_DN). NIRCam: the CRDS
    saturation reference named in R_SATURA (median 52,600-56,100 DN by detector), fetched into the CRDS cache (CRDS_PATH, default
    ~/crds_cache) on first use, cut to the subarray."""
    if header['INSTRUME'] == 'MIRI':
        return np.full(shape, SAT_DN, np.float32)
    name = header['R_SATURA'].replace('crds://', '')
    if name not in _SAT:
        os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
        import crds
        from crds.client import api
        api.dump_files(header.get('CRDS_CTX'), [name])
        _SAT[name] = fits.getdata(crds.locate_file(name, 'jwst'), 'SCI').astype(np.float32)
    y0, x0 = int(header.get('SUBSTRT2', 1)) - 1, int(header.get('SUBSTRT1', 1)) - 1
    return _SAT[name][y0:y0 + shape[0], x0:x0 + shape[1]]


def load(cal_path):
    """Arrays of one exposure (integration axis first where per integration):
    raw (nints, ngroups, ny, nx) uncal DN; lev (nints, ny, nx) first saturated group (ngroups if none); jump (nints, ny, nx);
    lin0 (nints, ny, nx) linearised group 0; pdq PIXELDQ; rates (nints, ny, nx) DN/s; sci, dq the _cal SCI and DQ; photmjsr;
    satdn (ny, nx) the raw saturation threshold."""
    base = cal_path[:-len('_cal.fits')]
    with fits.open(cal_path) as h:
        header, ro = h[0].header, Readout.from_header(h[0].header)
        sci, err, dq, photmjsr = h['SCI'].data.astype(np.float32), h['ERR'].data, h['DQ'].data, float(h['SCI'].header['PHOTMJSR'])
    sci[(dq & DO_NOT_USE) > 0] = np.nan      # jwst 1.15 keeps values there (coronagraph regions); 1.21 gives NaN
    raw = fits.getdata(f'{base}_uncal.fits', 'SCI').astype(np.float32)
    with fits.open(f'{base}_ramp.fits') as h:
        gdq, lin0, pdq = h['GROUPDQ'].data, h['SCI'].data[:, 0].astype(np.float32), h['PIXELDQ'].data
    rates = fits.getdata(f'{base}_rateints.fits', 'SCI') if os.path.exists(f'{base}_rateints.fits') \
        else fits.getdata(f'{base}_rate.fits', 'SCI')[None]
    assert raw.shape[:2] == (ro.nints, ro.ngroups) and rates.shape[0] == ro.nints, (raw.shape, rates.shape, ro)
    sat = (gdq & SATURATED) > 0
    lev = np.where(sat.any(1), sat.argmax(1), ro.ngroups).astype(np.int16)
    return dict(name=os.path.basename(base), path=cal_path, header=header, ro=ro, raw=raw, err=err, lev=lev, jump=((gdq & JUMP) > 0).any(1),
                lin0=lin0, pdq=pdq, rates=rates.astype(np.float32), sci=sci, dq=dq, photmjsr=photmjsr,
                satdn=saturation_map(header, sci.shape))


def flat_ratio(e):
    """SCI / (rate x PHOTMJSR) where the pipeline calibrated the pixel: 1 / flat, one exposure's estimate."""
    with np.errstate(all='ignore'):
        r = e['sci'] / (e['rates'].mean(0) * e['photmjsr'])
    return np.where(np.isfinite(r) & (r > 0), r, np.nan).astype(np.float32)


def inverse_flat(ratios):
    """1 / flat per detector pixel: the median of flat_ratio over the exposures (a pixel saturated in one dither is usually
    measured in another); pixels never calibrated take the mean of their 5x5 neighbourhood. DN/s x PHOTMJSR x this = MJy/sr."""
    from scipy import ndimage
    inv = np.nanmedian(np.array(ratios), 0)
    ok = np.isfinite(inv)
    with np.errstate(all='ignore'):
        near = ndimage.uniform_filter(np.where(ok, inv, 0.), 5) / ndimage.uniform_filter(ok.astype(float), 5)
    return np.where(ok, inv, near).astype(np.float32)
