"""python -m satrestore --out DIR [--mode notrain] <_cal.fits ...>   (one filter and readout per run; _uncal, _ramp, _rateints
or _rate files next to each _cal file)."""
import argparse
from .restore import run

p = argparse.ArgumentParser(prog='satrestore', description=__doc__)
p.add_argument('cal', nargs='+')
p.add_argument('--out', required=True)
p.add_argument('--mode', default='notrain', choices=['notrain', 'train'])
p.add_argument('--psf-cache', default=None, help='folder of cached PSF models (default: OUT/psf_cache)')
a = p.parse_args()
run(a.cal, a.out, a.mode, psf_cache=a.psf_cache)
