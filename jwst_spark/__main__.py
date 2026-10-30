"""jwst-spark --out DIR [--mode notrain|train] [--psf-cache DIR] <_cal.fits ...>   (or python -m jwst_spark; one filter and
readout per run; the _uncal, _ramp and _rateints or _rate files next to each _cal file)."""
import argparse
from .restore import run


def main(argv=None):
    p = argparse.ArgumentParser(prog='jwst-spark', description=__doc__)
    p.add_argument('cal', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--mode', default='notrain', choices=['notrain', 'train'])
    p.add_argument('--psf-cache', default=None, help='folder of cached PSF models (default: OUT/psf_cache)')
    a = p.parse_args(argv)
    run(a.cal, a.out, a.mode, psf_cache=a.psf_cache)


if __name__ == '__main__':
    main()
