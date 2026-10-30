# satrestore (test package)

Restores saturated pixels in JWST imaging, from the methods of the W51 saturation recovery (`../py`, `production/`).

Two versions:
- `--mode notrain`: methods measured on the dataset itself. MIRI: level-1 correction (one good group), first group
  linearised (pixels flagged only as neighbours of saturated ones: the dataset's raw -> linearised map of group 0, extended
  to saturation by the MIRI reference map of the same CRDS linearity file in `satrestore/data/`: 0032, 0038, 0040; FASTR1
  full frame only), response curve
  (pixels saturated in their first group, and the other flagged pixels from their groups at the saturation level; one curve
  each for even and odd detector rows where both have enough calibration pixels), direct PSF fit (STPSF) for compact regions
  the curve leaves empty, where the fit passes a quality cut.
  Each method reports a self-test in `summary.json`.
- `--mode train`: the same, with the PSF values corrected by a star predictor (gradient-boosted trees) trained on the
  program's own unsaturated stars with their brightest pixels hidden; cross-validated by star.

Usage: `python -m satrestore --out DIR <_cal.fits ...>`, one filter and readout per run, with the `_uncal`, `_ramp` and
`_rateints` (or `_rate`) files next to each `_cal` file. Output: `DIR/<exposure>_satrestore.fits` (SCI in MJy/sr, METHOD
index into the header table, ERR in dex) and `DIR/summary.json`.

Tests: `python -m pytest tests` (or call the test functions directly where pytest is not installed).
