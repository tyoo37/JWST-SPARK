# JWST-SPARK

**JWST-SPARK: Saturated Pixel Automated Recovery Kit.** It restores saturated pixels in JWST MIRI imaging from the detector ramps that the JWST pipeline already writes. Every method calibrates itself on the dataset being restored and reports a self-test, so you can see how far each restored value can be trusted.

Status: test version (0.1.0). It covers the MIRI imager and, in a first version, NIRCam imaging. It has been run on ten MIRI datasets of seven programs (F770W, F1500W, F2100W and F2550W, FASTR1 full frame, 5 to 40 groups, 1 to 15 integrations) and six NIRCam datasets (BRIGHT1, BRIGHT2, SHALLOW2, SHALLOW4; 2 to 7 groups; short and long wavelength). Its thresholds come from those data and from the W51 program it was developed for (JWST 6151).

## What it restores

For each pixel with no pipeline value, or a biased one, the package picks the first method that applies:

| Method | Pixels | How |
|---|---|---|
| level-1 corrected | only the first group below saturation | the pipeline's one-group value misses the first-group pedestal; corrected with `lin0 = a X + b`, fitted per exposure and integration on pixels with 2-4 good groups (pooled over the exposures when fewer than 2,000 such pixels). It needs only the linearised first group, so it also serves ramps the pipeline left without a rate: one-group ramps, and 2-4-group ramps in jwst 1.15, which drops MIRI's first group |
| first group linearised | flagged as saturated only because a neighbour saturated | the pipeline does not linearise a flagged group; the package maps the raw first group to its linearised value with the dataset's own map, extended to saturation by a reference map of the same CRDS linearity file, then applies the level-1 correction |
| response curve | saturated in their first group | how the raw signal keeps growing past saturation, measured on the dataset's own pixels of known brightness (groups 2-4, or later groups when those hold too few), then read backwards from the pixel's saturated groups. One curve per detector row parity where both parities have at least 20,000 calibration rows. Used only up to the charge where its self-test stays within 0.1 dex |
| PSF fit | compact saturated cores the curve leaves empty | the STPSF model for the observation date fitted to the measured pixels around the core; refused when the source does not look point-like (reduced chi-square above 5) or its surroundings are not measured. The fit uses the pixels within 4 of the core and the model integrated over each detector pixel. Used only when every bin of the PSF self-test (1-15 hidden pixels) and its agreement with the curve are within 0.05 dex; where no curve value exists to compare with, the self-test must be within 0.02 dex |
| not restored | everything else | detector defects (saturated at the same detector position in at least 90% of integrations, plus a 2-pixel margin) and pixels outside the flat field stay empty |

Two versions:
- `--mode notrain` (default): everything above, measured on the dataset itself.
- `--mode train`: the PSF values are also corrected by a star predictor (gradient-boosted trees) trained on the program's own unsaturated stars with their brightest pixels hidden. In the tests it halves the error in its own cross-validation, but shifts real saturated stars by about 0.09 dex against the response curve, so `notrain` is the default.

## Install

```
git clone https://github.com/tyoo37/JWST-SPARK.git
cd JWST-SPARK
pip install -e .
```

Requires Python 3.10 or later, numpy, scipy, astropy, photutils, scikit-learn and stpsf; NIRCam also needs crds (the saturation reference named in each file is fetched into the CRDS cache, `CRDS_PATH`). The PSF fit needs the STPSF data files (`STPSF_PATH`). STPSF downloads the telescope state for each observation date, so it needs network access the first time; the models are cached.

## Use

One filter, one readout and one detector per run. Each `_cal.fits` needs its `_uncal.fits`, `_ramp.fits` and `_rateints.fits` (or `_rate.fits`) next to it, as written by the JWST stage 1 and stage 2 pipelines. Tested with files from jwst 1.15 and 1.21.

```
jwst-spark --out restored/ path/to/jw*_mirimage_cal.fits
jwst-spark --out restored_train/ --mode train --psf-cache psf_cache/ path/to/jw*_mirimage_cal.fits
```

`python -m jwst_spark` works the same way.

## Output

- `<out>/<exposure>_spark.fits`, on the detector grid of the `_cal` file:
  - `SCI`: MJy/sr, the pipeline value or the restored one. Pixels the pipeline flags DO_NOT_USE stay empty unless restored (jwst 1.21 already leaves them empty; jwst 1.15 keeps dimmed values in the coronagraph regions). The conversion reproduces the pipeline's own calibration; the run checks this on every exposure.
  - `METHOD`: which method gave the value (index into the header table `METHOD0`, `METHOD1`, ...).
  - `ERR`: the uncertainty of restored values in dex, from the method's self-test.
- `<out>/summary.json`: the readout, the level-1 fits and self-tests, the curve self-test by charge and row parity, the first-group map self-test and reference-map agreement, the PSF self-test and its agreement with the curve, and the counts per method for every exposure.
- `<out>/curve.npz`: the response curves and detector offset maps.

## Self-tests

Each method is tested on the dataset before its values are used:
- level-1 correction: half the calibration pixels fit `(a, b)`, the other half predict their measured rate;
- first-group map: half the pixels build it, the other half test it. A reference map is used only if it agrees with the dataset's own map to 0.003 dex where both are sampled;
- response curve: exposures are held out in four folds, and their pixels of known brightness are restored from their saturated groups alone;
- PSF fit: unsaturated stars get their 1-63 brightest pixels hidden and restored.

On the ten test datasets (half-widths of the self-tests):
- response curve, first charge bin above saturation: 0.006-0.047 dex; even detector rows are 1.1 to 3 times wider than odd rows;
- first-group map at high signal: 0.0005 dex or better;
- level-1 correction: 0.007-0.045 dex.

PSF fit against the curve: within 0.01 dex at F2550W and F2100W, +0.041 dex at F1500W, and 0.13-0.20 dex too bright at F770W (the brighter-fatter effect broadens bright MIRI stars; STPSF does not model it), so the gate keeps PSF fills off at F770W.

A level-1 fit is used only if its self-test on the brightest 10% of its pixels is within 0.05 dex; else the fit pooled over the exposures, and for NIRCam then the pipeline's own one-group rate. The fit ignores points more than 5 robust sigma from a first line through binned medians (one unflagged NIRCam pixel with an impossible rate had pulled a fit to a = 0.002).

## Reference maps

`jwst_spark/data/miri_group0_map_<linearity reference>.csv` holds the raw-to-linearised relation of the first group for MIRI FASTR1 full-frame data, one file per CRDS linearity reference (`jwst_miri_linearity_0032`, `_0038`, `_0040`). The relation belongs to the linearity reference: datasets reduced with the same file agree to 0.0003 dex or better, while `0032` and `0040` differ by up to 21,600 DN near saturation. Data reduced with another linearity file (one test set used `0039`), readout or subarray use their own map only.

## Limits

- NIRCam, first version: per-pixel saturation levels from CRDS, the level-1 correction (it moves NIRCam values by 0.015 dex or less, since NIRCam's first group has no pedestal; ramps of fewer than 3 groups keep the pipeline rate), the neighbour-flagged level-1 error, and the PSF fill. No response curve (NIRCam's raw signal stops rising soon after saturation) and no group-0 map (raw group 0 carries per-pixel bias). Not yet used: the first group of pixels flagged by a saturated neighbour in group 0, and ZEROFRAME values the pipeline zeroed. Pipeline ZEROFRAME rates are kept.
- Reference maps exist for FASTR1 full frame and three linearity references.
- The response curve is tested only up to the charge its calibration pixels reach; brighter pixels stay empty. When the calibration pixels do not reach saturation at all (one F770W test set), the curve restores nothing.
- At F770W the PSF fit is biased high against the curve (brighter-fatter effect); PSF fills stay off there.
- Most of what stays empty in MIRI images (77% in the test sets) has every group above the top of the response curve and lies in regions too large for the PSF fit. Another filter of the same sky, scaled by the local colour, reached 86% of that area at 0.04-0.10 dex in tests; the package does not do this yet.
- Pixels the pipeline flags only because a neighbour saturated in group 1 look like level-1 pixels and keep the level-1 correction. Neighbours flagged from group 3 or 4 follow the level-1 relation to 0.00-0.04 dex against their own groups taken before the neighbour saturated. The later groups of the group-1 neighbours carry 0.1-0.2 dex of extra charge from the saturating pixel. Their first increment still reads 0.04-0.12 dex above the level-1 value; part of that is the same charge. The run measures this gap on each dataset (`summary.json`, `level1_neighbour_flagged`) and adds it to these pixels' `ERR`.
- Even detector rows near saturation scatter more than odd rows, and the scatter does not repeat between exposures. Their larger error is carried in `ERR`. Level-1 corrected pixels keep an even/odd pattern of about ±0.01 dex.

## Tests

```
python -m pytest tests
```

Synthetic checks of the level-1 fit, the curve inversion, the PSF fit and the star predictor.

## Origin

Built from the saturated-pixel recovery of the JWST W51 program (6151), then generalised and tested on MIRI data of programs 1266, 2092, 2221, 2526, 3571, 5365 and 7213.
