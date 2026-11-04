# aral-geo-map

Land cover of the Aral Sea and its dry bed from the open Landsat archive,
1984 to 2025. The pipeline turns raw scenes into masks of four classes, water,
salt crust, bare soil and vegetation, on one fixed grid, so that a pixel means
the same piece of ground in every year.

## The question

A spectral index with a fixed threshold (MNDWI, NDVI, BSI) already separates
water from land. The trouble is that the sensor changed underneath the
threshold. Landsat 5 flew until 2011, Landsat 8 started in 2013, and their
radiometry is not the same, so a fixed threshold across 40 years measures the
change of satellite as well as the change of the sea.

The project asks whether a trained segmentation model does better than that
threshold on the hard cases: salt crust against dry soil, and the boundary
between two sensors. The threshold is a fair baseline here, not a weak one
set up to lose, and either answer is worth having.

## Status

Working, and the numbers further down come from running it:

* stages 1 to 4, from catalogue search to patches ready for labelling;
* automatic labels from index thresholds, and a training loop that runs a
  U-Net and Prithvi-EO-2.0-300M-TL on them (`notebooks/aral_train.ipynb`).

Not done yet: hand-drawn labels, and therefore the real comparison. A model
trained on threshold labels learns to copy the threshold, mistakes included,
so the current scores say the loop works and nothing more. The map is not
written either.

## Install

Python 3.12 and the four dependencies in `pyproject.toml`.

```bash
uv sync
```

The package is not installed into the environment, so the stages run from
`src`:

```bash
PYTHONPATH=src .venv/bin/python -m aral.cli <stage>
```

The commands below assume this alias:

```bash
alias aral='PYTHONPATH=src .venv/bin/python -m aral.cli'
```

## Stages

Every stage can be re-run and picks up where it stopped. Over a thousand
scenes something always fails halfway, and a pipeline that cannot resume
never reaches the end.

```bash
# the analysis grid, before anything is downloaded
aral grid

# stage 1: search the catalogue, write the manifest
aral manifest --start 1984 --end 2025

# coverage summary, scenes and cells per year
aral qa

# stage 2: download. 'window' crops to the AOI with range requests,
# 'full' takes whole scenes
aral fetch --year 2019 --mode window --dry-run

# stage 3: scenes to tiles on the fixed grid
aral prepare data/raw/landsat/2019-07-14
```

Patch extraction and the spectral indices are library functions rather than
subcommands, because choosing which patches to label by hand needs eyes on
the images. They are driven from the notebooks.

## The notebooks

`notebooks/aral_colab.ipynb` runs stages 1 to 4 on Colab, where there is disk
and bandwidth to spare. Raw scenes and grid tiles stay on the machine and die
with the session, since they can be rebuilt from the manifest in minutes. Only
the manifest, the QA table and the chosen patches go back to Drive, which is
tens of megabytes.

`notebooks/aral_train.ipynb` takes those patches, labels them with the index
thresholds and trains the two models on them. It splits the data by place
rather than at random: neighbouring patches look alike, so a random split
fills the validation set with near-copies of the training set and the score
then measures memory instead of generalisation. Whole blocks of 4 x 4 patches,
about 30 km, go to one side or the other.

## Checking a change

`tools/run_notebook.py` executes every code cell of a notebook in order,
outside Colab, with `google.colab` stubbed and the `/content` paths redirected
into a scratch directory. The code is loaded through the same zip path the
notebook uses, so the run tests the working tree that is about to be pushed.

```bash
zip -qr <work>/drive/aral_src.zip src pyproject.toml -x '*__pycache__*'
uv run --with matplotlib python tools/run_notebook.py \
    notebooks/aral_colab.ipynb <work>
```

It downloads two 2019 scenes, about 600 MB, and stops at the first failing
cell. Settings can be overridden to keep a CPU run short, which is how the
training notebook is checked:

```bash
... run_notebook.py notebooks/aral_train.ipynb <work> \
    --set INSTALL=False --set MAX_PATCHES=40 --set EPOCHS_UNET=1
```

## What the pipeline measured

The analysis grid, from `aral grid`:

| | |
|---|---|
| CRS | EPSG:32640 (UTM 40N) |
| pixel | 30 m, native Landsat |
| size | 11821 x 16144 px, 355 x 484 km |
| patches | 47 x 64 of 256 px |

The committed manifest covers 1984 to 2025, paths 160 to 162, rows 027 to 030,
June to September. It holds 4559 scenes, of which 4433 are Tier 1 and 3376 of
those are under 20 percent cloud. Those 3376 come from Landsat 4 (57), 5 (965),
7 (1274), 8 (850) and 9 (230). The manifest predates the move to a May to
September season, so rebuilding it against the current config returns about a
thousand more scenes.

Coverage is the real constraint, and it is worse than the scene count
suggests. Requiring all 12 cells to have at least 3 clear scenes, only 22 of
the 42 years can produce a composite of the whole sea. Nine years miss a cell
completely: 1984, 1985, 1988, 1990, 1991, 1995, 1997, 1998 and 2003. Relaxing
the cloud filter barely helps, because in those years those scenes were never
acquired at all. The early part of the series is weaker than the late part,
and that is a property of the archive rather than of the search.

## Design notes

Things that took time to work out, written down so that nobody has to find
them twice.

**Fill is not water.** Landsat scenes sit rotated inside their bounding box
and the corners are nodata zeros. Converted to reflectance they become a dark
surface that every water index happily calls water. Validity is therefore
decided on the raw DN, before any arithmetic.

**Resample with nearest neighbour.** Bilinear invents values and blends valid
pixels with invalid ones. Nearest keeps every number a measurement.

**CFMask over-flags the coastline, and my fix for it was worse.** On some
clear dates QA_PIXEL marks a strip a few hundred metres wide along the shore
as cloud, dilated cloud or shadow, which is exactly where the classes change.
In one 2019 cell that happened on two of four clear dates, at 9 to 21 times
the rate measured away from the coast, and on a day with 1.2 percent scene
cloud it covered a fifth of the coastline. The 1985 strip is demonstrably
false: its surface temperature equals the average of water and land, so
nothing cooled it, its shadows are brighter than open water, and it follows
the inner lagoon. I wrote a thermal rule, that a cloud has to be colder than
clear water, with its own buffer and shadows only near confirmed cloud, and it
cleared 79 percent of the strip. I still dropped it. On 2019-05-23 it released
real thin cloud, and a persistence test, on the idea that cloud moves between
dates while salt does not, found the freed pixels flagged on about 30 percent
of dates, the same rate as the cloud the rule kept. So the flags stay as
published. The seasonal composite absorbs the strip because it moves between
dates, and years with a single usable date, such as 1985, keep the hole.

**QA_PIXEL bit 6 does not mean "not cloud".** It means clear *land*, and it is
never set over water, so using it as a validity mask deletes the sea. The code
rejects the specific bits instead, which is what `QA_REJECT_BITS` is for.
Shadow is only flagged at high confidence.

**Composite per cell, then mosaic.** Adjacent paths are imaged about a week
apart, so mosaicking raw scenes splices dates together. Build a seasonal
median per path and row first, then mosaic the composites.

**Reproject before mosaicking, even within one date.** Row 027 sits in UTM
zone 41 while rows 028 to 030 sit in zone 40, so a single-day strip already
spans two zones.

**Sign one asset and reuse the token.** Planetary Computer issues SAS tokens
per container rather than per file. Asking for one per asset earns an HTTP 429
within a few dozen files.

**Read statistics through the overviews.** A median over four million pixels
matches one over fifty-five million, and the full read costs gigabytes of RAM
per tile.

**The season is May to September.** Over this area the sun sits at 62.6
degrees in May and 60.6 in July, but only 33.0 in October. May costs nothing,
while October would mix two different illuminations into one median.

The manifest records the parameters it was built with and refuses to be reused
once they no longer match `config.py`. A stale manifest does not announce
itself. It simply lacks scenes the code expects, and the failure turns up
stages later as a missing file.

## Layout

```
src/aral/
  config.py     every constant, in one place
  grid.py       the fixed analysis grid
  manifest.py   stage 1, catalogue search
  pc.py         Planetary Computer tokens
  fetch.py      stage 2, download
  prepare.py    stage 3, scene to grid tile
  indices.py    MNDWI, NDVI, BSI
  patches.py    stage 4, tile to labelling patches
  labels.py     weak labels and model input
  qa.py         one row of statistics per tile
notebooks/      the Colab pipeline and the training run
tools/          the notebook test harness
data/
  raw/          downloaded scenes, reproducible, never committed
  interim/      the manifest and the grid tiles
  work/         composites and patches
reports/        QA tables and the patch index
```

Only the manifest is committed. Everything else can be rebuilt from it in
minutes, so it is not worth storing anywhere.

## Data

Landsat Collection 2 Level-2 surface reflectance, Tier 1 only, courtesy of the
U.S. Geological Survey. The scenes are fetched from Microsoft Planetary
Computer, which serves the same archive as cloud-optimised GeoTIFFs and allows
range requests.

Asset keys are sensor neutral (`green`, `nir08`, `swir16`, `qa_pixel`) because
TM, ETM+ and OLI number their bands differently. On TM band 1 is blue and band
5 is SWIR1, while on OLI the numbering shifts by one, so the code always works
from the names.

## Scope

The model classifies what is visible in a scene: an image goes in and a mask
of the same size comes out. It does not invent missing dates or fill in
imagery that was never acquired. Temporal dynamics come from running inference
year by year, and cloud gaps are handled by median composites.

The project measures class areas from imagery. It does not measure soil
salinity, crop yields, or the causes of the change.
