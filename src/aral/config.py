"""Single source of truth for every constant in the pipeline.

Nothing below is allowed to appear as a literal anywhere else: a threshold that
lives in two places will eventually differ in two places.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
WORK = ROOT / "data" / "work"
REPORTS = ROOT / "reports"

# --- area of interest -------------------------------------------------------
# Aral Sea, its dry bed and the Amu Darya delta (lon/lat, WGS84).
AOI_BBOX = (57.8, 43.0, 62.0, 47.2)

# --- analysis grid ----------------------------------------------------------
# One fixed grid for the whole 40-year archive. Every scene is resampled onto
# it, so a pixel at (row, col) means the same patch of ground in 1985 and 2025.
# Without this, patches from different dates are not comparable.
TARGET_CRS = "EPSG:32640"          # UTM zone 40N; most of the sea sits here
PIXEL = 30.0                       # metres, native Landsat resolution
PATCH = 256                        # model input size
TILE_M = PIXEL * PATCH             # 7680 m per patch

# UTM is conformal, not equal-area: near the eastern edge of the AOI it
# overstates area by ~0.4%. Areas are therefore measured in this instead.
AREA_CRS = ("+proj=aea +lat_1=43 +lat_2=47 +lat_0=45 +lon_0=60 "
            "+datum=WGS84 +units=m +no_defs")

# --- Landsat Collection 2 Level-2 -------------------------------------------
SR_SCALE = 2.75e-05                # DN -> surface reflectance
SR_OFFSET = -0.2
SR_VALID = (7273, 43636)           # the range USGS calls valid
# Water reflects almost nothing in SWIR, and atmospheric correction routinely
# overshoots by a fraction of a percent, so a genuine open-water pixel lands
# just below zero. Treating that as invalid punches single-pixel holes through
# the one class the project cares about most. Small excursions are clamped to
# the limit; only larger ones are rejected.
SR_CLAMP_TOL = 0.02                # reflectance units

# QA_PIXEL bit flags (Collection 2, all sensors)
QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW = 0, 1, 2, 3, 4
# CFMask's flags are used as they are, although along shorelines it over-flags:
# on some clear dates a coastal strip a few hundred metres wide is marked
# cloud, buffer or shadow (2019, cell 161/028: 9-21x the rate away from the
# coast, up to a fifth of the shoreline on a 1%-cloud day). A thermal test -
# "a cloud must be colder than open water" - cleared that strip on 1985-07-28
# but, checked across dates, also released real thin clouds over the dried
# seabed in May 2019. Not adopted. The strip is absent on other dates, so the
# seasonal composite is what fills it; years with a single date keep it.
QA_REJECT_BITS = (QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW)

# --- catalogues -------------------------------------------------------------
USGS_STAC = "https://landsatlook.usgs.gov/stac-server/search"
USGS_COLLECTION = "landsat-c2l2-sr"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_COLLECTION = "landsat-c2-l2"
PC_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="

# Sensor-neutral asset keys. TM/ETM+ and OLI number their bands differently;
# these names hide that difference and must be used instead of B1..B7.
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")
QA_ASSET = "qa_pixel"
MTL_ASSET = "mtl.json"             # carries the reflectance scaling
INDICES = ("mndwi", "ndvi", "bsi")

# --- selection rules --------------------------------------------------------
PATHS = ("160", "161", "162")      # three strips cover the sea east to west
ROWS = ("027", "028", "029", "030")
MAX_CLOUD = 20                     # percent, scene level
TIERS = ("T1",)                    # Tier 2 geometry is too loose for a time series
YEARS = (1984, 2025)
# May..September. Measured over the AOI, the sun sits at 62.6 deg in May and
# 60.6 in July, but only 33.0 in October: adding May costs nothing
# radiometrically, adding October would mix two different illuminations into
# the same median composite.
SEASON = (5, 9)

# --- storage ----------------------------------------------------------------
# Reflectance and indices are stored as int16 scaled by this factor: half the
# size of float32, and lossless at Landsat's radiometric precision.
STORE_SCALE = 10000
STORE_NODATA = -32768
MIN_VALID_FRACTION = 0.2           # skip patches that are mostly mask
# Statistics are read through the overviews, not at full resolution: a median
# over four million pixels is indistinguishable from one over fifty-five, and
# the full read costs gigabytes of RAM per tile.
QA_MAX_PIXELS = 4_000_000
# Patches offered for hand labelling must be near-complete. Landsat 7 after
# 2003 is shot through with gaps; drawing boundaries across them is slow and
# teaches the model a hole that is not in the ground.
LABEL_MIN_VALID = 0.95

# --- band numbering ---------------------------------------------------------
# The same physical band has a different number on different sensors. This
# table is the only place that knows it; the rest of the pipeline says
# "swir16" and stays sensor-agnostic.
BAND_FILE = {
    "TM":   {"blue": "B1", "green": "B2", "red": "B3",
             "nir08": "B4", "swir16": "B5", "swir22": "B7"},
    "ETM":  {"blue": "B1", "green": "B2", "red": "B3",
             "nir08": "B4", "swir16": "B5", "swir22": "B7"},
    "OLI":  {"blue": "B2", "green": "B3", "red": "B4",
             "nir08": "B5", "swir16": "B6", "swir22": "B7"},
}

# product id prefix -> sensor family
SENSOR = {"LT04": "TM", "LT05": "TM", "LE07": "ETM", "LC08": "OLI", "LC09": "OLI"}


def asset_filename(product_id: str, key: str) -> str:
    """Filename of one asset of one scene, e.g. ..._SR_B5.TIF."""
    if key == MTL_ASSET:
        return f"{product_id}_MTL.json"
    if key == QA_ASSET:
        return f"{product_id}_QA_PIXEL.TIF"
    sensor = SENSOR[product_id[:4]]
    return f"{product_id}_SR_{BAND_FILE[sensor][key]}.TIF"


# --- classes and labels -------------------------------------------------------
# Training ids. A published mask stores id + 1, keeping 0 for "no data".
CLASS_NAMES = ("water", "salt", "bare_soil", "vegetation")
IGNORE_INDEX = 255                 # no data: excluded from loss and metrics

# Thresholds for the automatic (weak) labels. The water one sits in a real gap
# between the classes. The salt one does not: bare-ground SWIR1 reflectance
# is continuous from ~0.15 to ~0.99 with no break, so 0.42 is a placeholder
# that hand labels are meant to replace, not a finding.
WEAK_WATER_MNDWI = 0.0
WEAK_VEG_NDVI = 0.20
WEAK_SALT_SWIR16 = 0.42

# Per-band normalisation for model input, in stored units (reflectance x
# 10000). Taken from Prithvi-EO-2.0's config, whose HLS training data uses the
# same scale; reused for every model so their inputs are identical.
MODEL_MEAN = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
MODEL_STD = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)
