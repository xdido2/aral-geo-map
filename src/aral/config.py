"""Every constant used by the pipeline lives here.

The other modules import these names and never repeat a number inline. One
place to look, and no chance of two copies of the same threshold drifting
apart over time.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
WORK = ROOT / "data" / "work"
REPORTS = ROOT / "reports"

# --- area of interest -------------------------------------------------------
# The Aral Sea, its dry bed and the Amu Darya delta (lon/lat, WGS84).
AOI_BBOX = (57.8, 43.0, 62.0, 47.2)

# --- analysis grid ----------------------------------------------------------
# One fixed grid for the whole 40-year archive. Every scene is resampled onto
# it, so pixel (row, col) covers the same piece of ground in 1985 and in 2025.
# Without that, patches from different years cannot be compared at all.
TARGET_CRS = "EPSG:32640"          # UTM zone 40N, where most of the sea sits
PIXEL = 30.0                       # metres, the native Landsat resolution
PATCH = 256                        # model input size
TILE_M = PIXEL * PATCH             # 7680 m per patch

# UTM preserves shape, not area. Near the eastern edge of the AOI it
# overstates area by about 0.4 percent, so areas are measured in this
# equal-area projection instead.
AREA_CRS = ("+proj=aea +lat_1=43 +lat_2=47 +lat_0=45 +lon_0=60 "
            "+datum=WGS84 +units=m +no_defs")

# --- Landsat Collection 2 Level-2 -------------------------------------------
SR_SCALE = 2.75e-05                # DN to surface reflectance
SR_OFFSET = -0.2
SR_VALID = (7273, 43636)           # the range USGS marks as valid
# Water reflects almost no SWIR, and the atmospheric correction often
# overshoots by a fraction of a percent, so a real open-water pixel can land
# just below the valid range. Dropping those pixels would punch holes through
# the class this project cares about most. Small excursions are clamped back
# to the limit and only larger ones are rejected.
SR_CLAMP_TOL = 0.02                # reflectance units

# QA_PIXEL bit flags (Collection 2, all sensors)
QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW = 0, 1, 2, 3, 4
# The CFMask flags are used as they come. Along the shoreline the mask does
# over-flag: on some clear dates a coastal strip a few hundred metres wide is
# marked as cloud, buffer or shadow (cell 161/028 in 2019 showed 9 to 21 times
# the rate measured away from the coast). I tested a thermal rule, that a
# cloud has to be colder than open water, and it cleared the strip on one
# date but also released real thin clouds over the dry seabed on another, so
# I did not adopt it. The strip is absent on other dates, so the seasonal
# composite covers it. Years with only one usable date keep it.
QA_REJECT_BITS = (QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW)

# --- catalogues -------------------------------------------------------------
USGS_STAC = "https://landsatlook.usgs.gov/stac-server/search"
USGS_COLLECTION = "landsat-c2l2-sr"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_COLLECTION = "landsat-c2-l2"
PC_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="

# Asset keys that work for any sensor. TM/ETM+ and OLI number their bands
# differently, so the code uses these names instead of B1..B7.
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")
QA_ASSET = "qa_pixel"
MTL_ASSET = "mtl.json"             # holds the reflectance scaling
INDICES = ("mndwi", "ndvi", "bsi")

# --- selection rules --------------------------------------------------------
PATHS = ("160", "161", "162")      # three strips cover the sea east to west
ROWS = ("027", "028", "029", "030")
MAX_CLOUD = 20                     # percent, per scene
TIERS = ("T1",)                    # Tier 2 geometry is too loose for a series
YEARS = (1984, 2025)
# May to September. Over the AOI the sun sits at 62.6 degrees in May and 60.6
# in July, but only 33.0 in October. Adding May costs nothing, while adding
# October would mix two very different illuminations into one composite.
SEASON = (5, 9)

# --- storage ----------------------------------------------------------------
# Reflectance and indices are stored as int16 multiplied by this factor. Half
# the size of float32, and nothing is lost: Landsat is not that precise.
STORE_SCALE = 10000
STORE_NODATA = -32768
MIN_VALID_FRACTION = 0.2           # skip patches that are mostly masked out
# Statistics are read from the overviews, not at full resolution. A median
# over four million pixels matches one over fifty-five million, and the full
# read would cost gigabytes of RAM per tile.
QA_MAX_PIXELS = 4_000_000
# Patches offered for hand labelling have to be nearly complete. Landsat 7
# scenes after 2003 are full of gaps, and drawing class boundaries across a
# gap is slow and teaches the model a hole that is not on the ground.
LABEL_MIN_VALID = 0.95

# --- band numbering ---------------------------------------------------------
# The same physical band has a different number on each sensor. This table is
# the only place that knows the mapping; everywhere else the code says
# "swir16" and does not care which satellite took the scene.
BAND_FILE = {
    "TM":   {"blue": "B1", "green": "B2", "red": "B3",
             "nir08": "B4", "swir16": "B5", "swir22": "B7"},
    "ETM":  {"blue": "B1", "green": "B2", "red": "B3",
             "nir08": "B4", "swir16": "B5", "swir22": "B7"},
    "OLI":  {"blue": "B2", "green": "B3", "red": "B4",
             "nir08": "B5", "swir16": "B6", "swir22": "B7"},
}

# product id prefix to sensor family
SENSOR = {"LT04": "TM", "LT05": "TM", "LE07": "ETM", "LC08": "OLI", "LC09": "OLI"}


def asset_filename(product_id: str, key: str) -> str:
    """Filename of one asset of one scene, for example ..._SR_B5.TIF."""
    if key == MTL_ASSET:
        return f"{product_id}_MTL.json"
    if key == QA_ASSET:
        return f"{product_id}_QA_PIXEL.TIF"
    sensor = SENSOR[product_id[:4]]
    return f"{product_id}_SR_{BAND_FILE[sensor][key]}.TIF"


# --- classes and labels -------------------------------------------------------
# Training ids. A published mask stores id + 1 and keeps 0 for "no data".
CLASS_NAMES = ("water", "salt", "bare_soil", "vegetation")
IGNORE_INDEX = 255                 # no data: left out of the loss and metrics

# Thresholds for the automatic (weak) labels. The water threshold sits in a
# real gap between the two classes. The salt one does not: bare-ground SWIR1
# reflectance runs from about 0.15 to 0.99 with no break in the histogram, so
# 0.42 is a placeholder for the hand labels to replace. It is not a result.
WEAK_WATER_MNDWI = 0.0
WEAK_VEG_NDVI = 0.20
WEAK_SALT_SWIR16 = 0.42

# Per-band normalisation for the model input, in stored units (reflectance
# times 10000). Taken from the Prithvi-EO-2.0 config, whose HLS training data
# uses the same scale. Every model here reuses it so they all see the same
# input and the comparison stays fair.
MODEL_MEAN = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
MODEL_STD = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)
