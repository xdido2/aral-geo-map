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
SR_VALID = (7273, 43636)           # anything outside is not a measurement

# QA_PIXEL bit flags (Collection 2, all sensors)
QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW = 0, 1, 2, 3, 4
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
