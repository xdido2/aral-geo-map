"""Spectral indices, computed from prepared tiles.

They are derived, never stored: recomputing from a tile costs milliseconds,
while a second copy on disk costs as much as the tile itself and can fall out
of step with it after a change upstream.

Every index is a normalised difference, so the reflectance scale cancels - but
the offset does not. That is why `prepare` applies both before anything here
runs, and why these functions take reflectance, never raw DN.
"""
import numpy as np

from . import config as C


def _nd(a, b):
    """Normalised difference (a - b) / (a + b), masked where it is undefined."""
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.ma.masked_invalid((a - b) / denom)
    return np.ma.masked_where(denom == 0, out)


def mndwi(bands):
    """Water. Green minus SWIR1: water absorbs SWIR, soil and salt reflect it."""
    return _nd(bands["green"], bands["swir16"])


def ndvi(bands):
    """Vegetation. NIR minus red."""
    return _nd(bands["nir08"], bands["red"])


def bsi(bands):
    """Bare soil / salt crust. Needs blue and SWIR2 as well as red and NIR."""
    bright = bands["swir16"] + bands["red"]
    dark = bands["nir08"] + bands["blue"]
    return _nd(bright, dark)


ALL = {"mndwi": mndwi, "ndvi": ndvi, "bsi": bsi}


def read_bands(path):
    """Prepared tile -> dict of masked reflectance arrays keyed by band name."""
    import rasterio
    with rasterio.open(path) as src:
        data = src.read(masked=True).astype(np.float32) / C.STORE_SCALE
        names = src.descriptions
    return {name: data[i] for i, name in enumerate(names)}


def compute(path, which=None):
    bands = read_bands(path)
    return {k: ALL[k](bands) for k in (which or C.INDICES)}
