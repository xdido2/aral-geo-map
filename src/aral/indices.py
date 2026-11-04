"""Spectral indices, computed from prepared tiles.

They are derived on the fly and never stored. Recomputing one from a tile
takes milliseconds, while a second copy on disk costs as much as the tile
itself and can fall out of step with it after a change upstream.

Every index is a normalised difference, so the reflectance scale cancels out.
The offset does not, which is why `prepare` applies both before anything here
runs and why these functions take reflectance and never raw DN.
"""
import numpy as np

from . import config as C


def _nd(a, b):
    """Normalised difference (a - b) / (a + b), masked where undefined."""
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
    """Bare soil and salt crust. Uses blue and SWIR2 as well as red and NIR."""
    bright = bands["swir16"] + bands["red"]
    dark = bands["nir08"] + bands["blue"]
    return _nd(bright, dark)


ALL = {"mndwi": mndwi, "ndvi": ndvi, "bsi": bsi}


def read_bands(path, window=None, max_pixels=None):
    """Read a prepared tile into masked reflectance arrays, keyed by band.

    `max_pixels` caps the read: anything larger is decimated through the
    tile's averaged overviews. Reading a whole scene-sized tile at full
    resolution as float32 costs several gigabytes, which is enough to end a
    Colab session, and it buys nothing when the answer is a median.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        h = int(window.height) if window is not None else src.height
        w = int(window.width) if window is not None else src.width
        kwargs = {"masked": True, "window": window}
        if max_pixels and h * w > max_pixels:
            shrink = (h * w / max_pixels) ** 0.5
            # GDAL only serves a read from an overview if that level's factor
            # is no larger than the shrink asked for. Ask for less than the
            # finest level and it silently decodes the full resolution
            # instead, which is two orders of magnitude slower for the same
            # median.
            levels = src.overviews(1)
            if levels and shrink < min(levels):
                shrink = float(min(levels))
            kwargs["out_shape"] = (src.count,
                                   max(int(h / shrink), 1),
                                   max(int(w / shrink), 1))
            kwargs["resampling"] = Resampling.average
        data = src.read(**kwargs)
        names = src.descriptions
    # Convert in place. Writing `.astype(...) / scale` would keep two full
    # copies of the array alive at the same time.
    data = data.astype(np.float32)
    data /= C.STORE_SCALE
    return {name: data[i] for i, name in enumerate(names)}


def compute(path, which=None, window=None, max_pixels=None):
    bands = read_bands(path, window=window, max_pixels=max_pixels)
    return {k: ALL[k](bands) for k in (which or C.INDICES)}
