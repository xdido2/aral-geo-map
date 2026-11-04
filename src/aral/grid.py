"""The fixed analysis grid.

Every scene, whatever UTM zone it came in, is resampled onto this one grid.
Grid lines fall on whole multiples of TILE_M from a fixed origin, so any two
dates line up pixel for pixel and patch boundaries never drift.
"""
import math

from affine import Affine
from rasterio.warp import transform_bounds

from . import config as C


def aoi_bounds():
    """The AOI in TARGET_CRS metres."""
    return transform_bounds("EPSG:4326", C.TARGET_CRS, *C.AOI_BBOX, densify_pts=21)


def origin():
    """Top-left corner of the grid, snapped outwards to a whole tile."""
    left, bottom, right, top = aoi_bounds()
    x0 = math.floor(left / C.TILE_M) * C.TILE_M
    y0 = math.ceil(top / C.TILE_M) * C.TILE_M
    return x0, y0


def snap(bounds):
    """Align arbitrary bounds to the grid, growing outwards.

    Returns (transform, width, height). Growing outwards instead of rounding
    to the nearest line keeps every source pixel inside the result.
    """
    x0, y0 = origin()
    left, bottom, right, top = bounds
    col_min = math.floor((left - x0) / C.PIXEL)
    col_max = math.ceil((right - x0) / C.PIXEL)
    row_min = math.floor((y0 - top) / C.PIXEL)
    row_max = math.ceil((y0 - bottom) / C.PIXEL)
    transform = Affine(C.PIXEL, 0.0, x0 + col_min * C.PIXEL,
                       0.0, -C.PIXEL, y0 - row_min * C.PIXEL)
    return transform, col_max - col_min, row_max - row_min


def clip_to_aoi(bounds):
    """Intersect bounds with the AOI. Returns None if they do not overlap."""
    al, ab, ar, at = aoi_bounds()
    left, bottom, right, top = bounds
    out = (max(left, al), max(bottom, ab), min(right, ar), min(top, at))
    return out if out[0] < out[2] and out[1] < out[3] else None


def describe():
    x0, y0 = origin()
    left, bottom, right, top = aoi_bounds()
    w = math.ceil((right - x0) / C.PIXEL)
    h = math.ceil((y0 - bottom) / C.PIXEL)
    return {
        "crs": C.TARGET_CRS,
        "pixel_m": C.PIXEL,
        "origin": (x0, y0),
        "aoi_bounds_m": (left, bottom, right, top),
        "size_px": (w, h),
        "size_km": (round(w * C.PIXEL / 1000, 1), round(h * C.PIXEL / 1000, 1)),
        "patches": (math.ceil(w / C.PATCH), math.ceil(h / C.PATCH)),
    }


def patch_index(transform, col_off=0, row_off=0):
    """Global patch coordinates of a window, counted from the grid origin.

    Computed from the map position and never from the tile, so the same
    ground gets the same (gx, gy) in 1985 and in 2025, no matter which scene
    it came from.
    """
    x0, y0 = origin()
    x = transform.c + col_off * C.PIXEL
    y = transform.f - row_off * C.PIXEL
    gx = int(round((x - x0) / C.TILE_M))
    gy = int(round((y0 - y) / C.TILE_M))
    return gx, gy
