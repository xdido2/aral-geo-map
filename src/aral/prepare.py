"""Stage 3: one scene -> one tile on the fixed grid.

Three things happen here, in this order, and the order matters:

1. validity is decided on the raw DN, before any arithmetic;
2. resampling onto the grid uses nearest neighbour, so no value is invented
   and no valid pixel is blended with an invalid one;
3. only then are DN turned into reflectance.

Doing it the other way round - converting first, masking later - lets fill
zeros and out-of-range values through into the arithmetic, where they look
like plausible reflectance and quietly bias every statistic downstream.
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from . import config as C
from . import grid


def read_scaling(scene_dir, product_id):
    """Scale and offset from the scene's own metadata, not from a constant.

    They are the same for all of Collection 2 today. Reading them anyway means
    the pipeline fails loudly if that ever changes, instead of silently
    producing reflectance that is wrong by a fixed amount.
    """
    mtl = json.loads((Path(scene_dir) / f"{product_id}_MTL.json").read_text())
    params = mtl["LANDSAT_METADATA_FILE"]["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]
    mult = {float(v) for k, v in params.items() if k.startswith("REFLECTANCE_MULT_BAND_")}
    add = {float(v) for k, v in params.items() if k.startswith("REFLECTANCE_ADD_BAND_")}
    if len(mult) != 1 or len(add) != 1:
        raise ValueError(f"{product_id}: per-band scaling differs {mult} {add}")
    scale, offset = mult.pop(), add.pop()
    if (scale, offset) != (C.SR_SCALE, C.SR_OFFSET):
        raise ValueError(f"{product_id}: scaling {scale}/{offset} "
                         f"!= expected {C.SR_SCALE}/{C.SR_OFFSET}")
    return scale, offset


def qa_reject(qa):
    """True where QA_PIXEL says the pixel is fill, cloud, shadow or cirrus."""
    bad = 0
    for bit in C.QA_REJECT_BITS:
        bad |= 1 << bit
    return (qa & bad) != 0


def target_window(scene_dir, product_id):
    """Grid-aligned transform and size for this scene, clipped to the AOI."""
    ref = Path(scene_dir) / C.asset_filename(product_id, "green")
    with rasterio.open(ref) as src:
        bounds = transform_bounds(src.crs, C.TARGET_CRS, *src.bounds, densify_pts=21)
    inside = grid.clip_to_aoi(bounds)
    if inside is None:
        return None
    return grid.snap(inside)


def _on_grid(path, transform, width, height):
    """Read one asset resampled onto the grid. Nearest: values are preserved."""
    with rasterio.open(path) as src, WarpedVRT(
            src, crs=C.TARGET_CRS, transform=transform,
            width=width, height=height,
            resampling=Resampling.nearest,
            src_nodata=0, nodata=0) as vrt:
        return vrt.read(1)


def prepare(scene_dir, product_id, out_path, window=None):
    """Write the scene as int16 reflectance on the grid. Returns QA stats."""
    scene_dir = Path(scene_dir)
    scale, offset = read_scaling(scene_dir, product_id)
    geom = window or target_window(scene_dir, product_id)
    if geom is None:
        return None
    transform, width, height = geom
    lo, hi = C.SR_VALID

    # pass 1 - validity, from raw DN and QA only
    qa = _on_grid(scene_dir / C.asset_filename(product_id, C.QA_ASSET),
                  transform, width, height)
    rejected = qa_reject(qa)
    valid = ~rejected
    for key in C.BANDS:                       # fill (0) is below lo, so the
        dn = _on_grid(scene_dir / C.asset_filename(product_id, key),
                      transform, width, height)
        valid &= (dn >= lo) & (dn <= hi)      # range check catches it too
        del dn

    # pass 2 - convert and write, one band at a time
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", width=width, height=height,
                   count=len(C.BANDS), dtype="int16", crs=C.TARGET_CRS,
                   transform=transform, nodata=C.STORE_NODATA,
                   tiled=True, blockxsize=256, blockysize=256,
                   compress="deflate", predictor=2)
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, key in enumerate(C.BANDS, start=1):
            dn = _on_grid(scene_dir / C.asset_filename(product_id, key),
                          transform, width, height)
            refl = dn.astype(np.float32) * scale + offset
            band = np.full(refl.shape, C.STORE_NODATA, dtype=np.int16)
            np.rint(refl * C.STORE_SCALE, out=refl)
            np.copyto(band, refl.astype(np.int16), where=valid)
            dst.write(band, i)
            dst.set_band_description(i, key)
            del dn, refl, band

        # Overviews are averaged, so a zoomed-out view reports the real mix of
        # a block instead of whichever single pixel decimation happened to
        # land on. Without them, periodic gaps (Landsat 7 after the 2003 SLC
        # failure above all) alias into a pattern that is not in the data.
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    total = valid.size
    return {
        "product_id": product_id,
        "width": width, "height": height,
        "valid_frac": round(float(valid.sum()) / total, 4),
        "qa_reject_frac": round(float(rejected.sum()) / total, 4),
        "out_bytes": out_path.stat().st_size,
    }
