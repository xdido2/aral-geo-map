"""Stage 3: turn one scene into one tile on the fixed grid.

Three things happen here, in this order, and the order is what matters:

1. validity is decided on the raw DN, before any arithmetic;
2. resampling onto the grid uses nearest neighbour, so no value is invented
   and no valid pixel is blended with an invalid one;
3. only then are the DN converted into reflectance.

Doing it the other way round, converting first and masking afterwards, lets
fill zeros and out-of-range values into the arithmetic, where they look like
plausible reflectance and quietly bias every statistic downstream.
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


OVERVIEWS = [8, 32]


def read_scaling(scene_dir, product_id):
    """Read scale and offset from the scene's own metadata, not a constant.

    They are the same across Collection 2 today. Reading them anyway means
    the pipeline fails loudly if that ever changes, instead of producing
    reflectance that is wrong by a fixed amount without telling anyone.
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
    """True where QA_PIXEL marks the pixel as fill, cloud, shadow or cirrus."""
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
    """Read one asset resampled onto the grid. Nearest, so values survive."""
    with rasterio.open(path) as src, WarpedVRT(
            src, crs=C.TARGET_CRS, transform=transform,
            width=width, height=height,
            resampling=Resampling.nearest,
            src_nodata=0, nodata=0) as vrt:
        return vrt.read(1)


def _clear(valid, fill):
    """Valid share of the pixels the scene actually covers.

    valid_frac counts the empty corners of a rotated scene as losses, about a
    third of the tile, so a completely cloudless scene would still score
    around 0.7. This measures the sky instead, which is what a quality
    ranking needs.
    """
    inside = int((~fill).sum())
    return float(valid.sum()) / inside if inside else 0.0


def tile_stats_from_file(tile_path):
    """Quality fractions of an existing tile, from its tags or measured.

    Measuring reads one band at a time, because a whole tile as a single
    array is several gigabytes.
    """
    with rasterio.open(tile_path) as src:
        tags = src.tags()
        if "valid_frac" in tags:
            return {"product_id": tags.get("product_id", tile_path.stem),
                    "width": src.width, "height": src.height,
                    "valid_frac": round(float(tags["valid_frac"]), 4),
                    "clear_frac": (round(float(tags["clear_frac"]), 4)
                                   if "clear_frac" in tags else None),
                    "qa_reject_frac": round(float(tags["qa_reject_frac"]), 4)}
        valid = None
        for i in range(1, src.count + 1):
            m = src.read_masks(i) > 0
            valid = m if valid is None else (valid & m)
        return {"product_id": tile_path.stem,
                "width": src.width, "height": src.height,
                "valid_frac": round(float(valid.mean()), 4),
                "clear_frac": None,
                "qa_reject_frac": None}


def prepare(scene_dir, product_id, out_path, window=None):
    """Write the scene as int16 reflectance on the grid. Returns QA stats."""
    scene_dir = Path(scene_dir)
    scale, offset = read_scaling(scene_dir, product_id)
    geom = window or target_window(scene_dir, product_id)
    if geom is None:
        return None
    transform, width, height = geom
    lo, hi = C.SR_VALID
    slack = int(round(C.SR_CLAMP_TOL / C.SR_SCALE))

    # Pass 1: decide validity from the raw DN and the QA band only.
    qa = _on_grid(scene_dir / C.asset_filename(product_id, C.QA_ASSET),
                  transform, width, height)
    rejected = qa_reject(qa)
    fill = (qa & (1 << C.QA_FILL)) != 0
    valid = ~rejected
    for key in C.BANDS:
        dn = _on_grid(scene_dir / C.asset_filename(product_id, key),
                      transform, width, height)
        # Fill pixels carry 0, which is far below lo, so this range test
        # catches them as well as the genuinely out-of-range values.
        valid &= (dn >= lo - slack) & (dn <= hi + slack)
        del dn

    # Pass 2: convert and write, one band at a time to keep memory flat.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", width=width, height=height,
                   count=len(C.BANDS), dtype="int16", crs=C.TARGET_CRS,
                   transform=transform, nodata=C.STORE_NODATA,
                   tiled=True, blockxsize=256, blockysize=256,
                   compress="deflate", predictor=2)
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), \
            rasterio.open(out_path, "w", **profile) as dst:
        for i, key in enumerate(C.BANDS, start=1):
            dn = _on_grid(scene_dir / C.asset_filename(product_id, key),
                          transform, width, height)
            np.clip(dn, lo, hi, out=dn)       # pull the slack back in
            refl = dn.astype(np.float32) * scale + offset
            band = np.full(refl.shape, C.STORE_NODATA, dtype=np.int16)
            np.rint(refl * C.STORE_SCALE, out=refl)
            np.copyto(band, refl.astype(np.int16), where=valid)
            dst.write(band, i)
            dst.set_band_description(i, key)
            del dn, refl, band

        # Overviews are averaged, so a zoomed-out view shows the real mix of
        # a block instead of whichever single pixel the decimation landed on.
        # Without them the periodic gaps of Landsat 7 after the 2003 SLC
        # failure alias into a pattern that is not in the data.
        #
        # Two levels are enough. A preview shrinks a scene about elevenfold,
        # so a third level would never be displayed, yet it would hold a
        # quarter of the full image and cost most of the build time.
        dst.build_overviews(OVERVIEWS, Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")
        # Stored in the file itself, so a tile that is skipped later can
        # still report its own quality. Recomputing this means re-reading
        # every band, and losing it leaves a hole in the QA table exactly
        # where it matters.
        dst.update_tags(clear_frac=f"{_clear(valid, fill):.6f}",
                        valid_frac=f"{valid.mean():.6f}",
                        qa_reject_frac=f"{rejected.mean():.6f}",
                        product_id=product_id)

    total = valid.size
    return {
        "product_id": product_id,
        "width": width, "height": height,
        "valid_frac": round(float(valid.sum()) / total, 4),
        "clear_frac": round(_clear(valid, fill), 4),
        "qa_reject_frac": round(float(rejected.sum()) / total, 4),
        "out_bytes": out_path.stat().st_size,
    }
