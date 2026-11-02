"""Stage 4: prepared tile -> patches on the fixed grid.

Patches are addressed by their position on the global grid, so a patch keeps
its name across dates and sensors. That is what makes a multi-temporal stack
possible at all: patch (31, 44) is the same 7.68 km square in every year.

Scanning and writing are separate on purpose. A single date offers tens of
thousands of candidate patches and only a few hundred are worth labelling;
writing them all first would spend almost all of its time on files that are
about to be deleted.
"""
import csv

import numpy as np
import rasterio
from rasterio.windows import Window

from . import config as C
from . import grid

BAND_INDEX = {name: i + 1 for i, name in enumerate(C.BANDS)}


def windows(src):
    """Grid-aligned patch windows fully inside the tile."""
    x0, y0 = grid.origin()
    off_c = int(round((src.transform.c - x0) / C.PIXEL)) % C.PATCH
    off_r = int(round((y0 - src.transform.f) / C.PIXEL)) % C.PATCH
    start_c = (C.PATCH - off_c) % C.PATCH
    start_r = (C.PATCH - off_r) % C.PATCH
    for row in range(start_r, src.height - C.PATCH + 1, C.PATCH):
        for col in range(start_c, src.width - C.PATCH + 1, C.PATCH):
            yield Window(col, row, C.PATCH, C.PATCH)


def scan(tile_path, product_id=None, min_valid=None):
    """Judge every candidate patch without writing anything.

    MNDWI is a ratio, so the stored scale factor cancels: water can be told
    from land on the raw int16 values, with no conversion to reflectance.
    """
    min_valid = C.MIN_VALID_FRACTION if min_valid is None else min_valid
    product_id = product_id or tile_path.stem
    gi, si = BAND_INDEX["green"], BAND_INDEX["swir16"]
    out = []
    with rasterio.open(tile_path) as src:
        for win in windows(src):
            valid = None
            for band in range(1, src.count + 1):
                m = src.read_masks(band, window=win) > 0
                valid = m if valid is None else (valid & m)
            frac = float(valid.mean())
            if frac < min_valid:
                continue
            green = src.read(gi, window=win).astype(np.int32)
            swir = src.read(si, window=win).astype(np.int32)
            water = float(((green > swir) & valid).sum() / max(valid.sum(), 1))
            gx, gy = grid.patch_index(src.window_transform(win))
            out.append({
                "patch": f"{product_id}_x{gx:03d}y{gy:03d}.tif",
                "product_id": product_id,
                "gx": gx, "gy": gy,
                "col_off": int(win.col_off), "row_off": int(win.row_off),
                "valid_frac": round(frac, 4),
                "water_frac": round(water, 4),
                # 1 at a 50/50 split, 0 at all-water or all-land: the
                # boundary is the part of a label worth drawing by hand.
                "mixedness": round(1 - abs(2 * water - 1), 4),
            })
    return out


def extract(tile_path, records, out_dir):
    """Write the chosen patches, and only those."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with rasterio.open(tile_path) as src:
        for rec in records:
            dest = out_dir / rec["patch"]
            if dest.exists():
                written.append(dest)
                continue
            win = Window(rec["col_off"], rec["row_off"], C.PATCH, C.PATCH)
            data = src.read(window=win)
            profile = src.profile | dict(
                width=C.PATCH, height=C.PATCH,
                transform=src.window_transform(win),
                compress="deflate", predictor=2)
            # A 256x256 image is a single block; leaving the parent's block
            # keys in place with tiled=False is what GDAL complains about.
            profile.update(tiled=True, blockxsize=C.PATCH, blockysize=C.PATCH)
            profile.pop("BIGTIFF", None)
            with rasterio.open(dest, "w", **profile) as dst:
                dst.write(data)
                for i, name in enumerate(src.descriptions, start=1):
                    if name:
                        dst.set_band_description(i, name)
            written.append(dest)
    return written


FIELDS = ("patch", "product_id", "gx", "gy", "col_off", "row_off",
          "valid_frac", "water_frac", "mixedness")


def write_index(records, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    return out_path
