"""Stage 4: prepared tile -> patches on the fixed grid.

Patches are addressed by their position on the global grid, so a patch keeps
its name across dates and sensors. That is what makes a multi-temporal stack
possible at all: patch (31, 44) is the same 7.68 km square in every year.
"""
import csv

import numpy as np
import rasterio
from rasterio.windows import Window

from . import config as C
from . import grid


def windows(src):
    """Grid-aligned patch windows fully inside the tile."""
    x0, y0 = grid.origin()
    # offset of the tile's own origin within the grid, in pixels
    off_c = int(round((src.transform.c - x0) / C.PIXEL)) % C.PATCH
    off_r = int(round((y0 - src.transform.f) / C.PIXEL)) % C.PATCH
    start_c = (C.PATCH - off_c) % C.PATCH
    start_r = (C.PATCH - off_r) % C.PATCH
    for row in range(start_r, src.height - C.PATCH + 1, C.PATCH):
        for col in range(start_c, src.width - C.PATCH + 1, C.PATCH):
            yield Window(col, row, C.PATCH, C.PATCH)


def cut(tile_path, out_dir, product_id=None, min_valid=None, limit=None):
    """Write every sufficiently complete patch. Returns one record each."""
    min_valid = C.MIN_VALID_FRACTION if min_valid is None else min_valid
    out_dir.mkdir(parents=True, exist_ok=True)
    product_id = product_id or tile_path.stem
    records = []
    with rasterio.open(tile_path) as src:
        for win in windows(src):
            data = src.read(masked=True, window=win)
            valid = float((~data.mask).all(axis=0).mean())
            if valid < min_valid:
                continue
            tr = src.window_transform(win)
            gx, gy = grid.patch_index(tr)
            dest = out_dir / f"{product_id}_x{gx:03d}y{gy:03d}.tif"
            profile = src.profile | dict(
                width=C.PATCH, height=C.PATCH, transform=tr,
                tiled=False, compress="deflate", predictor=2)
            if not dest.exists():
                with rasterio.open(dest, "w", **profile) as dst:
                    dst.write(data.filled(C.STORE_NODATA))
                    for i, name in enumerate(src.descriptions, start=1):
                        if name:
                            dst.set_band_description(i, name)
            records.append({"patch": dest.name, "product_id": product_id,
                            "gx": gx, "gy": gy,
                            "valid_frac": round(valid, 4),
                            "bytes": dest.stat().st_size})
            if limit and len(records) >= limit:
                break
    return records


def write_index(records, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("patch", "product_id", "gx", "gy",
                                           "valid_frac", "bytes"))
        w.writeheader()
        w.writerows(records)
    return out_path
