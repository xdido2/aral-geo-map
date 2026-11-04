"""Quality control: one row per tile, so 1400 tiles can be checked at once.

Every rule here exists because I made the same mistake by hand first: a tile
from the wrong cell, a tile that is mostly nodata, a tile whose statistics sit
far away from its neighbours in the same cell.
"""
import csv

import numpy as np

from . import config as C
from . import indices

FIELDS = ("product_id", "path", "row", "date", "width", "height",
          "valid_frac", "clear_frac", "qa_reject_frac",
          "mndwi_median", "ndvi_median", "bsi_median", "water_frac")


def tile_stats(tile_path, meta=None, max_pixels=None):
    """Numbers that make a bad tile stand out once the table is sorted.

    Read through the overviews: these are summary statistics, and a tile the
    size of a Landsat scene does not fit in memory as float32.
    """
    idx = indices.compute(tile_path,
                          max_pixels=max_pixels or C.QA_MAX_PIXELS)
    water = idx["mndwi"] > 0
    row = dict(meta or {})
    row.update({
        "mndwi_median": round(float(np.ma.median(idx["mndwi"])), 4),
        "ndvi_median": round(float(np.ma.median(idx["ndvi"])), 4),
        "bsi_median": round(float(np.ma.median(idx["bsi"])), 4),
        "water_frac": round(float(water.sum()) / max(idx["mndwi"].count(), 1), 4),
    })
    return row


def write(rows, out_path=None):
    out_path = out_path or (C.REPORTS / "tile_stats.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out_path


def flag(row):
    """Cheap anomaly rules. Not a model, just the things that are never fine."""
    out = []
    # Judge the sky, not the geometry. A third of every tile is the empty
    # corners of a rotated scene, whatever the weather was that day.
    clear = row.get("clear_frac")
    if (clear if clear is not None else row.get("valid_frac", 1)) < C.MIN_VALID_FRACTION:
        out.append("mostly-masked")
    if row.get("water_frac") in (0.0, 1.0):
        out.append("uniform-water-fraction")
    if row.get("path") not in C.PATHS or row.get("row") not in C.ROWS:
        out.append("outside-working-cells")
    return out
