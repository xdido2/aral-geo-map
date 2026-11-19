import pathlib
import numpy as np
from src.aral import config as C, labels as L

files = sorted(pathlib.Path.home().joinpath("aral-work/drive/patches").glob("*.tif"))
old_water_total = 0
went_to = np.zeros(len(C.CLASS_NAMES), np.int64)   # счётчик по новым классам

for f in files:
    data, valid = L.read_patch(f)
    new = L.weak_labels(data, valid)

    green = data[C.BANDS.index("green")].astype(np.float32)
    swir  = data[C.BANDS.index("swir16")].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (green - swir) / (green + swir)
    old_water = valid & (mndwi > C.WEAK_WATER_MNDWI)   # старое правило

    old_water_total += old_water.sum()
    went_to += np.bincount(new[old_water], minlength=len(C.CLASS_NAMES))

print(f"патчей: {len(files)}, бывшая вода: {old_water_total} пикселей")
for name, n in zip(C.CLASS_NAMES, went_to):
    print(f"  -> {name:11s} {n:9d}  {100 * n / old_water_total:5.1f}%")