"""Layer 2, the cheap half: labels from index thresholds.

These are the threshold's opinion of each pixel, not the truth. A model
trained on them learns to reproduce the threshold, mistakes included. That is
enough to prove the training loop works and not enough to ask whether a model
beats the threshold - that question needs the hand-drawn labels.
"""
import numpy as np
import rasterio

from . import config as C

WATER, SALT, SOIL, VEG = range(4)


def read_patch(path):
    """Stored int16 bands (reflectance x 10000) and a validity mask."""
    with rasterio.open(path) as src:
        data = src.read()
    valid = (data != C.STORE_NODATA).all(axis=0)
    return data, valid


def weak_labels(data, valid):
    """Threshold labels for one patch, in training ids; IGNORE_INDEX off-data."""
    b = {name: data[i].astype(np.float32) for i, name in enumerate(C.BANDS)}
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (b["green"] - b["swir16"]) / (b["green"] + b["swir16"])
        ndvi = (b["nir08"] - b["red"]) / (b["nir08"] + b["red"])
    lab = np.full(valid.shape, C.IGNORE_INDEX, np.uint8)
    water = valid & (mndwi > C.WEAK_WATER_MNDWI)
    veg = valid & ~water & (ndvi > C.WEAK_VEG_NDVI)
    bare = valid & ~water & ~veg
    salt_dn = C.WEAK_SALT_SWIR16 * C.STORE_SCALE
    lab[water] = WATER
    lab[veg] = VEG
    lab[bare & (b["swir16"] > salt_dn)] = SALT
    lab[bare & (b["swir16"] <= salt_dn)] = SOIL
    return lab


def normalise(data, valid):
    """Model input: per-band standardised, zero where there is no data."""
    mean = np.asarray(C.MODEL_MEAN, np.float32)[:, None, None]
    std = np.asarray(C.MODEL_STD, np.float32)[:, None, None]
    x = (data.astype(np.float32) - mean) / std
    x[:, ~valid] = 0.0
    return x
