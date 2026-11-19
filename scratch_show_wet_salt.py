"""Eyeball check of the wet salt rule: old labels vs new, side by side.

    uv run --with matplotlib python scratch_show_wet_salt.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.aral import config as C, labels as L

N_SHOW = 6
OUT = pathlib.Path.home() / "aral-work" / "wet_salt_check.png"
files = sorted((pathlib.Path.home() / "aral-work/drive/patches").glob("*.tif"))


def old_labels(data, valid):
    """The rule before today's change: water is just MNDWI > 0."""
    b = {n: data[i].astype(np.float32) for i, n in enumerate(C.BANDS)}
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (b["green"] - b["swir16"]) / (b["green"] + b["swir16"])
        ndvi = (b["nir08"] - b["red"]) / (b["nir08"] + b["red"])
    lab = np.full(valid.shape, C.IGNORE_INDEX, np.uint8)
    water = valid & (mndwi > C.WEAK_WATER_MNDWI)
    veg = valid & ~water & (ndvi > C.WEAK_VEG_NDVI)
    bare = valid & ~water & ~veg
    salt_dn = C.WEAK_SALT_SWIR16 * C.STORE_SCALE
    lab[water], lab[veg] = L.WATER, L.VEG
    lab[bare & (b["swir16"] > salt_dn)] = L.SALT
    lab[bare & (b["swir16"] <= salt_dn)] = L.SOIL
    return lab


# rank patches by how many pixels moved from water to salt
ranked = []
for f in files:
    data, valid = L.read_patch(f)
    old, new = old_labels(data, valid), L.weak_labels(data, valid)
    moved = int(((old == L.WATER) & (new == L.SALT)).sum())
    ranked.append((moved, f))
ranked.sort(reverse=True)
# "mixed": patches where only part moved, i.e. where water and salt meet
if "mixed" in __import__("sys").argv:
    ranked = [(m, f) for m, f in ranked if 0.05 < m / C.PATCH**2 < 0.6]
    OUT = OUT.with_name("wet_salt_check_mixed.png")

K = len(C.CLASS_NAMES)   # index K is drawn grey: no data
cmap = ListedColormap([(0.12, 0.37, 0.71), (0.95, 0.94, 0.90),
                       (0.65, 0.49, 0.32), (0.24, 0.57, 0.27), (0.45, 0.45, 0.45)])


def composite(data, valid, bands, top):
    img = np.stack([data[C.BANDS.index(b)] for b in bands], -1) / C.STORE_SCALE
    img = np.clip(img / top, 0, 1)
    img[~valid] = 0.45
    return img


fig, ax = plt.subplots(N_SHOW, 4, figsize=(13, 3.2 * N_SHOW), squeeze=False)
for r, (moved, f) in enumerate(ranked[:N_SHOW]):
    data, valid = L.read_patch(f)
    panels = [
        (composite(data, valid, ("red", "green", "blue"), 0.35), "natural colour"),
        (composite(data, valid, ("swir16", "nir08", "red"), 0.45), "SWIR1 / NIR / red"),
        (old_labels(data, valid), "old rule"),
        (L.weak_labels(data, valid), "new rule"),
    ]
    for c, (img, title) in enumerate(panels):
        a = ax[r, c]
        if c < 2:
            a.imshow(img)
        else:
            a.imshow(np.where(img == C.IGNORE_INDEX, K, img), cmap=cmap, vmin=0, vmax=K,
                     interpolation="nearest")
        a.set_title(title if r == 0 else "")
        a.set_xticks([]); a.set_yticks([])
    ax[r, 0].set_ylabel(f"{f.stem[17:25]} {f.stem[-8:]}\n{100 * moved / valid.size:.0f}% moved",
                        fontsize=9)
fig.suptitle("blue water  |  white salt  |  brown bare soil  |  green vegetation  |  grey no data")
fig.tight_layout()
fig.savefig(OUT, dpi=90)
print("saved", OUT)
