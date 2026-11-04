"""Stage 2: download the assets listed in the manifest.

Two modes work from the same manifest:

  full    whole scenes, about 305 MB each. For a fast link such as the lab.
  window  only the part of each scene inside the AOI, read out of the COG
          with range requests. Around 40 times less traffic at home.

Everything here assumes it will be interrupted. Each file is written under a
.part name and renamed only when it is complete, so a half-written file is
never mistaken for a finished one and a re-run skips what is already on disk.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from . import config as C
from . import pc

# GDAL has to be told that these URLs are single files. Otherwise it probes
# the "directory" on every open and spends more time collecting 404s than
# reading data.
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="33554432",
)

# The MTL file is not optional. Stage 3 reads the reflectance scaling out of
# it and refuses to run without it, so leaving it out here is what makes that
# stage fail halfway through a batch instead of at the start.
ASSETS = C.BANDS + (C.QA_ASSET, C.MTL_ASSET)


def scene_dir(root, row):
    return root / row["date"][:4] / row["product_id"]


def _download_whole(url, dest):
    """Plain byte copy. Nothing is decoded, so nothing can be misread."""
    import urllib.request
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            n += len(chunk)
    os.replace(tmp, dest)
    return n


def _download_window(url, dest):
    """Read only the AOI out of the remote COG and write it unchanged.

    The crop keeps the source CRS and grid, because this stage must not
    resample. Reprojection belongs to `prepare`, where it happens once, on
    purpose, with a documented resampling rule.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *C.AOI_BBOX, densify_pts=21)
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            return 0
        data = src.read(1, window=win)
        profile = src.profile | dict(
            height=int(win.height), width=int(win.width),
            transform=src.window_transform(win),
            tiled=True, blockxsize=256, blockysize=256,
            compress="deflate", predictor=2)
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data, 1)
    os.replace(tmp, dest)
    return dest.stat().st_size


def fetch_asset(row, key, root, mode="window", tries=4):
    dest = scene_dir(root, row) / C.asset_filename(row["product_id"], key)
    if dest.exists() and dest.stat().st_size > 0:
        return ("have", dest, dest.stat().st_size, 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Only rasters can be windowed, so metadata is always copied whole.
    how = _download_whole if (mode == "full" or dest.suffix != ".TIF") \
        else _download_window
    for attempt in range(tries):
        try:
            url = pc.asset_url(row["base_url"], row["product_id"], key)
            n = how(url, dest)
            return ("ok", dest, n, attempt)
        except Exception as exc:
            if attempt == tries - 1:
                return ("fail", dest, repr(exc), attempt)
            time.sleep(3 * 2 ** attempt)


def fetch(rows, root=None, mode="window", workers=8, assets=ASSETS,
          progress=True):
    """Download every asset of every manifest row. Safe to re-run."""
    root = root or (C.RAW / "landsat")
    jobs = [(r, k) for r in rows for k in assets]
    # Retries back off for several seconds each and used to be invisible, so
    # a slow run looked slow for no reason. Counting them explains it.
    done = {"ok": 0, "have": 0, "fail": 0, "retried": 0}
    nbytes = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_asset, r, k, root, mode): (r, k) for r, k in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            status, dest, info, retries = fut.result()
            done[status] += 1
            done["retried"] += retries > 0
            if status == "ok":
                nbytes += info
            elif status == "fail":
                print(f"  FAIL {dest.name}: {info}", flush=True)
            if progress and i % 25 == 0:
                rate = nbytes / max(time.time() - t0, 1e-9) / 1e6
                print(f"  {i}/{len(jobs)}  {nbytes/1e9:.2f} GB  {rate:.1f} MB/s",
                      flush=True)
    done["gb"] = round(nbytes / 1e9, 3)
    done["seconds"] = round(time.time() - t0, 1)
    return done
