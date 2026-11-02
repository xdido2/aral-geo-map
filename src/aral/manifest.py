"""Stage 1: build the manifest.

The manifest is the list of every scene the project may use. Nothing
downstream ever walks a directory; it reads this table. That makes the dataset
reproducible: the table is an artefact, and re-running it a year from now with
the same parameters yields the same rows.
"""
import csv
import json
import time
import urllib.error
import urllib.request

from . import config as C


def _last_day(year, month):
    import calendar
    return calendar.monthrange(year, month)[1]

FIELDS = ("product_id", "date", "year", "month", "path", "row",
          "platform", "tier", "cloud", "sun_elev", "epsg", "base_url")


def _post(url, body, tries=5):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def _search(path, year, season=None):
    """One page of scenes for one strip in one season. Deliberately small."""
    m0, m1 = season or C.SEASON
    body = {
        "collections": [C.PC_COLLECTION],
        "bbox": list(C.AOI_BBOX),
        # last day of m1, not a hard-coded 30: a 31-day end month would
        # otherwise lose its last day silently.
        "datetime": f"{year}-{m0:02d}-01T00:00:00Z/{year}-{m1:02d}-{_last_day(year, m1):02d}T23:59:59Z",
        "query": {"landsat:wrs_path": {"eq": path}},
        "limit": 500,
    }
    return _post(C.PC_STAC, body).get("features", [])


def _row(feature):
    p = feature["properties"]
    if p.get("landsat:wrs_row") not in C.ROWS:
        return None
    href = feature["assets"]["green"]["href"]
    base, _, _ = href.rpartition("/")
    product_id = base.rsplit("/", 1)[-1]
    if product_id[:4] not in C.SENSOR:          # MSS and other oddities
        return None
    dt = p["datetime"][:10]
    return {
        "product_id": product_id,
        "date": dt,
        "year": int(dt[:4]),
        "month": int(dt[5:7]),
        "path": p["landsat:wrs_path"],
        "row": p["landsat:wrs_row"],
        "platform": p.get("platform", ""),
        # Tier 2 has looser geometric accuracy; recorded, not dropped, so the
        # choice stays reversible without rebuilding the manifest.
        "tier": product_id[-2:],
        # Illumination differs across the season; a composite that mixes
        # high and low sun compares two different things.
        "sun_elev": round(float(p["view:sun_elevation"]), 2)
        if "view:sun_elevation" in p else "",
        "cloud": p.get("eo:cloud_cover", ""),
        "epsg": p.get("proj:epsg", ""),
        "base_url": base,
    }


def params_path(manifest_path):
    return manifest_path.with_suffix(".json")


def params(manifest_path):
    """Build parameters of an existing manifest, or None if unknown."""
    p = params_path(manifest_path)
    return json.loads(p.read_text()) if p.exists() else None


def matches_config(manifest_path, years=None, season=None):
    """Was this manifest built with the parameters in force now?

    A manifest that silently disagrees with the config defines a different
    dataset than the code thinks it does - the kind of mismatch that only
    shows up as a missing file three stages later.
    """
    got = params(manifest_path)
    if not got:
        return False
    want = {"season": list(season or C.SEASON), "years": list(years or C.YEARS),
            "paths": list(C.PATHS), "rows": list(C.ROWS),
            "collection": C.PC_COLLECTION}
    return all(got.get(k) == v for k, v in want.items())


def build(out_path=None, years=None, season=None, verbose=True):
    out_path = out_path or (C.INTERIM / "manifest.csv")
    y0, y1 = years or C.YEARS
    rows, seen = [], set()
    for year in range(y0, y1 + 1):
        n = 0
        for path in C.PATHS:
            for feature in _search(path, year, season):
                r = _row(feature)
                if r and r["product_id"] not in seen:
                    seen.add(r["product_id"])
                    rows.append(r)
                    n += 1
        if verbose:
            print(f"  {year}: {n:4d} scenes", flush=True)

    rows.sort(key=lambda r: (r["date"], r["path"], r["row"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params_path(out_path).write_text(json.dumps({
        "season": list(season or C.SEASON),
        "years": list(years or C.YEARS),
        "paths": list(C.PATHS),
        "rows": list(C.ROWS),
        "collection": C.PC_COLLECTION,
        "aoi_bbox": list(C.AOI_BBOX),
        "n_rows": len(rows),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_path, len(rows)
