"""Stage 1: build the manifest.

The manifest is the list of every scene the project may use. Nothing
downstream ever walks a directory, it reads this table instead. That is what
makes the dataset reproducible: the table is an artefact, and rebuilding it a
year from now with the same parameters gives the same rows.
"""
import csv
import json
import pathlib
import time
import urllib.error
import urllib.request

from . import config as C


def _last_day(year, month):
    import calendar
    return calendar.monthrange(year, month)[1]

FIELDS = ("product_id", "date", "year", "month", "path", "row",
          "platform", "tier", "cloud", "sun_elev", "epsg", "base_url")


def _post(url, body, tries=7):
    """POST with a backoff that handles 429 properly.

    Shared outbound addresses get throttled much harder than a home
    connection, and Colab is the worst of them. If 429 is treated like any
    other error, a pause the server asked for turns into a failed run.
    """
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if attempt == tries - 1:
                raise
            if exc.code == 429:
                wait = float(exc.headers.get("Retry-After") or 0) or 10 * 2 ** attempt
                print(f"    429, waiting {wait:.0f}s", flush=True)
            elif exc.code in (500, 502, 503, 504):
                wait = 5 * 2 ** attempt
            else:
                raise
            time.sleep(min(wait, 300))
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def _search(path, year, season=None):
    """One page of scenes for one strip in one season. Kept small on purpose."""
    m0, m1 = season or C.SEASON
    body = {
        "collections": [C.PC_COLLECTION],
        "bbox": list(C.AOI_BBOX),
        # The last day of m1 is computed, not hard-coded to 30, otherwise a
        # 31-day end month would quietly lose its last day.
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
    if product_id[:4] not in C.SENSOR:          # MSS and other odd products
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
        # Tier 2 has looser geometric accuracy. It is recorded rather than
        # dropped, so the choice can be changed without rebuilding the table.
        "tier": product_id[-2:],
        # Sun elevation changes through the season, and a composite that
        # mixes high and low sun is comparing two different things.
        "sun_elev": round(float(p["view:sun_elevation"]), 2)
        if "view:sun_elevation" in p else "",
        "cloud": p.get("eo:cloud_cover", ""),
        "epsg": p.get("proj:epsg", ""),
        "base_url": base,
    }


def params_path(manifest_path):
    return manifest_path.with_suffix(".json")


def params(manifest_path):
    """Build parameters of an existing manifest, or None if not recorded."""
    p = params_path(manifest_path)
    return json.loads(p.read_text()) if p.exists() else None


def matches_config(manifest_path, years=None, season=None):
    """Was this manifest built with the parameters in force right now?

    A manifest that disagrees with the config describes a different dataset
    than the code thinks it does, and that kind of mismatch usually shows up
    as a missing file three stages later.
    """
    got = params(manifest_path)
    if not got:
        return False
    want = {"season": list(season or C.SEASON), "years": list(years or C.YEARS),
            "paths": list(C.PATHS), "rows": list(C.ROWS),
            "collection": C.PC_COLLECTION}
    return all(got.get(k) == v for k, v in want.items())


def build(out_path=None, years=None, season=None, cache_dir=None,
          verbose=True, pause=0.3):
    """Search the catalogue cell by cell and cache every answer.

    One cache file per (path, year). A run that dies at request 100 of 126
    resumes from 100 instead of starting over. Without the cache a rate limit
    is not a delay but a loop, because nothing is written until the very last
    query succeeds.
    """
    out_path = out_path or (C.INTERIM / "manifest.csv")
    y0, y1 = years or C.YEARS
    season = tuple(season or C.SEASON)
    # Keep the cache somewhere that outlives the machine. On Colab the VM
    # disk disappears with the session, and then a rate limit costs the whole
    # build a second time.
    base = pathlib.Path(cache_dir) if cache_dir else out_path.parent
    cache = base / f"{out_path.stem}_cache_{season[0]}{season[1]}"
    cache.mkdir(parents=True, exist_ok=True)

    rows, seen = [], set()
    for year in range(y0, y1 + 1):
        n = 0
        for path in C.PATHS:
            shard = cache / f"{path}_{year}.json"
            if shard.exists():
                found = json.loads(shard.read_text())
            else:
                found = [r for r in (_row(f) for f in _search(path, year, season))
                         if r]
                shard.write_text(json.dumps(found))
                time.sleep(pause)          # be polite on a shared IP
            for r in found:
                if r["product_id"] not in seen:
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
