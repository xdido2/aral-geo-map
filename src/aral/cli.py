"""Command line entry point. One subcommand per stage, all re-runnable.

Being able to re-run a stage is not a nicety here. Over 1400 scenes something
always fails halfway, and a pipeline that cannot resume never reaches the end.
"""
import argparse
import csv
import sys
from pathlib import Path

from . import config as C
from . import fetch, grid, manifest, prepare, qa


def cmd_grid(args):
    for k, v in grid.describe().items():
        print(f"{k:14s} {v}")


def cmd_manifest(args):
    out, n = manifest.build(years=(args.start, args.end))
    print(f"\n{n} scenes -> {out}")


def cmd_prepare(args):
    rows = []
    for scene_dir in sorted(Path(args.scenes).iterdir()):
        if not scene_dir.is_dir():
            continue
        pid = scene_dir.name
        out = C.INTERIM / "tiles" / f"{pid}.tif"
        if out.exists() and not args.force:        # resume, do not redo
            print(f"[have] {pid}")
            continue
        st = prepare.prepare(scene_dir, pid, out)
        if st is None:
            print(f"[skip] {pid}: outside AOI")
            continue
        print(f"[ok]   {pid}  valid {st['valid_frac']:.3f}  "
              f"{st['out_bytes'] / 1e6:.0f} MB")
        rows.append(st)
    if rows:
        print(qa.write([qa.tile_stats(C.INTERIM / 'tiles' / f"{r['product_id']}.tif", r)
                        for r in rows]))


def cmd_fetch(args):
    with open(args.manifest) as fh:
        rows = [r for r in csv.DictReader(fh) if r["tier"] in C.TIERS]
    if args.max_cloud is not None:
        rows = [r for r in rows if r["cloud"] and float(r["cloud"]) <= args.max_cloud]
    if args.year:
        rows = [r for r in rows if int(r["year"]) in args.year]
    gb = len(rows) * len(fetch.ASSETS) * 44 / 1000      # about 44 MB per asset
    print(f"{len(rows)} scenes, {len(rows) * len(fetch.ASSETS)} assets, "
          f"~{gb:.0f} GB in '{args.mode}' mode")
    if args.dry_run:
        return
    print(fetch.fetch(rows, mode=args.mode, workers=args.workers))


def cmd_qa(args):
    with open(args.manifest) as fh:
        rows = list(csv.DictReader(fh))
    by_year = {}
    for r in rows:
        by_year.setdefault(r["year"], []).append(r)
    print(f"{'year':>6} {'scenes':>7} {'cloud<=20':>10} {'cells':>6}")
    for y in sorted(by_year):
        rs = by_year[y]
        clear = [r for r in rs if r["cloud"] and float(r["cloud"]) <= C.MAX_CLOUD]
        cells = {(r["path"], r["row"]) for r in clear}
        print(f"{y:>6} {len(rs):>7} {len(clear):>10} {len(cells):>6}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="aral")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("grid", help="print the analysis grid").set_defaults(fn=cmd_grid)

    m = sub.add_parser("manifest", help="stage 1: search the catalogue")
    m.add_argument("--start", type=int, default=C.YEARS[0])
    m.add_argument("--end", type=int, default=C.YEARS[1])
    m.set_defaults(fn=cmd_manifest)

    pr = sub.add_parser("prepare", help="stage 3: scenes -> grid tiles")
    pr.add_argument("scenes", help="directory of downloaded scene folders")
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(fn=cmd_prepare)

    f = sub.add_parser("fetch", help="stage 2: download assets")
    f.add_argument("--manifest", default=str(C.INTERIM / "manifest.csv"))
    f.add_argument("--mode", choices=("window", "full"), default="window")
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--year", type=int, nargs="*")
    f.add_argument("--max-cloud", type=float, default=C.MAX_CLOUD)
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(fn=cmd_fetch)

    q = sub.add_parser("qa", help="coverage summary from the manifest")
    q.add_argument("--manifest", default=str(C.INTERIM / "manifest.csv"))
    q.set_defaults(fn=cmd_qa)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
