"""Execute every code cell of the Colab notebook, in order, outside Colab.

Run before every push that touches src/ or the notebook:

    zip -qr <work>/drive/aral_src.zip src pyproject.toml -x '*__pycache__*'
    uv run --with matplotlib python tools/run_notebook.py \
        notebooks/aral_colab.ipynb <work>

Downloads two 2019 scenes (~600 MB) into <work>. A manifest left in
<work>/drive from an earlier run is reused, otherwise it is rebuilt (~6 min).

google.colab is stubbed, /content paths are redirected into a scratch
directory, and the code arrives through the zip path so the test covers
the working tree about to be pushed. Stops at the first failing cell.
"""
import json, os, pathlib, sys, time, traceback, types

nb_path, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
os.environ["MPLBACKEND"] = "Agg"
drive_dir, root_dir, mount_dir = work / "drive", work / "root", work / "mount"
for d in (drive_dir, mount_dir):
    d.mkdir(parents=True, exist_ok=True)

g, gc = types.ModuleType("google"), types.ModuleType("google.colab")
gd, gu = types.ModuleType("google.colab.drive"), types.ModuleType("google.colab.userdata")
gd.mount = lambda *a, **k: print("[stub] drive.mount")
gu.get = lambda key: None
gc.drive, gc.userdata, g.colab = gd, gu, gc
sys.modules.update({"google": g, "google.colab": gc,
                    "google.colab.drive": gd, "google.colab.userdata": gu})

SUBSET = ("rows = [next(r for r in rows if r['product_id'].startswith(p))"
          " for p in ('LC08', 'LE07')]\n"
          "print('[harness] subset:', [r['product_id'] for r in rows])")

cells = [c["source"] for c in json.loads(nb_path.read_text())["cells"]
         if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
for i, src in enumerate(cells, 1):
    src = "\n".join(l for l in src.split("\n") if not l.strip().startswith(("!", "%")))
    src = (src.replace("/content/drive/MyDrive/aral", str(drive_dir))
              .replace("/content/aral-geo-map", str(root_dir))
              .replace("/content/drive", str(mount_dir)))
    if src.startswith("REPO_URL"):
        src = "REPO_URL = ''" + src[src.index("\n"):]
    if "CELLS = " in src:
        src += "\n" + SUBSET
    head = src.strip().split("\n")[0][:70]
    t = time.time()
    print(f"\n===== cell {i}/{len(cells)}: {head}", flush=True)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except BaseException:
        print(f"\n##### FAILED at cell {i} after {time.time()-t:.1f}s", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print(f"----- cell {i} ok, {time.time()-t:.1f}s", flush=True)
print("\n##### ALL CELLS PASSED")
