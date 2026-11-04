"""Run every code cell of a Colab notebook, in order, outside Colab.

I run this before every push that touches src/ or a notebook, because a cell
that breaks on Colab costs a reload and a re-download to find out.

    zip -qr <work>/drive/aral_src.zip src pyproject.toml -x '*__pycache__*'
    uv run --with matplotlib python tools/run_notebook.py \
        notebooks/aral_colab.ipynb <work>

It downloads two 2019 scenes (about 600 MB) into <work>. A manifest left in
<work>/drive by an earlier run is reused, otherwise it is rebuilt, which
takes roughly 6 minutes.

Any top-level setting can be overridden to keep a CPU run short, which is how
the training notebook is checked:

    ... run_notebook.py notebooks/aral_train.ipynb <work> \
        --set INSTALL=False --set MAX_PATCHES=40 --set EPOCHS_UNET=1

google.colab is replaced by a stub, the /content paths are redirected into a
scratch directory, and the code is loaded through the zip path, so the run
tests the working tree that is about to be pushed. It stops at the first
failing cell.
"""
import json, os, pathlib, sys, time, traceback, types

nb_path, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
overrides = {}
args = sys.argv[3:]
while args:
    flag, pair, *args = args
    assert flag == "--set", f"unknown argument {flag}"
    name, value = pair.split("=", 1)
    overrides[name] = value
os.environ["MPLBACKEND"] = "Agg"
# Stand-ins for the three Colab directories the notebooks use.
drive_dir, root_dir, mount_dir = work / "drive", work / "root", work / "mount"
for d in (drive_dir, mount_dir):
    d.mkdir(parents=True, exist_ok=True)

# Fake google.colab, so `from google.colab import drive` works off Colab.
# mount() just prints, and userdata.get() returns None, which is the same
# answer Colab gives when a secret is not set.
g, gc = types.ModuleType("google"), types.ModuleType("google.colab")
gd, gu = types.ModuleType("google.colab.drive"), types.ModuleType("google.colab.userdata")
gd.mount = lambda *a, **k: print("[stub] drive.mount")
gu.get = lambda key: None
gc.drive, gc.userdata, g.colab = gd, gu, gc
sys.modules.update({"google": g, "google.colab": gc,
                    "google.colab.drive": gd, "google.colab.userdata": gu})

# Appended to the cell that picks scenes: keep one Landsat 8 and one Landsat
# 7 scene. Two sensors is what the stage has to handle, and two scenes is
# what a test can afford to download.
SUBSET = ("rows = [next(r for r in rows if r['product_id'].startswith(p))"
          " for p in ('LC08', 'LE07')]\n"
          "print('[harness] subset:', [r['product_id'] for r in rows])")

cells = [c["source"] for c in json.loads(nb_path.read_text())["cells"]
         if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
for i, src in enumerate(cells, 1):
    # Drop the shell and magic lines: they only mean something in Colab.
    src = "\n".join(l for l in src.split("\n") if not l.strip().startswith(("!", "%")))
    src = (src.replace("/content/drive/MyDrive/aral", str(drive_dir))
              .replace("/content/aral-geo-map", str(root_dir))
              .replace("/content/drive", str(mount_dir)))
    # Empty REPO_URL sends the cell down the zip path, so the run uses the
    # local working tree instead of cloning whatever is on GitHub.
    if src.startswith("REPO_URL"):
        src = "REPO_URL = ''" + src[src.index("\n"):]
    for name, value in overrides.items():
        src = "\n".join(f"{name} = {value}" if l.startswith(f"{name} =") else l
                        for l in src.split("\n"))
    if "CELLS = " in src:
        src += "\n" + SUBSET
    head = src.strip().split("\n")[0][:70]
    t = time.time()
    print(f"\n===== cell {i}/{len(cells)}: {head}", flush=True)
    # All cells share one namespace, exactly as they do in a live notebook.
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except BaseException:
        print(f"\n##### FAILED at cell {i} after {time.time()-t:.1f}s", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print(f"----- cell {i} ok, {time.time()-t:.1f}s", flush=True)
print("\n##### ALL CELLS PASSED")
