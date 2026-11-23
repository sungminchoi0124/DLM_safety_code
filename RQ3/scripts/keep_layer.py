"""Rewrite trace npz files in place keeping one layer (default b23) of every hidden* array. Numbers only."""
import glob, os, sys, numpy as np
layer = sys.argv[1]; files = sorted(glob.glob(sys.argv[2]))
done = skipped = 0; before = after = 0
for f in files:
    with np.load(f, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    names = [str(n) for n in a["layer_names"]]
    if names == [layer]:
        skipped += 1; continue
    i = names.index(layer)
    for k in [k for k in a if k.startswith("hidden")]:
        a[k] = a[k][:, i:i + 1, :]
    a["layer_names"] = np.array([layer])
    before += os.path.getsize(f)
    tmp = f[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **a)
    os.replace(tmp, f)
    after += os.path.getsize(f); done += 1
print(f"files {len(files)} rewritten {done} skipped {skipped} MB {before/1e6:.0f} -> {after/1e6:.0f}")
