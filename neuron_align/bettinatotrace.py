import pandas as pd
import numpy as np

# --- FILE PATHS ---
FILE1 = "/Users/amyzheng/Desktop/bettina_assigntotrace/006real.csv"  # napari output 
FILE2 = "/Users/amyzheng/Desktop/bettina_assigntotrace/4569_dob6-10-24_Cell1_60x_0006_xyzCoordinates (1).csv" #xyzcoordinate

file1 = pd.read_csv(FILE1)
file2 = pd.read_csv(FILE2)

f1_needed = ["axis-0", "axis-1", "axis-2"]
missing1 = [c for c in f1_needed if c not in file1.columns]
if missing1:
    raise ValueError(f"Missing in file1: {missing1}. Got columns: {list(file1.columns)}")
f1_xyz = pd.DataFrame({
    "x": file1["axis-2"],
    "y": file1["axis-1"],
    "z": file1["axis-0"],
}).copy()

for c in ["x","y","z"]:
    f1_xyz[c] = pd.to_numeric(f1_xyz[c], errors="coerce")

mask1 = f1_xyz.notna().all(axis=1)
coords1 = f1_xyz.loc[mask1, ["x","y","z"]].to_numpy(dtype=float)
idx1   = file1.index[mask1].to_numpy()  # original rows that survived

if {"x","y","z"}.issubset(file2.columns):
    f2_xyz = file2.loc[:, ["x","y","z"]].copy()
elif {"z","y","x"}.issubset(file2.columns):
    tmp = file2.loc[:, ["z","y","x"]].copy()
    tmp.columns = ["z","y","x"]
    f2_xyz = tmp.loc[:, ["x","y","z"]].copy()
else:
    raise ValueError(f"file2 must have either ['x','y','z'] or ['z','y','x']. Got: {list(file2.columns)}")

for c in ["x","y","z"]:
    f2_xyz[c] = pd.to_numeric(f2_xyz[c], errors="coerce")

mask2 = f2_xyz.notna().all(axis=1)
coords2 = f2_xyz.loc[mask2, ["x","y","z"]].to_numpy(dtype=float)
file2_clean = file2.loc[mask2].reset_index(drop=True)

def nearest_indices_xy(points, branch):
    """
    points: (N,3) float [x,y,z]
    branch: (M,3) float [x,y,z]
    returns: list of length N with argmin indices into branch
    """
    Bxy = branch[:, :2]  # x,y
    idxs = []
    for p in points:
        d2 = np.sum((Bxy - p[:2])**2, axis=1)
        idxs.append(int(np.argmin(d2)))
    return idxs

nearest_list = nearest_indices_xy(coords1, coords2)

if "path" in file2_clean.columns:
    matched_paths = [str(file2_clean.loc[i, "path"]).strip() for i in nearest_list]
    file1 = file1.copy()
    file1["nearest_path"] = pd.NA
    file1.loc[idx1, "nearest_path"] = matched_paths

# --- OPTIONAL: quick accuracy if ground truth 'path' exists in file1 ---
if "path" in file1.columns and "nearest_path" in file1.columns:
    valid = file1["nearest_path"].notna()
    truth = file1.loc[valid, "path"].astype(str).str.strip()
    pred  = file1.loc[valid, "nearest_path"].astype(str).str.strip()
    ok = (truth == pred)
    print(f"Evaluated rows: {valid.sum()} / {len(file1)}")
    print(f"Accuracy: {ok.mean():.2%}  (Correct {int(ok.sum())} / {ok.size})")


file1.to_csv("/Users/amyzheng/Desktop/bettina_assigntotrace/006dsfsdreal_with_nearest_path.csv", index=False)

print("coords1 shape/dtype:", coords1.shape, coords1.dtype)
print("coords2 shape/dtype:", coords2.shape, coords2.dtype)
print("nearest_list first 10:", nearest_list[:10])
