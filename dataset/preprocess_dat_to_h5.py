"""Convert raw AirCraft Tecplot surface files (Components.i.dat) into the .h5
samples expected by AirplaneDataset, driven by result.csv as the master index.

Each raw file is a Tecplot FEPOINT surface triangulation:

    VARIABLES = x,y,z, Cp, Rho, U, V, W, Pressure
    ZONE T="Surface", N = <#nodes>, E = <#tris>, F=FEPOINT, ET=TRIANGLE
    <N rows: x y z Cp Rho U V W Pressure>
    <E rows: i j k>   (1-indexed triangle connectivity)

Output .h5 (consumed by dataset/dataset.py):
    pos     (N, 3)  float32  -> x, y, z
    normals (N, 3)  float32  -> per-node area-weighted normals from the triangulation
    values  (N, 6)  float32  -> Cp, Rho, U, V, W, Pressure
    attrs:  Ma, alpha, beta

Canonical sample name (single source of truth for both training json and eval):
    {id}_{Ma}_{alpha}_{beta}.h5   e.g. "91_7_7_2.h5"
maps to <raw_dir>/{id}/Mach{Ma:05.2f}_Alpha{alpha:05.2f}_Beta{beta:05.2f}/Components.i.dat

To migrate to a NEW dataset: drop the raw folders + a result.csv with columns
[idx, Ma, alpha, beta, ...] under --raw_dir and rerun.

Usage:
    # convert all samples listed in result.csv and write a train/val/test split json
    python dataset/preprocess_dat_to_h5.py --write_split --val_ids 81 82 83 --test_ids 91 92 93

    # convert only a subset of aircraft ids
    python dataset/preprocess_dat_to_h5.py --ids 1 2 3 91
"""
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd

RAW_FILENAME = "Components.i.dat"


def sample_name(sid, ma, alpha, beta):
    """Canonical h5 filename. Ints render without a decimal point (matches result.csv)."""
    def fmt(v):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    return f"{int(sid)}_{fmt(ma)}_{fmt(alpha)}_{fmt(beta)}.h5"


def raw_dir_for(raw_dir, sid, ma, alpha, beta):
    sub = f"Mach{ma:05.2f}_Alpha{alpha:05.2f}_Beta{beta:05.2f}"
    return os.path.join(raw_dir, str(int(sid)), sub, RAW_FILENAME)


def read_tecplot_fepoint(dat_path):
    """Return (nodes (N,9) float64, tris (E,3) int64 0-indexed)."""
    with open(dat_path, "r") as f:
        lines = f.readlines()

    zone_idx = next(i for i, ln in enumerate(lines) if ln.lstrip().upper().startswith("ZONE"))
    header = lines[zone_idx].replace(" ", "").upper()

    def _grab(key):
        seg = header.split(key + "=", 1)[1]
        num = ""
        for ch in seg:
            if ch.isdigit():
                num += ch
            elif num:
                break
        return int(num)

    n_nodes, n_elems = _grab("N"), _grab("E")
    start = zone_idx + 1
    node_lines = lines[start:start + n_nodes]
    elem_lines = lines[start + n_nodes:start + n_nodes + n_elems]

    nodes = np.loadtxt(node_lines, dtype=np.float64)
    tris = np.loadtxt(elem_lines, dtype=np.int64) - 1  # 1-indexed -> 0-indexed
    assert nodes.shape == (n_nodes, 9), f"unexpected node block {nodes.shape}"
    assert tris.shape == (n_elems, 3), f"unexpected element block {tris.shape}"
    return nodes, tris


def compute_vertex_normals(pos, tris):
    """Area-weighted per-vertex normals (face normal magnitude == 2*area)."""
    v0, v1, v2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(pos)
    for k in range(3):
        np.add.at(normals, tris[:, k], face_n)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(norm, 1e-12, None)


def convert_one(dat_path, out_path, ma, alpha, beta):
    nodes, tris = read_tecplot_fepoint(dat_path)
    pos = nodes[:, 0:3]
    values = nodes[:, 3:9]  # Cp, Rho, U, V, W, Pressure
    normals = compute_vertex_normals(pos, tris)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("pos", data=pos.astype(np.float32))
        h.create_dataset("normals", data=normals.astype(np.float32))
        h.create_dataset("values", data=values.astype(np.float32))
        h.attrs["Ma"] = float(ma)
        h.attrs["alpha"] = float(alpha)
        h.attrs["beta"] = float(beta)
    return pos.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="dataset/aircraft_dataset")
    ap.add_argument("--out_dir", default="dataset/aircraft_dataset")
    ap.add_argument("--csv", default=None, help="defaults to <raw_dir>/result.csv")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="only convert these aircraft ids (default: all in csv)")
    ap.add_argument("--write_split", action="store_true",
                    help="also write airplane_dataset.json train/val/test split")
    ap.add_argument("--val_ids", type=int, nargs="*", default=[81, 82, 83],
                    help="aircraft ids assigned to the validation split")
    ap.add_argument("--test_ids", type=int, nargs="*", default=[91, 92, 93],
                    help="aircraft ids assigned to the test split")
    ap.add_argument("--split_json", default="airplane_dataset.json")
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(args.raw_dir, "result.csv")
    df = pd.read_csv(csv_path)
    os.makedirs(args.out_dir, exist_ok=True)

    train_names, val_names, test_names = [], [], []
    for _, row in df.iterrows():
        sid = int(row["idx"])
        ma, alpha, beta = row["Ma"], row["alpha"], row["beta"]
        if args.ids is not None and sid not in args.ids:
            continue
        name = sample_name(sid, ma, alpha, beta)
        dat_path = raw_dir_for(args.raw_dir, sid, ma, alpha, beta)
        out_path = os.path.join(args.out_dir, name)
        if not os.path.exists(dat_path):
            print(f"SKIP  {name}: raw file not found -> {dat_path}")
            continue
        n = convert_one(dat_path, out_path, ma, alpha, beta)
        print(f"OK    {name}: {n} nodes -> {out_path}")
        if sid in args.test_ids:
            test_names.append(name)
        elif sid in args.val_ids:
            val_names.append(name)
        else:
            train_names.append(name)

    if args.write_split:
        split = {"train_set": train_names, "val_set": val_names, "test_set": test_names}
        with open(args.split_json, "w") as f:
            json.dump(split, f, indent=4)
        print(f"split -> {args.split_json}: {len(train_names)} train / "
              f"{len(val_names)} val / {len(test_names)} test")


if __name__ == "__main__":
    main()
