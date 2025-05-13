import os
import shutil
import random
import re
import argparse
from pathlib import Path

def organize_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15):
    # 1. Check source
    if not os.path.isdir(source_dir):
        print(f"Error: source directory '{source_dir}' not found")
        return False
    # 2. Prepare destination splits
    dest_path = Path(dest_dir)
    splits = {
        "train": dest_path / "train",
        "val":   dest_path / "val",
        "test":  dest_path / "test"
    }
    for split_path in splits.values():
        if split_path.exists():
            shutil.rmtree(split_path)
        split_path.mkdir(parents=True, exist_ok=True)

    # 3. Gather and group files by ID
    files = [f for f in os.listdir(source_dir)
             if os.path.isfile(os.path.join(source_dir, f))
             and f.lower().endswith(('.png','.jpg','.jpeg','.tif','.bmp'))]

    groups = {}
    for fname in files:
        m = re.match(r'^(\d+)_([OMF])', fname)
        if m:
            idx = m.group(1)
            groups.setdefault(idx, []).append(fname)

    # 4. Filter valid sets: need at least original + mask
    valid_ids = []
    for idx, flist in groups.items():
        if f"{idx}_O.png" in flist and f"{idx}_M.png" in flist:
            valid_ids.append(idx)
    valid_ids.sort()

    if not valid_ids:
        print("Error: no valid image sets found in source")
        return False

    # 5. Shuffle & split
    random.shuffle(valid_ids)
    n = len(valid_ids)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))
    split_ids = {
        "train": valid_ids[:t_end],
        "val":   valid_ids[t_end:v_end],
        "test":  valid_ids[v_end:]
    }

    # 6. Copy files
    for split, ids in split_ids.items():
        for idx in ids:
            dst_folder = splits[split] / idx
            dst_folder.mkdir(exist_ok=True)
            for fname in groups[idx]:
                src = Path(source_dir) / fname
                dst = dst_folder / fname
                shutil.copy2(src, dst)

    # 7. Report
    for split in ["train","val","test"]:
        cnt = len(list((splits[split]).iterdir()))
        print(f"{split.capitalize()} sets: {cnt}")
    return True

def verify_dataset(dest_dir):
    base = Path(dest_dir)
    ok = True
    for split in ["train","val","test"]:
        spath = base / split
        if not spath.is_dir():
            print(f"Missing split directory: {split}")
            ok = False
            continue
        for folder in spath.iterdir():
            if not folder.is_dir():
                continue
            files = [f.name for f in folder.iterdir()]
            idx = folder.name
            req = {f"{idx}_O.png", f"{idx}_M.png"}
            if not req.issubset(files):
                print(f"Incomplete set {idx} in {split}: missing O/M")
                ok = False
            fs = [f for f in files if f.startswith(f"{idx}_F")]
            if not fs:
                print(f"Incomplete set {idx} in {split}: no forged images")
                ok = False
    if ok:
        print("Dataset verified successfully")
    return ok

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Organize flat CMFD dataset into train/val/test")
    p.add_argument("--source", required=True, help="path to flat image folder")
    p.add_argument("--dest",   required=True, help="output root for organized dataset")
    p.add_argument("--train",  type=float, default=0.7, help="train split ratio")
    p.add_argument("--val",    type=float, default=0.15, help="validation split ratio")
    args = p.parse_args()

    print(f"Organizing '{args.source}' → '{args.dest}'")
    if organize_dataset(args.source, args.dest, args.train, args.val):
        verify_dataset(args.dest)
