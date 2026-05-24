#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path


SPLITS = ["train", "val_seen", "val_unseen", "test"]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _find_split_file(source_dir: Path, split: str):
    candidates = [
        source_dir / f"REVERIE_{split}_enc.json",
        source_dir / f"REVERIE_{split}.json",
        source_dir / "tasks" / "REVERIE" / "data" / f"REVERIE_{split}_enc.json",
        source_dir / "tasks" / "REVERIE" / "data" / f"REVERIE_{split}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _prepare_split_files(source_dir: Path, target_anno_dir: Path):
    missing = []
    copied = []
    for split in SPLITS:
        src = _find_split_file(source_dir, split)
        dst = target_anno_dir / f"REVERIE_{split}_enc.json"
        if src is None:
            missing.append(split)
            continue
        if src.name.endswith("_enc.json"):
            shutil.copy2(src, dst)
        else:
            data = _read_json(src)
            _write_json(dst, data)
        copied.append((split, str(src), str(dst)))
    return copied, missing


def _maybe_copy_existing_bboxes(source_dir: Path, target_anno_dir: Path):
    candidates = [
        source_dir / "BBoxes.json",
        source_dir / "tasks" / "REVERIE" / "data" / "BBoxes.json",
    ]
    for c in candidates:
        if c.exists():
            dst = target_anno_dir / "BBoxes.json"
            shutil.copy2(c, dst)
            return str(c), str(dst)
    return None, None


def _build_bboxes_from_folder(source_dir: Path, target_anno_dir: Path):
    bbox_dirs = [
        source_dir / "BBox",
        source_dir / "tasks" / "REVERIE" / "data" / "BBox",
    ]
    bbox_dir = None
    for d in bbox_dirs:
        if d.exists() and d.is_dir():
            bbox_dir = d
            break
    if bbox_dir is None:
        return None, None, 0

    merged = {}
    count = 0
    for p in sorted(bbox_dir.glob("*.json")):
        stem = p.stem
        if "_" not in stem:
            continue
        scan, vp = stem.split("_", 1)
        scanvp = f"{scan}_{vp}"
        data = _read_json(p)
        if isinstance(data, dict) and vp in data and isinstance(data[vp], dict):
            merged[scanvp] = data[vp]
        elif isinstance(data, dict):
            # Fallback: keep as-is when file already stores object map only.
            merged[scanvp] = data
        count += 1

    dst = target_anno_dir / "BBoxes.json"
    _write_json(dst, merged)
    return str(bbox_dir), str(dst), count


def _check_shared_r2r_dependencies(root_dir: Path):
    needed = [
        root_dir / "R2R" / "annotations" / "scanvp_candidates.json",
        root_dir / "R2R" / "connectivity",
        root_dir / "R2R" / "features" / "MP3D_eva_clip_g_can.lmdb",
    ]
    status = []
    for p in needed:
        status.append((str(p), p.exists()))
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Prepare REVERIE files for NavGPT-2 layout."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to downloaded REVERIE data root (or REVERIE repo root).",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../datasets",
        help="NavGPT root_dir (contains R2R/, REVERIE/, Matterport3D/).",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()
    target_anno_dir = root_dir / "REVERIE" / "annotations"
    target_anno_dir.mkdir(parents=True, exist_ok=True)

    copied, missing = _prepare_split_files(source_dir, target_anno_dir)
    bbox_src, bbox_dst = _maybe_copy_existing_bboxes(source_dir, target_anno_dir)
    bbox_folder_src, bbox_folder_dst, bbox_files = (None, None, 0)
    if bbox_dst is None:
        bbox_folder_src, bbox_folder_dst, bbox_files = _build_bboxes_from_folder(source_dir, target_anno_dir)

    r2r_dep_status = _check_shared_r2r_dependencies(root_dir)

    print("=== REVERIE Preparation Summary ===")
    print(f"source_dir: {source_dir}")
    print(f"target annotations: {target_anno_dir}")
    for split, src, dst in copied:
        print(f"[OK] {split}: {src} -> {dst}")
    for split in missing:
        print(f"[MISSING] split file for: {split} (expected REVERIE_{split}.json or _enc.json)")

    if bbox_dst is not None:
        print(f"[OK] BBoxes copied: {bbox_src} -> {bbox_dst}")
    elif bbox_folder_dst is not None:
        print(f"[OK] BBoxes built from folder: {bbox_folder_src} ({bbox_files} files) -> {bbox_folder_dst}")
    else:
        print("[MISSING] BBoxes.json or BBox folder not found in source_dir")

    print("\n=== Shared Dependencies Check (required by this codebase) ===")
    for p, ok in r2r_dep_status:
        tag = "OK" if ok else "MISSING"
        print(f"[{tag}] {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
