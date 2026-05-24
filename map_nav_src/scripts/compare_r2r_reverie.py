#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_split(data, dataset_name: str):
    n_items = len(data)
    n_scans = len({x.get("scan") for x in data if x.get("scan") is not None})
    n_path_ids = len({x.get("path_id") for x in data if x.get("path_id") is not None})
    avg_path_len = 0.0
    path_lens = [len(x.get("path", [])) for x in data]
    if path_lens:
        avg_path_len = sum(path_lens) / len(path_lens)

    if dataset_name.upper() == "REVERIE":
        # REVERIE entries keep a list in "instructions"
        inst_counts = [len(x.get("instructions", [])) for x in data]
        total_instructions = sum(inst_counts)
        with_obj = sum(1 for x in data if x.get("objId") is not None)
        ids = [x.get("id") for x in data if x.get("id") is not None]
    else:
        # R2R enc files also have "instructions" list
        inst_counts = [len(x.get("instructions", [])) for x in data]
        total_instructions = sum(inst_counts)
        with_obj = 0
        ids = [x.get("path_id") for x in data if x.get("path_id") is not None]

    top_scans = Counter([x.get("scan") for x in data if x.get("scan") is not None]).most_common(5)

    return {
        "items": n_items,
        "total_instructions": total_instructions,
        "avg_instructions_per_item": (total_instructions / n_items) if n_items else 0.0,
        "unique_scans": n_scans,
        "unique_path_ids": n_path_ids,
        "avg_path_len": avg_path_len,
        "has_objId_count": with_obj,
        "unique_ids": len(set(ids)),
        "top_scans": top_scans,
    }


def summarize_dataset(root: Path, dataset: str, expected_splits):
    anno_dir = root / dataset / "annotations"
    out = {
        "dataset": dataset,
        "annotation_dir": str(anno_dir),
        "annotation_dir_exists": anno_dir.exists(),
        "splits": {},
    }
    if not anno_dir.exists():
        return out

    for split in expected_splits:
        p = anno_dir / f"{dataset}_{split}_enc.json"
        # folder is upper-case dataset names
        p_alt = anno_dir / f"{dataset.upper()}_{split}_enc.json"
        split_path = p if p.exists() else p_alt
        if not split_path.exists():
            out["splits"][split] = {"exists": False, "path": str(split_path)}
            continue
        data = read_json(split_path)
        stats = analyze_split(data, dataset)
        out["splits"][split] = {"exists": True, "path": str(split_path), "stats": stats}

    # extra files often needed in this codebase
    out["extra"] = {
        "bboxes_json": str(anno_dir / "BBoxes.json"),
        "bboxes_exists": (anno_dir / "BBoxes.json").exists(),
    }
    return out


def shared_dependency_status(root: Path):
    checks = {
        "r2r_connectivity_dir": root / "R2R" / "connectivity",
        "r2r_candidates_json": root / "R2R" / "annotations" / "scanvp_candidates.json",
        "r2r_feature_lmdb": root / "R2R" / "features" / "MP3D_eva_clip_g_can.lmdb",
        "matterport_scans_dir": root / "Matterport3D" / "v1_unzip_scans",
    }
    return {k: {"path": str(v), "exists": v.exists()} for k, v in checks.items()}


def print_dataset_report(report):
    print(f"\n=== {report['dataset']} ===")
    print(f"annotation_dir: {report['annotation_dir']}")
    print(f"annotation_dir_exists: {report['annotation_dir_exists']}")
    for split, s in report["splits"].items():
        if not s["exists"]:
            print(f"- {split:14s} MISSING ({s['path']})")
            continue
        st = s["stats"]
        print(
            f"- {split:14s} items={st['items']}, total_instructions={st['total_instructions']}, "
            f"avg_inst/item={st['avg_instructions_per_item']:.2f}, scans={st['unique_scans']}, "
            f"path_ids={st['unique_path_ids']}, avg_path_len={st['avg_path_len']:.2f}, "
            f"objId_count={st['has_objId_count']}"
        )
    if "extra" in report:
        print(f"bboxes_exists: {report['extra']['bboxes_exists']} ({report['extra']['bboxes_json']})")


def main():
    parser = argparse.ArgumentParser(description="Compare R2R and REVERIE dataset readiness/statistics.")
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="/coss/ywyoun/VLN/NavGPT-2/datasets",
        help="Root path that contains R2R/ and REVERIE/ folders.",
    )
    args = parser.parse_args()
    root = Path(args.datasets_root).resolve()

    print(f"datasets_root: {root}")
    if not root.exists():
        raise FileNotFoundError(f"datasets_root does not exist: {root}")

    r2r = summarize_dataset(root, "R2R", ["train", "val_train_seen", "val_seen", "val_unseen", "test"])
    rev = summarize_dataset(root, "REVERIE", ["train", "val_seen", "val_unseen", "test"])

    print_dataset_report(r2r)
    print_dataset_report(rev)

    print("\n=== Shared Dependencies (NavGPT-2 runtime) ===")
    dep = shared_dependency_status(root)
    for k, v in dep.items():
        print(f"- {k:24s} exists={v['exists']} ({v['path']})")


if __name__ == "__main__":
    main()
