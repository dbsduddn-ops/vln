import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from huggingface_hub import snapshot_download


def _build_reverie_bboxes(bbox_dir: Path, out_path: Path):
    merged = {}
    for p in sorted(bbox_dir.glob("*.json")):
        stem = p.stem
        if "_" not in stem:
            continue
        scan, vp = stem.split("_", 1)
        scanvp = f"{scan}_{vp}"
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and vp in data and isinstance(data[vp], dict):
            merged[scanvp] = data[vp]
        elif isinstance(data, dict):
            merged[scanvp] = data
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)


def download_reverie_dataset():
    repo_url = "https://github.com/YuankaiQi/REVERIE.git"
    root = Path("datasets")
    reverie_root = root / "REVERIE"
    anno_dir = reverie_root / "annotations"
    raw_dir = reverie_root / "original_tasks_data"
    anno_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="reverie_clone_") as tmpdir:
        tmp_path = Path(tmpdir)
        clone_dir = tmp_path / "REVERIE"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            check=True,
        )
        src_data = clone_dir / "tasks" / "REVERIE" / "data"
        if not src_data.exists():
            raise FileNotFoundError(f"Cannot find REVERIE data folder: {src_data}")

        # Keep an original copy for reference/debug.
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        shutil.copytree(src_data, raw_dir)

        # Convert split file names to this codebase format: *_enc.json
        for split in ["train", "val_seen", "val_unseen", "test"]:
            src = src_data / f"REVERIE_{split}.json"
            if src.exists():
                dst = anno_dir / f"REVERIE_{split}_enc.json"
                shutil.copy2(src, dst)

        bbox_dir = src_data / "BBox"
        if bbox_dir.exists():
            _build_reverie_bboxes(bbox_dir, anno_dir / "BBoxes.json")

    print("[INFO] REVERIE original data downloaded and prepared:")
    print(f"       raw: {raw_dir}")
    print(f"       annotations: {anno_dir}")


def main(args):
    if args.data:
        if args.dataset == "r2r" or args.dataset == "all":
            # Download the R2R dataset
            snapshot_download(repo_id="ZGZzz/NavGPT-R2R", repo_type="dataset", allow_patterns="*.zip.*", local_dir="datasets", local_dir_use_symlinks=False)

        if args.dataset == "instruct" or args.dataset == "all":
            snapshot_download(repo_id="ZGZzz/NavGPT-Instruct", repo_type="dataset", allow_patterns="*.json", local_dir="datasets/NavGPT-Instruct", local_dir_use_symlinks=False)
        
        if args.dataset == "reverie" or args.dataset == "all":
            download_reverie_dataset()
    
    if args.checkpoints:
        if args.model == "xl" or args.model == "all":
            # Download the NavGPT-2 policy model
            snapshot_download(repo_id="ZGZzz/NavGPT2-FlanT5-XL", repo_type="model", allow_patterns="best_val_unseen_xl", local_dir="datasets/R2R/trained_models", local_dir_use_symlinks=False)

            # Download the NavGPT-2 pretrained Q-former
            snapshot_download(repo_id="ZGZzz/NavGPT2-FlanT5-XL", repo_type="model", allow_patterns="*.pth", local_dir="map_nav_src/models/lavis/output", local_dir_use_symlinks=False)
        
        if args.model == "xxl" or args.model == "all":
            # Download the NavGPT-2 policy model
            snapshot_download(repo_id="ZGZzz/NavGPT2-FlanT5-XXL", repo_type="model", allow_patterns="best_val_unseen_xxl", local_dir="datasets/R2R/trained_models", local_dir_use_symlinks=False)

            # Download the NavGPT-2 pretrained Q-former
            snapshot_download(repo_id="ZGZzz/NavGPT2-FlanT5-XXL", repo_type="model", allow_patterns="*.pth", local_dir="map_nav_src/models/lavis/output", local_dir_use_symlinks=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true", default=False, help="Download the R2R dataset and instruction tuning data for NavGPT-2")
    parser.add_argument("--dataset", type=str, default="all", choices=["r2r", "instruct", "reverie", "all"], help="Dataset to download")
    parser.add_argument("--checkpoints", action="store_true", default=False, help="Download the NavGPT-2 policy model and pretrained Q-former")
    parser.add_argument("--model", type=str, default="all", choices=["xl", "xxl", "all"], help="Model type to download")
    args = parser.parse_args()
    main(args)