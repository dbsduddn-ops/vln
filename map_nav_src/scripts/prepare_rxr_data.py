#!/usr/bin/env python3
"""
Convert official RxR guide annotations (JSON Lines) to NavGPT-2 / DUET layout.

Official RxR guide schema:
  https://github.com/google-research-datasets/RxR

Expected outputs (for --dataset rxr-en):
  datasets/RXR-EN/annotations/RXR-EN_{split}_enc.json

Each entry is one guide instruction (instructions=[single string]) with
instruction_id for instr_id = {path_id}_{instruction_id}.
"""
import argparse
import gzip
import json
from pathlib import Path

# NavGPT split name -> RxR official guide file stems (without extension)
SPLIT_SOURCES = {
    "train": ["rxr_train_guide", "train_guide"],
    "val_seen": ["rxr_val_seen_guide", "val_seen_guide"],
    "val_unseen": ["rxr_val_unseen_guide", "val_unseen_guide"],
    "test": [
        "rxr_test_standard_public_guide",
        "rxr_test_standard_guide",
        "rxr_test_standard",
        "test_standard_guide",
    ],
}

DEFAULT_EN_LANGUAGES = ("en-IN", "en-US")


def _open_text(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_records(path: Path):
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            raise ValueError(f"Expected JSON list in {path}")
        return
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _find_guide_file(source_dir: Path, split: str):
    stems = SPLIT_SOURCES[split]
    exts = [".jsonl", ".jsonl.gz", ".json", ".json.gz"]
    for stem in stems:
        for ext in exts:
            p = source_dir / f"{stem}{ext}"
            if p.exists():
                return p
    for stem in stems:
        matches = sorted(source_dir.glob(f"{stem}*"))
        if matches:
            return matches[0]
    return None


def _convert_entry(raw, languages):
    lang = raw.get("language", "")
    if languages and lang not in languages:
        return None
    required = ("instruction_id", "scan", "path", "heading", "instruction")
    for key in required:
        if key not in raw:
            raise KeyError(
                f"Missing field {key!r} in RxR entry "
                f"path_id={raw.get('path_id')} instruction_id={raw.get('instruction_id')}"
            )
    out = {
        "instruction_id": raw["instruction_id"],
        "scan": raw["scan"],
        "path": raw["path"],
        "heading": raw["heading"],
        "instructions": [raw["instruction"]],
        "language": lang,
    }
    if "path_id" in raw:
        out["path_id"] = raw["path_id"]
    return out


def _prepare_split(source_dir: Path, split: str, out_path: Path, languages):
    src = _find_guide_file(source_dir, split)
    if src is None:
        return None, 0, 0
    kept, skipped = [], 0
    for raw in _iter_records(src):
        item = _convert_entry(raw, languages)
        if item is None:
            skipped += 1
            continue
        kept.append(item)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)
    return str(src), len(kept), skipped


def _check_shared_r2r_dependencies(root_dir: Path):
    needed = [
        root_dir / "R2R" / "annotations" / "scanvp_candidates.json",
        root_dir / "R2R" / "connectivity",
        root_dir / "R2R" / "features" / "MP3D_eva_clip_g_can.lmdb",
    ]
    return [(str(p), p.exists()) for p in needed]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RxR-EN guide annotations for NavGPT-2 (--dataset rxr-en)."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing official RxR guide files (json/jsonl), "
        "e.g. rxr_val_unseen_guide.jsonl from https://github.com/google-research-datasets/RxR",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../datasets",
        help="NavGPT datasets root (will write RXR-EN/annotations/).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_EN_LANGUAGES),
        help="Comma-separated RxR language tags to keep (default: en-IN,en-US for RxR-EN).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val_seen,val_unseen,test",
        help="NavGPT split names to export.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()
    anno_dir = root_dir / "RXR-EN" / "annotations"
    languages = tuple(x.strip() for x in args.languages.split(",") if x.strip())
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    print("=== RxR -> NavGPT-2 (RXR-EN) Preparation ===")
    print(f"source_dir: {source_dir}")
    print(f"target: {anno_dir}")
    print(f"languages: {languages}")

    for split in splits:
        if split not in SPLIT_SOURCES:
            print(f"[SKIP] unknown split: {split}")
            continue
        out_path = anno_dir / f"RXR-EN_{split}_enc.json"
        src, n_kept, n_skip = _prepare_split(source_dir, split, out_path, languages)
        if src is None:
            print(f"[MISSING] {split}: no guide file found (stems: {SPLIT_SOURCES[split]})")
        else:
            print(f"[OK] {split}: {src} -> {out_path} ({n_kept} kept, {n_skip} lang-filtered)")

    print("\n=== Shared MP3D dependencies (same as R2R eval) ===")
    for p, ok in _check_shared_r2r_dependencies(root_dir):
        print(f"[{'OK' if ok else 'MISSING'}] {p}")

    print("\nNext: bash scripts/test_rxr_zeroshot_from_r2r_xl.sh")
    print("Ref: RxR guide annotations — https://github.com/google-research-datasets/RxR")


if __name__ == "__main__":
    main()
