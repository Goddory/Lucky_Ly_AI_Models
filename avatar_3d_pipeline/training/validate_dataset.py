"""
Dataset Validation — Check quality of training pairs.

Checks:
  - Image resolution meets minimum
  - All 3 maps exist for each pair
  - No corrupt/unreadable images
  - Color distribution analysis
  - Duplicate detection

Usage:
    python training/validate_dataset.py --data-root ./data --min-size 256

    python training/validate_dataset.py \
        --data-root /content/drive/MyDrive/avatar_data \
        --min-size 512 --fix
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def image_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_image(path: Path, min_size: int) -> dict:
    """Check a single image for quality issues."""
    result = {"path": str(path), "ok": True, "issues": []}

    if not path.exists():
        result["ok"] = False
        result["issues"].append("FILE_MISSING")
        return result

    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path).convert("RGB")
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"CORRUPT: {e}")
        return result

    w, h = img.size
    if w < min_size or h < min_size:
        result["ok"] = False
        result["issues"].append(f"TOO_SMALL: {w}x{h} (min {min_size})")

    arr = np.array(img)
    mean_val = arr.mean()
    if mean_val < 10:
        result["issues"].append("TOO_DARK")
    if mean_val > 245:
        result["issues"].append("TOO_BRIGHT")

    std_val = arr.std()
    if std_val < 5:
        result["issues"].append("LOW_CONTRAST")

    result["size"] = (w, h)
    result["mean"] = float(mean_val)
    result["std"] = float(std_val)
    return result


def validate_dataset(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    selfie_dir = root / "selfies"
    albedo_dir = root / "albedo"
    normal_dir = root / "normal"

    if not selfie_dir.exists():
        print(f"❌ selfies/ directory not found in {root}")
        return

    selfie_files = sorted(
        list(selfie_dir.glob("*.png"))
        + list(selfie_dir.glob("*.jpg"))
        + list(selfie_dir.glob("*.jpeg"))
    )

    print(f"🔍 Validating {len(selfie_files)} samples in {root}")
    print(f"   Min size: {args.min_size}px")

    valid = []
    invalid = []
    missing_pairs = []
    duplicates = set()
    seen_hashes: dict[str, Path] = {}

    for sf in tqdm(selfie_files, desc="Checking"):
        stem = sf.stem

        # Check all 3 maps exist
        albedo_path = None
        normal_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            if (albedo_dir / f"{stem}{ext}").exists():
                albedo_path = albedo_dir / f"{stem}{ext}"
            if (normal_dir / f"{stem}{ext}").exists():
                normal_path = normal_dir / f"{stem}{ext}"

        if albedo_path is None or normal_path is None:
            missing_pairs.append(stem)
            continue

        # Check each image
        checks = []
        for p in [sf, albedo_path, normal_path]:
            checks.append(check_image(p, args.min_size))

        all_ok = all(c["ok"] for c in checks)
        has_warnings = any(c.get("issues") for c in checks)

        # Duplicate check (by selfie hash)
        h = image_hash(sf)
        if h in seen_hashes:
            duplicates.add(stem)
            if not args.keep_dupes:
                continue
        seen_hashes[h] = sf

        if all_ok and not (has_warnings and args.strict):
            valid.append({
                "stem": stem,
                "selfie": sf,
                "albedo": albedo_path,
                "normal": normal_path
            })
        else:
            issues = []
            for c in checks:
                issues.extend(c.get("issues", []))
            invalid.append({"stem": stem, "issues": issues})

    # Report
    print(f"\n{'='*50}")
    print(f"📊 Validation Results")
    print(f"{'='*50}")
    print(f"  Total selfies found: {len(selfie_files)}")
    print(f"  ✅ Valid pairs: {len(valid)}")
    print(f"  ❌ Invalid: {len(invalid)}")
    print(f"  ⚠️  Missing albedo/normal: {len(missing_pairs)}")
    print(f"  🔁 Duplicates: {len(duplicates)}")

    if invalid and args.verbose:
        print(f"\n  Invalid samples:")
        for item in invalid[:20]:
            print(f"    {item['stem']}: {', '.join(item['issues'])}")
        if len(invalid) > 20:
            print(f"    ... and {len(invalid) - 20} more")

    # Fix mode: remove invalid samples and regenerate manifest
    if args.fix:
        print(f"\n🔧 Fix mode: removing invalid samples...")
        removed = 0
        for item in invalid:
            stem = item["stem"]
            for d in [selfie_dir, albedo_dir, normal_dir]:
                for ext in [".png", ".jpg", ".jpeg"]:
                    p = d / f"{stem}{ext}"
                    if p.exists():
                        p.unlink()
                        removed += 1
        print(f"   Removed {removed} files")

    # Regenerate manifest
    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["selfie", "albedo", "normal"])
        for item in valid:
            writer.writerow([
                f"selfies/{item['stem']}.png",
                f"albedo/{item['stem']}.png",
                f"normal/{item['stem']}.png",
            ])

    print(f"\n📄 Updated manifest: {manifest_path} ({len(valid)} entries)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate avatar training dataset quality")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--min-size", type=int, default=256)
    p.add_argument("--fix", action="store_true",
                    help="Remove invalid samples and regenerate manifest")
    p.add_argument("--strict", action="store_true",
                    help="Treat warnings as errors")
    p.add_argument("--keep-dupes", action="store_true",
                    help="Keep duplicate images")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dataset(args)
