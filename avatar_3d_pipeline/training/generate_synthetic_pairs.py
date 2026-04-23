"""
Generate Synthetic Training Pairs from selfie images using DECA reconstruction.

Pipeline:
    Selfie → DECA encode/decode → Render Albedo Map + Normal Map
    Output: (selfie, albedo, normal) triplet

Usage:
    python training/generate_synthetic_pairs.py \
        --input-dir ./data/ffhq_raw \
        --output-dir ./data \
        --weights-dir ./weights \
        --max-samples 5000

Colab:
    !python training/generate_synthetic_pairs.py \
        --input-dir /content/drive/MyDrive/avatar_data/ffhq_raw \
        --output-dir /content/drive/MyDrive/avatar_data \
        --weights-dir /content/drive/MyDrive/avatar_weights \
        --max-samples 5000
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def estimate_albedo_from_image(image: np.ndarray) -> np.ndarray:
    """
    Create a simplified albedo map from a face image.

    For a proper pipeline, this should use DECA's texture UV-unwrap.
    This fallback uses bilateral filtering to approximate diffuse-only albedo.
    """
    # Bilateral filter removes specular highlights while keeping edges
    albedo = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Reduce shadows via adaptive histogram equalization
    lab = cv2.cvtColor(albedo, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    albedo = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return albedo


def estimate_normal_from_image(image: np.ndarray) -> np.ndarray:
    """
    Estimate a pseudo-normal map from a face image using Sobel gradients.

    For a proper pipeline, this should come from DECA's 3D reconstruction.
    This fallback gives a reasonable approximation for initial training.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Sobel gradients for surface orientation estimation
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    # Normalize gradients
    strength = 2.0
    dx = dx / (np.max(np.abs(dx)) + 1e-8) * strength
    dy = dy / (np.max(np.abs(dy)) + 1e-8) * strength

    # Build normal map (right-hand convention)
    normal = np.zeros((*gray.shape, 3), dtype=np.float32)
    normal[:, :, 0] = -dx  # X (red channel)
    normal[:, :, 1] = -dy  # Y (green channel)
    normal[:, :, 2] = 1.0  # Z (blue channel)

    # Normalize to unit vectors
    norm = np.sqrt(np.sum(normal ** 2, axis=-1, keepdims=True))
    normal = normal / (norm + 1e-8)

    # Map from [-1, 1] to [0, 255]
    normal = ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

    return normal


def try_deca_generation(
    image: np.ndarray,
    deca_model,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Try to generate albedo and normal using DECA reconstruction.
    Returns None if DECA is not available or fails.
    """
    try:
        result = deca_model.reconstruct(image)
        if result and hasattr(result, "uv_texture") and result.uv_texture is not None:
            albedo = cv2.resize(result.uv_texture, (target_size, target_size))
            normal = estimate_normal_from_image(albedo)
            return albedo, normal
    except Exception as e:
        print(f"DECA generation failed: {e}")
        pass
    return None



def process_single_image(
    image_path: Path,
    output_selfie_dir: Path,
    output_albedo_dir: Path,
    output_normal_dir: Path,
    target_size: int,
    deca_model=None,
    skip_existing: bool = False,
) -> str | None:
    """Process one selfie image → (selfie, albedo, normal) triplet."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        # Skip images that are too small
        if img_np.shape[0] < 64 or img_np.shape[1] < 64:
            return None

        stem = image_path.stem

        # Resume: skip if all 3 output files already exist
        if skip_existing:
            selfie_out = output_selfie_dir / f"{stem}.png"
            albedo_out = output_albedo_dir / f"{stem}.png"
            normal_out = output_normal_dir / f"{stem}.png"
            if selfie_out.exists() and albedo_out.exists() and normal_out.exists():
                return stem

        selfie_resized = cv2.resize(img_np, (target_size, target_size))

        # Try DECA first, fallback to estimation
        deca_result = None
        if deca_model is not None:
            deca_result = try_deca_generation(img_np, deca_model, target_size)

        if deca_result is not None:
            albedo, normal = deca_result
        else:
            albedo = estimate_albedo_from_image(selfie_resized)
            normal = estimate_normal_from_image(selfie_resized)

        # Save outputs
        cv2.imwrite(str(output_selfie_dir / f"{stem}.png"), cv2.cvtColor(selfie_resized, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_albedo_dir / f"{stem}.png"), cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_normal_dir / f"{stem}.png"), cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))

        return stem
    except Exception as e:
        print(f"⚠️ Failed to process {image_path.name}: {e}")
        return None


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_selfie_dir = output_dir / "selfies"
    output_albedo_dir = output_dir / "albedo"
    output_normal_dir = output_dir / "normal"

    output_selfie_dir.mkdir(parents=True, exist_ok=True)
    output_albedo_dir.mkdir(parents=True, exist_ok=True)
    output_normal_dir.mkdir(parents=True, exist_ok=True)

    # Collect input images
    if args.ext:
        extensions = {f"*{args.ext}"}
    else:
        extensions = {"*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.jfif"}
        
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))

    image_files = sorted(set(image_files))

    if args.max_samples and len(image_files) > args.max_samples:
        image_files = image_files[:args.max_samples]

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        sys.exit(1)

    print(f"📸 Found {len(image_files)} images in {input_dir}")
    print(f"📁 Output: {output_dir}")

    # Try to load DECA for better quality
    deca_model = None
    if not args.skip_deca:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.reconstructor import DECAReconstructor
            deca_model = DECAReconstructor(
                weights_dir=args.weights_dir,
                device=args.device,
                fallback_on_error=False,
            )
            print("✅ DECA loaded — using 3D reconstruction for map generation")
        except Exception as e:
            print(f"⚠️ DECA not available ({e}). Using image-based estimation fallback.")

    # Process images
    valid_stems = []
    for img_path in tqdm(image_files, desc="Generating pairs"):
        stem = process_single_image(
            img_path,
            output_selfie_dir,
            output_albedo_dir,
            output_normal_dir,
            target_size=args.image_size,
            deca_model=deca_model,
            skip_existing=args.resume,
        )
        if stem:
            valid_stems.append(stem)

    # Generate manifest CSV
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["selfie", "albedo", "normal"])
        for stem in valid_stems:
            writer.writerow([
                f"selfies/{stem}.png",
                f"albedo/{stem}.png",
                f"normal/{stem}.png",
            ])

    print(f"\n✅ Generated {len(valid_stems)} pairs")
    print(f"   Failed: {len(image_files) - len(valid_stems)}")
    print(f"   Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic (selfie, albedo, normal) training pairs")
    p.add_argument("--input-dir", type=str, required=True,
                    help="Directory containing raw selfie/face images")
    p.add_argument("--output-dir", type=str, required=True,
                    help="Output directory (will create selfies/, albedo/, normal/)")
    p.add_argument("--weights-dir", type=str, default="weights",
                    help="Model weights directory (for DECA)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ext", type=str, default=None,
                    help="Filter by specific extension (e.g. '.jpg')")
    p.add_argument("--resume", action="store_true",
                    help="Skip images that already have all 3 output files")
    p.add_argument("--skip-deca", action="store_true",
                    help="Skip DECA and use image-based estimation only")
    return p.parse_args()


if __name__ == "__main__":
    main()
