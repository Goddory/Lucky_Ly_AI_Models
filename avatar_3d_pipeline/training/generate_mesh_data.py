"""
Generate Mesh Data for Geometry VAE Training.

Converts face images into FLAME mesh vertices using DECA reconstruction,
then saves as .npy files for geometry VAE training.

Output structure:
    output_dir/
        meshes/        ← .npy vertex arrays (autoencoder mode)
        faces.npy      ← shared triangle topology

Usage:
    python training/generate_mesh_data.py \
        --input-dir ./data/selfies \
        --output-dir ./data/meshes_dataset \
        --weights-dir ./weights

Colab:
    !python training/generate_mesh_data.py \
        --input-dir /content/drive/MyDrive/avatar_data/selfies \
        --output-dir /content/drive/MyDrive/avatar_data/meshes \
        --weights-dir /content/drive/MyDrive/avatar_weights \
        --max-samples 5000 --resume
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Mesh .npy files are saved directly inside output_dir.
    # train_geometry_vae.py expects them in --data-root/meshes/
    meshes_dir = output_dir
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Collect input images
    extensions = {"*.jpg", "*.jpeg", "*.png", "*.webp", "*.jfif"}
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

    # Load DECA reconstructor
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from core.reconstructor import DECAReconstructor
        reconstructor = DECAReconstructor(
            weights_dir=args.weights_dir,
            device=args.device,
            fallback_on_error=True,
        )
        print("✅ DECA reconstructor loaded")
    except Exception as e:
        print(f"❌ Cannot load reconstructor: {e}")
        sys.exit(1)

    # Load shared faces if already saved (for resume)
    faces_path = output_dir / "faces.npy"
    shared_faces = np.load(str(faces_path)) if faces_path.exists() else None

    # Count already done
    already_done = {p.stem for p in meshes_dir.glob("*.npy")}
    if args.resume and already_done:
        print(f"⏩ Resume mode: skipping {len(already_done)} already-processed meshes")

    # Process images → mesh vertices
    saved = 0

    for img_path in tqdm(image_files, desc="Generating meshes"):
        stem = img_path.stem

        # Resume: skip if .npy already exists
        if args.resume and stem in already_done:
            saved += 1
            continue

        try:
            result = reconstructor.reconstruct(img_path)

            if result.vertices.size == 0:
                continue

            # Save vertices
            np.save(str(meshes_dir / f"{stem}.npy"), result.vertices)

            # Save faces topology (shared across all meshes, save only once)
            if shared_faces is None and result.faces.size > 0:
                shared_faces = result.faces
                np.save(str(faces_path), shared_faces)

            saved += 1
        except Exception as e:
            print(f"⚠️ Failed {img_path.name}: {e}")
            continue

    print(f"\n✅ Generated {saved} mesh files → {meshes_dir}")
    print(f"   Faces topology: {output_dir / 'faces.npy'}")

    if shared_faces is not None:
        print(f"   Vertices per mesh: {shared_faces.max() + 1}")
        print(f"   Triangles: {len(shared_faces)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mesh .npy files from face images via DECA")
    p.add_argument("--input-dir", type=str, required=True,
                    help="Directory of face images")
    p.add_argument("--output-dir", type=str, required=True,
                    help="Output directory for mesh data")
    p.add_argument("--weights-dir", type=str, default="weights",
                    help="Model weights directory")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                    help="Skip images whose .npy mesh already exists in output dir")
    return p.parse_args()


if __name__ == "__main__":
    main()
