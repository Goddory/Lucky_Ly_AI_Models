from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    import gdown
except Exception:
    gdown = None


@dataclass
class WeightSpec:
    name: str
    target_relative_path: Path
    description: str
    required: bool = True
    hf_candidates: List[Tuple[str, str]] = field(default_factory=list)
    gdown_ids: List[str] = field(default_factory=list)
    url_candidates: List[str] = field(default_factory=list)


def _download_from_url(url: str, output_path: Path, timeout: int = 60) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _try_hf(spec: WeightSpec, output_path: Path) -> bool:
    if hf_hub_download is None:
        return False

    for repo_id, filename in spec.hf_candidates:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(output_path.parent),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path != output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded_path), str(output_path))
            return True
        except Exception:
            continue
    return False


def _try_gdown(spec: WeightSpec, output_path: Path) -> bool:
    if gdown is None:
        return False

    for file_id in spec.gdown_ids:
        if not file_id:
            continue
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = gdown.download(
                id=file_id,
                output=str(output_path),
                quiet=False,
                fuzzy=True,
            )
            if result is not None and output_path.exists():
                return True
        except Exception:
            continue
    return False


def _try_url(spec: WeightSpec, output_path: Path) -> bool:
    for url in spec.url_candidates:
        if not url:
            continue
        try:
            _download_from_url(url=url, output_path=output_path)
            return True
        except Exception:
            continue
    return False


def resolve_specs() -> Iterable[WeightSpec]:
    """
    Default weight definitions.

    Some assets (especially FLAME) may require license acceptance. Environment
    variables let users override mirrors without changing source code.
    """

    return [
        WeightSpec(
            name="deca_model",
            target_relative_path=Path("deca/deca_model.tar"),
            description="DECA pre-trained checkpoint (Phase 1 base reconstruction)",
            required=True,
            hf_candidates=[
                (os.getenv("DECA_HF_REPO", "YadiraF/DECA"), os.getenv("DECA_HF_FILE", "deca_model.tar")),
            ],
            gdown_ids=[os.getenv("DECA_GDOWN_ID", "")],
            url_candidates=[os.getenv("DECA_URL", "")],
        ),
        WeightSpec(
            name="hrn_model",
            target_relative_path=Path("hrn/hrn_model.pth"),
            description="Optional HRN checkpoint (alternative to DECA)",
            required=False,
            hf_candidates=[
                (os.getenv("HRN_HF_REPO", "3DHuman/HRN"), os.getenv("HRN_HF_FILE", "hrn_model.pth")),
            ],
            gdown_ids=[os.getenv("HRN_GDOWN_ID", "")],
            url_candidates=[os.getenv("HRN_URL", "")],
        ),
        WeightSpec(
            name="flame_2020_topology",
            target_relative_path=Path("flame2020/head_template.obj"),
            description="FLAME 2020 topology mesh",
            required=True,
            hf_candidates=[
                (os.getenv("FLAME_HF_REPO", "flame-model/FLAME-2020"), os.getenv("FLAME_HF_FILE", "head_template.obj")),
            ],
            gdown_ids=[os.getenv("FLAME_GDOWN_ID", "")],
            url_candidates=[os.getenv("FLAME_URL", "")],
        ),
        WeightSpec(
            name="texture_g_ffhq",
            target_relative_path=Path("ffhq/texture_g.pt"),
            description="Texture prior G initialized from FFHQ-pretrained generator",
            required=False,
            hf_candidates=[
                (os.getenv("FFHQ_G_HF_REPO", "ffhq/stylegan2"), os.getenv("FFHQ_G_HF_FILE", "texture_g.pt")),
            ],
            gdown_ids=[os.getenv("FFHQ_G_GDOWN_ID", "")],
            url_candidates=[os.getenv("FFHQ_G_URL", "")],
        ),
        WeightSpec(
            name="texture_ga_facescape",
            target_relative_path=Path("facescape/texture_ga.pt"),
            description="G_A initialization from FaceScape/face texture pretraining",
            required=False,
            hf_candidates=[
                (os.getenv("FACESCAPE_GA_HF_REPO", "facescape/pbr-texture"), os.getenv("FACESCAPE_GA_HF_FILE", "texture_ga.pt")),
            ],
            gdown_ids=[os.getenv("FACESCAPE_GA_GDOWN_ID", "")],
            url_candidates=[os.getenv("FACESCAPE_GA_URL", "")],
        ),
        WeightSpec(
            name="texture_gc_facescape",
            target_relative_path=Path("facescape/texture_gc.pt"),
            description="G_C initialization from FaceScape/face texture pretraining",
            required=False,
            hf_candidates=[
                (os.getenv("FACESCAPE_GC_HF_REPO", "facescape/pbr-texture"), os.getenv("FACESCAPE_GC_HF_FILE", "texture_gc.pt")),
            ],
            gdown_ids=[os.getenv("FACESCAPE_GC_GDOWN_ID", "")],
            url_candidates=[os.getenv("FACESCAPE_GC_URL", "")],
        ),
        WeightSpec(
            name="texture_ge_facescape",
            target_relative_path=Path("facescape/texture_ge.pt"),
            description="G_E PBR extractor initialization from FaceScape-like priors",
            required=False,
            hf_candidates=[
                (os.getenv("FACESCAPE_GE_HF_REPO", "facescape/pbr-texture"), os.getenv("FACESCAPE_GE_HF_FILE", "texture_ge.pt")),
            ],
            gdown_ids=[os.getenv("FACESCAPE_GE_GDOWN_ID", "")],
            url_candidates=[os.getenv("FACESCAPE_GE_URL", "")],
        ),
    ]


def _try_drive_fallback(spec: WeightSpec, output_path: Path) -> bool:
    """Copy weights from Google Drive if running on Colab with mounted Drive."""
    drive_base = Path(os.getenv("AVATAR_DRIVE_WEIGHTS", "/content/drive/MyDrive/avatar_weights"))
    if not drive_base.exists():
        return False

    candidate = drive_base / spec.target_relative_path
    if candidate.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(candidate), str(output_path))
        return True
    return False


def download_weights(weights_dir: Path, include_optional: bool = True) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[str] = []
    missing_required: List[str] = []
    missing_optional: List[str] = []

    for spec in resolve_specs():
        if not include_optional and not spec.required:
            continue

        target = weights_dir / spec.target_relative_path
        if target.exists():
            downloaded.append(f"[SKIP] {spec.name} already exists")
            continue

        # Try Google Drive first (fastest on Colab)
        ok = _try_drive_fallback(spec, target)
        if ok:
            downloaded.append(f"[DRIVE] {spec.name} -> {target}")
            continue

        ok = _try_hf(spec, target)
        if not ok:
            ok = _try_gdown(spec, target)
        if not ok:
            ok = _try_url(spec, target)

        if ok:
            downloaded.append(f"[OK]   {spec.name} -> {target}")
        else:
            if spec.required:
                missing_required.append(f"{spec.name}: {spec.description}")
            else:
                missing_optional.append(f"{spec.name}: {spec.description}")

    for line in downloaded:
        print(line)

    if missing_optional:
        print("\nOptional weights were not downloaded:")
        for line in missing_optional:
            print(f"  - {line}")

    if missing_required:
        print("\nMissing required weights:")
        for line in missing_required:
            print(f"  - {line}")
        print(
            "\nSet mirror environment variables (e.g., DECA_HF_REPO / DECA_HF_FILE, "
            "FLAME_URL) and re-run this script."
        )
        raise RuntimeError("Required weight download failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download avatar pipeline pre-trained weights.")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "weights"),
        help="Target directory for checkpoints.",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Download only required weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_weights(weights_dir=Path(args.weights_dir), include_optional=not args.required_only)


if __name__ == "__main__":
    main()
