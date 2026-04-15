from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, RandomResizedCrop
from torchvision.transforms import functional as TF


@dataclass
class SampleEntry:
    selfie: Path
    albedo: Path
    normal: Path


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _random_lighting(image: Image.Image) -> Image.Image:
    tensor = TF.to_tensor(image)
    gain = random.uniform(0.75, 1.25)
    gamma = random.uniform(0.80, 1.25)
    bias = random.uniform(-0.05, 0.05)

    tensor = torch.clamp((tensor + bias).pow(gamma) * gain, 0.0, 1.0)
    return TF.to_pil_image(tensor)


class SelfiePBRPairDataset(Dataset):
    """
    Dataset for (selfie, target albedo, target normal) pairs.

    Supported input formats:
    - manifest CSV with columns: selfie, albedo, normal
    - manifest JSON list of objects with keys: selfie, albedo, normal
    - automatic directory scan with structure:
      root/selfies/*, root/albedo/*, root/normal/* (matched by filename stem)
    """

    def __init__(
        self,
        root_dir: str | Path,
        manifest_path: str | Path | None = None,
        image_size: int = 512,
        train: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.image_size = image_size
        self.train = train

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError("No dataset samples found for SelfiePBRPairDataset.")

        self.color_jitter = ColorJitter(
            brightness=0.35,
            contrast=0.35,
            saturation=0.35,
            hue=0.08,
        )

    def _build_samples(self) -> List[SampleEntry]:
        if self.manifest_path is not None:
            suffix = self.manifest_path.suffix.lower()
            if suffix == ".csv":
                return self._from_csv(self.manifest_path)
            if suffix == ".json":
                return self._from_json(self.manifest_path)
            raise ValueError("manifest_path must be .csv or .json")
        return self._from_directory_scan(self.root_dir)

    def _from_csv(self, path: Path) -> List[SampleEntry]:
        rows: List[SampleEntry] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    SampleEntry(
                        selfie=(self.root_dir / row["selfie"]).resolve(),
                        albedo=(self.root_dir / row["albedo"]).resolve(),
                        normal=(self.root_dir / row["normal"]).resolve(),
                    )
                )
        return [s for s in rows if s.selfie.exists() and s.albedo.exists() and s.normal.exists()]

    def _from_json(self, path: Path) -> List[SampleEntry]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows: List[SampleEntry] = []
        for row in payload:
            rows.append(
                SampleEntry(
                    selfie=(self.root_dir / row["selfie"]).resolve(),
                    albedo=(self.root_dir / row["albedo"]).resolve(),
                    normal=(self.root_dir / row["normal"]).resolve(),
                )
            )
        return [s for s in rows if s.selfie.exists() and s.albedo.exists() and s.normal.exists()]

    def _from_directory_scan(self, root_dir: Path) -> List[SampleEntry]:
        selfies_dir = root_dir / "selfies"
        albedo_dir = root_dir / "albedo"
        normal_dir = root_dir / "normal"

        stem_to_paths: Dict[str, Dict[str, Path]] = {}

        for kind, folder in (("selfie", selfies_dir), ("albedo", albedo_dir), ("normal", normal_dir)):
            if not folder.exists():
                continue
            for path in folder.glob("*"):
                if path.is_file():
                    stem_to_paths.setdefault(path.stem, {})[kind] = path

        rows: List[SampleEntry] = []
        for _, bundle in stem_to_paths.items():
            if {"selfie", "albedo", "normal"}.issubset(set(bundle.keys())):
                rows.append(
                    SampleEntry(
                        selfie=bundle["selfie"],
                        albedo=bundle["albedo"],
                        normal=bundle["normal"],
                    )
                )
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_shared_geometric_aug(
        self,
        images: Sequence[Image.Image],
    ) -> List[Image.Image]:
        selfie, albedo, normal = images

        i, j, h, w = RandomResizedCrop.get_params(
            selfie,
            scale=(0.65, 1.0),
            ratio=(0.9, 1.1),
        )
        selfie = TF.resized_crop(selfie, i, j, h, w, (self.image_size, self.image_size))
        albedo = TF.resized_crop(albedo, i, j, h, w, (self.image_size, self.image_size))
        normal = TF.resized_crop(normal, i, j, h, w, (self.image_size, self.image_size))

        angle = random.uniform(-18.0, 18.0)
        translate = (
            int(random.uniform(-0.06, 0.06) * self.image_size),
            int(random.uniform(-0.06, 0.06) * self.image_size),
        )
        scale = random.uniform(0.92, 1.08)
        shear = [random.uniform(-6.0, 6.0), random.uniform(-4.0, 4.0)]

        selfie = TF.affine(selfie, angle=angle, translate=translate, scale=scale, shear=shear)
        albedo = TF.affine(albedo, angle=angle, translate=translate, scale=scale, shear=shear)
        normal = TF.affine(normal, angle=angle, translate=translate, scale=scale, shear=shear)

        if random.random() < 0.5:
            selfie = TF.hflip(selfie)
            albedo = TF.hflip(albedo)
            normal = TF.hflip(normal)

        return [selfie, albedo, normal]

    def _apply_selfie_appearance_aug(self, selfie: Image.Image) -> Image.Image:
        selfie = self.color_jitter(selfie)
        selfie = _random_lighting(selfie)

        if random.random() < 0.30:
            sigma = random.uniform(0.1, 1.2)
            selfie = selfie.filter(ImageFilter.GaussianBlur(radius=sigma))

        return selfie

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        selfie = _load_rgb(sample.selfie)
        albedo = _load_rgb(sample.albedo)
        normal = _load_rgb(sample.normal)

        if self.train:
            selfie, albedo, normal = self._apply_shared_geometric_aug([selfie, albedo, normal])
            selfie = self._apply_selfie_appearance_aug(selfie)
        else:
            selfie = TF.resize(selfie, [self.image_size, self.image_size])
            albedo = TF.resize(albedo, [self.image_size, self.image_size])
            normal = TF.resize(normal, [self.image_size, self.image_size])

        selfie_t = TF.to_tensor(selfie)
        albedo_t = TF.to_tensor(albedo)
        normal_t = TF.to_tensor(normal)

        selfie_t = (selfie_t * 2.0) - 1.0
        albedo_t = (albedo_t * 2.0) - 1.0
        normal_t = (normal_t * 2.0) - 1.0

        return {
            "selfie": selfie_t,
            "target_albedo": albedo_t,
            "target_normal": normal_t,
        }
