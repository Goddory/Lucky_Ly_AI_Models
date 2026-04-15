from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh
from PIL import Image


@dataclass
class ExportResult:
    glb_path: Path
    texture_paths: Dict[str, Path]


class AvatarExporter:
    """Phase-4 exporter that packs mesh + PBR textures into GLB/GLTF assets."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_uv(vertices: np.ndarray, uv_coords: Optional[np.ndarray]) -> np.ndarray:
        if uv_coords is not None and uv_coords.size > 0:
            uv = uv_coords.astype(np.float32)
            if uv.max() > 1.0 or uv.min() < 0.0:
                uv_min = uv.min(axis=0, keepdims=True)
                uv_max = uv.max(axis=0, keepdims=True)
                uv = (uv - uv_min) / (uv_max - uv_min + 1e-8)
            return uv

        centered = vertices - vertices.mean(axis=0, keepdims=True)
        x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
        u = 0.5 + np.arctan2(z, x) / (2 * np.pi)
        v = 0.5 - np.arcsin(np.clip(y / (np.linalg.norm(centered, axis=1) + 1e-6), -1, 1)) / np.pi
        return np.stack([u, v], axis=-1).astype(np.float32)

    def _save_texture_maps(
        self,
        textures: Dict[str, np.ndarray],
        asset_name: str,
    ) -> Dict[str, Path]:
        texture_dir = self.output_dir / f"{asset_name}_textures"
        texture_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}
        for key, image in textures.items():
            image = image.astype(np.uint8)
            path = texture_dir / f"{key.lower()}.png"
            Image.fromarray(image).save(path)
            paths[key] = path
        return paths

    def export_glb(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uv_coords: np.ndarray,
        textures: Dict[str, np.ndarray],
        output_name: Optional[str] = None,
    ) -> ExportResult:
        asset_name = output_name or f"avatar_{uuid.uuid4().hex[:8]}"
        uv = self._ensure_uv(vertices, uv_coords)

        tex_paths = self._save_texture_maps(
            {
                "Color": textures["C"],
                "Normal": textures["N"],
                "Specular": textures["S"],
                "Albedo": textures["A"],
            },
            asset_name=asset_name,
        )

        color_img = Image.open(tex_paths["Color"])
        normal_img = Image.open(tex_paths["Normal"])
        specular_img = Image.open(tex_paths["Specular"])

        material = trimesh.visual.material.PBRMaterial(
            name=f"{asset_name}_material",
            baseColorTexture=color_img,
            normalTexture=normal_img,
            # GLTF does not have native specular texture in core PBR; we map to
            # metallicRoughness slot as a practical fallback.
            metallicRoughnessTexture=specular_img,
            metallicFactor=0.1,
            roughnessFactor=0.75,
        )

        mesh = trimesh.Trimesh(
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            process=False,
            maintain_order=True,
        )
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=uv,
            image=color_img,
            material=material,
        )

        glb_path = self.output_dir / f"{asset_name}.glb"
        scene = trimesh.Scene(mesh)
        scene.export(glb_path)

        metadata = {
            "asset": asset_name,
            "glb": str(glb_path.name),
            "textures": {k: str(v.name) for k, v in tex_paths.items()},
        }
        (self.output_dir / f"{asset_name}.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        return ExportResult(glb_path=glb_path, texture_paths=tex_paths)
