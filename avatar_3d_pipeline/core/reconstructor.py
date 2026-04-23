from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh

LOGGER = logging.getLogger(__name__)


@dataclass
class ReconstructionOutput:
    """Container for phase-1 2D-to-3D reconstruction outputs."""

    vertices: np.ndarray
    faces: np.ndarray
    uv_coords: np.ndarray
    uv_faces: np.ndarray
    shape_params: np.ndarray
    expression_params: np.ndarray
    camera_pose: np.ndarray
    uv_texture: np.ndarray


class DECAReconstructor:
    """
    DECA/EMOCA-backed reconstructor that estimates a base FLAME mesh from one selfie.

    The implementation prefers DECA pre-trained weights and falls back to a template mesh
    only when DECA is unavailable in the runtime environment.
    """

    def __init__(
        self,
        weights_dir: str | Path,
        device: Optional[str] = None,
        use_emoca: bool = False,
        image_size: int = 224,
        fallback_on_error: bool = False,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_emoca = use_emoca
        self.image_size = image_size
        self.fallback_on_error = fallback_on_error

        self._deca = None
        self._deca_loaded = False
        self._flame_template = self._try_load_flame_template()

    def _try_load_flame_template(self) -> Optional[trimesh.Trimesh]:
        """Loads a FLAME-2020 topology mesh if available in weights."""
        candidate_files = [
            self.weights_dir / "flame2020" / "head_template.obj",
            self.weights_dir / "flame2020" / "flame_template.obj",
            self.weights_dir / "flame2020" / "generic_model.obj",
            self.weights_dir / "flame_template.obj",
        ]
        for candidate in candidate_files:
            if candidate.exists():
                mesh = trimesh.load_mesh(candidate, process=False)
                if isinstance(mesh, trimesh.Trimesh):
                    LOGGER.info("Loaded FLAME template from %s", candidate)
                    return mesh
        return None

    def _load_deca(self) -> None:
        if self._deca_loaded:
            return

        # Python 3.12 compatibility patches
        import inspect
        if not hasattr(inspect, 'getargspec'):
            inspect.getargspec = inspect.getfullargspec
        if not hasattr(np, 'bool'):
            np.bool = np.bool_
            np.int = np.int_
            np.float = np.float64
            np.complex = np.complex128

        try:
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg

            deca_cfg.model.defrost()
            deca_cfg.model.use_tex = True
            deca_cfg.model.extract_tex = True
            deca_cfg.model.freeze()

            pretrained_candidates = [
                self.weights_dir / "deca" / "deca_model.tar",
                self.weights_dir / "deca_model.tar",
            ]
            for path in pretrained_candidates:
                if path.exists() and hasattr(deca_cfg, "pretrained_modelpath"):
                    deca_cfg.pretrained_modelpath = str(path)
                    break

            self._deca = DECA(config=deca_cfg, device=str(self.device))
            self._deca_loaded = True
            LOGGER.info("DECA loaded on %s", self.device)
        except Exception as exc:
            if not self.fallback_on_error:
                raise RuntimeError(
                    "DECA initialization failed. Run utils/download_weights.py and install "
                    "DECA dependencies before inference."
                ) from exc
            LOGGER.warning(
                "DECA unavailable (%s). Falling back to template reconstruction.", exc
            )
            self._deca = None
            self._deca_loaded = True

    def _preprocess(self, image_path: str | Path | np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        if isinstance(image_path, np.ndarray):
            # Already a numpy array (RGB)
            image = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image from {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        image_float = image_resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor, image_resized

    @staticmethod
    def _to_numpy(tensor_or_array: Any) -> np.ndarray:
        if tensor_or_array is None:
            return np.array([])
        if isinstance(tensor_or_array, np.ndarray):
            return tensor_or_array
        if torch.is_tensor(tensor_or_array):
            return tensor_or_array.detach().cpu().numpy()
        return np.asarray(tensor_or_array)

    @staticmethod
    def _normalize_uv(uv: np.ndarray) -> np.ndarray:
        if uv.size == 0:
            return uv
        uv = uv.astype(np.float32)
        if uv.max() > 1.0 or uv.min() < 0.0:
            uv_min = uv.min(axis=0, keepdims=True)
            uv_max = uv.max(axis=0, keepdims=True)
            uv = (uv - uv_min) / (uv_max - uv_min + 1e-8)
        return uv

    def _extract_faces(self) -> np.ndarray:
        if self._deca is None:
            return np.array([], dtype=np.int32)

        candidates = [
            getattr(getattr(self._deca, "flame", object()), "faces_tensor", None),
            getattr(self._deca, "faces_tensor", None),
            getattr(self._deca, "faces", None),
        ]
        for candidate in candidates:
            arr = self._to_numpy(candidate)
            if arr.size > 0:
                return arr.astype(np.int32)
        return np.array([], dtype=np.int32)

    def _extract_uv(self, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._deca is None:
            return np.array([], dtype=np.float32), faces.astype(np.int32)

        render = getattr(self._deca, "render", None)
        if render is None:
            return np.array([], dtype=np.float32), faces.astype(np.int32)

        uv_candidates = [
            getattr(render, "raw_uvcoords", None),
            getattr(render, "uvcoords", None),
        ]
        uv_faces_candidates = [
            getattr(render, "uvfaces", None),
            getattr(render, "uv_face_index", None),
        ]

        uv = np.array([], dtype=np.float32)
        uv_faces = faces.astype(np.int32)

        for candidate in uv_candidates:
            arr = self._to_numpy(candidate)
            if arr.size > 0:
                uv = arr.squeeze().astype(np.float32)
                break

        for candidate in uv_faces_candidates:
            arr = self._to_numpy(candidate)
            if arr.size > 0:
                uv_faces = arr.squeeze().astype(np.int32)
                break

        return self._normalize_uv(uv), uv_faces

    def _extract_uv_texture(self, vis_dict: Dict[str, Any], image_rgb: np.ndarray) -> np.ndarray:
        texture_keys = ["uv_texture", "uv_texture_gt", "albedo", "texture"]
        for key in texture_keys:
            if key not in vis_dict:
                continue
            value = self._to_numpy(vis_dict[key])
            if value.size == 0:
                continue
            tex = value[0] if value.ndim == 4 else value
            if tex.ndim == 3 and tex.shape[0] in (1, 2, 3):
                tex = np.transpose(tex, (1, 2, 0))
            tex = np.clip(tex, 0.0, 1.0)
            return (tex * 255.0).astype(np.uint8)

        fallback = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        return fallback

    def _fallback_reconstruction(self, image_rgb: np.ndarray) -> ReconstructionOutput:
        if self._flame_template is not None:
            mesh = self._flame_template.copy()
        else:
            mesh = trimesh.creation.icosphere(subdivisions=5, radius=95.0)

        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)

        # Spherical UV as a robust fallback if no UV template exists.
        centered = vertices - vertices.mean(axis=0, keepdims=True)
        x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
        u = 0.5 + np.arctan2(z, x) / (2 * np.pi)
        v = 0.5 - np.arcsin(np.clip(y / (np.linalg.norm(centered, axis=1) + 1e-6), -1, 1)) / np.pi
        uv = np.stack([u, v], axis=-1).astype(np.float32)

        return ReconstructionOutput(
            vertices=vertices,
            faces=faces,
            uv_coords=uv,
            uv_faces=faces.copy(),
            shape_params=np.zeros(100, dtype=np.float32),
            expression_params=np.zeros(50, dtype=np.float32),
            camera_pose=np.array([7.0, 0.0, 0.0], dtype=np.float32),
            uv_texture=cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC),
        )

    def reconstruct(self, image_path: str | Path | np.ndarray) -> ReconstructionOutput:
        self._load_deca()
        image_tensor, image_rgb = self._preprocess(image_path)

        if self._deca is None:
            return self._fallback_reconstruction(image_rgb=image_rgb)

        try:
            with torch.no_grad():
                codedict = self._deca.encode(image_tensor)
                opdict, visdict = self._deca.decode(codedict)

            vertices = self._to_numpy(opdict.get("verts", opdict.get("trans_verts")))
            if vertices.ndim == 3:
                vertices = vertices[0]
            vertices = vertices.astype(np.float32)

            faces = self._extract_faces()
            if faces.size == 0 and self._flame_template is not None:
                faces = self._flame_template.faces.astype(np.int32)

            uv_coords, uv_faces = self._extract_uv(faces)
            if uv_coords.size == 0:
                # In case DECA does not expose UV coordinates in this runtime build.
                uv_coords = self._fallback_reconstruction(image_rgb).uv_coords

            uv_texture = self._extract_uv_texture({**visdict, **opdict}, image_rgb)

            shape_params = self._to_numpy(codedict.get("shape", np.array([]))).reshape(-1)
            expression_params = self._to_numpy(codedict.get("exp", np.array([]))).reshape(-1)
            camera_pose = self._to_numpy(codedict.get("cam", np.array([]))).reshape(-1)

            return ReconstructionOutput(
                vertices=vertices,
                faces=faces,
                uv_coords=uv_coords,
                uv_faces=uv_faces,
                shape_params=shape_params.astype(np.float32),
                expression_params=expression_params.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                uv_texture=uv_texture,
            )
        except Exception as exc:
            if not self.fallback_on_error:
                raise RuntimeError("DECA forward pass failed.") from exc
            LOGGER.warning("DECA inference failed (%s). Using fallback mesh.", exc)
            return self._fallback_reconstruction(image_rgb=image_rgb)
