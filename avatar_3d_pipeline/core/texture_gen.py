from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from models.generators import (
    PBRExtractorGE,
    ReflectanceNetworkGC,
    SkinToneControlGA,
    TextureGeneratorG,
)


@dataclass
class TextureGenerationOutput:
    """Container for all generated texture maps in uint8 image format."""

    m_map: np.ndarray
    h_map: np.ndarray
    albedo_map: np.ndarray
    color_map: np.ndarray
    specular_map: np.ndarray
    normal_map: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "M": self.m_map,
            "H": self.h_map,
            "A": self.albedo_map,
            "C": self.color_map,
            "S": self.specular_map,
            "N": self.normal_map,
        }


class TextureSynthesisPipeline:
    """Phase-3 geometry-aware texture synthesis and PBR decomposition."""

    def __init__(
        self,
        weights_dir: str | Path,
        latent_dim: int = 256,
        device: Optional[str] = None,
        use_fp16: bool = True,
        strict_pretrained: bool = True,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.strict_pretrained = strict_pretrained

        self.g_texture = TextureGeneratorG(latent_dim=latent_dim).to(self.device)
        self.g_a = SkinToneControlGA().to(self.device)
        self.g_c = ReflectanceNetworkGC().to(self.device)
        self.g_e = PBRExtractorGE().to(self.device)

        self._load_pretrained()

        self.g_texture.eval()
        self.g_a.eval()
        self.g_c.eval()
        self.g_e.eval()

    def _load_module_weights(self, module: torch.nn.Module, candidate_paths: list[Path]) -> bool:
        for path in candidate_paths:
            if not path.exists():
                continue
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            module.load_state_dict(state, strict=False)
            return True
        return False

    def _load_pretrained(self) -> None:
        loaded_g = self._load_module_weights(
            self.g_texture,
            [
                self.weights_dir / "texture_g.pt",
                self.weights_dir / "ffhq" / "texture_g.pt",
            ],
        )
        loaded_ga = self._load_module_weights(
            self.g_a,
            [
                self.weights_dir / "texture_ga.pt",
                self.weights_dir / "facescape" / "texture_ga.pt",
            ],
        )
        loaded_gc = self._load_module_weights(
            self.g_c,
            [
                self.weights_dir / "texture_gc.pt",
                self.weights_dir / "facescape" / "texture_gc.pt",
            ],
        )
        loaded_ge = self._load_module_weights(
            self.g_e,
            [
                self.weights_dir / "texture_ge.pt",
                self.weights_dir / "facescape" / "texture_ge.pt",
            ],
        )

        if self.strict_pretrained:
            missing = []
            if not loaded_g:
                missing.append("texture_g.pt")
            if not loaded_ga:
                missing.append("texture_ga.pt")
            if not loaded_gc:
                missing.append("texture_gc.pt")
            if not loaded_ge:
                missing.append("texture_ge.pt")

            if missing:
                raise FileNotFoundError(
                    "Missing required texture checkpoints: "
                    + ", ".join(missing)
                    + ". Run utils/download_weights.py and ensure FFHQ/FaceScape mirrors are configured."
                )

    @staticmethod
    def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        image = np.clip(image, 0.0, 1.0)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)

    @staticmethod
    def _ensure_3_channel(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.repeat(image[..., None], 3, axis=-1)
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=-1)
        return image

    @staticmethod
    def _to_uint8(tensor: torch.Tensor, mode: str = "sigmoid") -> np.ndarray:
        arr = tensor.detach().cpu().numpy()[0]
        arr = np.transpose(arr, (1, 2, 0))

        if mode == "tanh":
            arr = (arr * 0.5) + 0.5
        elif mode == "normal":
            arr = (arr * 0.5) + 0.5

        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)

    def generate(
        self,
        uv_texture: np.ndarray,
        z_g: np.ndarray,
        alpha: float = 0.55,
        output_size: int = 1024,
    ) -> TextureGenerationOutput:
        uv_texture = self._ensure_3_channel(uv_texture)
        if uv_texture.shape[0] != output_size or uv_texture.shape[1] != output_size:
            uv_texture = cv2.resize(uv_texture, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

        uv_tensor = self._to_tensor(uv_texture, self.device)
        z_tensor = torch.as_tensor(z_g, dtype=torch.float32, device=self.device).view(1, -1)
        alpha_tensor = torch.tensor([alpha], dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                m, h = self.g_texture(uv_tensor, z_tensor)
                a = self.g_a(m, alpha=alpha_tensor)
                c = self.g_c(a, h)
                s, n = self.g_e(c)

        m_vis = self._to_uint8(torch.cat([m, torch.zeros_like(m[:, :1])], dim=1), mode="sigmoid")
        h_map = self._to_uint8(h, mode="tanh")
        a_map = self._to_uint8(a, mode="sigmoid")
        c_map = self._to_uint8(c, mode="sigmoid")

        s_3 = torch.cat([s, s, s], dim=1)
        s_map = self._to_uint8(s_3, mode="sigmoid")
        n_map = self._to_uint8(n, mode="normal")

        return TextureGenerationOutput(
            m_map=m_vis,
            h_map=h_map,
            albedo_map=a_map,
            color_map=c_map,
            specular_map=s_map,
            normal_map=n_map,
        )
