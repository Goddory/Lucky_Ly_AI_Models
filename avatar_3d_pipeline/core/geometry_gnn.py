from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from models.gnn_decoder import MeshGNNDecoder
from models.gnn_encoder import PartBasedGNNEncoder


@dataclass
class GeometryRefinementOutput:
    """Phase-2 output containing refined geometry and latent embeddings."""

    refined_vertices: np.ndarray
    faces: np.ndarray
    z_g: np.ndarray
    mu: np.ndarray
    logvar: np.ndarray


def faces_to_edge_index(faces: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Converts triangular faces [F, 3] into an undirected graph edge index [2, E]."""
    edges = torch.cat(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        dim=0,
    )
    rev_edges = torch.stack([edges[:, 1], edges[:, 0]], dim=1)
    edges = torch.cat([edges, rev_edges], dim=0)
    edges = torch.unique(edges, dim=0)
    return edges.t().contiguous().to(device=device, dtype=torch.long)


def laplacian_smooth(
    vertices: torch.Tensor,
    edge_index: torch.Tensor,
    iterations: int = 2,
    smoothing_lambda: float = 0.15,
) -> torch.Tensor:
    """Fast Laplacian smoothing used as a post-decode denoiser."""
    src, dst = edge_index
    n_vertices = vertices.size(0)

    for _ in range(iterations):
        neighbor_sum = torch.zeros_like(vertices)
        degree = torch.zeros((n_vertices, 1), dtype=vertices.dtype, device=vertices.device)

        neighbor_sum.index_add_(0, src, vertices[dst])
        degree.index_add_(
            0,
            src,
            torch.ones((src.size(0), 1), dtype=vertices.dtype, device=vertices.device),
        )

        mean_neighbor = neighbor_sum / (degree + 1e-6)
        vertices = vertices + smoothing_lambda * (mean_neighbor - vertices)

    return vertices


class GeometryRefiner:
    """Runs the part-based VAE geometry refinement (Phase 2)."""

    def __init__(
        self,
        weights_dir: str | Path,
        latent_dim: int = 256,
        device: Optional[str] = None,
        use_fp16: bool = True,
        strict_checkpoint: bool = True,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.strict_checkpoint = strict_checkpoint

        self.encoder = PartBasedGNNEncoder(latent_dim=latent_dim).to(self.device)
        self.decoder = MeshGNNDecoder(latent_dim=latent_dim).to(self.device)

        loaded = self._load_weights()
        if self.strict_checkpoint and not loaded:
            raise FileNotFoundError(
                "Geometry VAE checkpoint not found. Expected one of: "
                "weights/geometry_vae.pt, weights/geometry_vae.pth, "
                "weights/geometry/geometry_vae.pt"
            )
        self.encoder.eval()
        self.decoder.eval()

    def _load_weights(self) -> bool:
        checkpoint_paths = [
            self.weights_dir / "geometry_vae.pt",
            self.weights_dir / "geometry_vae.pth",
            self.weights_dir / "geometry" / "geometry_vae.pt",
        ]

        for path in checkpoint_paths:
            if not path.exists():
                continue
            state = torch.load(path, map_location=self.device)
            encoder_state = state.get("encoder", state)
            decoder_state = state.get("decoder", state)

            self.encoder.load_state_dict(encoder_state, strict=False)
            self.decoder.load_state_dict(decoder_state, strict=False)
            return True

        return False

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def refine(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        deterministic: bool = True,
        smooth_iterations: int = 2,
    ) -> GeometryRefinementOutput:
        v = torch.as_tensor(vertices, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        edge_index = faces_to_edge_index(f, device=self.device)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                enc_out = self.encoder(v, edge_index)
                if deterministic:
                    z_g = enc_out.mu
                else:
                    z_g = self.encoder.reparameterize(enc_out.mu, enc_out.logvar)

                displacement = self.decoder(
                    z_g=z_g,
                    edge_index=edge_index,
                    template_vertices=v,
                )
                refined = v + displacement

                if smooth_iterations > 0:
                    refined = laplacian_smooth(
                        vertices=refined,
                        edge_index=edge_index,
                        iterations=smooth_iterations,
                    )

        return GeometryRefinementOutput(
            refined_vertices=refined.detach().cpu().numpy().astype(np.float32),
            faces=faces.astype(np.int32),
            z_g=z_g.detach().cpu().numpy().reshape(-1).astype(np.float32),
            mu=enc_out.mu.detach().cpu().numpy().reshape(-1).astype(np.float32),
            logvar=enc_out.logvar.detach().cpu().numpy().reshape(-1).astype(np.float32),
        )

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
