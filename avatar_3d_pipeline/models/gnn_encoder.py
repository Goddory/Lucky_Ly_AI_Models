from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
    from torch_geometric.utils import subgraph

    HAS_PYG = True
except Exception:
    GATv2Conv = GraphNorm = global_mean_pool = None
    subgraph = None
    HAS_PYG = False


@dataclass
class EncoderOutput:
    """Output container for the part-based mesh encoder."""

    mu: torch.Tensor
    logvar: torch.Tensor
    part_embeddings: torch.Tensor


class SemanticFacePartitioner:
    """
    Splits FLAME-like face vertices into 8 semantic spatial regions.

    This partitioning is topology-agnostic and uses stable coordinate thresholds:
    - Left/Right x Top, Middle, Bottom (6 regions)
    - Central Nose bridge region (1)
    - Central mouth/chin strip region (1)
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def split(self, vertices: torch.Tensor) -> List[torch.Tensor]:
        if vertices.dim() != 2 or vertices.size(-1) != 3:
            raise ValueError("vertices must be shaped [N, 3]")

        x = vertices[:, 0]
        y = vertices[:, 1]

        y_top = torch.quantile(y, 0.66)
        y_mid = torch.quantile(y, 0.33)

        x_center_lo = torch.quantile(x, 0.40)
        x_center_hi = torch.quantile(x, 0.60)

        left = x < 0
        right = ~left

        top = y >= y_top
        middle = (y < y_top) & (y >= y_mid)
        bottom = y < y_mid

        center = (x >= x_center_lo) & (x <= x_center_hi)

        masks = [
            left & top,
            right & top,
            left & middle,
            right & middle,
            left & bottom,
            right & bottom,
            center & middle,
            center & bottom,
        ]

        # Guarantee each part has vertices to avoid invalid subgraphs.
        fallback = torch.ones_like(x, dtype=torch.bool)
        safe_masks = [m if m.any() else fallback for m in masks]
        return safe_masks


class MeshPartEncoder(nn.Module):
    """Single region encoder using attention-based graph convolutions."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        out_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        if HAS_PYG:
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=2, concat=False)
            self.norm1 = GraphNorm(hidden_channels)
            self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=2, concat=False)
            self.norm2 = GraphNorm(hidden_channels)
            self.conv3 = GATv2Conv(hidden_channels, out_channels, heads=1, concat=False)
            self.norm3 = GraphNorm(out_channels)
        else:
            self.fallback = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, out_channels),
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if HAS_PYG:
            batch = x.new_zeros((x.size(0),), dtype=torch.long)
            x = self.conv1(x, edge_index)
            x = self.norm1(x, batch)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv2(x, edge_index)
            x = self.norm2(x, batch)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv3(x, edge_index)
            x = self.norm3(x, batch)
            x = F.gelu(x)
            pooled = global_mean_pool(x, batch)
            return pooled

        pooled = self.fallback(x).mean(dim=0, keepdim=True)
        return pooled


class PartBasedGNNEncoder(nn.Module):
    """
    8-part VAE encoder for geometry latent extraction.

    Input: vertices [N, 3], edge_index [2, E]
    Output: global mean/logvar and per-part embeddings.
    """

    def __init__(
        self,
        num_parts: int = 8,
        hidden_channels: int = 128,
        part_embedding_dim: int = 128,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_parts = num_parts
        self.latent_dim = latent_dim
        self.partitioner = SemanticFacePartitioner()

        self.part_encoders = nn.ModuleList(
            [
                MeshPartEncoder(
                    in_channels=3,
                    hidden_channels=hidden_channels,
                    out_channels=part_embedding_dim,
                    dropout=dropout,
                )
                for _ in range(num_parts)
            ]
        )

        fusion_dim = num_parts * part_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_channels),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_channels, latent_dim)
        self.logvar_head = nn.Linear(hidden_channels, latent_dim)

    @staticmethod
    def _fallback_subgraph(mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        selected = mask.nonzero(as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            return selected.new_zeros((2, 0), dtype=torch.long)

        mapping = torch.full(
            (mask.numel(),),
            fill_value=-1,
            dtype=torch.long,
            device=mask.device,
        )
        mapping[selected] = torch.arange(selected.numel(), device=mask.device)

        src, dst = edge_index
        keep = mask[src] & mask[dst]
        src_new = mapping[src[keep]]
        dst_new = mapping[dst[keep]]
        return torch.stack([src_new, dst_new], dim=0)

    def _extract_part_graph(
        self,
        vertices: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        node_indices = mask.nonzero(as_tuple=False).squeeze(-1)
        if node_indices.numel() == 0:
            node_indices = torch.arange(vertices.size(0), device=vertices.device)
            mask = torch.ones(vertices.size(0), dtype=torch.bool, device=vertices.device)

        part_vertices = vertices[node_indices]

        if HAS_PYG and subgraph is not None:
            subgraph_out = subgraph(
                subset=mask,
                edge_index=edge_index,
                relabel_nodes=True,
                return_edge_mask=True,
            )
            # PyG may return either edge_index only or a tuple depending on version.
            if isinstance(subgraph_out, tuple):
                part_edge_index = subgraph_out[0]
            else:
                part_edge_index = subgraph_out
        else:
            part_edge_index = self._fallback_subgraph(mask=mask, edge_index=edge_index)

        if part_edge_index.numel() == 0:
            chain = torch.arange(part_vertices.size(0), device=vertices.device)
            if chain.numel() >= 2:
                part_edge_index = torch.stack([chain[:-1], chain[1:]], dim=0)
            else:
                part_edge_index = torch.zeros((2, 0), dtype=torch.long, device=vertices.device)

        return {"x": part_vertices, "edge_index": part_edge_index}

    def forward(self, vertices: torch.Tensor, edge_index: torch.Tensor) -> EncoderOutput:
        masks = self.partitioner.split(vertices)

        part_embeddings: List[torch.Tensor] = []
        for part_idx in range(self.num_parts):
            part_graph = self._extract_part_graph(vertices, edge_index, masks[part_idx])
            embedding = self.part_encoders[part_idx](
                part_graph["x"],
                part_graph["edge_index"],
            )
            part_embeddings.append(embedding)

        parts = torch.cat(part_embeddings, dim=-1)
        fused = self.fusion(parts)
        mu = self.mu_head(fused)
        logvar = self.logvar_head(fused)

        stacked_parts = torch.stack(part_embeddings, dim=1)
        return EncoderOutput(mu=mu, logvar=logvar, part_embeddings=stacked_parts)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
