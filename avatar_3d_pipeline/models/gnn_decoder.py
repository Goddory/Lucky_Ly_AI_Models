from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, GraphNorm

    HAS_PYG = True
except Exception:
    GATv2Conv = GraphNorm = None
    HAS_PYG = False


class MeshGNNDecoder(nn.Module):
    """
    Graph decoder F that maps geometry latent Z_g to vertex displacements.

    The decoder predicts residual offsets for each vertex to preserve identity while
    removing UV-flattening and reconstruction artifacts.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        template_vertex_count: int = 5023,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.template_vertex_count = template_vertex_count
        self.dropout = dropout

        self.register_buffer(
            "template_vertices",
            torch.zeros(template_vertex_count, 3, dtype=torch.float32),
            persistent=False,
        )

        self.node_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.style = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if HAS_PYG:
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
            self.norm1 = GraphNorm(hidden_dim)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
            self.norm2 = GraphNorm(hidden_dim)
            self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False)
            self.norm3 = GraphNorm(hidden_dim)
        else:
            self.fallback = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

        self.displacement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def _get_template(self, template_vertices: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        if template_vertices is not None:
            return template_vertices.to(device=device, dtype=torch.float32)
        return self.template_vertices.to(device=device)

    def forward(
        self,
        z_g: torch.Tensor,
        edge_index: torch.Tensor,
        template_vertices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_g.dim() == 1:
            z_g = z_g.unsqueeze(0)

        if z_g.size(0) != 1:
            raise ValueError("MeshGNNDecoder currently expects batch size of 1.")

        template = self._get_template(template_vertices, device=z_g.device)
        n_vertices = template.size(0)

        x = self.node_embed(template)
        style = self.style(z_g).expand(n_vertices, -1)
        x = x + style

        if HAS_PYG:
            batch = torch.zeros((n_vertices,), dtype=torch.long, device=z_g.device)

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
        else:
            x = self.fallback(x)

        displacement = self.displacement_head(x)
        return displacement

    def decode_mesh(
        self,
        z_g: torch.Tensor,
        edge_index: torch.Tensor,
        template_vertices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        template = self._get_template(template_vertices, device=z_g.device)
        displacement = self.forward(z_g=z_g, edge_index=edge_index, template_vertices=template)
        return template + displacement
