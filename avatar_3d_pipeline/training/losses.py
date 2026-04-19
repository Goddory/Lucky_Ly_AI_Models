"""Shared loss functions for avatar geometry and texture training."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry losses
# ---------------------------------------------------------------------------

def vertex_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-vertex L1 reconstruction loss. Shapes: [N, 3]."""
    return F.l1_loss(pred, target)


def vertex_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-vertex L2 (MSE) reconstruction loss. Shapes: [N, 3]."""
    return F.mse_loss(pred, target)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence for VAE regularization. Shapes: [B, D]."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def laplacian_smoothness_loss(
    vertices: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Penalises non-smooth surfaces by measuring deviation from neighbourhood mean.

    Args:
        vertices: [N, 3] vertex positions.
        edge_index: [2, E] undirected edge indices.
    """
    src, dst = edge_index
    n_vertices = vertices.size(0)

    neighbor_sum = torch.zeros_like(vertices)
    degree = torch.zeros((n_vertices, 1), dtype=vertices.dtype, device=vertices.device)

    neighbor_sum.index_add_(0, src, vertices[dst])
    degree.index_add_(
        0, src,
        torch.ones((src.size(0), 1), dtype=vertices.dtype, device=vertices.device),
    )

    mean_neighbor = neighbor_sum / (degree + 1e-8)
    laplacian = vertices - mean_neighbor
    return laplacian.pow(2).sum(dim=-1).mean()


def edge_length_preservation_loss(
    pred_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Penalises edge length changes relative to a target mesh.

    This prevents the refined mesh from distorting the overall scale
    while still allowing local detail corrections.
    """
    src, dst = edge_index

    pred_edges = pred_vertices[dst] - pred_vertices[src]
    target_edges = target_vertices[dst] - target_vertices[src]

    pred_len = pred_edges.norm(dim=-1)
    target_len = target_edges.norm(dim=-1)

    return F.l1_loss(pred_len, target_len)


def normal_consistency_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Encourages adjacent face normals to be consistent (smooth shading).

    Args:
        vertices: [N, 3] vertex positions.
        faces: [F, 3] triangle indices (int64).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = F.normalize(face_normals, dim=-1, eps=1e-8)

    # Build face adjacency from shared edges
    n_faces = faces.size(0)
    edge_to_face: dict = {}
    for fi in range(n_faces):
        for ei in range(3):
            e = tuple(sorted((faces[fi, ei].item(), faces[fi, (ei + 1) % 3].item())))
            edge_to_face.setdefault(e, []).append(fi)

    adj_pairs = []
    for face_list in edge_to_face.values():
        if len(face_list) == 2:
            adj_pairs.append(face_list)

    if not adj_pairs:
        return torch.tensor(0.0, device=vertices.device, requires_grad=True)

    adj_tensor = torch.tensor(adj_pairs, dtype=torch.long, device=vertices.device)
    n1 = face_normals[adj_tensor[:, 0]]
    n2 = face_normals[adj_tensor[:, 1]]

    cosine = (n1 * n2).sum(dim=-1)
    return (1.0 - cosine).mean()


# ---------------------------------------------------------------------------
# Texture / image losses
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss using early feature layers.

    Computes L1 distance between VGG feature maps of predicted and target images.
    """

    def __init__(self, layers: Optional[list[int]] = None) -> None:
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        except Exception:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features.eval()

        self.layers = layers or [3, 8, 15, 22]
        self.blocks = nn.ModuleList()

        prev = 0
        for layer_idx in self.layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer_idx + 1]))
            prev = layer_idx + 1

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Input x is in [0, 1] range."""
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = torch.tensor(0.0, device=pred.device)
        x_pred, x_target = pred, target
        for block in self.blocks:
            x_pred = block(x_pred)
            with torch.no_grad():
                x_target = block(x_target)
            loss = loss + F.l1_loss(x_pred, x_target)

        return loss / len(self.blocks)


def adversarial_loss_generator(
    fake_preds: list[torch.Tensor],
    mode: str = "hinge",
) -> torch.Tensor:
    """Generator adversarial loss (multi-scale)."""
    loss = torch.tensor(0.0, device=fake_preds[0].device)
    for pred in fake_preds:
        if mode == "hinge":
            loss = loss + (-pred).mean()
        elif mode == "lsgan":
            loss = loss + F.mse_loss(pred, torch.ones_like(pred))
        else:
            loss = loss + F.binary_cross_entropy_with_logits(
                pred, torch.ones_like(pred)
            )
    return loss / len(fake_preds)


def adversarial_loss_discriminator(
    real_preds: list[torch.Tensor],
    fake_preds: list[torch.Tensor],
    mode: str = "hinge",
) -> torch.Tensor:
    """Discriminator adversarial loss (multi-scale)."""
    loss = torch.tensor(0.0, device=real_preds[0].device)
    for real, fake in zip(real_preds, fake_preds):
        if mode == "hinge":
            loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
        elif mode == "lsgan":
            loss = loss + F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fake, torch.zeros_like(fake))
        else:
            loss = loss + F.binary_cross_entropy_with_logits(
                real, torch.ones_like(real)
            ) + F.binary_cross_entropy_with_logits(
                fake, torch.zeros_like(fake)
            )
    return loss / len(real_preds)
