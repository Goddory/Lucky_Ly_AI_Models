"""
Geometry VAE Training — Train PartBasedGNNEncoder + MeshGNNDecoder.

Usage (local):
    python training/train_geometry_vae.py \
        --data-root ./data/meshes --output-dir ./weights --epochs 100

Usage (Colab):
    !python training/train_geometry_vae.py \
        --data-root /content/drive/MyDrive/avatar_data/meshes \
        --output-dir /content/drive/MyDrive/avatar_checkpoints/geometry \
        --epochs 100 --batch-size 2 --save-every 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.gnn_encoder import PartBasedGNNEncoder
from models.gnn_decoder import MeshGNNDecoder
from training.losses import (
    kl_divergence,
    laplacian_smoothness_loss,
    edge_length_preservation_loss,
    vertex_l1_loss,
)


# ---------------------------------------------------------------------------
# Dataset: loads mesh pairs (input_mesh → target_mesh)
# ---------------------------------------------------------------------------

class MeshPairDataset(Dataset):
    """
    Loads .npy or .npz mesh files for geometry VAE training.

    Directory structure:
        data_root/
            input/   ← raw DECA reconstructions (noisy)
            target/  ← clean reference meshes (from 3D scans / refined)
            faces.npy ← shared triangle topology [F, 3]

    If only a 'meshes/' directory exists, the same mesh is used as both
    input and target (autoencoder mode).
    """

    def __init__(self, data_root: str | Path, max_samples: Optional[int] = None) -> None:
        self.root = Path(data_root)
        self.pairs: list[tuple[Path, Path]] = []

        input_dir = self.root / "input"
        target_dir = self.root / "target"
        auto_dir = self.root / "meshes"

        if input_dir.exists() and target_dir.exists():
            input_files = sorted(input_dir.glob("*.npy")) + sorted(input_dir.glob("*.npz"))
            for inp in input_files:
                tgt = target_dir / inp.name
                if tgt.exists():
                    self.pairs.append((inp, tgt))
        elif auto_dir.exists():
            mesh_files = sorted(auto_dir.glob("*.npy")) + sorted(auto_dir.glob("*.npz"))
            self.pairs = [(m, m) for m in mesh_files]
        else:
            raise RuntimeError(
                f"Expected 'input/' + 'target/' or 'meshes/' inside {self.root}. "
                "See training/train_geometry_vae.py docstring for format."
            )

        if max_samples:
            self.pairs = self.pairs[:max_samples]

        # Load shared topology
        faces_path = self.root / "faces.npy"
        if faces_path.exists():
            self.faces = np.load(str(faces_path)).astype(np.int64)
        else:
            self.faces = None

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_vertices(path: Path) -> np.ndarray:
        data = np.load(str(path), allow_pickle=True)
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        if hasattr(data, "files"):
            key = data.files[0]
            return data[key].astype(np.float32)
        return np.array(data, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        inp_path, tgt_path = self.pairs[idx]
        input_verts = self._load_vertices(inp_path)
        target_verts = self._load_vertices(tgt_path)
        result = {
            "input_vertices": torch.from_numpy(input_verts),
            "target_vertices": torch.from_numpy(target_verts),
        }
        if self.faces is not None:
            result["faces"] = torch.from_numpy(self.faces.copy())
        return result


# ---------------------------------------------------------------------------
# Edge index builder (reused from core.geometry_gnn)
# ---------------------------------------------------------------------------

def faces_to_edge_index(faces: torch.Tensor, device: torch.device) -> torch.Tensor:
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], dim=0)
    rev = torch.stack([edges[:, 1], edges[:, 0]], dim=1)
    edges = torch.cat([edges, rev], dim=0)
    edges = torch.unique(edges, dim=0)
    return edges.t().contiguous().to(device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GeometryVAETrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.encoder = PartBasedGNNEncoder(latent_dim=args.latent_dim).to(self.device)
        self.decoder = MeshGNNDecoder(latent_dim=args.latent_dim).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        self.dataset = MeshPairDataset(args.data_root, max_samples=args.max_samples)
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,  # GNN processes one mesh at a time
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        self.start_epoch = 0
        self._maybe_resume()

        print(f"🔧 Geometry VAE Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Samples: {len(self.dataset)}")
        print(f"   Epochs: {args.epochs} (start from {self.start_epoch})")
        print(f"   Latent dim: {args.latent_dim}")

    def _maybe_resume(self) -> None:
        latest = self.output_dir / "geometry_vae_latest.pt"
        if not latest.exists():
            return

        print(f"🔄 Resuming from {latest}")
        ckpt = torch.load(latest, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        self.decoder.load_state_dict(ckpt["decoder"], strict=False)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt.get("epoch", 0)

    def train(self) -> None:
        best_loss = float("inf")

        for epoch in range(self.start_epoch, self.args.epochs):
            self.encoder.train()
            self.decoder.train()

            epoch_losses = {
                "total": 0.0, "recon": 0.0, "kl": 0.0,
                "laplacian": 0.0, "edge": 0.0,
            }
            t0 = time.time()

            for step, batch in enumerate(self.loader):
                input_v = batch["input_vertices"].squeeze(0).to(self.device)
                target_v = batch["target_vertices"].squeeze(0).to(self.device)

                if "faces" in batch:
                    faces = batch["faces"].squeeze(0).to(self.device)
                    edge_index = faces_to_edge_index(faces, self.device)
                else:
                    n = input_v.size(0)
                    idx = torch.arange(n, device=self.device)
                    edge_index = torch.stack([
                        idx[:-1], idx[1:]
                    ], dim=0)
                    faces = None

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    # Encode
                    enc_out = self.encoder(input_v, edge_index)
                    z_g = self.encoder.reparameterize(enc_out.mu, enc_out.logvar)

                    # Decode
                    displacement = self.decoder(
                        z_g=z_g, edge_index=edge_index, template_vertices=input_v,
                    )
                    refined = input_v + displacement

                    # Losses
                    loss_recon = vertex_l1_loss(refined, target_v)
                    loss_kl = kl_divergence(enc_out.mu, enc_out.logvar)
                    loss_lap = laplacian_smoothness_loss(refined, edge_index)

                    if faces is not None:
                        loss_edge = edge_length_preservation_loss(
                            refined, target_v, edge_index,
                        )
                    else:
                        loss_edge = torch.tensor(0.0, device=self.device)

                    total = (
                        self.args.w_recon * loss_recon
                        + self.args.w_kl * loss_kl
                        + self.args.w_laplacian * loss_lap
                        + self.args.w_edge * loss_edge
                    )

                self.scaler.scale(total).backward()

                if self.args.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        self.args.max_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_losses["total"] += total.item()
                epoch_losses["recon"] += loss_recon.item()
                epoch_losses["kl"] += loss_kl.item()
                epoch_losses["laplacian"] += loss_lap.item()
                epoch_losses["edge"] += loss_edge.item()

            self.scheduler.step()
            n_steps = max(1, len(self.loader))
            avg = {k: v / n_steps for k, v in epoch_losses.items()}
            elapsed = time.time() - t0

            print(
                f"[Epoch {epoch + 1}/{self.args.epochs}] "
                f"loss={avg['total']:.5f} "
                f"recon={avg['recon']:.5f} "
                f"kl={avg['kl']:.6f} "
                f"lap={avg['laplacian']:.5f} "
                f"edge={avg['edge']:.5f} "
                f"lr={self.scheduler.get_last_lr()[0]:.2e} "
                f"({elapsed:.1f}s)"
            )

            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0 or (epoch + 1) == self.args.epochs:
                self._save_checkpoint(epoch + 1, avg)

            if avg["total"] < best_loss:
                best_loss = avg["total"]
                self._save_checkpoint(epoch + 1, avg, tag="best")

        print(f"\n✅ Training complete! Best loss: {best_loss:.5f}")
        print(f"   Checkpoints saved to: {self.output_dir}")

    def _save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest") -> None:
        ckpt = {
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(self.args),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        if tag == "latest":
            path = self.output_dir / "geometry_vae_latest.pt"
        elif tag == "best":
            path = self.output_dir / "geometry_vae_best.pt"
        else:
            path = self.output_dir / f"geometry_vae_epoch_{epoch:03d}.pt"

        torch.save(ckpt, path)

        # Also save in format compatible with GeometryRefiner
        production_ckpt = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
        torch.save(production_ckpt, self.output_dir / "geometry_vae.pt")

        metrics_path = self.output_dir / f"metrics_{tag}.json"
        metrics_path.write_text(
            json.dumps({"epoch": epoch, **metrics}, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Geometry VAE (Encoder + Decoder)")
    p.add_argument("--data-root", type=str, required=True,
                    help="Path containing input/, target/, faces.npy (or meshes/)")
    p.add_argument("--output-dir", type=str, default="weights",
                    help="Directory to save checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-every", type=int, default=1,
                    help="Save checkpoint every N epochs (important for Colab!)")

    p.add_argument("--w-recon", type=float, default=1.0, help="Vertex reconstruction weight")
    p.add_argument("--w-kl", type=float, default=0.001, help="KL divergence weight")
    p.add_argument("--w-laplacian", type=float, default=0.1, help="Laplacian smoothness weight")
    p.add_argument("--w-edge", type=float, default=0.05, help="Edge length preservation weight")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = GeometryVAETrainer(args)
    trainer.train()
