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

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm  # auto-detects Colab/Jupyter vs terminal

# Suppress noisy deprecation warnings from triton/torch internals
warnings.filterwarnings("ignore", category=UserWarning, module="triton")
warnings.filterwarnings("ignore", message=".*tl.where.*non-boolean.*")
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*weights.*deprecated.*")

# Suppress torch.compile / dynamo verbose graph-break logs
import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

import sys
from pathlib import Path as _Path
# Ensure project root is on sys.path so `models/` and `training/` packages resolve correctly
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

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

    def __init__(
        self,
        data_root: str | Path,
        max_samples: Optional[int] = None,
        cache_in_ram: bool = False,
    ) -> None:
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
            # Fallback: read directly from data_root (flat structure)
            mesh_files = sorted(self.root.glob("*.npy")) + sorted(self.root.glob("*.npz"))
            mesh_files = [m for m in mesh_files if m.name != "faces.npy"]

            if mesh_files:
                self.pairs = [(m, m) for m in mesh_files]
            else:
                raise RuntimeError(
                    f"Expected 'input/' + 'target/' dirs, a 'meshes/' dir, or .npy files directly inside {self.root}."
                )

        if max_samples:
            self.pairs = self.pairs[:max_samples]

        # Load shared topology
        faces_path = self.root / "faces.npy"
        if faces_path.exists():
            self.faces = np.load(str(faces_path)).astype(np.int64)
        else:
            self.faces = None

        self._cache: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
        if cache_in_ram:
            print(f"📦 Caching {len(self.pairs)} meshes into RAM (Parallel I/O)...")
            from concurrent.futures import ThreadPoolExecutor

            def _load_pair(pair):
                return self._load_vertices(pair[0]), self._load_vertices(pair[1])

            with ThreadPoolExecutor(max_workers=32) as executor:
                self._cache = list(tqdm(
                    executor.map(_load_pair, self.pairs),
                    total=len(self.pairs),
                    desc="Loading",
                    unit="mesh"
                ))
            used_mb = sum(a.nbytes + b.nbytes for a, b in self._cache) / 1024 ** 2
            print(f"✅ Cached {len(self._cache)} meshes ({used_mb:.0f} MB used in RAM)")

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
        if self._cache is not None:
            input_verts, target_verts = self._cache[idx]
        else:
            inp_path, tgt_path = self.pairs[idx]
            input_verts = self._load_vertices(inp_path)
            target_verts = self._load_vertices(tgt_path)

        result = {
            "input_vertices": torch.from_numpy(input_verts.copy()),
            "target_vertices": torch.from_numpy(target_verts.copy()),
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
            torch.backends.cudnn.allow_tf32 = True

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.encoder = PartBasedGNNEncoder(latent_dim=args.latent_dim).to(self.device)
        self.decoder = MeshGNNDecoder(latent_dim=args.latent_dim).to(self.device)

        # Optional: torch.compile for extra speed (PyTorch 2.x)
        if getattr(args, 'compile', False) and hasattr(torch, 'compile'):
            print("⚡ Compiling models with torch.compile...")
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        self.dataset = MeshPairDataset(
            args.data_root,
            max_samples=args.max_samples,
            cache_in_ram=args.cache_in_ram,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        # Pre-compute and cache edge_index once (topology is shared across all meshes)
        self._cached_edge_index = None
        if self.dataset.faces is not None:
            faces_t = torch.from_numpy(self.dataset.faces)
            self._cached_edge_index = faces_to_edge_index(faces_t, self.device)
            print(f"   Edge index cached: {self._cached_edge_index.shape}")

            # Pre-compute partition masks for encoder (same FLAME topology for all meshes)
            ref_verts = torch.from_numpy(self.dataset[0]["input_vertices"].numpy()).to(self.device)
            self.encoder.precompute_partitions(ref_verts, self._cached_edge_index)

            # Pre-compute batched graph tensors for B meshes
            N = ref_verts.size(0)  # vertices per mesh (5023 for FLAME)
            B = args.batch_size
            E = self._cached_edge_index.size(1)

            # Batched edge_index: offset each mesh's edges by i * N
            offsets = torch.arange(B, device=self.device) * N
            repeated = self._cached_edge_index.repeat(1, B)  # [2, B*E]
            offset_vec = offsets.repeat_interleave(E)          # [B*E]
            self._batched_edge_index = repeated + offset_vec.unsqueeze(0)

            # Batch tensor: maps each vertex to its mesh index
            self._batch_tensor = torch.arange(B, device=self.device).repeat_interleave(N)
            self._n_verts = N

            # Pre-compute encoder batch partitions
            self.encoder.precompute_batch_partitions(B, N)

            print(f"   ⚡ Batched graph tensors cached (B={B}, N={N})")

        self.start_epoch = 0
        self._maybe_resume()

        print(f"🔧 Geometry VAE Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {args.batch_size}")
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
        n_steps = len(self.loader)
        log_every = max(1, n_steps // 10)  # In tiến độ 10 lần mỗi epoch

        print(f"\n🚀 Bắt đầu training: Epoch {self.start_epoch + 1} → {self.args.epochs}", flush=True)
        print(f"   Mỗi epoch = {n_steps} steps | Log mỗi {log_every} steps\n", flush=True)

        for epoch in range(self.start_epoch, self.args.epochs):
            self.encoder.train()
            self.decoder.train()

            epoch_losses = {
                "total": 0.0, "recon": 0.0, "kl": 0.0,
                "laplacian": 0.0, "edge": 0.0,
            }
            t0 = time.time()

            print(f"── Epoch {epoch + 1}/{self.args.epochs} ──────────────────", flush=True)

            for step, batch in enumerate(self.loader):
                input_batch = batch["input_vertices"].to(self.device, non_blocking=True)
                target_batch = batch["target_vertices"].to(self.device, non_blocking=True)
                B = input_batch.size(0)
                N = input_batch.size(1)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    # === CHUNKED BATCHED FORWARD (Maximize VRAM, Avoid OOM) ===
                    chunk_size = 16
                    for i in range(0, B, chunk_size):
                        sub_B = min(chunk_size, B - i)
                        sub_input = input_batch[i:i+sub_B]
                        sub_target = target_batch[i:i+sub_B]
                        
                        input_flat = sub_input.reshape(-1, 3)      # [sub_B*N, 3]
                        target_flat = sub_target.reshape(-1, 3)    # [sub_B*N, 3]

                        # Handle chunk batch indexing
                        if sub_B == self.args.batch_size and hasattr(self, '_batched_edge_index'):
                            b_edge = self._batched_edge_index
                            b_tensor = self._batch_tensor
                        else:
                            # Recompute for sub-batch
                            E = self._cached_edge_index.size(1)
                            offsets = torch.arange(sub_B, device=self.device) * N
                            repeated = self._cached_edge_index.repeat(1, sub_B)
                            offset_vec = offsets.repeat_interleave(E)
                            b_edge = repeated + offset_vec.unsqueeze(0)
                            b_tensor = torch.arange(sub_B, device=self.device).repeat_interleave(N)

                        enc_out = self.encoder.forward_batched(input_flat, sub_B)

                        z_g = self.encoder.reparameterize(enc_out.mu, enc_out.logvar)  # [sub_B, latent]

                        displacement = self.decoder.forward_batched(
                            z_g=z_g, edge_index=b_edge,
                            template_vertices=input_flat, batch=b_tensor,
                        )  # [sub_B*N, 3]
                        refined = input_flat + displacement

                        loss_recon = vertex_l1_loss(refined, target_flat)
                        loss_kl = kl_divergence(enc_out.mu, enc_out.logvar)
                        loss_lap = laplacian_smoothness_loss(refined, b_edge)
                        loss_edge = edge_length_preservation_loss(refined, target_flat, b_edge)

                        # Scale loss properly for gradient accumulation across chunks
                        chunk_ratio = sub_B / B
                        total = (
                            self.args.w_recon * loss_recon
                            + self.args.w_kl * loss_kl
                            + self.args.w_laplacian * loss_lap
                            + self.args.w_edge * loss_edge
                        ) * chunk_ratio

                        self.scaler.scale(total).backward()
                        
                        # Accumulate metrics
                        epoch_losses["total"] += total.item()
                        epoch_losses["recon"] += (loss_recon * chunk_ratio).item()
                        epoch_losses["kl"] += (loss_kl * chunk_ratio).item()
                        epoch_losses["laplacian"] += (loss_lap * chunk_ratio).item()
                        epoch_losses["edge"] += (loss_edge * chunk_ratio).item()

                if self.args.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        self.args.max_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # In tiến độ mỗi 10%
                if (step + 1) % log_every == 0 or (step + 1) == n_steps:
                    done = step + 1
                    pct = done / n_steps * 100
                    elapsed = time.time() - t0
                    speed = done / elapsed if elapsed > 0 else 0
                    eta = (n_steps - done) / speed if speed > 0 else 0
                    cur_loss = epoch_losses["total"] / done
                    print(
                        f"  [{pct:5.1f}%] {done}/{n_steps} | "
                        f"loss={cur_loss:.5f} | "
                        f"{speed:.1f} step/s | ETA {eta:.0f}s",
                        flush=True,
                    )

            self.scheduler.step()
            n_steps_final = max(1, len(self.loader))
            avg = {k: v / n_steps_final for k, v in epoch_losses.items()}
            elapsed = time.time() - t0

            # Hiển thị kết quả epoch
            is_best = avg["total"] < best_loss
            if is_best:
                best_loss = avg["total"]
            marker = " ⭐ BEST" if is_best else ""
            print(
                f"✅ Epoch {epoch + 1}/{self.args.epochs} | "
                f"loss={avg['total']:.5f} recon={avg['recon']:.5f} "
                f"kl={avg['kl']:.6f} lr={self.scheduler.get_last_lr()[0]:.2e} "
                f"({elapsed:.0f}s){marker}\n",
                flush=True,
            )
            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0 or (epoch + 1) == self.args.epochs:
                self._save_checkpoint(epoch + 1, avg)
            if is_best:
                self._save_checkpoint(epoch + 1, avg, tag="best")

        print(f"\n✅ Training hoàn tất! Best loss: {best_loss:.5f}", flush=True)
        print(f"   Checkpoints: {self.output_dir}", flush=True)


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
    p.add_argument("--batch-size", type=int, default=4,
                    help="Batch size (L4=4-8, T4=2-4, A100=8-16)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--save-every", type=int, default=5,
                    help="Save checkpoint every N epochs")
    p.add_argument("--compile", action="store_true",
                    help="Use torch.compile for extra speed (PyTorch 2.x)")
    p.add_argument("--cache-in-ram", action="store_true",
                    help="Preload all meshes into RAM (recommended: use when RAM > 10GB free)")

    p.add_argument("--w-recon", type=float, default=1.0, help="Vertex reconstruction weight")
    p.add_argument("--w-kl", type=float, default=0.001, help="KL divergence weight")
    p.add_argument("--w-laplacian", type=float, default=0.1, help="Laplacian smoothness weight")
    p.add_argument("--w-edge", type=float, default=0.05, help="Edge length preservation weight")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = GeometryVAETrainer(args)
    trainer.train()
