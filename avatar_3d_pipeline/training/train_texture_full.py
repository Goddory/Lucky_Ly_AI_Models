"""
Full Texture Network Training — G, G_A, G_C, G_E with adversarial loss.

Usage:
    python training/train_texture_full.py \
        --data-root ./data --output-dir ./weights --epochs 50

Colab:
    !python training/train_texture_full.py \
        --data-root /content/drive/MyDrive/avatar_data \
        --output-dir /content/drive/MyDrive/avatar_checkpoints/texture \
        --epochs 50 --save-every 1

Training stages (run sequentially or pick one with --stage):
    1: Train G (texture prior) only
    2: Train G_A (skin tone) with frozen G
    3: Train G_C (reflectance) with frozen G, G_A
    4: Train G_E (PBR extractor) with frozen G, G_A, G_C
    all: End-to-end fine-tune G_A + G_C (G and G_E frozen)
"""
from __future__ import annotations

import argparse
import json
import sys
import logging
import time
from pathlib import Path

# Suppress torch.compile inductor low-level log noise
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# Ensure project root is on sys.path regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.generators import (
    TextureGeneratorG,
    SkinToneControlGA,
    ReflectanceNetworkGC,
    PBRExtractorGE,
)
from models.discriminators import MultiScaleDiscriminator
from utils.dataset import SelfiePBRPairDataset
from training.losses import (
    PerceptualLoss,
    adversarial_loss_generator,
    adversarial_loss_discriminator,
)


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True
    module.train()


class TextureTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")  # TF32 aggressively

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.g = TextureGeneratorG(latent_dim=args.latent_dim).to(self.device)
        self.ga = SkinToneControlGA().to(self.device)
        self.gc = ReflectanceNetworkGC().to(self.device)
        self.ge = PBRExtractorGE().to(self.device)
        self.disc = MultiScaleDiscriminator(in_channels=3).to(self.device)

        # channels_last: faster conv on Ampere+ GPUs (NHWC layout)
        _cl = torch.channels_last
        self.g   = self.g.to(memory_format=_cl)
        self.ga  = self.ga.to(memory_format=_cl)
        self.gc  = self.gc.to(memory_format=_cl)
        self.ge  = self.ge.to(memory_format=_cl)
        self.disc = self.disc.to(memory_format=_cl)

        # Losses
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss().to(self.device)
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.scaler_d = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Load any existing weights
        self._load_existing_weights()

        # Configure stage
        self._configure_stage(args.stage)

        # torch.compile: A100 has 108 SMs + 40-80GB VRAM → max-autotune safe
        if self.device.type == "cuda":
            self.g    = torch.compile(self.g,    mode="max-autotune")
            self.disc = torch.compile(self.disc, mode="max-autotune")

        # Optimizers
        self.opt_g = torch.optim.AdamW(
            [p for p in self._generator_params() if p.requires_grad],
            lr=args.lr_g, betas=(0.0, 0.999), weight_decay=args.weight_decay,
        )
        self.opt_d = torch.optim.AdamW(
            self.disc.parameters(),
            lr=args.lr_d, betas=(0.0, 0.999), weight_decay=args.weight_decay,
        )

        # LR Schedulers (CosineAnnealing for smooth convergence)
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g, T_max=args.epochs, eta_min=1e-6,
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, T_max=args.epochs, eta_min=1e-6,
        )

        # Dataset
        self.dataset = SelfiePBRPairDataset(
            root_dir=args.data_root,
            manifest_path=args.manifest,
            image_size=args.image_size,
            train=True,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

        self.start_epoch = 0
        self._maybe_resume()

        print(f"🎨 Texture Trainer — Stage: {args.stage}")
        print(f"   Device: {self.device}")
        print(f"   Samples: {len(self.dataset)}")
        print(f"   Epochs: {args.epochs} (start from {self.start_epoch})")

    def _generator_params(self):
        return (
            list(self.g.parameters())
            + list(self.ga.parameters())
            + list(self.gc.parameters())
            + list(self.ge.parameters())
        )

    def _load_existing_weights(self) -> None:
        """Load any previously trained stage weights."""
        w = Path(self.args.weights_dir) if self.args.weights_dir else self.output_dir
        weight_map = {
            "g": (self.g, [w / "texture_g.pt", w / "ffhq" / "texture_g.pt"]),
            "ga": (self.ga, [w / "texture_ga.pt", w / "facescape" / "texture_ga.pt"]),
            "gc": (self.gc, [w / "texture_gc.pt", w / "facescape" / "texture_gc.pt"]),
            "ge": (self.ge, [w / "texture_ge.pt", w / "facescape" / "texture_ge.pt"]),
        }
        for name, (module, paths) in weight_map.items():
            for path in paths:
                if path.exists():
                    state = torch.load(path, map_location=self.device)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    module.load_state_dict(state, strict=False)
                    print(f"   ✅ Loaded {name} from {path}")
                    break

    def _configure_stage(self, stage: str) -> None:
        """Freeze/unfreeze modules depending on training stage."""
        if stage == "1":
            unfreeze(self.g)
            freeze(self.ga)
            freeze(self.gc)
            freeze(self.ge)
        elif stage == "2":
            freeze(self.g)
            unfreeze(self.ga)
            freeze(self.gc)
            freeze(self.ge)
        elif stage == "3":
            freeze(self.g)
            freeze(self.ga)
            unfreeze(self.gc)
            freeze(self.ge)
        elif stage == "4":
            freeze(self.g)
            freeze(self.ga)
            freeze(self.gc)
            unfreeze(self.ge)
        elif stage == "all":
            freeze(self.g)
            unfreeze(self.ga)
            unfreeze(self.gc)
            freeze(self.ge)
        else:
            raise ValueError(f"Unknown stage: {stage}. Use 1, 2, 3, 4, or all.")

    def _maybe_resume(self) -> None:
        latest = self.output_dir / f"texture_stage_{self.args.stage}_latest.pt"
        if not latest.exists():
            return

        print(f"🔄 Resuming from {latest}")
        ckpt = torch.load(latest, map_location=self.device)
        if "g" in ckpt:
            self.g.load_state_dict(ckpt["g"], strict=False)
        if "ga" in ckpt:
            self.ga.load_state_dict(ckpt["ga"], strict=False)
        if "gc" in ckpt:
            self.gc.load_state_dict(ckpt["gc"], strict=False)
        if "ge" in ckpt:
            self.ge.load_state_dict(ckpt["ge"], strict=False)
        if "disc" in ckpt:
            self.disc.load_state_dict(ckpt["disc"], strict=False)
        if "opt_g" in ckpt:
            self.opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            self.opt_d.load_state_dict(ckpt["opt_d"])
        self.start_epoch = ckpt.get("epoch", 0)

    def _forward_pipeline(self, selfie: torch.Tensor, z_g: torch.Tensor, alpha: torch.Tensor):
        """Run the full texture pipeline G → G_A → G_C → G_E."""
        uv = (selfie + 1.0) * 0.5  # [-1,1] → [0,1]
        m, h = self.g(uv, z_g)
        a = self.ga(m, alpha)
        c = self.gc(a, h)
        s, n = self.ge(c)
        return {"m": m, "h": h, "a": a, "c": c, "s": s, "n": n}

    def train(self) -> None:
        best_loss = float("inf")
        n_steps = len(self.loader)
        log_every = max(1, n_steps // 10)

        print(f"\n🎨 Bắt đầu training Texture: Epoch {self.start_epoch + 1} → {self.args.epochs}")
        print(f"   Mỗi epoch = {n_steps} steps | Log mỗi {log_every} steps\n", flush=True)

        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_metrics = {"g_total": 0.0, "d_total": 0.0, "recon": 0.0, "percep": 0.0}
            t0 = time.time()

            print(f"── Epoch {epoch + 1}/{self.args.epochs} ──────────────────", flush=True)

            for step, batch in enumerate(self.loader):
                selfie = batch["selfie"].to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                target_albedo = batch["target_albedo"].to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                target_normal = batch["target_normal"].to(self.device, non_blocking=True).to(memory_format=torch.channels_last)

                bsz = selfie.size(0)
                z_g = torch.randn(bsz, self.args.latent_dim, device=self.device)
                alpha = torch.empty(bsz, device=self.device).uniform_(0.3, 0.9)

                target_a = (target_albedo + 1.0) * 0.5  # [-1,1] → [0,1]
                target_n = torch.nn.functional.normalize(target_normal, dim=1)

                # ---- Train Discriminator ----
                self.opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    with torch.no_grad():
                        pred = self._forward_pipeline(selfie, z_g, alpha)
                    real_preds = self.disc(target_a)
                    fake_preds = self.disc(pred["a"].detach())
                    loss_d = adversarial_loss_discriminator(
                        real_preds, fake_preds, mode=self.args.adv_mode,
                    )

                self.scaler_d.scale(loss_d).backward()
                self.scaler_d.step(self.opt_d)
                self.scaler_d.update()

                # ---- Train Generator(s) ----
                self.opt_g.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    pred = self._forward_pipeline(selfie, z_g, alpha)

                    loss_recon_a = self.l1(pred["a"], target_a)
                    loss_recon_c = self.l1(pred["c"], target_a)
                    loss_recon_n = self.l1(pred["n"], target_n)
                    loss_percep = self.perceptual(pred["a"], target_a)

                    fake_preds = self.disc(pred["a"])
                    loss_adv_g = adversarial_loss_generator(fake_preds, mode=self.args.adv_mode)

                    loss_g = (
                        self.args.w_recon * loss_recon_a
                        + self.args.w_recon_c * loss_recon_c
                        + self.args.w_normal * loss_recon_n
                        + self.args.w_percep * loss_percep
                        + self.args.w_adv * loss_adv_g
                    )

                self.scaler_g.scale(loss_g).backward()

                if self.args.max_grad_norm > 0:
                    self.scaler_g.unscale_(self.opt_g)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self._generator_params() if p.requires_grad],
                        self.args.max_grad_norm,
                    )

                self.scaler_g.step(self.opt_g)
                self.scaler_g.update()

                epoch_metrics["g_total"] += loss_g.item()
                epoch_metrics["d_total"] += loss_d.item()
                epoch_metrics["recon"] += loss_recon_a.item()
                epoch_metrics["percep"] += loss_percep.item()

                # In tiến độ mỗi 10%
                if (step + 1) % log_every == 0 or (step + 1) == n_steps:
                    done = step + 1
                    pct = done / n_steps * 100
                    elapsed_step = time.time() - t0
                    speed = done / elapsed_step if elapsed_step > 0 else 0
                    eta = (n_steps - done) / speed if speed > 0 else 0
                    cur_g = epoch_metrics["g_total"] / done
                    cur_d = epoch_metrics["d_total"] / done
                    print(
                        f"  [{pct:5.1f}%] {done}/{n_steps} | "
                        f"G={cur_g:.4f} D={cur_d:.4f} | "
                        f"{speed:.1f} step/s | ETA {eta:.0f}s",
                        flush=True,
                    )

            self.scheduler_g.step()
            self.scheduler_d.step()

            n_steps_final = max(1, len(self.loader))
            avg = {k: v / n_steps_final for k, v in epoch_metrics.items()}
            elapsed = time.time() - t0
            lr_g = self.scheduler_g.get_last_lr()[0]

            is_best = avg["g_total"] < best_loss
            if is_best:
                best_loss = avg["g_total"]
            marker = " ⭐ BEST" if is_best else ""

            print(
                f"✅ Epoch {epoch + 1}/{self.args.epochs} | "
                f"G={avg['g_total']:.4f} D={avg['d_total']:.4f} "
                f"recon={avg['recon']:.4f} percep={avg['percep']:.4f} "
                f"lr={lr_g:.2e} ({elapsed:.0f}s){marker}\n",
                flush=True,
            )

            if (epoch + 1) % self.args.save_every == 0:
                self._save_checkpoint(epoch + 1, avg)

            if is_best:
                self._save_checkpoint(epoch + 1, avg, tag="best")

        print(f"\n✅ Texture training (stage {self.args.stage}) complete!")

    def _save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest") -> None:
        ckpt = {
            "epoch": epoch,
            "stage": self.args.stage,
            "metrics": metrics,
            "g": self.g.state_dict(),
            "ga": self.ga.state_dict(),
            "gc": self.gc.state_dict(),
            "ge": self.ge.state_dict(),
            "disc": self.disc.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }

        stage = self.args.stage
        path = self.output_dir / f"texture_stage_{stage}_{tag}.pt"
        torch.save(ckpt, path)

        # Save production-compatible per-module weights
        stage_module_map = {
            "1": [("texture_g.pt", self.g)],
            "2": [("texture_ga.pt", self.ga)],
            "3": [("texture_gc.pt", self.gc)],
            "4": [("texture_ge.pt", self.ge)],
            "all": [
                ("texture_ga.pt", self.ga),
                ("texture_gc.pt", self.gc),
            ],
        }
        for filename, module in stage_module_map.get(stage, []):
            torch.save(module.state_dict(), self.output_dir / filename)

        (self.output_dir / f"metrics_stage_{stage}_{tag}.json").write_text(
            json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full texture network adversarial training")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--weights-dir", type=str, default=None,
                    help="Directory with existing pretrained weights to load")
    p.add_argument("--output-dir", type=str, default="weights")
    p.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "4", "all"],
                    help="Training stage: 1=G, 2=GA, 3=GC, 4=GE, all=GA+GC end-to-end")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--save-every", type=int, default=1)

    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--adv-mode", type=str, default="hinge", choices=["hinge", "lsgan", "bce"])

    p.add_argument("--w-recon", type=float, default=10.0)
    p.add_argument("--w-recon-c", type=float, default=5.0)
    p.add_argument("--w-normal", type=float, default=3.0)
    p.add_argument("--w-percep", type=float, default=1.0)
    p.add_argument("--w-adv", type=float, default=1.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = TextureTrainer(args)
    trainer.train()
