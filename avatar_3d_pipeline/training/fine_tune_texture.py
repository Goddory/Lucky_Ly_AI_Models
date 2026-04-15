from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.generators import (
    PBRExtractorGE,
    ReflectanceNetworkGC,
    SkinToneControlGA,
    TextureGeneratorG,
    inject_lora,
    lora_parameters,
)
from utils.dataset import SelfiePBRPairDataset


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def keep_only_lora_trainable(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    state = module.state_dict()
    return {k: v for k, v in state.items() if "lora_" in k}


class TextureFineTuner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.weights_dir = Path(args.weights_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.g_texture = TextureGeneratorG(latent_dim=args.latent_dim).to(self.device)
        self.g_a = SkinToneControlGA().to(self.device)
        self.g_c = ReflectanceNetworkGC().to(self.device)
        self.g_e = PBRExtractorGE().to(self.device)

        self._load_pretrained()
        self._configure_trainable_modules()

        self.optimizer = torch.optim.AdamW(
            self._trainable_params(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.l1 = nn.L1Loss()

        self.train_dataset = SelfiePBRPairDataset(
            root_dir=args.data_root,
            manifest_path=args.manifest,
            image_size=args.image_size,
            train=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )

    def _trainable_params(self) -> Iterable[nn.Parameter]:
        return list(lora_parameters(self.g_a)) + list(lora_parameters(self.g_c))

    def _maybe_load_module(self, module: nn.Module, candidates: list[Path]) -> bool:
        for path in candidates:
            if not path.exists():
                continue
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            module.load_state_dict(state, strict=False)
            return True
        return False

    def _load_pretrained(self) -> None:
        loaded_g = self._maybe_load_module(
            self.g_texture,
            [
                self.weights_dir / "texture_g.pt",
                self.weights_dir / "ffhq" / "texture_g.pt",
            ],
        )
        loaded_ga = self._maybe_load_module(
            self.g_a,
            [
                self.weights_dir / "texture_ga.pt",
                self.weights_dir / "facescape" / "texture_ga.pt",
            ],
        )
        loaded_gc = self._maybe_load_module(
            self.g_c,
            [
                self.weights_dir / "texture_gc.pt",
                self.weights_dir / "facescape" / "texture_gc.pt",
            ],
        )
        loaded_ge = self._maybe_load_module(
            self.g_e,
            [
                self.weights_dir / "texture_ge.pt",
                self.weights_dir / "facescape" / "texture_ge.pt",
            ],
        )

        if not self.args.allow_missing_pretrained:
            missing = []
            if not loaded_g:
                missing.append("texture_g.pt")
            if not loaded_ga:
                missing.append("texture_ga.pt")
            if not loaded_gc:
                missing.append("texture_gc.pt")
            if not loaded_ge and self.args.lambda_n > 0:
                missing.append("texture_ge.pt")

            if missing:
                raise FileNotFoundError(
                    "Missing pretrained checkpoints for fine-tuning: "
                    + ", ".join(missing)
                    + ". Run utils/download_weights.py first."
                )

    def _configure_trainable_modules(self) -> None:
        # Backbone priors are frozen; only LoRA adapters update in G_A/G_C.
        freeze_module(self.g_texture)
        freeze_module(self.g_a)
        freeze_module(self.g_c)
        freeze_module(self.g_e)

        inject_lora(
            self.g_a,
            rank=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            target_keywords=("conv", "attn", "down", "up", "bottleneck"),
        )
        inject_lora(
            self.g_c,
            rank=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            target_keywords=("conv", "attn", "net"),
        )

        keep_only_lora_trainable(self.g_a)
        keep_only_lora_trainable(self.g_c)

        self.g_texture.eval()
        self.g_e.eval()
        self.g_a.train()
        self.g_c.train()

    def train(self) -> None:
        global_step = 0
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.args.epochs):
            epoch_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                selfie = batch["selfie"].to(self.device, non_blocking=True)
                target_albedo = batch["target_albedo"].to(self.device, non_blocking=True)
                target_normal = batch["target_normal"].to(self.device, non_blocking=True)

                bsz = selfie.size(0)
                z_g = torch.zeros((bsz, self.args.latent_dim), dtype=torch.float32, device=self.device)
                alpha = torch.empty((bsz,), device=self.device).uniform_(
                    self.args.alpha_min,
                    self.args.alpha_max,
                )

                with torch.no_grad():
                    uv_proxy = (selfie + 1.0) * 0.5
                    m, h = self.g_texture(uv_proxy, z_g)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred_a = self.g_a(m, alpha)
                    pred_c = self.g_c(pred_a, h)
                    _, pred_n = self.g_e(pred_c)

                    target_a = (target_albedo + 1.0) * 0.5
                    target_n = F.normalize(target_normal, dim=1)

                    loss_a = self.l1(pred_a, target_a)
                    loss_c = self.l1(pred_c, target_a)
                    loss_n = self.l1(pred_n, target_n)

                    total_loss = (
                        self.args.lambda_a * loss_a
                        + self.args.lambda_c * loss_c
                        + self.args.lambda_n * loss_n
                    ) / self.args.grad_accum_steps

                self.scaler.scale(total_loss).backward()

                if (step + 1) % self.args.grad_accum_steps == 0:
                    if self.args.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self._trainable_params()),
                            self.args.max_grad_norm,
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += total_loss.item() * self.args.grad_accum_steps
                global_step += 1

                if global_step % self.args.log_interval == 0:
                    print(
                        f"[epoch {epoch+1}/{self.args.epochs}] "
                        f"step={global_step} "
                        f"loss={epoch_loss / (step + 1):.5f} "
                        f"loss_a={loss_a.item():.5f} "
                        f"loss_c={loss_c.item():.5f} "
                        f"loss_n={loss_n.item():.5f}"
                    )

            self._save_checkpoint(epoch + 1, epoch_loss / max(1, len(self.train_loader)))

    def _save_checkpoint(self, epoch: int, avg_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "args": vars(self.args),
            "ga_lora": lora_state_dict(self.g_a),
            "gc_lora": lora_state_dict(self.g_c),
            "ga_full": self.g_a.state_dict(),
            "gc_full": self.g_c.state_dict(),
        }

        out_file = self.output_dir / f"texture_lora_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, out_file)

        latest = self.output_dir / "texture_lora_latest.pt"
        torch.save(checkpoint, latest)

        metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss,
        }
        (self.output_dir / "latest_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for G_A and G_C texture modules")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default="weights")
    parser.add_argument("--output-dir", type=str, default="checkpoints/texture_lora")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)

    parser.add_argument("--alpha-min", type=float, default=0.3)
    parser.add_argument("--alpha-max", type=float, default=0.9)

    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-c", type=float, default=0.7)
    parser.add_argument("--lambda-n", type=float, default=0.3)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--allow-missing-pretrained",
        action="store_true",
        help="Allow fallback random initialization (not recommended).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = TextureFineTuner(args)
    trainer.train()
