"""
Evaluation & Benchmarking — Compute quality metrics for avatar outputs.

Metrics:
  - SSIM: Structural Similarity (texture quality)
  - LPIPS: Learned Perceptual Image Patch Similarity (perceptual quality)
  - FID: Fréchet Inception Distance (distribution quality, needs ≥50 samples)
  - Chamfer Distance: 3D geometry accuracy (if reference mesh available)

Usage:
    python training/evaluate.py \
        --pred-dir ./outputs/textures \
        --gt-dir ./data/albedo \
        --metrics ssim lpips

    python training/evaluate.py \
        --pred-dir ./outputs/textures \
        --gt-dir ./data/albedo \
        --metrics all --report-path ./outputs/eval_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def load_image_tensor(path: Path, size: int = 512) -> torch.Tensor:
    """Load image as [1, 3, H, W] tensor in [0, 1] range."""
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [size, size])
    return TF.to_tensor(img).unsqueeze(0)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> float:
    """Compute SSIM between two image tensors [B, C, H, W] in [0,1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel_size = window_size
    sigma = 1.5
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.size(1), 1, -1, -1).to(pred.device)

    pad = kernel_size // 2
    mu1 = torch.nn.functional.conv2d(pred, window, padding=pad, groups=pred.size(1))
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=pred.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=pad, groups=pred.size(1)) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=pad, groups=pred.size(1)) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=pad, groups=pred.size(1)) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Optional[float]:
    """Compute LPIPS if the lpips package is available."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex", verbose=False).to(pred.device)
        with torch.no_grad():
            score = loss_fn(pred * 2 - 1, target * 2 - 1)
        return score.mean().item()
    except ImportError:
        print("⚠️  lpips package not installed. Skipping LPIPS metric.")
        return None


def compute_chamfer_distance(
    pred_vertices: np.ndarray,
    target_vertices: np.ndarray,
) -> float:
    """One-directional Chamfer distance (in mm if mesh is in mm)."""
    from scipy.spatial import cKDTree

    tree_target = cKDTree(target_vertices)
    tree_pred = cKDTree(pred_vertices)

    d_pred_to_target, _ = tree_target.query(pred_vertices)
    d_target_to_pred, _ = tree_pred.query(target_vertices)

    return float(np.mean(d_pred_to_target) + np.mean(d_target_to_pred)) / 2.0


class Evaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_dir = Path(args.pred_dir)
        self.gt_dir = Path(args.gt_dir)

        self.metrics_to_compute = set(args.metrics)
        if "all" in self.metrics_to_compute:
            self.metrics_to_compute = {"ssim", "lpips"}

    def evaluate(self) -> dict:
        pred_files = sorted(self.pred_dir.glob("*.png")) + sorted(self.pred_dir.glob("*.jpg"))
        gt_files = sorted(self.gt_dir.glob("*.png")) + sorted(self.gt_dir.glob("*.jpg"))

        # Match by filename stem
        gt_stems = {f.stem: f for f in gt_files}
        pairs = []
        for pf in pred_files:
            if pf.stem in gt_stems:
                pairs.append((pf, gt_stems[pf.stem]))

        if not pairs:
            print("❌ No matching pred/gt pairs found.")
            return {}

        print(f"📊 Evaluating {len(pairs)} pairs...")

        results: dict = {"samples": len(pairs)}
        ssim_scores = []
        lpips_scores = []

        for pred_path, gt_path in pairs:
            pred = load_image_tensor(pred_path, self.args.image_size).to(self.device)
            gt = load_image_tensor(gt_path, self.args.image_size).to(self.device)

            if "ssim" in self.metrics_to_compute:
                ssim_scores.append(compute_ssim(pred, gt))

            if "lpips" in self.metrics_to_compute:
                score = compute_lpips(pred, gt)
                if score is not None:
                    lpips_scores.append(score)

        if ssim_scores:
            results["ssim_mean"] = float(np.mean(ssim_scores))
            results["ssim_std"] = float(np.std(ssim_scores))
            results["ssim_min"] = float(np.min(ssim_scores))
            results["ssim_max"] = float(np.max(ssim_scores))
            status = "✅" if results["ssim_mean"] >= 0.80 else "⚠️"
            print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f} {status}")

        if lpips_scores:
            results["lpips_mean"] = float(np.mean(lpips_scores))
            results["lpips_std"] = float(np.std(lpips_scores))
            status = "✅" if results["lpips_mean"] <= 0.20 else "⚠️"
            print(f"  LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f} {status}")

        # Save report
        if self.args.report_path:
            report_path = Path(self.args.report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\n📄 Report saved to {report_path}")

        return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate avatar texture/geometry quality")
    p.add_argument("--pred-dir", type=str, required=True,
                    help="Directory with predicted textures/meshes")
    p.add_argument("--gt-dir", type=str, required=True,
                    help="Directory with ground truth textures/meshes")
    p.add_argument("--metrics", nargs="+", default=["all"],
                    choices=["ssim", "lpips", "all"],
                    help="Metrics to compute")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--report-path", type=str, default=None,
                    help="Path to save JSON report")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluator(args)
    evaluator.evaluate()
