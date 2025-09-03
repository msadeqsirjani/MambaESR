#!/usr/bin/env python3
"""
Evaluation script for standard SR benchmarks: Set5, Set14, BSD100, Urban100
"""

from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CONFIG
from src.data import EvalImageFolder
from src.model import MambaLiteSR
from src.utils import calculate_psnr_y, calculate_ssim_y, load_checkpoint
from src.experiment_manager import (
    ExperimentManager,
    get_latest_experiment_dir,
)


def evaluate_dataset(
    model: torch.nn.Module, dataset_path: Path, device: torch.device, scale: int
) -> tuple[float, float]:
    """Evaluate model on a single dataset"""
    if not dataset_path.exists():
        print(f"Warning: Dataset {dataset_path} not found, skipping...")
        return 0.0, 0.0

    dataset = EvalImageFolder(dataset_path, scale=scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    psnrs, ssims = [], []
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {dataset_path.name}"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr)

            psnrs.append(calculate_psnr_y(sr, hr))
            ssims.append(calculate_ssim_y(sr, hr))

    avg_psnr = sum(psnrs) / len(psnrs) if psnrs else 0.0
    avg_ssim = sum(ssims) / len(ssims) if ssims else 0.0

    return avg_psnr, avg_ssim


def main() -> None:
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create evaluation experiment manager under runs/latest/evaluation
    experiment = ExperimentManager("evaluation")

    # Load model
    model = MambaLiteSR(
        scale=cfg.scale,
        embed_dim=cfg.embed_dim,
        num_rmmb=cfg.num_rmmb,
        mixers_per_block=cfg.mixers_per_block,
        low_rank=cfg.low_rank,
    ).to(device)

    # Try to find the latest student experiment under runs/latest
    latest_root = get_latest_experiment_dir()
    checkpoint_path = None
    if latest_root and latest_root.exists():
        student_dir = latest_root / "student"
        if (student_dir / "best.pt").exists():
            checkpoint_path = student_dir / "best.pt"
        elif (student_dir / "last.pt").exists():
            checkpoint_path = student_dir / "last.pt"

    if checkpoint_path and checkpoint_path.exists():
        load_checkpoint(model, checkpoint_path, map_location=device)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        return

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Standard benchmark datasets
    data_root = Path(cfg.data_root)
    datasets = {
        "Set5": data_root / "Set5" / "HR",
        "Set14": data_root / "Set14" / "HR",
        "BSD100": data_root / "BSD100" / "HR",
        "Urban100": data_root / "Urban100" / "HR",
    }

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 60)

    results = {}
    for dataset_name, dataset_path in datasets.items():
        psnr, ssim = evaluate_dataset(model, dataset_path, device, cfg.scale)
        results[dataset_name] = (psnr, ssim)

        if psnr > 0:  # Only print if dataset was found
            print(f"{dataset_name:>10}: PSNR: {psnr:5.2f} dB | SSIM: {ssim:.4f}")

            # Log to experiment manager
            baseline_psnr = baseline_results[dataset_name][0]
            baseline_ssim = baseline_results[dataset_name][1]
            experiment.log_benchmark_results(
                dataset_name, psnr, ssim, baseline_psnr, baseline_ssim
            )

    print("=" * 60)

    # Compare with baseline
    baseline_results = {
        "Set5": (28.28, 0.837),
        "Set14": (25.39, 0.727),
        "BSD100": (25.20, 0.693),
        "Urban100": (22.57, 0.686),
    }

    print("\nCOMPARISON WITH BASELINE:")
    print("-" * 60)
    print(
        f"{'Dataset':<10} {'Your PSNR':<10} {'Baseline':<10} {'Diff':<8} {'Your SSIM':<10} {'Baseline':<10} {'Diff':<8}"
    )
    print("-" * 60)

    for dataset_name in datasets.keys():
        if results[dataset_name][0] > 0:  # Only compare if we have results
            your_psnr, your_ssim = results[dataset_name]
            base_psnr, base_ssim = baseline_results[dataset_name]

            psnr_diff = your_psnr - base_psnr
            ssim_diff = your_ssim - base_ssim

            print(
                f"{dataset_name:<10} {your_psnr:>9.2f} {base_psnr:>9.2f} "
                f"{psnr_diff:>+7.2f} {your_ssim:>9.4f} {base_ssim:>9.4f} {ssim_diff:>+7.4f}"
            )

    # Calculate and log overall summary
    valid_results = [(psnr, ssim) for psnr, ssim in results.values() if psnr > 0]
    if valid_results:
        avg_psnr = sum(psnr for psnr, _ in valid_results) / len(valid_results)
        avg_ssim = sum(ssim for _, ssim in valid_results) / len(valid_results)
        baseline_avg_psnr = sum(
            baseline_results[name][0] for name in results.keys() if results[name][0] > 0
        ) / len(valid_results)

        experiment.log_evaluation_summary(avg_psnr, avg_ssim, baseline_avg_psnr)

        print(f"\n📊 SUMMARY:")
        print(f"Average PSNR: {avg_psnr:.2f} dB (Baseline: {baseline_avg_psnr:.2f} dB)")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"📁 Evaluation results saved to: {experiment.metrics_file}")


if __name__ == "__main__":
    main()
