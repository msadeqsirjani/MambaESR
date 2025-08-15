#!/usr/bin/env python3
"""
Complete training pipeline to match baseline results:
1. Train teacher model (large, no KD)
2. Train student model with knowledge distillation
3. Evaluate on standard benchmarks
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=Path.cwd())
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False


def main():
    """Complete training pipeline"""
    print("🚀 MAMBALITESR BASELINE REPRODUCTION PIPELINE")
    print("Goal: Match baseline PSNR of 28.28 dB on Set5")
    
    # Step 1: Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Step 2: Train teacher model
    print("\n📚 Training large teacher model (no knowledge distillation)")
    if not run_command("python3 train_teacher.py", "Teacher model training"):
        return False
    
    # Step 3: Train student with KD
    print("\n🎓 Training student model with knowledge distillation")
    if not run_command("python3 train_student.py", "Student model training with KD"):
        return False
    
    # Step 4: Evaluate on benchmarks
    print("\n📊 Evaluating on standard benchmarks")
    if not run_command("python3 eval_benchmarks.py", "Benchmark evaluation"):
        return False
    
    print("\n🎉 PIPELINE COMPLETED!")
    print("\nNext steps:")
    print("1. Check results in eval_benchmarks.py output")
    print("2. Compare with baseline targets:")
    print("   - Set5: PSNR: 28.28, SSIM: 0.837")
    print("   - Set14: PSNR: 25.39, SSIM: 0.727")
    print("   - BSD100: PSNR: 25.20, SSIM: 0.693")
    print("   - Urban100: PSNR: 22.57, SSIM: 0.686")
    print("3. If results are below target, try:")
    print("   - Increase training epochs")
    print("   - Adjust alpha (try 0.4, 0.6)")
    print("   - Enable perceptual loss (beta=0.1)")


if __name__ == "__main__":
    main()
