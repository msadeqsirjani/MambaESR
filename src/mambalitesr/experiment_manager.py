#!/usr/bin/env python3
"""
Experiment management utilities for versioned runs and metrics tracking
"""

from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentManager:
    """Manages experiment directories and metrics tracking"""
    
    def __init__(self, base_name: str = "mambalitesr", base_dir: str = "runs"):
        self.base_name = base_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / f"{base_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Update latest symlink
        latest_link = self.base_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.is_dir():
                import shutil
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        latest_link.symlink_to(self.experiment_dir.name)
        
        # Initialize metrics file
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.metrics = {
            "experiment_name": base_name,
            "timestamp": timestamp,
            "training": {},
            "evaluation": {},
            "config": {}
        }
        self._save_metrics()
        
        print(f"🔬 Experiment: {self.experiment_dir}")
        print(f"📊 Metrics: {self.metrics_file}")
    
    def get_experiment_dir(self) -> Path:
        """Get the current experiment directory"""
        return self.experiment_dir
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration"""
        self.metrics["config"] = config
        self._save_metrics()
    
    def log_training_metric(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log training metrics for an epoch"""
        self.metrics["training"][f"epoch_{epoch}"] = {
            **metrics,
            "timestamp": datetime.now().isoformat()
        }
        self._save_metrics()
    
    def log_final_training_results(self, best_psnr: float, total_epochs: int, 
                                 total_params: int, early_stopped: bool = False) -> None:
        """Log final training summary"""
        self.metrics["training"]["summary"] = {
            "best_psnr": best_psnr,
            "total_epochs": total_epochs,
            "total_parameters": total_params,
            "early_stopped": early_stopped,
            "completed_at": datetime.now().isoformat()
        }
        self._save_metrics()
    
    def log_benchmark_results(self, dataset_name: str, psnr: float, ssim: float,
                            baseline_psnr: Optional[float] = None, 
                            baseline_ssim: Optional[float] = None) -> None:
        """Log benchmark evaluation results"""
        result = {
            "psnr": psnr,
            "ssim": ssim,
            "evaluated_at": datetime.now().isoformat()
        }
        
        if baseline_psnr is not None and baseline_ssim is not None:
            result["baseline_psnr"] = baseline_psnr
            result["baseline_ssim"] = baseline_ssim
            result["psnr_diff"] = psnr - baseline_psnr
            result["ssim_diff"] = ssim - baseline_ssim
        
        self.metrics["evaluation"][dataset_name] = result
        self._save_metrics()
    
    def log_evaluation_summary(self, avg_psnr: float, avg_ssim: float,
                             baseline_avg_psnr: Optional[float] = None) -> None:
        """Log overall evaluation summary"""
        summary = {
            "average_psnr": avg_psnr,
            "average_ssim": avg_ssim,
            "evaluated_at": datetime.now().isoformat()
        }
        
        if baseline_avg_psnr is not None:
            summary["baseline_average_psnr"] = baseline_avg_psnr
            summary["psnr_improvement"] = avg_psnr - baseline_avg_psnr
        
        self.metrics["evaluation"]["summary"] = summary
        self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def create_subdir(self, name: str) -> Path:
        """Create a subdirectory in the experiment directory"""
        subdir = self.experiment_dir / name
        subdir.mkdir(exist_ok=True)
        return subdir
    
    def copy_code_snapshot(self, source_dir: str = "src") -> None:
        """Copy source code snapshot to experiment directory"""
        source_path = Path(source_dir)
        if source_path.exists():
            dest_path = self.experiment_dir / "code_snapshot"
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
            print(f"📁 Code snapshot saved to {dest_path}")


def get_latest_experiment_dir(base_dir: str = "runs") -> Optional[Path]:
    """Get the latest experiment directory"""
    latest_link = Path(base_dir) / "latest"
    if latest_link.exists() and latest_link.is_symlink():
        return Path(base_dir) / latest_link.readlink()
    return None


def load_metrics(experiment_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics from an experiment directory"""
    metrics_file = experiment_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None
