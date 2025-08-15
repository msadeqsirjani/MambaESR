# AlphaTune-MambaSR

**Baseline reproduction for MambaLiteSR with Knowledge Distillation**

Reproducing baseline results: **Set5: 28.28 dB PSNR** using a 370K parameter student model with knowledge distillation.

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python scripts/download_datasets.py
```
This downloads benchmark datasets (Set5, Set14, BSD100, Urban100) and sets up the data structure.

### 3. Train Complete Pipeline
```bash
python train_pipeline.py
```
This runs the complete baseline reproduction:
1. **Teacher training** (large model, no KD)
2. **Student training** (with KD from teacher)  
3. **Benchmark evaluation** (all 4 datasets)

## 📊 Expected Results

| Dataset  | Baseline PSNR | Target SSIM |
|----------|---------------|-------------|
| Set5     | 28.28 dB      | 0.837       |
| Set14    | 25.39 dB      | 0.727       |
| BSD100   | 25.20 dB      | 0.693       |
| Urban100 | 22.57 dB      | 0.686       |

## 🗂️ Project Structure

```
📁 AlphaTune-MambaSR/
├── 🎯 Training Scripts
│   ├── train_teacher.py      # Step 1: Train large teacher model
│   ├── train_student.py      # Step 2: Train student with KD
│   ├── train_pipeline.py     # Complete automated pipeline
│   └── eval_benchmarks.py    # Step 3: Evaluate all benchmarks
├── 🔧 Core Implementation  
│   └── src/mambalitesr/
│       ├── model.py          # MambaLiteSR architecture
│       ├── blocks.py         # Mamba blocks & components
│       ├── losses.py         # Distillation & perceptual losses
│       ├── data.py           # Datasets & data loading
│       ├── utils.py          # Metrics & utilities
│       ├── config.py         # Hyperparameters
│       └── experiment_manager.py  # Versioned experiment tracking
├── 📊 Outputs
│   └── runs/                 # Timestamped experiment results
│       ├── latest/           # Symlink to latest experiment
│       ├── mambalitesr_teacher_*/    # Teacher experiments
│       ├── mambalitesr_student_*/    # Student experiments  
│       └── mambalitesr_evaluation_*/ # Evaluation results
└── 📁 Data & Docs
    ├── data/                 # Datasets (auto-downloaded)
    ├── scripts/              # Dataset download utilities
    └── docs/                 # Paper & documentation
```

## ⚙️ Manual Training

If you prefer step-by-step control:

```bash
# Step 1: Train teacher model (large, no KD)
python train_teacher.py

# Step 2: Train student with knowledge distillation  
python train_student.py

# Step 3: Evaluate on all benchmarks
python eval_benchmarks.py
```

## 📊 Experiment Tracking

All experiments are automatically versioned with:
- ✅ **Timestamped directories** - No overwriting
- ✅ **Complete metrics** - PSNR, SSIM, training curves
- ✅ **Configuration backup** - All hyperparameters saved
- ✅ **Code snapshots** - Source code backup
- ✅ **Baseline comparisons** - Automatic performance gaps

**View latest results:**
```bash
python -c "
import json
from pathlib import Path
metrics_file = Path('runs/latest/metrics.json')
if metrics_file.exists():
    with open(metrics_file) as f:
        metrics = json.load(f)
    if 'evaluation' in metrics and 'summary' in metrics['evaluation']:
        avg_psnr = metrics['evaluation']['summary']['average_psnr']
        print(f'Latest Average PSNR: {avg_psnr:.2f} dB')
    else:
        print('Run eval_benchmarks.py to see results')
else:
    print('No experiments found. Run train_pipeline.py first.')
"
```

## 🔧 Model Architecture

**Student Model (396K params):**
- Embed dim: 40, Blocks: 6, Low-rank: 6
- Scale: 4x, Mixers per block: 2

**Teacher Model (larger):**  
- Embed dim: 128, Blocks: 12, Low-rank: 16
- Used for knowledge distillation only

## 📈 Hyperparameters

Key settings in `src/mambalitesr/config.py`:
- **Knowledge Distillation:** `alpha=0.2` (best from baseline)
- **Training:** 1000 epochs, batch size 32, LR 1e-4
- **Optimization:** Learning rate warmup, early stopping
- **Data:** Random crops, augmentation (flips, rotations)

## 🚀 Advanced Usage

**Tune hyperparameters:**
Edit `src/mambalitesr/config.py` and rerun training.

**Enable perceptual loss:**
```python
beta = 0.1  # VGG19 perceptual loss
```

**Different KD weights:**
```python
alpha = 0.4  # Try 0.2, 0.4, 0.6 from baseline paper
```

## 📄 License

Apache-2.0 for this implementation. See original Mamba license if using `mamba_ssm`.
