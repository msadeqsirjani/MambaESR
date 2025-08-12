## AlphaTune-MambaSR Baseline

Baseline implementation of the MambaLiteSR student (x4) plus optional knowledge distillation (KD) support with a teacher model.

### Features
- Student model: 4 RMMBs (8 mixers), embed dim 32, pixel-shuffle upsampler
- Optional KD with fixed alpha (teacher forward in eval/no-grad)
- Data pipeline for DF2K (HR), validation on DIV2K_valid_HR, benchmarks (Set5/Set14/BSD100/Urban100)
- Metrics: PSNR/SSIM on Y channel

### Setup
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### Datasets
```powershell
.\.venv\Scripts\python scripts\download_datasets.py
```
This creates `data/DF2K/HR`, `data/DIV2K_valid_HR`, and downloads benchmarks when available. You can copy `DIV2K_train_HR` into `DF2K/HR` as a placeholder; add Flickr2K to complete DF2K.

### Train
- Configure parameters in `src/mambalitesr/config.py`.
- Student-only baseline (default):
```powershell
.\.venv\Scripts\python train.py
```
- KD baseline with fixed alpha (set `use_kd=True` and optionally `TEACHER.ckpt` if you have teacher weights):
```powershell
.\.venv\Scripts\python train.py
```

### Evaluate
```powershell
.\.venv\Scripts\python eval.py
```
Uses `EVAL.ckpt` and `CONFIG.data_root`/`CONFIG.scale` from `config.py`.

### Notes
- If `mamba_ssm` is not installed, the mixer uses a lightweight fallback and still trains.
- To enable `mamba_ssm` with GPU kernels, follow the instructions in the official repo.

### License
Apache-2.0 for code in this repo. See original Mamba license if you use `mamba_ssm`.
