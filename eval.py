from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mambalitesr.config import CONFIG, EVAL
from src.mambalitesr.data import EvalImageFolder
from src.mambalitesr.model import MambaLiteSR
from src.mambalitesr.utils import calculate_psnr_y, calculate_ssim_y, load_checkpoint


def main() -> None:
    cfg = CONFIG
    ecfg = EVAL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MambaLiteSR(scale=cfg.scale)
    load_checkpoint(model, Path(ecfg.ckpt), map_location=device)
    model.to(device).eval()

    hr_dir = Path(cfg.data_root) / ecfg.eval_set / "HR"
    ds = EvalImageFolder(hr_dir, scale=cfg.scale)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    psnrs, ssims = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Eval {ecfg.eval_set}"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr)
            psnrs.append(calculate_psnr_y(sr, hr))
            ssims.append(calculate_ssim_y(sr, hr))

    print(f"{ecfg.eval_set} PSNR-Y: {sum(psnrs)/len(psnrs):.3f} | SSIM-Y: {sum(ssims)/len(ssims):.4f}")


if __name__ == "__main__":
    main()
