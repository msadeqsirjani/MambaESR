#!/usr/bin/env python
import ssl
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen
import shutil

# ==========================
# Hardcoded settings
# ==========================
ROOT = Path("data")
SKIP_DIV2K = False  # You already have DIV2K; set to False to auto-download
DOWNLOAD_BENCHMARKS = True  # Download Set5/Set14/BSD100/Urban100 if missing
POPULATE_DF2K_FROM_DIV2K = True  # Copy DIV2K_train_HR into DF2K/HR if DF2K/HR is empty

# ==========================
# URLs
# ==========================
DIV2K_TRAIN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

# Common SR benchmark archive (contains Set5, Set14, B100, Urban100)
# Primary and fallback mirrors
BENCHMARK_URLS = [
    "https://cv.snu.ac.kr/research/EDSR/benchmark.tar",
    "https://kr.object.ncloudstorage.com/snu-ailab/EDSR/benchmark.tar",
]


def safe_zip_extract(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_with_fallback(urls: list[str], dest_path: Path) -> None:
    last_err: Exception | None = None
    
    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for url in urls:
        try:
            print(f"Downloading: {url}")
            # Use custom SSL context
            import urllib.request
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            urlretrieve(url, dest_path)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  Failed: {e}")
            last_err = e
    if last_err is not None:
        raise last_err


def extract_benchmark_tar(tar_path: Path, root: Path, needed_sets: set[str]) -> None:
    """
    Extract HR images for needed sets from EDSR benchmark archive.
    Archive layout: benchmark/{Set5,Set14,B100,Urban100}/HR/*.png
    We map B100 -> BSD100 in destination.
    """
    mapping = {"Set5": "Set5", "Set14": "Set14", "B100": "BSD100", "Urban100": "Urban100"}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 4:
                continue
            # Expect: benchmark/<SET>/<SPLIT>/<filename>
            _, set_name, split_name, *rest = parts
            if split_name != "HR":
                continue
            if set_name not in mapping:
                continue
            dest_root_name = mapping[set_name]
            if dest_root_name not in needed_sets:
                continue
            if member.isdir():
                continue
            dest_dir = root / dest_root_name / "HR"
            ensure_dir(dest_dir)
            dest_file = dest_dir / Path(*rest)
            # Extract file bytes and write
            f = tf.extractfile(member)
            if f is None:
                continue
            with open(dest_file, "wb") as out_f:
                shutil.copyfileobj(f, out_f)


def dir_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def maybe_download_benchmarks(root: Path) -> None:
    # Determine which sets are missing or empty
    sets = ["Set5", "Set14", "BSD100", "Urban100"]
    needed: set[str] = set()
    for s in sets:
        hr_dir = root / s / "HR"
        if not dir_has_files(hr_dir):
            needed.add(s)
    if not needed:
        print("Benchmark sets already present; skipping download.")
        return
    tar_path = root / "benchmark.tar"
    if not tar_path.exists():
        ensure_dir(root)
        download_with_fallback(BENCHMARK_URLS, tar_path)
    print(f"Extracting benchmark HR images for: {sorted(needed)}")
    extract_benchmark_tar(tar_path, root, needed)


def maybe_populate_df2k_from_div2k(root: Path) -> None:
    df2k_hr = root / "DF2K" / "HR"
    ensure_dir(df2k_hr)
    if dir_has_files(df2k_hr):
        return
    div2k_train = root / "DIV2K_train_HR"
    if div2k_train.exists() and any(div2k_train.iterdir()):
        print("DF2K/HR is empty. Populating it with a copy of DIV2K_train_HR as a placeholder (Flickr2K not included).")
        for p in div2k_train.iterdir():
            if p.is_file():
                shutil.copy2(p, df2k_hr / p.name)


def main() -> None:
    root = ROOT
    ensure_dir(root)

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Install opener with custom SSL context
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    urllib.request.install_opener(opener)

    # DIV2K Train/Valid HR
    if not SKIP_DIV2K:
        div2k_train_zip = root / "DIV2K_train_HR.zip"
        div2k_valid_zip = root / "DIV2K_valid_HR.zip"
        if not (root / "DIV2K_train_HR").exists():
            print("Downloading DIV2K train HR...")
            urlretrieve(DIV2K_TRAIN_URL, div2k_train_zip)
            safe_zip_extract(div2k_train_zip, root)
            div2k_train_zip.unlink(missing_ok=True)
        if not (root / "DIV2K_valid_HR").exists():
            print("Downloading DIV2K valid HR...")
            urlretrieve(DIV2K_VALID_URL, div2k_valid_zip)
            safe_zip_extract(div2k_valid_zip, root)
            div2k_valid_zip.unlink(missing_ok=True)
    else:
        print("Skipping DIV2K downloads per SKIP_DIV2K=True")

    # DF2K expected layout and optional population
    df2k_hr = root / "DF2K" / "HR"
    ensure_dir(df2k_hr)
    if POPULATE_DF2K_FROM_DIV2K:
        maybe_populate_df2k_from_div2k(root)

    print("\nDF2K expected content:")
    print(f"  Place/copy DIV2K_train_HR/* and Flickr2K_HR/* into: {df2k_hr}")

    print("\nValidation (DIV2K_valid_HR) at:")
    print(f"  {root / 'DIV2K_valid_HR'}")

    # Benchmarks
    print("\nBenchmarks (Set5, Set14, BSD100, Urban100) HR folders under:")
    for name in ["Set5", "Set14", "BSD100", "Urban100"]:
        ensure_dir(root / name / "HR")
        print(f"  {root / name / 'HR'}")

    if DOWNLOAD_BENCHMARKS:
        try:
            maybe_download_benchmarks(root)
        except Exception as e:  # noqa: BLE001
            print("\nFailed to download EDSR benchmark archive.")
            print("You can also add datasets manually (HR images) into the folders above.")
            print("Optional mirrors:")
            print("  Set5:    https://huggingface.co/datasets/eugenesiow/Set5")
            print("  Set14:   https://huggingface.co/datasets/eugenesiow/Set14")
            print("  BSD100:  https://huggingface.co/datasets/eugenesiow/BSD100")
            print("  Urban100:https://huggingface.co/datasets/eugenesiow/Urban100")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
