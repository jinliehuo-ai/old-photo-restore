"""下载修复用到的模型到 .models/（只需运行一次，约 600MB）。

    python download_models.py            # 必需的模型：人脸修复、放大、人脸检测/解析、补画
    python download_models.py --scratch  # 另外下载可选的自动划痕检测模型（需先下载 2GB 的官方压缩包，只保留其中 450MB）

已下载过的文件会跳过。
"""
import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS = Path(__file__).resolve().parent / ".models"

REQUIRED = {
    # 人脸修复 GFPGAN v1.4（TencentARC/GFPGAN，Apache-2.0）
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    # 背景放大 Real-ESRGAN x2（xinntao/Real-ESRGAN，BSD-3-Clause）
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    # 人脸检测、人脸解析（xinntao/facexlib，MIT）
    "detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
    # 补画 LaMa（advimman/lama，Apache-2.0；TorchScript 版本来自 enesmsahin/simple-lama-inpainting）
    "big-lama.pt": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
}

# 自动划痕检测（microsoft/Bringing-Old-Photos-Back-to-Life，MIT）
BOPB_REPO = "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git"
BOPB_ZIP = "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip"


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"已存在，跳过：{dest.name}")
        return
    print(f"下载 {dest.name} …")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("content-length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}  {done >> 20} / {total >> 20} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def scratch_model():
    src = MODELS / "bopb_src"
    if not (src / "Global" / "detection_models").is_dir():
        print("获取划痕检测网络代码 …")
        subprocess.run(["git", "clone", "-q", "--depth", "1", BOPB_REPO, str(src)], check=True)
    # 推理用不到同步 BatchNorm，用普通 BatchNorm 代替，免去编译依赖
    stub = src / "Global" / "detection_models" / "sync_batchnorm"
    stub.mkdir(exist_ok=True)
    (stub / "__init__.py").write_text("import torch.nn as nn\nDataParallelWithCallback = nn.DataParallel\n"
                                      "SynchronizedBatchNorm2d = nn.BatchNorm2d\n")
    ckpt = MODELS / "bopb" / "detection" / "FT_Epoch_latest.pt"
    if ckpt.exists():
        print("已存在，跳过：划痕检测模型")
        return
    zpath = MODELS / "global_checkpoints.zip"
    download(BOPB_ZIP, zpath)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z, z.open("checkpoints/detection/FT_Epoch_latest.pt") as fin, open(ckpt, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    zpath.unlink()
    print("划痕检测模型就绪")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", action="store_true", help="同时下载可选的自动划痕检测模型")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    for name, url in REQUIRED.items():
        download(url, MODELS / name)
    if args.scratch:
        scratch_model()
    print("完成。")


if __name__ == "__main__":
    sys.exit(main())
