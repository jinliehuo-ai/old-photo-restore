"""自动检测划痕、折痕：用微软 Bringing-Old-Photos-Back-to-Life 的划痕检测 UNet，输出修补掩膜。
这个模型是可选的（在 2GB 的官方压缩包里，用 download_models.py --scratch 下载）；没有时这一步自动跳过。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MODELS = Path(__file__).resolve().parent / ".models"
CKPT = MODELS / "bopb" / "detection" / "FT_Epoch_latest.pt"
_net = None


def available():
    return CKPT.exists() and (MODELS / "bopb_src" / "Global" / "detection_models").is_dir()


def _load(device):
    global _net
    if _net is None:
        sys.path.insert(0, str(MODELS / "bopb_src" / "Global"))
        from detection_models import networks

        net = networks.UNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6, padding=True,
                            batch_norm=True, up_mode="upsample", with_tanh=False, sync_bn=False, antialiasing=True)
        ckpt = torch.load(CKPT, map_location="cpu")
        net.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model_state"].items()})
        _net = net.to(device).eval()
    return _net


def crease_mask(img, device="cpu", short_side=256, thresh=0.4, protect=None):
    """返回整图大小的划痕/折痕掩膜。protect 区域（如人脸五官）内的检测结果会被去掉。"""
    if not available():
        return np.zeros(img.shape[:2], np.uint8)
    net = _load(device)
    h, w = img.shape[:2]
    s = short_side / min(h, w)
    th, tw = max(16, int(round(h * s / 16)) * 16), max(16, int(round(w * s / 16)) * 16)
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (tw, th), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy((gray.astype(np.float32) / 255 - 0.5) / 0.5)[None, None].to(device)
    with torch.no_grad():
        p = torch.sigmoid(net(x))
        p = F.interpolate(p, (h, w), mode="bilinear")[0, 0].cpu().numpy()
    m = (p >= thresh).astype(np.uint8) * 255
    if protect is not None:
        m[protect > 0] = 0
    # 去掉零碎噪点，稍微膨胀盖住折痕的高光边
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(12, h * w // 40000)
    m = np.where(keep[lab], 255, 0).astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)
