"""预处理：去反光（LaMa 修补）+ 去雾/去褪色 + 自动色彩 + 局部对比度 + 自然饱和度。"""
import cv2
import numpy as np

from defects import global_only, neutralize_highlights

_lama = None


def glare_mask(img, faces=()):
    """反光区域：接近过曝、几乎无色、比周围亮很多；人脸区域不动。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    k = max(31, (min(img.shape[:2]) // 12) | 1)
    bg = cv2.medianBlur(v.astype(np.uint8), k).astype(np.int16)
    m = (v > 232) & (s < 40) & (v - bg > 45)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    for x1, y1, x2, y2 in faces:
        pw, ph = (x2 - x1) * 0.4, (y2 - y1) * 0.4
        m[max(0, int(y1 - ph)):int(y2 + ph), max(0, int(x1 - pw)):int(x2 + pw)] = 0
    # 去掉过大的连通域（白墙、天空、白衣服），保留点状/条状反光
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    area_max = img.shape[0] * img.shape[1] * 0.04
    keep = np.zeros(n, bool)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 6) & (stats[1:, cv2.CC_STAT_AREA] <= area_max)
    m = np.where(keep[lab], 255, 0).astype(np.uint8)
    return cv2.dilate(m, np.ones((7, 7), np.uint8))


def _lama_full(img, mask):
    global _lama
    if _lama is None:
        import os
        from pathlib import Path

        import torch
        from simple_lama_inpainting import SimpleLama
        local = Path(__file__).resolve().parent / ".models" / "big-lama.pt"  # download_models.py 下载的位置
        if local.exists():
            os.environ.setdefault("LAMA_MODEL", str(local))
        _lama = SimpleLama(device=torch.device("cpu"))
    from PIL import Image
    res = _lama(Image.fromarray(img[..., ::-1]), Image.fromarray(mask))
    return np.array(res)[: img.shape[0], : img.shape[1], ::-1]


def lama_raw(img, mask):
    """LaMa 补画，返回整图大小的补画结果（掩膜外为原图）。
    只把掩膜附近的区域送进模型：相邻的缺损合并成块，每块外扩一圈作为上下文。
    整张大图送进 LaMa 在 CPU 上又慢又占内存（8GB 机器上 1600px 的图会被系统杀掉）。"""
    h, w = img.shape[:2]
    out = img.copy()
    if not mask.any():
        return out
    pad = max(48, int(min(h, w) * 0.08))
    groups = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad + 1, pad + 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats((groups > 0).astype(np.uint8))
    for j in range(1, n):
        x, y, bw, bh = stats[j, :4]
        x0, y0 = max(0, x - pad // 2), max(0, y - pad // 2)
        x1, y1 = min(w, x + bw + pad // 2), min(h, y + bh + pad // 2)
        m = mask[y0:y1, x0:x1]
        if m.any():
            out[y0:y1, x0:x1] = _lama_full(img[y0:y1, x0:x1], m)
    return out


def inpaint(img, mask):
    """用 LaMa 补画 mask 区域。"""
    if not mask.any():
        return img
    return np.where(mask[..., None] > 0, lama_raw(img, mask), img)


def remove_glare(img, faces=(), extra_mask=None):
    mask = glare_mask(img, faces)
    if extra_mask is not None:
        mask = cv2.bitwise_or(mask, extra_mask)
    if mask.mean() < 0.05:  # 几乎没有反光
        return img, 0.0
    res = lama_raw(img, mask)
    # 掩膜内完全替换，只在掩膜外围羽化（小斑点如果整体羽化，中心只会被淡化而不是去掉）
    a = np.maximum(cv2.GaussianBlur(cv2.dilate(mask, np.ones((5, 5), np.uint8)), (0, 0), 2), mask)
    a = a.astype(np.float32)[..., None] / 255
    out = (img * (1 - a) + res * a).astype(np.uint8)
    return out, mask.mean() / 255 * 100


def remove_veil(img, faces=(), strength=1.0):
    """去朦胧反光/白色条纹：反光相当于在原色上叠了一层半透明白光，深色区域三通道最小值会被整体抬高。
    用三通道最小值的顶帽运算估计这层白光，只在底色偏深的区域扣掉；白衣服、鞋带这类本身就白的物体不动。"""
    f = img.astype(np.float32) / 255
    mn = f.min(2)
    k = max(15, (min(img.shape[:2]) // 25) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    base = cv2.morphologyEx(mn, cv2.MORPH_OPEN, ker)  # 去掉细条纹后的“真实”底色暗度
    base = cv2.GaussianBlur(base, (0, 0), k / 4)
    veil = np.clip(mn - base, 0, None)
    # 只处理半透明的白光：底色越暗越可信；非常亮、很不透明的（真实白色细节）不处理
    dark_bg = np.clip((0.62 - base) / 0.3, 0, 1)
    veil = np.minimum(veil, 0.35) * dark_bg
    # 大片均匀的蒙光：每个区域最暗处本应接近黑色，整片发灰说明被白光蒙住（局部黑点校正）
    big = max(31, (min(img.shape[:2]) // 7) | 1)
    local_black = cv2.erode(mn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big, big)))
    local_black = cv2.GaussianBlur(local_black, (0, 0), big / 2)
    broad = np.clip(local_black - np.percentile(mn, 1) - 0.03, 0, 0.22)
    veil = np.maximum(veil, broad) * strength
    for x1, y1, x2, y2 in faces:  # 人脸交给人脸修复
        p = (x2 - x1) * 0.2
        veil[max(0, int(y1 - p)):int(y2 + p), max(0, int(x1 - p)):int(x2 + p)] = 0
    veil = cv2.GaussianBlur(veil, (0, 0), 1.2)[..., None]
    # I = J*(1-v) + v  =>  J = (I - v) / (1 - v)
    out = (f - veil) / np.clip(1 - veil, 0.5, 1)
    return np.clip(out * 255, 0, 255).astype(np.uint8), float(veil.mean())


def _guided(I, p, r, eps):
    mean = lambda x: cv2.boxFilter(x, -1, (r, r))
    mI, mp = mean(I), mean(p)
    a = (mean(I * p) - mI * mp) / (mean(I * I) - mI * mI + eps)
    b = mp - a * mI
    return mean(a) * I + mean(b)


def dehaze(img, omega=0.35):
    """暗通道先验去雾，去掉褪色照片和翻拍时的发灰“雾层”。"""
    f = img.astype(np.float32) / 255
    dark = cv2.erode(f.min(2), np.ones((15, 15), np.uint8))
    idx = np.argsort(dark.ravel())[-max(1, dark.size // 1000):]
    A = np.full(3, np.clip(f.reshape(-1, 3)[idx].mean(), 0.6, 1.0), np.float32)  # 中性大气光，不引入偏色
    t = 1 - omega * cv2.erode((f / A).min(2), np.ones((15, 15), np.uint8))
    t = _guided(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, t, 41, 1e-3)
    t = np.clip(t, 0.6, 1)[..., None]
    return np.clip(((f - A) / t + A) * 255, 0, 255).astype(np.uint8)


def auto_color(img, clip=0.3):
    """逐通道色阶拉伸（类似 Photoshop 自动颜色），同时校正偏色和对比度。"""
    out = np.empty_like(img)
    for c in range(3):
        lo, hi = np.percentile(img[..., c], (clip, 100 - clip))
        hi = max(hi, lo + 40)
        out[..., c] = np.clip((img[..., c].astype(np.float32) - lo) * 255 / (hi - lo), 0, 255)
    # 与原图亮度通道混合一点，避免个别照片偏色矫枉过正
    return cv2.addWeighted(out, 0.7, img, 0.3, 0)


def tone(img, is_gray):
    """光影：导向滤波分出“大块明暗”和“细节”两层，细节单独加强、大块明暗只做柔和 S 曲线。
    （原先用 CLAHE，会在浅色背景上的深色线条周围压出暗晕。导向滤波保边，不产生暗晕。）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32) / 255
    r = max(8, max(img.shape[:2]) // 60)
    base = _guided(L, L, r, 0.004)
    detail = L - base
    base = base + 0.06 * np.sin(np.pi * (base - 0.5)) * (1 - np.abs(2 * base - 1)) * 2  # 柔和 S 曲线
    L = base + 1.35 * detail
    lab[..., 0] = np.clip(L * 255, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if is_gray:
        return out
    # 自然饱和度：低饱和像素提升更多，已经很艳的少提
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255
    hsv[..., 1] = np.clip((s + 0.30 * s * (1 - s) + 0.03 * s) * 255, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def match_brightness(img, ref):
    """用 gamma 把亮度均值拉回原片水平，保留原片的整体光影。"""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cur = L[..., 0].astype(np.float32) / 255
    target = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[..., 0].mean() / 255
    gamma = np.clip(np.log(max(target, 1e-3)) / np.log(max(cur.mean(), 1e-3)), 0.6, 1.4)
    L[..., 0] = np.clip(cur ** gamma * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(L, cv2.COLOR_LAB2BGR)


def fade_level(img):
    """0 = 对比度、饱和度都已经很好（如数码照片），1 = 明显褪色发灰。"""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    lo, hi = np.percentile(L, (2, 98))
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()
    contrast = np.clip((235 - (hi - lo)) / 90, 0, 1)
    dull = np.clip((110 - sat) / 70, 0, 1)
    return float(np.clip(0.6 * contrast + 0.4 * dull, 0, 1))


def is_monochrome(img, thresh=5.0):
    """黑白照片（含泛黄、泛褐的老照片）：颜色只沿一个色调变化，Lab 色度的离散程度很小。
    淡彩色照片饱和度也低，但会有好几种不同的颜色，离散程度明显更大（实测黑白 ≤3，彩色 ≥9）。"""
    ab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 1:].reshape(-1, 2).astype(np.float32)
    ab = ab[:: max(1, len(ab) // 50000)]
    return float(np.sqrt(np.linalg.eigvalsh(np.cov((ab - ab.mean(0)).T))[-1])) < thresh


def enhance(img, faces=(), extra_mask=None, veil=False, auto_glare=True, digital=False):
    is_gray = is_monochrome(img)
    if auto_glare:
        img, glare_pct = remove_glare(img, faces, extra_mask)
    else:  # 黑白照片里亮白的墙容易被误判成反光点，只修补手动/其他机制标出的区域
        img, glare_pct = inpaint(img, extra_mask) if extra_mask is not None else img, 0.0
    if veil:  # 只在标注了条纹反光的照片上用：它会把深色线条之间的窄条白底误当成反光压暗
        img, _ = remove_veil(img, faces)
    orig = img
    if digital:  # 数码照片：没有褪色偏色，不做校色去雾，只轻度加强光影和细节
        out = tone(img, is_gray)
        k = 0.4
    else:
        out = auto_color(img) if not is_gray else img
        out = dehaze(out)
        out = tone(out, is_gray)
        # 褪色越重校正越强；本来就清晰的照片只做轻度增强
        k = 0.35 + 0.65 * fade_level(orig)
    out = cv2.addWeighted(out, k, orig, 1 - k, 0)
    out = global_only(out, orig)  # 局部只加细节，不放大污渍、不产生暗晕
    out = match_brightness(out, orig)
    if not is_gray:
        out = neutralize_highlights(out, orig, faces)  # 天空、白墙保持原片的中性色
    if is_gray:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return out, is_gray, glare_pct
