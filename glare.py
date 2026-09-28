"""翻拍照片时的大块反光（灯光、窗户照在相纸上）：

反光可以看作一层叠在原画面上的白光：I = J·(1-a) + G·a
  I 拍到的颜色，J 本来的颜色，G 反光的颜色（接近白色），a 反光强度（0~1）
1. 在 repairs.json 标注的 box 内，以接近过曝、几乎无色的亮点为种子，向外扩出反光范围；
2. 用反光范围外紧邻的正常像素，估计每个位置“本来应有”的最暗通道值，算出反光强度 a；
3. 边缘较弱的光晕按公式扣除白光还原 J；反光较强（a ≥ lost_at，底下内容基本看不清、而且常带蓝紫色偏色）
   的部分整片交给 LaMa 按周围内容补画。
"""
import cv2
import numpy as np


def _nconv(x, weight, sigma):
    """归一化卷积：只用 weight>0 的像素做平滑插值。"""
    num = cv2.GaussianBlur(x * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / (den + 1e-4), den


def remove_glare_blob(img, box, lost_at=0.3, reach_frac=0.07):
    """返回 (修复后的图, 需要 LaMa 补画的中心掩膜, 反光强度图)。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    f = img.astype(np.float32) / 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mn = f.min(2)

    # 1. 种子：接近过曝、低饱和
    seed = np.zeros((h, w), np.uint8)
    sub = (hsv[y0:y1, x0:x1, 2] > 215) & (hsv[y0:y1, x0:x1, 1] < 70)
    seed[y0:y1, x0:x1] = sub.astype(np.uint8)
    if seed.sum() < 20:
        return img, np.zeros((h, w), np.uint8), np.zeros((h, w), np.float32)
    G = f[seed > 0].mean(0)  # 反光颜色
    reach = max(15, int(min(h, w) * reach_frac))  # 反光外围淡雾能延伸多远
    zone = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2))
    zone[:y0, :] = 0; zone[y1:, :] = 0; zone[:, :x0] = 0; zone[:, x1:] = 0

    # 2. 用反光范围外的像素估计“本来的”最暗通道值
    outside = (zone == 0).astype(np.float32)
    ref, den = _nconv(mn, outside, reach * 0.8)
    ref = np.minimum(ref, mn)
    Gmin = G.min()
    a = np.clip((mn - ref) / np.maximum(Gmin - ref, 0.05), 0, 1) * zone
    a = cv2.GaussianBlur(a, (0, 0), 3)  # 反光本身是平滑的
    a[den < 1e-3] = 0

    # 3. 扣掉白光；中心过曝部分交给 LaMa
    ac = np.minimum(a, lost_at)[..., None]
    J = (f - ac * G) / (1 - ac)
    out = np.clip(J * 255, 0, 255).astype(np.uint8)
    lost = ((a >= lost_at) | ((seed > 0) & (a > 0.5))).astype(np.uint8) * 255
    lost = cv2.dilate(lost, np.ones((5, 5), np.uint8), iterations=2)
    return out, lost, a


def recover_glare(img, box, max_a=0.85, lost_at=0.7, reach_frac=0.07):
    """还原而不是重画：亮度按 I = J·(1-a) + G·a 反推出 J，保留反光底下的建筑细节；
    颜色（反光带的蓝紫色和原色分不开）改用反光范围外相邻正常画面的颜色；
    只有反光强度超过 lost_at（底下已经没有信息）的中心部分交给补画，补画时参照的是已还原的周边。返回 (还原后的图, 需补画的掩膜)。"""
    h, w = img.shape[:2]
    _, _, a = remove_glare_blob(img, box, lost_at=1.01, reach_frac=reach_frac)
    if not a.any():
        return img, np.zeros((h, w), np.uint8)
    reach = max(15, int(min(h, w) * reach_frac))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 255
    seed = (a > 0.5)
    LG = float(np.percentile(L[seed], 90)) if seed.any() else 1.0
    ac = np.minimum(a, max_a)
    Lr = (L - ac * LG) / (1 - ac)
    # 反推会放大噪点：a 越大，越多地做保边平滑
    Ls = cv2.bilateralFilter(np.clip(Lr, 0, 1).astype(np.float32), 7, 0.08, 5)
    wden = np.clip((a - 0.3) / 0.4, 0, 1)
    Lr = Lr * (1 - wden) + Ls * wden
    # 颜色参照反光范围外的正常画面
    outside = (a < 0.02).astype(np.float32)
    cw = np.clip(a * 4, 0, 1)
    for c in (1, 2):
        nb, _ = _nconv(lab[..., c], outside, reach * 0.6)
        lab[..., c] = lab[..., c] * (1 - cw) + nb * cw
    lab[..., 0] = np.clip(Lr, 0, 1) * 255
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    lost = (a >= lost_at).astype(np.uint8) * 255
    lost = cv2.dilate(lost, np.ones((5, 5), np.uint8), iterations=1)
    return out, lost


def lift_haze(img, polygon, ref_box, max_a=0.7):  # 坐标系与传入的 img 一致
    """局部去雾：反光外围发灰发白的一层淡雾（尤其挨着照片边缘、没有外部参照的地方）。
    ref_box 框一块没受影响的同类内容（例如同一片屋顶），它的暗部亮度就是“本来应有”的黑位；
    区域内每处的雾浓度 a 由局部暗部比参照亮多少来估计，按 I = J·(1-a) + 白·a 扣除。
    颜色同样向参照区靠拢，纹理保留。"""
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255
    mn = f.min(2)
    x0, y0, x1, y1 = [int(v) for v in ref_box]
    ref_black = float(np.percentile(mn[y0:y1, x0:x1], 10))
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [np.array(polygon, np.int32).reshape(-1, 1, 2)], 255)
    k = max(9, int(min(h, w) / 60)) | 1
    dark = cv2.erode(mn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dark = cv2.GaussianBlur(dark, (0, 0), k)
    a = np.clip((dark - ref_black) / (1 - ref_black + 1e-6), 0, max_a)
    soft = cv2.GaussianBlur(region.astype(np.float32) / 255, (0, 0), k / 2)
    a = (a * soft)[..., None]
    J = (f - a) / (1 - a)
    # 颜色：去雾后色度向参照区靠拢一半
    lab = cv2.cvtColor(np.clip(J * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    for c in (1, 2):
        m = np.median(ref_lab[:, c])
        lab[..., c] = lab[..., c] + (m - lab[..., c]) * 0.5 * soft
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
