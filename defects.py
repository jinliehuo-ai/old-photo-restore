"""按缺陷类型分别处理的机制（本文件）：

- ink_mask               笔迹：repairs.json 标出区域，区域内按颜色找细笔画，交给 LaMa 补画
- sr_guard               超分/人脸修复凭空画出的纹理：全自动，成片偏离输入的地方退回普通放大
- global_only            污渍/阴影被局部增强放大：全自动，颜色只做全局校正，局部只加细节
- anchor_midtones        处理引入的暗晕、光影不均：全自动，中频明暗跟随原片
- neutralize_highlights  天空、白墙等亮区偏色：全自动，以原片为参照，原本中性的亮区保持中性

- mono_from_channel      黑白原片上的彩色倒影：repairs.json 的 mono_channel 指定用哪个颜色通道重建
- unmix_reflection       屏幕上黑白照片的彩色倒影：线性解混分离照片与倒影（repairs.json 的 unmix），最彻底
- remove_reflection_edges  倒影边界留下的明暗台阶：色度边界处抹掉梯度后泊松重建（repairs.json 的 reflect_edges）
- reflection_color_mask / recover_reflection  黑白原片上仍有残影的彩色倒影块：repairs.json 的 reflect_colors 列出颜色，扣除倒影亮度还原底下画面
- sheen_correct          塑料膜折叠处整片发白：repairs.json 的 sheen 标出多边形，按周围做过渡校正

其余机制：塑料膜折痕（repairs.json 的 folds，rectify.repair_mask 画线后 LaMa 补画）、
相册膜压边（repairs.json 的 trim，rectify.trim_box 裁掉）、脸上斑点（spots.py）、
细碎划痕（creases.py）、点状和朦胧反光（enhance.py）。
"""
import cv2
import numpy as np


# ---------------------------------------------------------------- 笔迹（按区域）
INK_COLORS = {
    # OpenCV 色相 0-180：红笔色相约 0-10，饱和度高（>100），肤色饱和度低、橙色背景色相 11-25
    "red": lambda h, s, v: ((h >= 170) | (h <= 10)) & (s > 100) & (v > 80),
    "blue": lambda h, s, v: (h >= 100) & (h <= 130) & (s > 90) & (v > 60),
}


def ink_mask(img, box, color="red"):
    """在 box 内找指定颜色的细笔画。宽的同色区域（衣服、横幅）用开运算排除，只留细线。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    c = INK_COLORS[color](hue, sat, val).astype(np.uint8) * 255
    k = max(7, int(min(h, w) / 80)) | 1
    thin = cv2.subtract(c, cv2.morphologyEx(c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))))
    # 先把断开的笔画段连起来，再按形状筛：笔画是细长的线，嘴唇、红扣子是短小的一块
    thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thin)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        bw, bh, a = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT], stats[j, cv2.CC_STAT_AREA]
        long_side, short_side = max(bw, bh), max(1, min(bw, bh))
        keep[j] = long_side >= 15 and (long_side / short_side >= 3 or a <= 0.3 * bw * bh)
    thin = np.where(keep[lab], 255, 0).astype(np.uint8)
    # 笔画边缘有一圈淡色晕，外扩盖住
    thin = cv2.dilate(thin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = thin
    return m


def ink_trace_mask(img, points, color="red", band=14, core=7):
    """沿手描路径去笔迹：路径先自动吸附到笔迹颜色最浓的位置，再在两侧 band 像素内
    按宽松颜色条件找笔画，另加一条 core 宽的路径保底。"""
    from rectify import snap_polyline

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    if color == "red":
        loose = ((hue >= 160) | (hue <= 12)) & (sat > 50)
    else:
        loose = (hue >= 95) & (hue <= 135) & (sat > 50)
    strength = cv2.GaussianBlur(loose.astype(np.float32) * sat.astype(np.float32), (0, 0), 1.5)
    pts = snap_polyline(img, points, radius=12, response=strength)
    pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    zone = np.zeros((h, w), np.uint8)
    cv2.polylines(zone, [pts], False, 255, 2 * band + 1)
    m = (loose & (zone > 0)).astype(np.uint8) * 255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    cv2.polylines(m, [pts], False, 255, core)
    # 笔画外面还有一圈很淡的色晕，不盖住的话 AI 补画会顺着它画出一道“幽灵线”
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=4)


# ---------------------------------------------------------------- 膜反光区域（按区域）
def sheen_correct(img, region, sigma=None):
    """塑料膜折叠处等整片发白发灰的区域：用区域外紧邻的像素估计“本来的”亮度和颜色，
    对区域内做平滑的偏移校正。用归一化卷积，区域内每个位置都参照离它最近的外部像素，
    所以牛仔裤参照牛仔裤、墙参照墙，内容本身不变。"""
    if not region.any():
        return img
    h, w = img.shape[:2]
    sigma = sigma or max(8.0, min(h, w) / 30)
    inside = (region > 0).astype(np.float32)
    ring = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(sigma * 2) | 1,) * 2))
    ring = ((ring > 0) & (region == 0)).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    wo, wi = g(ring), g(inside)
    out = lab.copy()
    for c in range(3):
        est_out = g(lab[..., c] * ring) / (wo + 1e-4)
        est_in = g(lab[..., c] * inside) / (wi + 1e-4)
        out[..., c] = lab[..., c] + (est_out - est_in)
    # 外部参照太远（wo 很小）的地方不校正；区域边缘羽化
    conf = np.clip(wo / (wo.max() * 0.15 + 1e-6), 0, 1)
    alpha = (cv2.GaussianBlur(inside, (0, 0), 2) * conf)[..., None]
    res = lab * (1 - alpha) + out * alpha
    return cv2.cvtColor(np.clip(res, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- 彩色倒影（黑白原片）
def mono_from_channel(img, channel="blue", comp=0.2):
    """原片是黑白照片、上面叠着彩色倒影（比如屏幕上映出的红旗）时：
    倒影只抬高某些颜色通道，挑受影响最小的通道重建黑白照片，倒影就基本消失。
    黑白原片三通道本应相等，其他通道高出所选通道的部分就是倒影，按 comp 比例再扣掉一点残影。"""
    ch = {"blue": 0, "green": 1, "red": 2}[channel]
    f = img.astype(np.float32)
    x = f[..., ch].copy()
    for o in range(3):
        if o != ch:
            x -= comp * np.maximum(f[..., o] - f[..., ch], 0)
    lo, hi = np.percentile(x, (0.5, 99.5))
    x = np.clip((x - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


def reflection_color_mask(img, color="yellow"):
    """黑白原片上某种颜色的倒影（屏幕映出的五角星等）按颜色圈出，交给 LaMa 整块补画。
    只用于黑白原片：画面本身没有颜色，有颜色的地方一定是倒影。"""
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    v = f.max(2)
    if color == "yellow":
        m = ((g - b) > 30) & ((r - b) > 30) & (v > 100)
    elif color == "red":
        m = ((r - b) > 45) & ((r - g) > 45) & (v > 90)
    else:
        raise ValueError(color)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 细的尖角也要保留
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 40
    m = np.where(keep[lab], 255, 0).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=3)


def recover_reflection(mono, mask, strength=1.0, repaint=0.0):
    """还原被倒影盖住的黑白画面（而不是重画）：倒影在一块区域里叠加的亮度大致均匀，
    对每个倒影块，用边界内外两侧窄带的亮度中位数之差估计这个增量并扣掉；
    倒影里带来的屏幕网纹用保边平滑去掉。建筑、窗户等真实内容保留。
    strength 调扣除力度；repaint（0~1）与补画结果混合的比例，用来在“真实”和“干净”之间取平衡。"""
    g = cv2.cvtColor(mono, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = np.ones((7, 7), np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(mask)
    offset = np.zeros_like(g)
    for j in range(1, n):
        comp = (lab == j).astype(np.uint8)
        inner = comp & ~cv2.erode(comp, k)
        outer = cv2.dilate(comp, k) & ~comp
        if inner.sum() < 10 or outer.sum() < 10:
            continue
        # 用低分位数比较：深色结构（窗户）受倒影影响最明显、也最稳定
        d = np.percentile(g[inner > 0], 30) - np.percentile(g[outer > 0], 30)
        offset[comp > 0] = max(d, 0) * strength
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 2)
    g = g - cv2.GaussianBlur(offset, (0, 0), 2) * np.clip(soft * 1.5, 0, 1)
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    sm = cv2.medianBlur(cv2.bilateralFilter(g8, 9, 30, 5), 3)
    out = g8 * (1 - soft) + sm * soft
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if repaint > 0:  # 与补画结果按比例混合：越大越干净，但越可能出现不真实的结构
        from enhance import inpaint
        fill = inpaint(mono, mask)
        wgt = (soft * repaint)[..., None]
        out = (out * (1 - wgt) + fill * wgt).astype(np.uint8)
    return out


def remove_reflection_edges(mono, color_src, top=0.07, keep_low=True):
    """黑白原片本身没有颜色边界，彩色翻拍里颜色突变的地方就是倒影的边界（星形轮廓、红旗边缘）。
    在黑白图上把这些位置的明暗梯度抹掉，再用泊松方程（DCT 求解）重新积分回图像，
    倒影留下的“台阶”就消失，真实的明暗变化保留。纯数值方法，不用神经网络。"""
    from scipy.fft import dctn, idctn

    g = cv2.cvtColor(mono, cv2.COLOR_BGR2GRAY).astype(np.float64)
    lab = cv2.cvtColor(cv2.GaussianBlur(color_src, (0, 0), 2.5), cv2.COLOR_BGR2LAB).astype(np.float32)  # 先滤掉屏幕网纹的颜色抖动
    ce = np.zeros(g.shape, np.float32)
    for c in (1, 2):  # 色度梯度
        ce += np.abs(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3))
    ce = cv2.dilate(ce, np.ones((3, 3), np.uint8))
    thresh = np.percentile(ce, 100 * (1 - top))  # 只处理色度变化最强的 top 比例（倒影边界）
    wgt = 1 - np.clip((ce - thresh) / thresh, 0, 1)  # 色度边界处梯度权重降为 0
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, :-1] = np.diff(g, axis=1); gy[:-1, :] = np.diff(g, axis=0)
    gx *= wgt; gy *= wgt
    div = np.zeros_like(g)
    div[:, 0] = gx[:, 0]; div[:, 1:] = gx[:, 1:] - gx[:, :-1]
    div[0, :] += gy[0, :]; div[1:, :] += gy[1:, :] - gy[:-1, :]
    h, w = g.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    denom = (2 * np.cos(np.pi * xx / w) - 2) + (2 * np.cos(np.pi * yy / h) - 2)
    denom[0, 0] = 1
    d = dctn(div, type=2, norm="ortho") / denom
    d[0, 0] = 0
    out = idctn(d, type=2, norm="ortho")
    out += g.mean() - out.mean()
    if keep_low:  # 超大尺度的整体明暗仍以原图为准，避免泊松积分带来整体偏移
        sig = max(h, w) / 6
        out = out - cv2.GaussianBlur(out, (0, 0), sig) + cv2.GaussianBlur(g, (0, 0), sig)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _poisson_reconstruct(gray, grad_w):
    """把梯度按 grad_w 加权后用 DCT 解泊松方程重建图像；超大尺度明暗仍以原图为准。"""
    from scipy.fft import dctn, idctn

    g = gray.astype(np.float64)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, :-1] = np.diff(g, axis=1); gy[:-1, :] = np.diff(g, axis=0)
    gx *= grad_w; gy *= grad_w
    div = np.zeros_like(g)
    div[:, 0] = gx[:, 0]; div[:, 1:] = gx[:, 1:] - gx[:, :-1]
    div[0, :] += gy[0, :]; div[1:, :] += gy[1:, :] - gy[:-1, :]
    h, w = g.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    den = (2 * np.cos(np.pi * xx / w) - 2) + (2 * np.cos(np.pi * yy / h) - 2)
    den[0, 0] = 1
    d = dctn(div, type=2, norm="ortho") / den
    d[0, 0] = 0
    o = idctn(d, type=2, norm="ortho")
    o += g.mean() - o.mean()
    sig = max(h, w) / 6
    return o - cv2.GaussianBlur(o, (0, 0), sig) + cv2.GaussianBlur(g, (0, 0), sig)


def unmix_reflection(src, star_bottom=None, edge_bottom=None, sample_bottom=None):
    """黑白照片显示在屏幕上被翻拍、屏幕映出彩色倒影（红旗、五角星）时，用线性解混分离：
    每个像素 = 照片亮度×屏幕白 + 红色倒影强度×红色 + 黄色倒影强度×黄色，三个通道解三个未知数。
    1. 三种颜色方向从画面里估计：屏幕白=无倒影的亮墙，红色=红旗下的亮墙减屏幕白，黄色=五角星减红旗下亮墙；
    2. 解出照片亮度；
    3. 五角星中心接近白色、与屏幕白同方向解不开，用倒影强度图圈出（只圈亮墙上的，窗户等深色内容不动）后补画；
    4. 红旗倒影边界处的残余明暗台阶，用泊松重建抹掉。
    star_bottom / edge_bottom / sample_bottom：星形补画、边界处理、颜色取样的最下一行（倒影只在上方时用来保护人物）。"""
    from enhance import inpaint

    H, W = src.shape[:2]
    star_bottom = star_bottom or H
    edge_bottom = edge_bottom or H
    sample_bottom = sample_bottom or H
    f = src.astype(np.float32)
    lab = cv2.cvtColor(cv2.GaussianBlur(src, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    top = np.zeros((H, W), bool); top[:sample_bottom] = True
    star = reflection_color_mask(src, "yellow")
    st = star > 0
    screen = (B < -12) & (L > 150) & top
    red = (A > 18) & (B > -5) & (B < 15) & (L > 150) & top & ~st
    yel = st & (L > 150)
    w0 = np.median(f[screen], 0)
    v1 = np.median(f[red], 0) - w0
    v2 = np.median(f[yel], 0) - np.median(f[red], 0)
    M = np.stack([w0 / np.linalg.norm(w0), v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)], 1)
    X = np.linalg.solve(M, f.reshape(-1, 3).T).T.reshape(H, W, 3)
    a1, a2 = np.maximum(X[..., 1], 0), np.maximum(X[..., 2], 0)  # 倒影强度不能为负
    P = (f - a1[..., None] * M[:, 1] - a2[..., None] * M[:, 2]) @ M[:, 0]
    lo, hi = np.percentile(P, (0.5, 99.5))
    P = np.clip((P - lo) * 255 / (hi - lo), 0, 255)

    # 五角星：倒影强度图圈出（含细尖角），只取亮墙上的部分
    a2s = cv2.GaussianBlur(a2, (0, 0), 1.5)
    s2 = (a2s > np.percentile(a2s[:sample_bottom], 88)).astype(np.uint8) * 255
    s2 = cv2.morphologyEx(s2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(s2)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 30
    s2 = cv2.dilate(np.where(keep[lab_], 255, 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
    bright = cv2.GaussianBlur(P, (0, 0), 2) > np.percentile(P[:star_bottom], 45)
    arm = ((s2 > 0) & bright).astype(np.uint8) * 255
    arm = cv2.dilate(arm, np.ones((3, 3), np.uint8)) | star
    arm[star_bottom:] = 0
    P3 = cv2.cvtColor(P.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    P3 = inpaint(P3, arm)

    # 红旗倒影边界：抹掉该处梯度后泊松重建
    a1s = cv2.GaussianBlur(a1, (0, 0), 2)
    ge = np.hypot(cv2.Sobel(a1s, cv2.CV_32F, 1, 0), cv2.Sobel(a1s, cv2.CV_32F, 0, 1))
    th = np.percentile(ge[:sample_bottom], 92)
    wgt = 1 - np.clip((ge - th) / th, 0, 1)
    wgt[edge_bottom:] = 1
    out = np.clip(_poisson_reconstruct(P3[..., 0], wgt), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------- 4. 暗晕 / 光影不均
def anchor_midtones(out, ref, strength=1.0):
    """增强后保留细节和整体色调，但中等尺度（几十像素）的明暗起伏跟随原片，消掉处理引入的暗晕、脏斑。"""
    D = max(out.shape[:2])
    s1, s2 = max(2.0, D * 0.006), max(8.0, D * 0.1)
    lo = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    lr = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    band = lambda x: cv2.GaussianBlur(x, (0, 0), s1) - cv2.GaussianBlur(x, (0, 0), s2)
    bo, br = band(lo[..., 0]), band(lr)
    gain = np.clip(bo.std() / (br.std() + 1e-6), 1.0, 1.15)  # 保留适度的对比度提升
    lo[..., 0] = np.clip(lo[..., 0] + strength * (gain * br - bo), 0, 255)
    return cv2.cvtColor(lo.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- 防止污渍放大
def global_only(out, ref, detail_sigma=None):
    """颜色校正只允许“整张图统一”的变化：用最小二乘拟合一个 原片→结果 的全局颜色变换（3x3 矩阵+偏移），
    作用在原片上得到“只做了全局校色”的版本；结果里超出全局变换的成片局部明暗（污渍被放大、暗晕）去掉，
    只保留细节层。这样原片上若有若无的污渍、阴影不会被局部增强放大成明显色块。"""
    D = max(out.shape[:2])
    sigma = detail_sigma or max(3.0, D * 0.008)
    o, r = out.astype(np.float32), ref.astype(np.float32)
    step = max(1, int(np.sqrt(o.shape[0] * o.shape[1] / 40000)))
    X = r[::step, ::step].reshape(-1, 3)
    Y = o[::step, ::step].reshape(-1, 3)
    A = np.hstack([X, np.ones((len(X), 1), np.float32)])
    M, *_ = np.linalg.lstsq(A, Y, rcond=None)
    G = (np.concatenate([r, np.ones(r.shape[:2] + (1,), np.float32)], 2).reshape(-1, 4) @ M).reshape(o.shape)
    resid = o - G
    low = cv2.GaussianBlur(resid, (0, 0), sigma)  # 成片的局部变化
    return np.clip(o - low, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 超分防护
def sr_guard(up, src, tol=12.0):
    """超分辨率/人脸修复后，成片的明暗颜色不应偏离输入。偏离超过 tol 的区域（模型凭空画出的纹理）
    退回普通放大，只保留模型生成的细节里与输入一致的部分。"""
    H, W = up.shape[:2]
    base = cv2.resize(src, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    u = up.astype(np.float32)
    sig = max(4.0, max(H, W) / 150)
    dev = np.abs(cv2.GaussianBlur(u, (0, 0), sig) - cv2.GaussianBlur(base, (0, 0), sig)).max(2)
    bad = np.clip((dev - tol) / tol, 0, 1)
    bad = cv2.GaussianBlur(cv2.dilate(bad, np.ones((5, 5), np.uint8)), (0, 0), sig)[..., None]
    return np.clip(u * (1 - bad) + base * bad, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 5. 亮区偏色
def neutralize_highlights(img, ref, faces=(), keep=1.2):
    """原片里本来接近中性色的亮区（天空、白墙、白衣服），修复后不能被色彩校正染上颜色：
    色度拉回到原片色度的 keep 倍左右。肤色、人脸不动。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lr = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lr = cv2.GaussianBlur(lr[..., 0], (0, 0), 3) / 255
    ar, br = cv2.GaussianBlur(lr[..., 1] - 128, (0, 0), 3), cv2.GaussianBlur(lr[..., 2] - 128, (0, 0), 3)
    cr = np.sqrt(ar * ar + br * br)
    wgt = np.clip((Lr - 0.55) / 0.15, 0, 1) * np.clip((14 - cr) / 8, 0, 1)
    for x1, y1, x2, y2 in faces:
        p = (x2 - x1) * 0.3
        wgt[max(0, int(y1 - p)):int(y2 + p), max(0, int(x1 - p)):int(x2 + p)] = 0
    wgt = cv2.GaussianBlur(wgt, (0, 0), 3)
    lab[..., 1] = 128 + (lab[..., 1] - 128) * (1 - wgt) + ar * keep * wgt
    lab[..., 2] = 128 + (lab[..., 2] - 128) * (1 - wgt) + br * keep * wgt
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
