"""翻拍照片透视校正：找到照片（相纸）的四个角，拉正并裁掉相册膜、桌面等背景。

角点优先取 corners.json 中手动标注的值（原图像素坐标，顺序 左上、右上、右下、左下），
没有标注时自动检测；文件里写 null 表示这张不做校正。
"""
import json
from pathlib import Path

import cv2
import numpy as np

CORNERS_FILE = Path(__file__).resolve().parent / "corners.json"
REPAIRS_FILE = Path(__file__).resolve().parent / "repairs.json"


def order_quad(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def _grad(small):
    """亮度和色度里最强的梯度（照片与相册页亮度相近、颜色不同时也能看到边）。"""
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    lab = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32), (3, 3), 0)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    for ch in (g, lab[..., 1] * 2, lab[..., 2] * 2):
        cx, cy = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        m = np.hypot(cx, cy) > np.hypot(gx, gy)
        gx[m], gy[m] = cx[m], cy[m]
    return gx, gy


def _side_support(gx, gy, a, b, thr, n=50):
    """线段 a→b 上有多少比例的点落在与它平行的明显边缘上。线段贴着画面边框时视为 0.5（照片延伸出画面）。"""
    h, w = gx.shape
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 10:
        return 0.0
    if (max(a[0], b[0]) < 3 or min(a[0], b[0]) > w - 4 or max(a[1], b[1]) < 3 or min(a[1], b[1]) > h - 4):
        return 0.5
    nrm = np.array([-d[1], d[0]]) / L
    hit = 0
    for t in np.linspace(0.1, 0.9, n):
        p = a + d * t
        for off in (-2, -1, 0, 1, 2):
            x, y = (p + off * nrm).round().astype(int)
            if 0 <= x < w and 0 <= y < h and abs(gx[y, x] * nrm[0] + gy[y, x] * nrm[1]) > thr:
                hit += 1
                break
    return hit / n


def _side_contrast(lab, a, b, band=8, n=24):
    """边两侧窄带的颜色差（Lab 距离中位数）。相纸边缘两侧一边是照片、一边是相册/桌面，差别明显；
    画面内部的线（横幅边、背景图案）两侧常是同一场景。贴着画面边框的边返回中性值。"""
    h, w = lab.shape[:2]
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 10:
        return 0.0
    if (max(a[0], b[0]) < 3 or min(a[0], b[0]) > w - 4 or max(a[1], b[1]) < 3 or min(a[1], b[1]) > h - 4):
        return 15.0
    nrm = np.array([-d[1], d[0]]) / L
    diffs = []
    for t in np.linspace(0.1, 0.9, n):
        p = a + d * t
        x1, y1 = (p + band * nrm).round().astype(int)
        x2, y2 = (p - band * nrm).round().astype(int)
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            diffs.append(np.linalg.norm(lab[y1, x1] - lab[y2, x2]))
    return float(np.median(diffs)) if diffs else 0.0


def _lines(small, gx, gy):
    """找长直线，按“横向/竖向”分组，返回 [(点, 方向), ...]；画面四条边框也作为候选。"""
    h, w = small.shape[:2]
    mag = np.hypot(gx, gy)
    edges = (mag > np.percentile(mag, 90)).astype(np.uint8) * 255
    raw = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40, minLineLength=min(h, w) * 0.2, maxLineGap=12)
    horiz, vert = [], []
    if raw is not None:
        for x1, y1, x2, y2 in raw[:, 0]:
            p, d = np.array([x1, y1], np.float32), np.array([x2 - x1, y2 - y1], np.float32)
            L = np.linalg.norm(d)
            d /= L
            (horiz if abs(d[0]) > abs(d[1]) else vert).append((p, d, L))

    def dedup(ls, key):
        # 同一条边会被检测出很多段：按位置合并，保留最长的
        ls.sort(key=lambda x: -x[2])
        out = []
        for p, d, L in ls:
            if all(abs(key(p, d) - key(q, e)) > 6 or abs(d @ e) < 0.995 for q, e, _ in out):
                out.append((p, d, L))
        return out[:10]

    at_x = lambda p, d, x: p[1] + (x - p[0]) * d[1] / d[0]
    at_y = lambda p, d, y: p[0] + (y - p[1]) * d[0] / d[1]
    horiz = dedup(horiz, lambda p, d: at_x(p, d, w / 2))
    vert = dedup(vert, lambda p, d: at_y(p, d, h / 2))
    border = lambda p, d: (np.array(p, np.float32), np.array(d, np.float32), 0)
    horiz += [border((0, 0), (1, 0)), border((0, h - 1), (1, 0))]
    vert += [border((0, 0), (0, 1)), border((w - 1, 0), (0, 1))]
    return horiz, vert


def _intersect(l1, l2):
    (p, d), (q, e) = l1[:2], l2[:2]
    m = np.array([d, -e]).T
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    t = np.linalg.solve(m, q - p)[0]
    return p + d * t


def detect_quad(img, min_score=0.6, return_score=False):
    """自动找照片（相纸）的四个角（文档扫描常用的直线法）：
    1. 找画面里的长直线，分成横线、竖线两组，画面四条边框也算候选（照片常常延伸出画面）；
    2. 上、下、左、右各取一条组合成四边形，要求是凸的、面积占画面 15% 以上；
    3. 打分 = 四条边落在真实边缘上的比例（贴着画面边框的边只算半分）× 边两侧颜色差 × 面积，
       画面内部的线（横幅、背景图案）两侧颜色相近，会被压低；相纸通常是画面里最大的矩形。
    最高分的四边形如果就是整个画面，或分数低于 min_score，返回 None（不校正）。
    实测不需要校正的照片最高分 ≤0.56，检测正确的 ≥0.61，所以 min_score 取 0.6：宁可不校正，也不误裁正常照片。"""
    h, w = img.shape[:2]
    scale = 700 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gx, gy = _grad(small)
    thr = np.percentile(np.hypot(gx, gy), 80)
    horiz, vert = _lines(small, gx, gy)
    mid_x = lambda l: l[0][1] + (sw / 2 - l[0][0]) * l[1][1] / l[1][0]
    mid_y = lambda l: l[0][0] + (sh / 2 - l[0][1]) * l[1][0] / l[1][1]
    tops = [l for l in horiz if mid_x(l) < sh * 0.6]
    bottoms = [l for l in horiz if mid_x(l) > sh * 0.4]
    lefts = [l for l in vert if mid_y(l) < sw * 0.6]
    rights = [l for l in vert if mid_y(l) > sw * 0.4]
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    cache, ccache = {}, {}
    best, best_score = None, min_score
    for T in tops:
        for B in bottoms:
            if mid_x(B) - mid_x(T) < sh * 0.25:
                continue
            for Lf in lefts:
                for Rt in rights:
                    if mid_y(Rt) - mid_y(Lf) < sw * 0.25:
                        continue
                    pts = [_intersect(T, Lf), _intersect(T, Rt), _intersect(B, Rt), _intersect(B, Lf)]
                    if any(p is None for p in pts):
                        continue
                    q = np.array(pts, np.float32)
                    if (q[:, 0] < -sw * 0.03).any() or (q[:, 0] > sw * 1.03).any() or \
                       (q[:, 1] < -sh * 0.03).any() or (q[:, 1] > sh * 1.03).any():
                        continue
                    area = cv2.contourArea(q) / (sh * sw)
                    if area < 0.15 or not cv2.isContourConvex(q.reshape(-1, 1, 2).round().astype(np.int32)):
                        continue
                    sups, cons = [], []
                    for a_, b_ in zip(q, np.roll(q, -1, axis=0)):
                        k = tuple(np.round(np.r_[a_, b_] / 2).astype(int))
                        if k not in cache:
                            cache[k] = _side_support(gx, gy, a_, b_, thr)
                            ccache[k] = _side_contrast(lab, a_, b_)
                        sups.append(cache[k])
                        cons.append(ccache[k])
                    if min(sups) < 0.2 or min(cons) < 6:
                        continue
                    contrast = min(1.0, float(np.mean(cons)) / 25)
                    score = float(np.mean(sups)) * (0.4 + 0.6 * contrast) * (0.4 + 0.6 * area)
                    if score > best_score:
                        best, best_score = q, score
    if best is None:
        return (None, 0.0) if return_score else None
    q = np.clip(best, 0, [sw - 1, sh - 1]).astype(np.float32)
    if cv2.contourArea(q) > 0.95 * sw * sh:  # 就是整个画面
        return (None, 0.0) if return_score else None
    q = order_quad(q / scale)
    return (q, best_score) if return_score else q


def load_manual():
    if CORNERS_FILE.exists():
        return json.loads(CORNERS_FILE.read_text())
    return {}


def get_quad(img, key):
    manual = load_manual()
    if key in manual:
        q = manual[key]
        return None if q is None else order_quad(q)
    return detect_quad(img)


def warp(img, quad, interp=cv2.INTER_CUBIC):
    tl, tr, br, bl = quad
    w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def load_repairs(key):
    if REPAIRS_FILE.exists():
        return json.loads(REPAIRS_FILE.read_text()).get(key, {})
    return {}


def snap_polyline(img, points, radius=12, step=3, response=None):
    """手描的折痕线、笔迹路径只需大致准确：沿线每 step 像素，在法线方向 ±radius 内找线状特征最强的位置，
    再对偏移做中值+均值平滑，避免被旁边的物体轮廓带跑。返回贴合后的折线。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 默认找线状明暗变化（折痕）；也可以传入别的响应图（比如笔迹的颜色浓度）
    resp = response if response is not None else np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F, ksize=3))
    pts = np.asarray(points, np.float32)
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        dense += [a + (b - a) * t for t in np.arange(n) / n]
    dense.append(pts[-1])
    dense = np.array(dense)
    tang = np.gradient(dense, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    h, w = gray.shape
    offs = np.arange(-radius, radius + 1)
    best = []
    for p, nrm in zip(dense, normal):
        q = np.clip(np.round(p + offs[:, None] * nrm), 0, [w - 1, h - 1]).astype(int)
        r = resp[q[:, 1], q[:, 0]] * np.exp(-(offs / radius) ** 2)  # 偏向原位置
        best.append(offs[int(np.argmax(r))])
    best = np.array(best, np.float32)
    k = 7
    pad = np.pad(best, k, mode="edge")
    med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(best))])
    smooth = np.convolve(np.pad(med, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    return dense + smooth[:, None] * normal


def repair_mask_cfg(img, cfg, part):
    """按一张照片的修复配置（repairs.json 里的一项，或网页传来的配置）生成原图大小的修补掩膜。
    part="traced"：手工标注的折痕折线、沿路径描的笔迹（可信，人脸内也修）；
    part="ink"：只给了框、按颜色自动找的笔迹（人脸内不修）；
    part="sheen"：膜反光区域多边形。"""
    from defects import ink_mask, ink_trace_mask

    mask = np.zeros(img.shape[:2], np.uint8)
    if part == "traced":
        for line in cfg.get("folds", []):
            pts = line["points"]
            if len(pts) < 2:
                continue
            if line.get("snap", True):  # 自动贴合到真实折痕
                pts = snap_polyline(img, pts)
            pts = np.array(pts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 255, int(line.get("width", 7)))
        for ink in cfg.get("ink", []):
            if len(ink.get("points", [])) >= 2:
                mask |= ink_trace_mask(img, ink["points"], ink.get("color", "red"))
    elif part == "sheen":
        for sh in cfg.get("sheen", []):
            if len(sh.get("polygon", [])) >= 3:
                cv2.fillPoly(mask, [np.array(sh["polygon"], np.int32).reshape(-1, 1, 2)], 255)
    elif part == "ink":
        for ink in cfg.get("ink", []):
            if "box" in ink:
                mask |= ink_mask(img, ink["box"], ink.get("color", "red"))
    return mask


def trim_box_cfg(shape, cfg):
    """校正后四边各裁多少（相册膜压边、相册边）。返回 (y0, y1, x0, x1) 切片范围。"""
    t = cfg.get("trim", {}) or {}
    h, w = shape[:2]
    return (int(t.get("top", 0)), h - int(t.get("bottom", 0)),
            int(t.get("left", 0)), w - int(t.get("right", 0)))


def quad_from_corners(img, corners):
    """corners: 四个角的列表 / None（不校正）/ "auto"（自动检测）。"""
    if corners == "auto":
        return detect_quad(img)
    return None if corners is None else order_quad(corners)


def rect_transform(quad, trim):
    """原图坐标 → 校正裁边后坐标 的变换（3x3 矩阵）。quad 为 None 时只做裁边平移。"""
    T = np.array([[1, 0, -trim[2]], [0, 1, -trim[0]], [0, 0, 1]], np.float64)
    if quad is None:
        return T
    tl, tr, br, bl = quad
    w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    M = cv2.getPerspectiveTransform(quad, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32))
    return T @ M


# ---- 以下为按编号读取 repairs.json / corners.json 的包装，命令行批量处理用 ----
def repair_mask(img, key, part):
    return repair_mask_cfg(img, load_repairs(key), part)


def auto_creases_enabled(key):
    return load_repairs(key).get("auto_scratch", True)


def trim_box(shape, key):
    return trim_box_cfg(shape, load_repairs(key))
