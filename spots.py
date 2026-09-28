"""自动找脸上的斑点（霉点、灰尘、墨点）：只在面部皮肤区域内找，眼睛、鼻孔、眉毛、嘴都不动。"""
import cv2
import numpy as np
import torch

_parser = None
SKIN = 1  # CelebAMask-HQ 标签：1 皮肤，2-5 眉眼，10 鼻子，11-13 嘴唇，17 头发


def _parse(face_bgr, device):
    global _parser
    if _parser is None:
        from facexlib.parsing import init_parsing_model
        _parser = init_parsing_model(model_name="parsenet", device=device, model_rootpath="gfpgan/weights")
    x = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)[..., ::-1].astype(np.float32) / 255
    x = torch.from_numpy(((x - 0.5) / 0.5).transpose(2, 0, 1).copy())[None].to(device)
    with torch.no_grad():
        return _parser(x)[0].argmax(1)[0].cpu().numpy().astype(np.uint8)


def _feature_zones(shape, lm, fw):
    """用 5 个关键点（左眼、右眼、鼻尖、左右嘴角）画出眼睛、眉毛、鼻孔、嘴的保护区。"""
    z = np.zeros(shape, np.uint8)
    (lx, ly), (rx, ry), (nx, ny), (ml, mly), (mr, mry) = np.asarray(lm).reshape(5, 2)
    e = lambda c, ax: cv2.ellipse(z, (int(c[0]), int(c[1])), (max(1, int(ax[0])), max(1, int(ax[1]))), 0, 0, 360, 1, -1)
    for ex, ey in ((lx, ly), (rx, ry)):
        e((ex, ey), (0.22 * fw, 0.13 * fw))  # 眼睛
        e((ex, ey - 0.17 * fw), (0.25 * fw, 0.11 * fw))  # 眉毛
    e((nx, ny + 0.03 * fw), (0.18 * fw, 0.11 * fw))  # 鼻头和鼻孔
    bx, by = ((lx + rx) / 2 + nx) / 2, ((ly + ry) / 2 + ny) / 2
    e((bx, by), (0.09 * fw, abs(ny - (ly + ry) / 2) / 2 + 0.04 * fw))  # 鼻梁两侧阴影
    e(((ml + mr) / 2, (mly + mry) / 2 + 0.03 * fw), (abs(mr - ml) / 2 + 0.1 * fw, 0.15 * fw))  # 嘴和下唇
    return z


def face_spot_mask(img, faces, device="cpu", landmarks=None):
    """返回整图大小的掩膜，标出人脸皮肤上的小黑点/小白点。landmarks 与 faces 一一对应。"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for fi, (x1, y1, x2, y2) in enumerate(faces):
        fw = x2 - x1
        if fw < 40:  # 太小的脸（集体照）斑点和五官分不开，跳过
            continue
        p = fw * 0.25
        cx1, cy1 = max(0, int(x1 - p)), max(0, int(y1 - p))
        cx2, cy2 = min(w, int(x2 + p)), min(h, int(y2 + p))
        crop = img[cy1:cy2, cx1:cx2]
        seg = cv2.resize(_parse(crop, device), (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_NEAREST)
        skin = cv2.erode((seg == SKIN).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(2, int(fw / 30)))  # 收掉脸部轮廓边缘
        if landmarks is not None:
            lm = np.asarray(landmarks[fi]).reshape(5, 2) - [cx1, cy1]
            skin[_feature_zones(skin.shape, lm, fw) > 0] = 0

        g = gray[cy1:cy2, cx1:cx2]
        k = max(5, int(fw * 0.07)) | 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dark = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker)
        bright = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ker)
        # 阈值随皮肤纹理自适应：比皮肤区域的正常起伏高很多才算斑点
        vals = dark[skin > 0]
        if vals.size < 50:
            continue
        t_dark = max(22, np.percentile(vals, 99.0) * 1.4)
        t_bright = max(30, np.percentile(bright[skin > 0], 99.0) * 1.6)
        cand = (((dark > t_dark) | (bright > t_bright)) & (skin > 0)).astype(np.uint8)

        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand)
        amax = (fw * 0.06) ** 2
        keep = np.zeros(n, bool)
        for j in range(1, n):
            a = stats[j, cv2.CC_STAT_AREA]
            bw, bh = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
            # 斑点是圆的一小团；细长的是皱纹、唇线、阴影，不动
            keep[j] = 2 <= a <= amax and max(bw, bh) <= 2.2 * min(bw, bh) + 1
        spots = np.where(keep[lab], 255, 0).astype(np.uint8)
        # 斑点边缘有一圈晕，外扩多一些连晕一起盖住，否则后续增强会把残影重新拉出来
        spots = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=max(2, int(fw / 45)))
        mask[cy1:cy2, cx1:cx2] |= spots
    return mask


def stain_mask(img, faces=(), max_frac=0.012, contrast=28):
    """整张照片上的霉点、墨点、污渍（不限于人脸）：比周围明显暗的小块。
    用黑帽运算找局部暗斑，只保留面积小、近似成团的；人脸区域交给 face_spot_mask，这里不动。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(9, int(min(h, w) * max_frac * 2)) | 1
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cand = (dark > contrast).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand)
    amax = (min(h, w) * max_frac) ** 2 * 2
    keep = np.zeros(n, bool)
    for j in range(1, n):
        a = stats[j, cv2.CC_STAT_AREA]
        bw, bh = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        keep[j] = 3 <= a <= amax and max(bw, bh) <= 7 * min(bw, bh) + 2  # 短的斜划痕也算
    m = np.where(keep[lab], 255, 0).astype(np.uint8)
    for x1, y1, x2, y2 in faces:
        p = (x2 - x1) * 0.2
        m[max(0, int(y1 - p)):int(y2 + p), max(0, int(x1 - p)):int(x2 + p)] = 0
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)


def thin_scratch_mask(img, box, polarity="bright", max_width=3, min_len=10, thresh=7):
    """框内又细又长的划痕（亮线或暗线）：顶帽/黑帽只留下比 max_width 细的结构，
    再按细长程度筛选，圆形光斑、花点不会被选中。返回整图大小的掩膜。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max_width + 1,) * 2)
    op = cv2.MORPH_TOPHAT if polarity == "bright" else cv2.MORPH_BLACKHAT
    r = cv2.morphologyEx(g, op, k)
    cand = (r > thresh).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 接上断开的淡划痕
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    m = np.zeros(g.shape, np.uint8)
    for j in range(1, n):
        bw, bh, a = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT], stats[j, cv2.CC_STAT_AREA]
        if max(bw, bh) >= min_len and a <= 0.35 * bw * bh + max(bw, bh):  # 细长：面积远小于外接框
            m[lab == j] = 255
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = m
    return out


def remove_vertical_streaks(img, box, max_width=9, thresh=10):
    """相纸表面磨损造成的竖向白色擦痕：用水平方向的线形结构元素做开运算，
    横向比 max_width 窄的亮竖条会被去掉、换成左右两侧的颜色；真实纹理（衣服、椅子）横向连续，基本不受影响。
    只替换明显变亮（超过 thresh）的像素，边缘羽化。"""
    x0, y0, x1, y1 = [int(v) for v in box]
    out = img.copy()
    roi = img[y0:y1, x0:x1]
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (max_width, 1))
    opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, ker)
    opened = cv2.GaussianBlur(opened, (3, 3), 0)
    diff = cv2.cvtColor(cv2.absdiff(roi, opened), cv2.COLOR_BGR2GRAY).astype(np.float32)
    w = np.clip((diff - thresh) / thresh, 0, 1)
    w = cv2.GaussianBlur(cv2.dilate(w, np.ones((1, 3), np.uint8)), (0, 0), 1)[..., None]
    out[y0:y1, x0:x1] = (roi * (1 - w) + opened * w).astype(np.uint8)
    return out
