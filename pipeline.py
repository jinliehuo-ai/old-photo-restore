"""修复流程核心：模型只加载一次，对单张照片按配置执行完整修复。命令行（restore.py）和网页版（webapp/）共用。

配置 cfg 的格式与 repairs.json 中的一项相同（字段含义见 repairs.json 的“说明”）；
corners 为照片四个角（原图坐标）/ None（不做透视校正）/ "auto"（自动检测）。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / ".models"

# facexlib 从当前目录下的 gfpgan/weights 读取检测/解析模型
_weights = MODELS / "gfpgan" / "weights"
_weights.mkdir(parents=True, exist_ok=True)
for _name in ("detection_Resnet50_Final.pth", "parsing_parsenet.pth"):
    _link = _weights / _name
    if not _link.exists():
        _link.symlink_to(MODELS / _name)
os.chdir(MODELS)
sys.path.insert(0, str(ROOT))

# basicsr 1.4.2 引用了新版 torchvision 已删除的模块，这里补一个别名
import torchvision.transforms.functional as _tvf  # noqa: E402
sys.modules.setdefault("torchvision.transforms.functional_tensor", _tvf)

from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402
from gfpgan import GFPGANer  # noqa: E402
from realesrgan import RealESRGANer  # noqa: E402

from creases import crease_mask  # noqa: E402
from defects import (mono_from_channel, recover_reflection, reflection_color_mask,  # noqa: E402
                     remove_reflection_edges, sheen_correct, sr_guard, unmix_reflection)
from enhance import enhance, inpaint  # noqa: E402
from glare import lift_haze, recover_glare  # noqa: E402
from rectify import quad_from_corners, repair_mask_cfg, trim_box_cfg, warp  # noqa: E402
from spots import face_spot_mask, remove_vertical_streaks, stain_mask, thin_scratch_mask  # noqa: E402

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Restorer:
    def __init__(self):
        self.bg = RealESRGANer(
            scale=2, model_path=str(MODELS / "RealESRGAN_x2plus.pth"),
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            tile=0, tile_pad=64, pre_pad=0, half=False, device=device,
        )
        self.gfpgan = GFPGANer(
            model_path=str(MODELS / "GFPGANv1.4.pth"), upscale=2, arch="clean",
            channel_multiplier=2, bg_upsampler=self.bg, device=device,
        )

    def run(self, img, cfg=None, corners="auto", progress=None, debug_dir=None, debug_name=""):
        """返回 dict：result 修复结果、quad 实际使用的四角、faces 人脸数、is_gray、glare 自动去反光面积%。"""
        cfg = cfg or {}
        say = progress or (lambda msg: None)

        say("去除翻拍反光")
        for g in cfg.get("glare", []):  # 翻拍时的大块反光（原图坐标）
            img, lost = recover_glare(img, g["box"], lost_at=g.get("lost_at", 0.5))
            img = inpaint(img, lost)
        for hz in cfg.get("haze", []):  # 反光外围的淡雾（原图坐标）
            img = lift_haze(img, hz["polygon"], hz["ref_box"], hz.get("max_a", 0.7))

        say("透视校正、裁边")
        quad = quad_from_corners(img, corners)
        src = warp(img, quad) if quad is not None else img
        folds, ink, sheen = (repair_mask_cfg(img, cfg, part) for part in ("traced", "ink", "sheen"))
        if quad is not None:
            folds, ink, sheen = (warp(m, quad, cv2.INTER_NEAREST) for m in (folds, ink, sheen))
        y0, y1, x0, x1 = trim_box_cfg(src.shape, cfg)  # 裁掉相册膜压边、相册边
        src, folds, ink, sheen = (m[y0:y1, x0:x1] for m in (src, folds, ink, sheen))

        say("去除膜反光、擦痕、划痕")
        src = sheen_correct(src, sheen)  # 膜反光区域先校正亮度颜色，再补画边缘线
        for st in cfg.get("streaks", []):  # 相纸磨损的竖向白色擦痕
            src = remove_vertical_streaks(src, st["box"], st.get("max_width", 13), st.get("thresh", 8))
        for hz in cfg.get("haze_rect", []):  # 局部磨损发白（校正裁边后坐标）
            src = lift_haze(src, hz["polygon"], hz["ref_box"], hz.get("max_a", 0.5))
            for st in cfg.get("streaks", []):  # 去白后擦痕更明显，再清一遍
                src = remove_vertical_streaks(src, st["box"], st.get("max_width", 13), 6)
        for sc in cfg.get("scratches", []):  # 细划痕：传统邻域填补，不凭空生成内容
            sm = thin_scratch_mask(src, sc["box"], sc.get("polarity", "bright"), thresh=sc.get("thresh", 7))
            src = cv2.inpaint(src, sm, 3, cv2.INPAINT_TELEA)

        mono_path = None
        if cfg.get("mono_channel"):  # 黑白原片上的彩色倒影：单通道重建 + 扣除倒影亮度 + 泊松去边界
            say("去除屏幕倒影")
            reflect = np.zeros(src.shape[:2], np.uint8)
            for color in cfg.get("reflect_colors", []):
                reflect |= reflection_color_mask(src, color)
            mono_path = recover_reflection(mono_from_channel(src, cfg["mono_channel"]), reflect,
                                           cfg.get("reflect_strength", 1.0), cfg.get("reflect_repaint", 0.0))
            if cfg.get("reflect_edges", 0) > 0:
                mono_path = remove_reflection_edges(mono_path, src, cfg["reflect_edges"])
        if cfg.get("unmix"):  # 屏幕上黑白照片的彩色倒影：线性解混
            u = cfg["unmix"]
            um = unmix_reflection(src, u.get("star_bottom"), u.get("edge_bottom"), u.get("sample_bottom"))
            if mono_path is not None and u.get("blend_rows"):
                # 上方（楼上的倒影）用解混，下方（人物）用单通道方案，中间渐变过渡
                r0, r1 = u["blend_rows"]
                t = np.clip((np.arange(src.shape[0]) - r0) / max(r1 - r0, 1), 0, 1)[:, None, None]
                src = (um * (1 - t) + mono_path * t).astype(np.uint8)
            else:
                src = um
        elif mono_path is not None:
            src = mono_path

        say("检测人脸、霉点、折痕")
        with torch.no_grad():
            dets = self.gfpgan.face_helper.face_det.detect_faces(src, 0.9)
        faces = [b[:4] for b in dets]
        for fx1, fy1, fx2, fy2 in faces:  # 笔迹修补不进人脸，避免误伤嘴唇、红润的皮肤
            ink[max(0, int(fy1)):int(fy2), max(0, int(fx1)):int(fx2)] = 0
        spots = face_spot_mask(src, faces, device, [b[5:15] for b in dets])
        repair = folds | ink | spots
        if cfg.get("stains"):  # 整张照片上的霉点、污渍
            opts = cfg["stains"] if isinstance(cfg["stains"], dict) else {}
            repair |= stain_mask(src, faces, **opts)
        if cfg.get("auto_scratch", True):
            face_zone = np.zeros(src.shape[:2], np.uint8)
            for fx1, fy1, fx2, fy2 in faces:  # 人脸交给斑点检测处理，划痕检测不碰
                p = (fx2 - fx1) * 0.15
                face_zone[max(0, int(fy1 - p)):int(fy2 + p), max(0, int(fx1 - p)):int(fx2 + p)] = 1
            repair = cv2.bitwise_or(repair, crease_mask(src, device, protect=face_zone))

        say("补画缺损、调整颜色光影")
        corrected, is_gray, glare = enhance(src, faces, repair if repair.any() else None,
                                            veil=cfg.get("veil", False), auto_glare=cfg.get("auto_glare", True),
                                            digital=cfg.get("digital", False))
        if debug_dir is not None:  # 保存各阶段中间结果，便于排查
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            for name, im in (("1_校正裁边", src), ("2_增强", corrected), ("repair", repair)):
                cv2.imencode(".png", im)[1].tofile(str(Path(debug_dir) / f"{debug_name}_{name}.png"))

        say("人脸修复、放大 2 倍")
        # 小图整张放大（分块太小时会凭空画出纹理）；大图分大块放大，避免 8GB 内存机器内存不够
        self.bg.tile_size = 0 if corrected.shape[0] * corrected.shape[1] <= 500_000 else 512
        _, restored_faces, restored = self.gfpgan.enhance(corrected, has_aligned=False, only_center_face=False,
                                                          paste_back=True, weight=0.5)
        restored = sr_guard(restored, corrected)
        if is_gray:  # 黑白照片保持黑白
            restored = cv2.cvtColor(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return {"result": restored, "quad": None if quad is None else quad.tolist(),
                "faces": len(restored_faces), "is_gray": bool(is_gray), "glare": float(glare)}


def side_by_side(before, after, height=900):
    def fit(im):
        return cv2.resize(im, (round(im.shape[1] * height / im.shape[0]), height), interpolation=cv2.INTER_AREA)

    a, b = fit(before), fit(after)
    bar = np.full((height, 12, 3), 255, np.uint8)
    canvas = np.hstack([a, bar, b])
    label = np.full((60, canvas.shape[1], 3), 255, np.uint8)
    for text, x in (("Before", 20), ("After", a.shape[1] + 32)):
        cv2.putText(label, text, (x, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2, cv2.LINE_AA)
    return np.vstack([label, canvas])


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img, quality=95):
    ext = Path(path).suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    cv2.imencode(ext, img, params)[1].tofile(str(path))
