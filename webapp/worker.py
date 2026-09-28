"""网页版的修复进程：和网页服务分开运行。
- 取消/删除正在修复的照片时，网页服务直接结束这个进程，再启动一个新的（重新加载模型约 20 秒）；
- 内存不够被系统杀掉时，网页服务仍然正常，会把这张标记为出错。
与网页服务通过两个队列通信：jobs 收 (照片id, 配置, 四角)；events 发 ("ready",) / ("stage", id, 说明) /
("done", id, 结果信息) / ("error", id, 说明)。结果图片由这个进程直接写到照片目录。
"""
import time
import traceback
from pathlib import Path


def run(jobs, events, data_dir):
    import cv2
    import numpy as np

    try:
        from pipeline import Restorer, side_by_side
        from rectify import trim_box_cfg, warp
        restorer = Restorer()
    except Exception as e:  # noqa: BLE001
        events.put(("fatal", f"模型加载失败：{e}"))
        return
    events.put(("ready",))

    def imwrite(p, img, q):
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tofile(str(p))

    while True:
        pid, cfg, corners = jobs.get()
        d = Path(data_dir) / pid
        try:
            img = cv2.imdecode(np.fromfile(str(d / "original.jpg"), np.uint8), cv2.IMREAD_COLOR)
            t0 = time.time()
            out = restorer.run(img, cfg, corners, progress=lambda s: events.put(("stage", pid, s)))
            imwrite(d / "result.jpg", out["result"], 95)
            imwrite(d / "compare.jpg", side_by_side(img, out["result"]), 88)
            # 与结果对齐的原片（同样拉正、裁边、放大到同尺寸），网页上的滑动对比用
            q = None if out["quad"] is None else np.array(out["quad"], np.float32)
            src = warp(img, q) if q is not None else img
            y0, y1, x0, x1 = trim_box_cfg(src.shape, cfg)
            src = src[y0:y1, x0:x1]
            rh, rw = out["result"].shape[:2]
            imwrite(d / "aligned.jpg", cv2.resize(src, (rw, rh), interpolation=cv2.INTER_CUBIC), 90)
            events.put(("done", pid, {"seconds": round(time.time() - t0), "faces": out["faces"],
                                      "is_gray": out["is_gray"]}))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            events.put(("error", pid, str(e) or type(e).__name__))
