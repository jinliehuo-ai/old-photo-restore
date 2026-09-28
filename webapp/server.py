"""老照片修复网页版（本机运行）。

启动：双击项目根目录的「启动网页版.command」，或
    python -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000
浏览器打开 http://localhost:8000 ；同一 Wi-Fi 下的手机打开页面顶部显示的局域网地址。

每张照片存在 网页版数据/<id>/：
    original.jpg  上传的照片（长边超过 1200 会缩小，避免 8GB 内存机器处理不动）
    meta.json     名称、状态、标注（全部为原图坐标，见 ann_to_cfg）
    result.jpg    修复结果；compare.jpg 左右对比图；aligned.jpg 与结果对齐的原片（滑动对比用）；thumb.jpg 缩略图
"""
import json
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rectify import detect_quad, order_quad, rect_transform  # noqa: E402

DATA = ROOT / "网页版数据"
DATA.mkdir(exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"
MAX_SIDE = 1200  # 更大的图在 8GB 内存的 Mac 上放大 2 倍时容易内存不够

app = FastAPI(title="老照片修复")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_state = {"model": "loading", "current": None, "error": None, "started": None}
_lock = threading.RLock()


# ------------------------------------------------------------------ 存取
def _dir(pid):
    d = DATA / pid
    if not d.is_dir() or "/" in pid or ".." in pid:
        raise HTTPException(404, "照片不存在")
    return d


def _meta(pid):
    return json.loads((_dir(pid) / "meta.json").read_text(encoding="utf-8"))


def _save_meta(pid, meta):
    with _lock:
        tmp = DATA / pid / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(DATA / pid / "meta.json")


def _update(pid, **kw):
    meta = _meta(pid)
    meta.update(kw, updated=time.time())
    _save_meta(pid, meta)
    return meta


def _imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def _imwrite(p, img, q=92):
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tofile(str(p))


def _thumb(img, side=240):
    s = side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _default_ann(corners):
    return {
        "corners": corners,
        "trim": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "options": {"digital": False, "stains": False, "veil": False, "auto_glare": True,
                    "auto_scratch": True, "screen_reflection": False},
        "folds": [], "ink": [], "glare": [], "haze": [], "sheen": [], "scratches": [], "streaks": [],
    }


# ------------------------------------------------------------------ 标注 → 修复配置
def _box_to_rect(box, T):
    x0, y0, x1, y1 = box
    pts = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float64)
    r = cv2.perspectiveTransform(pts, T)[0]
    return [max(0, int(r[:, 0].min())), max(0, int(r[:, 1].min())), int(r[:, 0].max()), int(r[:, 1].max())]


def ann_to_cfg(ann):
    """网页上的标注（全部为原图坐标）转成 pipeline 的配置和四角。
    细划痕、竖向擦痕在流程里作用于“校正裁边后”的图，这里换算坐标。"""
    o = ann.get("options", {})
    trim = {k: int(v or 0) for k, v in (ann.get("trim") or {}).items()}
    corners = ann.get("corners")
    cfg = {
        "trim": trim,
        "folds": [{"points": f["points"], "width": int(f.get("width", 10))} for f in ann.get("folds", [])],
        "ink": [{"points": i["points"], "color": i.get("color", "red")} for i in ann.get("ink", [])],
        "glare": [{"box": g["box"]} for g in ann.get("glare", [])],
        "haze": [{"polygon": h["polygon"], "ref_box": h["ref_box"]} for h in ann.get("haze", []) if h.get("ref_box")],
        "sheen": [{"polygon": s["polygon"]} for s in ann.get("sheen", [])],
        "digital": bool(o.get("digital")),
        "stains": bool(o.get("stains")),
        "veil": bool(o.get("veil")),
        "auto_glare": bool(o.get("auto_glare", True)),
        "auto_scratch": bool(o.get("auto_scratch", True)),
    }
    if o.get("screen_reflection"):  # 屏幕上的黑白照片映出彩色倒影
        cfg.update(mono_channel="blue", reflect_colors=["yellow", "red"], reflect_strength=1.6,
                   reflect_repaint=0.4, reflect_edges=0.05, auto_glare=False)
    quad = None if corners is None else order_quad(corners)
    T = rect_transform(quad, (trim.get("top", 0), 0, trim.get("left", 0), 0))
    cfg["scratches"] = [{"box": _box_to_rect(s["box"], T)} for s in ann.get("scratches", [])]
    cfg["streaks"] = [{"box": _box_to_rect(s["box"], T)} for s in ann.get("streaks", [])]
    return cfg, (None if corners is None else [list(map(float, p)) for p in corners])


# ------------------------------------------------------------------ 修复进程管理
# 修复在单独的进程里跑（webapp/worker.py）：可随时结束（取消、删除），内存不足被系统杀掉也不影响网页。
import multiprocessing as mp  # noqa: E402

_mp = mp.get_context("spawn")
_proc = None
_proc_jobs = None
_events = None
_pending: "list[str]" = []  # 排队中的照片 id（按顺序）


def _start_worker():
    global _proc, _proc_jobs, _events
    from webapp import worker
    _proc_jobs, _events = _mp.Queue(), _mp.Queue()
    _proc = _mp.Process(target=worker.run, args=(_proc_jobs, _events, str(DATA)), daemon=True)
    _proc.start()
    _state.update(model="loading", current=None, error=None)


def _kill_worker():
    if _proc is not None and _proc.is_alive():
        _proc.kill()
        _proc.join(timeout=5)


def _safe_update(pid, **kw):
    try:
        _update(pid, **kw)
    except HTTPException:
        pass  # 照片已被删除


def _dispatch():
    """模型就绪且空闲时，把下一张发给修复进程。"""
    with _lock:
        if _state["model"] != "ready" or _state["current"] or not _pending:
            return
        pid = _pending.pop(0)
    try:
        meta = _meta(pid)
        cfg, corners = ann_to_cfg(meta["ann"])
    except HTTPException:
        return _dispatch()
    except Exception as e:  # noqa: BLE001
        _safe_update(pid, status="error", stage="出错", error=f"标注有误：{e}")
        return _dispatch()
    _state["current"] = pid
    _state["started"] = time.time()
    _safe_update(pid, status="running", stage="准备中", error=None)
    _proc_jobs.put((pid, cfg, corners))


def _monitor():
    """接收修复进程的消息；进程意外退出（通常是内存不足）时标记出错并重启进程。"""
    while True:
        try:
            ev = _events.get(timeout=1)
        except Exception:  # noqa: BLE001  队列超时
            if _proc is not None and not _proc.is_alive():
                pid = _state["current"]
                if pid:
                    _safe_update(pid, status="error", stage="出错",
                                 error="修复进程意外退出，多半是电脑内存不够。可以关掉其他软件后点「重新修复」。")
                _start_worker()
            continue
        kind = ev[0]
        if kind == "ready":
            _state["model"] = "ready"
        elif kind == "fatal":
            _state.update(model="error", error=ev[1])
        elif kind == "stage" and ev[1] == _state["current"]:
            _safe_update(ev[1], stage=ev[2])
        elif kind in ("done", "error") and ev[1] == _state["current"]:
            pid = ev[1]
            if kind == "done":
                info = ev[2]
                try:
                    v = _meta(pid).get("version", 0) + 1
                    _safe_update(pid, status="done", stage="完成", seconds=info["seconds"],
                                 info={"faces": info["faces"], "is_gray": info["is_gray"]}, version=v)
                except HTTPException:
                    pass
            else:
                _safe_update(pid, status="error", stage="出错", error=ev[2])
            _state["current"] = None
        _dispatch()


def _enqueue(pid):
    with _lock:
        if pid not in _pending and _state["current"] != pid:
            _pending.append(pid)
    _dispatch()


def cancel(pid):
    """从队列移除；正在修复的直接结束修复进程并重启。"""
    with _lock:
        if pid in _pending:
            _pending.remove(pid)
    if _state["current"] == pid:
        _state["current"] = None
        _kill_worker()
        _start_worker()


_start_worker()
threading.Thread(target=_monitor, daemon=True).start()


def _requeue_unfinished():
    for d in sorted(DATA.iterdir()):
        m = d / "meta.json"
        if m.exists():
            meta = json.loads(m.read_text(encoding="utf-8"))
            if meta.get("status") in ("queued", "running"):
                meta["status"], meta["stage"] = "queued", "排队中"
                _save_meta(d.name, meta)
                _pending.append(d.name)


_requeue_unfinished()


# ------------------------------------------------------------------ 接口
def _lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # 不会真的发包，只是让系统选出局域网网卡
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"model": _state["model"], "error": _state["error"], "current": _state["current"],
            "queued": len(_pending), "lan": [f"http://{ip}:8000" for ip in _lan_ips()]}


def _summary(pid, meta):
    return {"id": pid, "name": meta["name"], "status": meta["status"], "stage": meta.get("stage"),
            "error": meta.get("error"), "created": meta["created"], "version": meta.get("version", 0),
            "seconds": meta.get("seconds"), "info": meta.get("info")}


@app.get("/api/photos")
def list_photos():
    items = []
    for d in DATA.iterdir():
        if (d / "meta.json").exists():
            items.append(_summary(d.name, json.loads((d / "meta.json").read_text(encoding="utf-8"))))
    return sorted(items, key=lambda x: x["created"], reverse=True)


@app.post("/api/photos")
async def upload(files: List[UploadFile] = File(...), auto_restore: bool = True):
    created = []
    for f in files:
        data = np.frombuffer(await f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"{f.filename}：无法读取，请上传 JPG 或 PNG 照片（iPhone 的 HEIC 请先转成 JPG）")
        s = MAX_SIDE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = DATA / pid
        d.mkdir()
        _imwrite(d / "original.jpg", img, 95)
        _imwrite(d / "thumb.jpg", _thumb(img), 85)
        q = detect_quad(img)
        meta = {"name": f.filename or pid, "created": time.time(), "updated": time.time(),
                "width": img.shape[1], "height": img.shape[0], "status": "new", "stage": None,
                "ann": _default_ann(None if q is None else q.round(1).tolist()),
                "auto_corners": None if q is None else q.round(1).tolist()}
        _save_meta(pid, meta)
        if auto_restore:
            _update(pid, status="queued", stage="排队中")
            _enqueue(pid)
        created.append(_summary(pid, _meta(pid)))
    return created


@app.get("/api/photos/{pid}")
def get_photo(pid: str):
    meta = _meta(pid)
    return {**_summary(pid, meta), "width": meta["width"], "height": meta["height"], "ann": meta["ann"],
            "auto_corners": meta.get("auto_corners"), "has_result": (DATA / pid / "result.jpg").exists(),
            "has_aligned": (DATA / pid / "aligned.jpg").exists()}


@app.put("/api/photos/{pid}/ann")
def save_ann(pid: str, ann: dict):
    try:
        ann_to_cfg(ann)  # 先检查能否转换
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"标注格式有误：{e}")
    _update(pid, ann=ann)
    return {"ok": True}


@app.post("/api/photos/{pid}/restore")
def restore(pid: str):
    meta = _meta(pid)
    if meta["status"] in ("queued", "running"):
        return {"ok": True, "status": meta["status"]}
    _update(pid, status="queued", stage="排队中", error=None)
    _enqueue(pid)
    return {"ok": True, "status": "queued"}


@app.post("/api/photos/{pid}/cancel")
def cancel_restore(pid: str):
    meta = _meta(pid)
    if meta["status"] not in ("queued", "running"):
        return {"ok": True}
    cancel(pid)
    has = (DATA / pid / "result.jpg").exists()
    _update(pid, status="done" if has else "new", stage="已停止" if not has else "完成", error=None)
    return {"ok": True}


@app.get("/api/photos/{pid}/image/{kind}")
def image(pid: str, kind: str, download: bool = False):
    names = {"original": "original.jpg", "result": "result.jpg", "compare": "compare.jpg", "thumb": "thumb.jpg",
             "aligned": "aligned.jpg"}
    if kind not in names:
        raise HTTPException(404)
    p = _dir(pid) / names[kind]
    if not p.exists():
        raise HTTPException(404, "还没有修复结果")
    if download:
        stem = Path(_meta(pid)["name"]).stem
        suffix = {"result": "_修复", "compare": "_对比", "original": ""}.get(kind, "")
        return FileResponse(p, media_type="image/jpeg", filename=f"{stem}{suffix}.jpg")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.delete("/api/photos/{pid}")
def delete(pid: str):
    d = _dir(pid)
    cancel(pid)  # 正在修复或排队的先停掉
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    return {"ok": True}
