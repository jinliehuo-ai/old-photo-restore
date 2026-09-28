"""老照片批量修复（命令行）。修复流程在 pipeline.py，网页版在 webapp/。

用法: python restore.py [编号或文件名前缀 ...]  （不带参数处理全部）
照片编号 01、02……写在 编号对照.txt，编号一旦分配不变，新照片接在最后。
输入 原照片/，输出 修复照片/（修复结果）和 对比/（左旧右新拼图）。
每张照片的四角在 corners.json、各类缺陷标注在 repairs.json（按编号，格式见 *.example.json）。
设置环境变量 RESTORE_DEBUG=1 会把中间结果存到 调试/。
"""
import os
import sys
import time

from pipeline import ROOT, Restorer, imread, imwrite, side_by_side
from rectify import load_manual, load_repairs

SRC, DST, CMP = ROOT / "原照片", ROOT / "修复照片", ROOT / "对比"


def log(msg):
    """进度同时打印并写入 运行日志.txt，方便在另一个终端或编辑器里查看是否在运行。"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(ROOT / "运行日志.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def assign_numbers():
    """编号一旦分配就不变（corners.json、repairs.json 和输出文件名都按编号）：
    已在 编号对照.txt 里的照片保持原编号，新放进 原照片/ 的照片按文件名顺序接在最后。"""
    table = ROOT / "编号对照.txt"
    known = {}
    if table.exists():
        for line in table.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                no, name = line.split("\t", 1)
                known[name] = no
    files = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    nxt = max((int(n) for n in known.values()), default=0) + 1
    for p in files:
        if p.name not in known:
            known[p.name] = f"{nxt:02d}"
            nxt += 1
    table.write_text("".join(f"{no}\t{name}\n" for name, no in sorted(known.items(), key=lambda x: x[1])),
                     encoding="utf-8")
    present = {p.name: p for p in files}
    return sorted(((no, present[name]) for name, no in known.items() if name in present), key=lambda x: x[0])


def main():
    for d in (SRC, DST, CMP):
        d.mkdir(exist_ok=True)
    if not any(p.suffix.lower() in (".jpg", ".jpeg", ".png") for p in SRC.iterdir()):
        print(f"请先把要修复的照片（JPG/PNG）放进 {SRC}")
        return 1
    restorer = Restorer()

    numbered = assign_numbers()
    if len(sys.argv) > 1:
        want = sys.argv[1:]
        numbered = [(no, p) for no, p in numbered
                    if any(a.zfill(2) == no or (not a.isdigit() and p.name.startswith(a)) for a in want)]
    corners_all = load_manual()
    debug = ROOT / "调试" if os.environ.get("RESTORE_DEBUG") else None
    log(f"开始：共 {len(numbered)} 张（{' '.join(no for no, _ in numbered)}）")
    for i, (no, path) in enumerate(numbered, 1):
        log(f"[{i}/{len(numbered)}] {no} 处理中…")
        img = imread(path)
        out = restorer.run(img, load_repairs(no), corners_all.get(no, "auto"), debug_dir=debug, debug_name=no)
        imwrite(DST / f"{no}_修复.jpg", out["result"])
        imwrite(CMP / f"{no}_对比.jpg", side_by_side(img, out["result"]), 88)
        log(f"[{i}/{len(numbered)}] {no} 完成 {path.name}: {'rectified, ' if out['quad'] else ''}"
            f"{out['faces']} faces, glare {out['glare']:.2f}%{' (B/W)' if out['is_gray'] else ''}")
    log("全部完成")


if __name__ == "__main__":
    sys.exit(main())
