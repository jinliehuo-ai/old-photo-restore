#!/bin/zsh
# 在「访达」(Finder) 里双击启动老照片修复网页版（在 VS Code 里双击只会打开文本，不会运行）。
# 启动后自动打开浏览器；关闭这个终端窗口即停止。
cd "$(dirname "$0")"
URL="http://localhost:8000"

# 找 Python：环境变量 PHOTO_RESTORE_PYTHON → 项目里的 .venv → conda 环境 photo_restore → 系统 python3
PY="${PHOTO_RESTORE_PYTHON:-}"
[ -z "$PY" ] && [ -x .venv/bin/python ] && PY=.venv/bin/python
if [ -z "$PY" ] && command -v conda >/dev/null; then
  C="$(conda info --base 2>/dev/null)/envs/photo_restore/bin/python"
  [ -x "$C" ] && PY="$C"
fi
[ -z "$PY" ] && for C in /opt/anaconda3 /opt/miniconda3 "$HOME/anaconda3" "$HOME/miniconda3" /opt/homebrew/anaconda3; do
  [ -x "$C/envs/photo_restore/bin/python" ] && PY="$C/envs/photo_restore/bin/python" && break
done
[ -z "$PY" ] && PY=python3

if curl -s -m 2 "$URL/api/status" >/dev/null; then
  echo "网页版已经在运行，直接打开浏览器：$URL"
  open "$URL"
  sleep 2
  exit 0
fi

if ! "$PY" -c "import gfpgan, fastapi" 2>/dev/null; then
  echo "找不到装好依赖的 Python（当前尝试的是：$PY）。"
  echo "请按 README 的「安装」步骤操作，或设置环境变量 PHOTO_RESTORE_PYTHON 指向正确的 Python。"
  echo "按回车键关闭窗口"; read
  exit 1
fi

echo "正在启动老照片修复网页版…（首次加载模型约需 20 秒）"
echo "启动后浏览器会自动打开 $URL ；关闭这个窗口即停止。"
(for i in {1..60}; do curl -s -m 1 "$URL/api/status" >/dev/null && open "$URL" && break; sleep 1; done) &
"$PY" -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000
echo ""
echo "网页版已停止（如上面有报错信息，可以截图反馈）。按回车键关闭窗口"
read
