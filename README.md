# 老照片修复 · Old Photo Restore

把用手机翻拍的老照片修成“像现在手机拍的”：拉正、去反光和屏幕倒影、去褪色偏色、去折痕划痕霉点、修复人脸、放大 2 倍。
**全部在本机运行，照片不会上传到任何地方，不需要联网。**

*Restore old printed photos that were re-shot with a phone: perspective correction, glare and screen-reflection removal,
fade/colour-cast correction, crease/scratch/stain removal, face restoration and 2× upscaling. Runs fully offline on your own computer.
Includes a local web app with upload, before/after slider and in-browser annotation for hard cases.*

## 功能

- **网页版**：上传照片自动修复；滑动对比修复前后；效果不满意时在网页上拖动照片四角、描出折痕和笔迹、框出反光，再重新修复。
  同一 Wi-Fi 下的手机也能用。
- **命令行批量处理**：把一整个文件夹的照片按编号批量修复，并输出左右对比图。
- 按缺陷类型分别处理：

| 问题 | 做法 |
|---|---|
| 翻拍角度歪、带着相册膜/桌面/手机边框 | 自动找相纸四角（直线检测 + 打分），透视校正并裁边；不准时可手动拖四角 |
| 灯光在相纸上的反光 | 点状反光自动识别补画；大块反光按“叠加白光”模型扣除还原，只有完全过曝处才补画 |
| 屏幕上显示的黑白照片映出彩色倒影 | 线性解混：逐像素把颜色拆成“照片 × 屏幕白 + 倒影颜色”，纯数学方法 |
| 塑料膜折痕、红笔/蓝笔圈 | 手动描大致位置，程序自动吸附到真实位置后补画 |
| 霉点、污渍、脸上黑点、细划痕、相纸擦痕 | 自动或框选检测后补画；脸上只修皮肤，避开五官 |
| 褪色、偏色、发灰 | 自动色阶、去雾、保边细节增强；褪色越重校正越强；黑白照片保持黑白，数码照片只轻度增强 |
| 人脸模糊、分辨率低 | GFPGAN 修复人脸，Real-ESRGAN 放大 2 倍，并防止放大模型凭空画出纹理 |

## 安装

需要 Python 3.10。在 Apple 芯片的 Mac 上会用 MPS 加速；其他电脑用 CPU（慢一些）。8GB 内存可以运行。

```bash
git clone https://github.com/jinliehuo-ai/old-photo-restore.git
cd old-photo-restore
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py            # 下载模型，约 600MB
.venv/bin/python download_models.py --scratch  # 可选：自动划痕检测（需下载 2GB 官方压缩包）
```

也可以用 conda：`conda create -n photo_restore python=3.10`，然后用这个环境的 python 执行上面的 pip 和下载命令。

## 使用

### 网页版

macOS：在「访达」里双击 **启动网页版.command**，浏览器会自动打开。其他系统：

```bash
.venv/bin/python -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000
```

然后打开 http://localhost:8000 。手机和电脑连同一个 Wi-Fi 时，打开页面顶部显示的局域网地址即可。
上传的照片和结果保存在本机的 `网页版数据/` 文件夹。

### 命令行批量处理

```bash
.venv/bin/python restore.py          # 处理 原照片/ 里的全部照片
.venv/bin/python restore.py 4 14     # 只处理 04、14 号
```

- 把照片放进 `原照片/`；结果在 `修复照片/`，左右对比图在 `对比/`，进度在 `运行日志.txt`。
- 编号写在 `编号对照.txt`，一旦分配不会变，新照片接在最后。
- 每张照片的手动标注：四角写在 `corners.json`，其余缺陷写在 `repairs.json`，格式见 `corners.example.json`、
  `repairs.example.json`（里面的“说明”列出了全部字段）。
- `RESTORE_DEBUG=1` 会把中间结果存到 `调试/`。

## 代码结构

| 文件 | 作用 |
|---|---|
| `pipeline.py` | 修复流程核心：加载模型、按配置修复一张照片（网页版和命令行共用） |
| `rectify.py` | 自动/手动透视校正、裁边、折痕和笔迹标注转掩膜 |
| `enhance.py` | 点状反光、褪色偏色、光影和饱和度、黑白/数码照片判断、LaMa 补画 |
| `glare.py` | 翻拍大块反光、反光淡雾 |
| `defects.py` | 屏幕倒影解混、膜反光区域、笔迹、超分防护、亮区偏色 |
| `spots.py` | 脸上斑点、整张霉点、细划痕、竖向擦痕 |
| `creases.py` | 自动划痕检测（可选模型） |
| `webapp/` | 网页版：`server.py` 接口和任务队列，`worker.py` 独立修复进程（可随时停止），`static/` 页面 |
| `download_models.py` | 下载模型 |

## 隐私

本项目不收集、不上传任何照片。仓库的 `.gitignore` 采用白名单，只放行代码文件；
`原照片/`、`修复照片/`、`对比/`、`调试/`、`网页版数据/`、`corners.json`、`repairs.json`、`编号对照.txt`
等个人数据都不会被提交。

## 致谢

本项目的修复效果依赖以下开源项目和模型，使用时请同时遵守它们的许可证：

- [GFPGAN](https://github.com/TencentARC/GFPGAN)（人脸修复，Apache-2.0）
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)、[BasicSR](https://github.com/XPixelGroup/BasicSR)（放大，BSD-3-Clause / Apache-2.0）
- [facexlib](https://github.com/xinntao/facexlib)（人脸检测与解析，MIT）
- [LaMa](https://github.com/advimman/lama)（补画，Apache-2.0），通过 [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
- [Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)（划痕检测，MIT）

## 许可证

[MIT](LICENSE)
