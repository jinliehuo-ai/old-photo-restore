// 老照片修复网页版前端：照片列表、结果对比、标注编辑。标注全部使用原图像素坐标。
const $ = (id) => document.getElementById(id);
const api = async (url, opt = {}) => {
  const r = await fetch(url, opt);
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : null;
  if (!r.ok) throw new Error(body?.detail || body?.message || `请求失败（${r.status}）`);
  return body;
};
const toast = (msg, ms = 2600) => {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => (t.hidden = true), ms);
};
const STATUS = { new: "未修复", queued: "排队中", running: "修复中", done: "已完成", error: "出错" };

let photos = [];
let current = null;      // 当前照片详情
let view = "compare";

// ------------------------------------------------------------------ 列表与状态
async function refresh() {
  try {
    const [st, list] = await Promise.all([api("/api/status"), api("/api/photos")]);
    const pill = $("model-pill");
    pill.textContent = st.model === "ready" ? (st.current ? `修复中，另有 ${st.queued} 张排队` : "就绪")
      : st.model === "error" ? st.error : "修复程序准备中（约 20 秒）…";
    pill.className = "pill " + (st.model === "ready" ? "ok" : st.model === "error" ? "err" : "");
    $("lan").innerHTML = st.lan.length ? `同一 Wi-Fi 的手机打开：${st.lan.map((u) => `<code>${u}</code>`).join(" ")}` : "";
    photos = list;
    $("offline").hidden = true;
    refresh.fails = 0;
    renderList();
    if (current) {
      const s = photos.find((p) => p.id === current.id);
      if (!s) { current = null; showMain(); return; }
      const changed = s.status !== current.status || s.version !== current.version || s.stage !== current.stage;
      Object.assign(current, s);
      if (changed) renderCompare();
    }
  } catch (e) {
    // 连续几次连不上，说明网页服务停了
    refresh.fails = (refresh.fails || 0) + 1;
    if (refresh.fails >= 3) {
      $("offline").hidden = false;
      const pill = $("model-pill"); pill.textContent = "服务已停止"; pill.className = "pill err";
    }
  }
}

function renderList() {
  $("empty").hidden = photos.length > 0;
  $("list").innerHTML = photos.map((p) => `
    <li><button type="button" data-id="${p.id}" aria-current="${current?.id === p.id}">
      <img src="/api/photos/${p.id}/image/thumb" alt="" loading="lazy">
      <span class="meta"><span class="name">${escapeHtml(p.name)}</span>
      <span class="chip ${p.status}">${p.status === "running" ? (p.stage || "修复中") : STATUS[p.status]}</span></span>
    </button></li>`).join("");
}
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

$("list").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-id]");
  if (b) open(b.dataset.id);
});

async function open(id, v = view) {
  current = await api(`/api/photos/${id}`);
  ann = structuredClone(current.ann);
  undoStack.length = 0; selected = null; drawing = null;
  view = v;
  renderList(); showMain();
}

function showMain() {
  $("main").hidden = !current;
  if (!current) return;
  $("title").textContent = current.name;
  $("tab-compare").setAttribute("aria-selected", view === "compare");
  $("tab-edit").setAttribute("aria-selected", view === "edit");
  $("view-compare").hidden = view !== "compare";
  $("view-edit").hidden = view !== "edit";
  renderCompare();
  if (view === "edit") initEditor();
}
$("tab-compare").onclick = () => { view = "compare"; showMain(); };
$("tab-edit").onclick = () => { view = "edit"; showMain(); };
$("go-edit").onclick = () => { view = "edit"; showMain(); };

// 对比方式：滑动 / 并排 / 对比图（记住上次的选择）
let mode = (() => { try { return localStorage.getItem("cmpMode") || "slider"; } catch { return "slider"; } })();
function setMode(m) {
  mode = m;
  try { localStorage.setItem("cmpMode", m); } catch { /* 无痕模式 */ }
  renderCompare();
}
document.querySelectorAll(".modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
$("cmp-range").addEventListener("input", (e) => $("cmp").style.setProperty("--p", e.target.value + "%"));
$("cmp-before").onerror = () => { if (mode === "slider") { mode = "pair"; renderCompare(); } };

function renderCompare() {
  if (!current) return;
  const c = $("chip");
  c.className = "chip " + current.status;
  c.textContent = current.status === "running" ? (current.stage || "修复中") : STATUS[current.status];
  const base = `/api/photos/${current.id}/image/`;
  if ($("img-before").dataset.id !== current.id) {
    $("img-before").src = base + "original"; $("img-before").dataset.id = current.id;
  }
  const key = `${current.id}:${current.version}`;
  const hasResult = current.version > 0;
  if (hasResult && $("img-after").dataset.key !== key) {
    $("img-after").src = `${base}result?v=${current.version}`; $("img-after").dataset.key = key;
  }
  $("img-after").hidden = !hasResult;
  // 对比方式：没有修复结果时只能看并排（显示原片和进度）
  const m = hasResult ? mode : "pair";
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === m));
  document.querySelector(".modes").hidden = !hasResult;
  $("mode-slider").hidden = m !== "slider";
  $("mode-sheet").hidden = m !== "sheet";
  $("mode-pair").hidden = m !== "pair";
  if (hasResult && $("cmp").dataset.key !== key) {
    $("cmp").dataset.key = key;
    $("cmp-before").src = `${base}aligned?v=${current.version}`;
    $("cmp-after").src = `${base}result?v=${current.version}`;
    $("img-sheet").src = `${base}compare?v=${current.version}`;
    $("cmp-range").value = 50; $("cmp").style.setProperty("--p", "50%");
  }
  // 修复中时，在滑动/对比图模式也要看得到进度，所以进度提示放在并排视图里：这时切回并排
  if (current.status === "queued" || current.status === "running") {
    $("mode-slider").hidden = true; $("mode-sheet").hidden = true; $("mode-pair").hidden = false;
    document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === "pair"));
  }
  const wait = $("wait");
  if (current.status === "queued" || current.status === "running") {
    wait.innerHTML = `<div><div class="spinner"></div><b>${current.status === "queued" ? "排队中" : escapeHtml(current.stage || "修复中")}</b>一张通常需要 1–5 分钟，可以先去看别的照片</div>`;
  } else if (current.status === "error") {
    wait.innerHTML = `<div><b>修复出错</b>${escapeHtml(current.error || "")}</div>`;
  } else if (!hasResult) {
    wait.innerHTML = `<div><b>${current.stage === "已停止" ? "已停止修复" : "还没有修复"}</b>到「标注调整」检查后点「保存并重新修复」</div>`;
  } else wait.innerHTML = "";
  $("stop").hidden = !(current.status === "queued" || current.status === "running");
  for (const [id, kind] of [["dl-result", "result"], ["dl-compare", "compare"]]) {
    const a = $(id);
    a.href = hasResult ? `${base}${kind}?download=true` : "#";
    a.setAttribute("aria-disabled", !hasResult);
  }
  $("info").textContent = hasResult && current.info
    ? `上次修复用时 ${current.seconds ?? "?"} 秒，找到 ${current.info.faces} 张人脸${current.info.is_gray ? "，识别为黑白照片" : ""}。`
    : "";
}

$("stop").onclick = async () => {
  try {
    await api(`/api/photos/${current.id}/cancel`, { method: "POST" });
    toast(current.status === "running" ? "已停止。修复程序需要约 20 秒重新准备" : "已从队列移除");
    refresh();
  } catch (e) { toast(e.message); }
};

$("delete").onclick = async () => {
  const b = $("delete");
  if (b.dataset.confirm !== "1") { b.dataset.confirm = "1"; b.textContent = "再点一次确认删除"; setTimeout(() => { b.dataset.confirm = ""; b.textContent = "删除这张"; }, 3000); return; }
  try { await api(`/api/photos/${current.id}`, { method: "DELETE" }); current = null; toast("已删除"); refresh(); showMain(); }
  catch (e) { toast(e.message); }
  b.dataset.confirm = ""; b.textContent = "删除这张";
};

// ------------------------------------------------------------------ 上传
async function upload(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  toast(`正在上传 ${files.length} 张…`, 60000);
  try {
    const created = await api("/api/photos", { method: "POST", body: fd });
    toast(`已上传 ${created.length} 张，开始自动修复`);
    await refresh();
    open(created[0].id, "compare");
  } catch (e) { toast(e.message, 5000); }
}
$("file").onchange = (e) => { upload([...e.target.files]); e.target.value = ""; };
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
document.addEventListener("drop", (e) => upload([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"))));

// ------------------------------------------------------------------ 标注编辑
const TOOLS = [
  { id: "select", label: "选择 / 拖四角", color: "#2C6A6E", hint: "拖动四个圆点对齐照片的四个角；点一下标注可以选中，再按「删除」或 Delete 键。" },
  { id: "folds", label: "折痕", color: "#E07A1F", kind: "line", hint: "沿折痕依次点几个点（大致准确即可，程序会自动贴合），然后点「完成这一条」或双击。" },
  { id: "ink", label: "笔迹", color: "#D6336C", kind: "line", hint: "沿红笔/蓝笔画的线依次点几个点，然后点「完成这一条」。右侧可切换笔迹颜色。" },
  { id: "glare", label: "翻拍反光", color: "#E8B400", kind: "box", hint: "拖一个框，框住翻拍时灯光造成的大块白色反光（不要框进画面里本来就白的东西）。" },
  { id: "haze", label: "反光淡雾", color: "#1F9BD6", kind: "poly", hint: "先依次点出发灰发白的区域，点「完成这一条」；再拖一个框，框住同类但没被雾盖住的地方作参照。" },
  { id: "sheen", label: "膜反光区域", color: "#6C5CE7", kind: "poly", hint: "依次点出塑料膜折叠处整片发白的区域，点「完成这一条」。" },
  { id: "scratches", label: "细划痕", color: "#2FA84F", kind: "box", hint: "拖框框住细长的白色划痕所在区域，程序只去掉框里又细又长的亮线。" },
  { id: "streaks", label: "竖向擦痕", color: "#8BC34A", kind: "box", hint: "拖框框住相纸磨损造成的竖向白色擦痕，框不要包含真实的竖直白线（如鞋带）。" },
];
const LABEL = Object.fromEntries(TOOLS.map((t) => [t.id, t.label]));
const COLOR = Object.fromEntries(TOOLS.map((t) => [t.id, t.color]));

let ann = null, tool = "select", selected = null, drawing = null, dragCorner = -1;
const undoStack = [];
const img = new Image();
const canvas = $("canvas"), ctx = canvas.getContext("2d");
let scale = 1, ox = 0, oy = 0;

$("tools").innerHTML = TOOLS.map((t) => `<button type="button" data-tool="${t.id}"><i style="background:${t.color}"></i>${t.label}</button>`).join("")
  + `<select id="ink-color" aria-label="笔迹颜色"><option value="red">红笔</option><option value="blue">蓝笔</option></select>`;
$("tools").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-tool]");
  if (!b) return;
  tool = b.dataset.tool; drawing = null; renderTools(); draw();
});
function renderTools() {
  document.querySelectorAll("#tools button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.tool === tool));
  $("hint").textContent = TOOLS.find((t) => t.id === tool).hint;
  $("drawing").hidden = !(drawing && (drawing.kind === "line" || drawing.kind === "poly"));
}

function initEditor() {
  if (img.dataset.id !== current.id) {
    img.onload = () => { layout(); draw(); };
    img.src = `/api/photos/${current.id}/image/original`;
    img.dataset.id = current.id;
  }
  for (const k of ["top", "bottom", "left", "right"]) $("trim-" + k).value = ann.trim?.[k] ?? 0;
  for (const k of ["digital", "screen_reflection", "stains", "veil", "auto_glare", "auto_scratch"]) $("opt-" + k).checked = !!ann.options[k];
  renderTools(); renderAnns(); cornersNote(); layout(); draw();
}

function layout() {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!img.naturalWidth) return;
  scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
  ox = (r.width - img.naturalWidth * scale) / 2; oy = (r.height - img.naturalHeight * scale) / 2;
}
window.addEventListener("resize", () => { if (view === "edit" && current) { layout(); draw(); } });

const toScreen = ([x, y]) => [ox + x * scale, oy + y * scale];
const toImage = (e) => {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left - ox) / scale, y = (e.clientY - r.top - oy) / scale;
  return [Math.max(0, Math.min(img.naturalWidth, x)), Math.max(0, Math.min(img.naturalHeight, y))].map((v) => Math.round(v));
};

function path(pts, close) {
  ctx.beginPath();
  pts.forEach((p, i) => { const [x, y] = toScreen(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  if (close) ctx.closePath();
}
function boxPts([x0, y0, x1, y1]) { return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]; }

function draw() {
  const r = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  if (!img.naturalWidth || !ann) return;
  ctx.drawImage(img, ox, oy, img.naturalWidth * scale, img.naturalHeight * scale);
  ctx.lineJoin = ctx.lineCap = "round";
  // 四角
  if (ann.corners) {
    path(ann.corners, true);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 4; ctx.stroke();
    ctx.strokeStyle = COLOR.select; ctx.lineWidth = 2; ctx.stroke();
    ann.corners.forEach((p) => {
      const [x, y] = toScreen(p);
      ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.fillStyle = "#fff"; ctx.fill();
      ctx.lineWidth = 3; ctx.strokeStyle = COLOR.select; ctx.stroke();
    });
  }
  for (const t of TOOLS.slice(1)) {
    (ann[t.id] || []).forEach((a, i) => drawAnn(t.id, a, selected?.type === t.id && selected.i === i));
  }
  if (drawing) drawAnn(drawing.type, drawing.data, true, true);
}

function drawAnn(type, a, sel, live = false) {
  const color = type === "ink" && a.color === "blue" ? "#2B6CD8" : COLOR[type];
  ctx.strokeStyle = color; ctx.fillStyle = color + "33";
  ctx.lineWidth = sel ? 4 : 2.5;
  ctx.setLineDash(live ? [6, 4] : []);
  if (a.points) {
    path(a.points, false); ctx.stroke();
    a.points.forEach((p) => { const [x, y] = toScreen(p); ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fillStyle = color; ctx.fill(); });
  }
  if (a.polygon) {
    path(a.polygon, !live || a.ref_box); ctx.fill(); ctx.stroke();
  }
  if (a.box) { path(boxPts(a.box), true); ctx.fill(); ctx.stroke(); }
  if (a.ref_box) {
    ctx.setLineDash([4, 4]); path(boxPts(a.ref_box), true); ctx.stroke();
    const [x, y] = toScreen([a.ref_box[0], a.ref_box[1]]);
    ctx.fillStyle = color; ctx.font = "12px sans-serif"; ctx.fillText("参照", x + 4, y + 14);
  }
  ctx.setLineDash([]);
}

// 指针事件：鼠标和触摸通用
let lastTap = 0;
canvas.addEventListener("pointerdown", (e) => {
  if (!ann) return;
  canvas.setPointerCapture(e.pointerId);
  const p = toImage(e);
  const t = TOOLS.find((x) => x.id === tool);
  if (tool === "select") {
    dragCorner = -1;
    if (ann.corners) {
      let best = 1e9;
      ann.corners.forEach((c, i) => { const d = Math.hypot(c[0] - p[0], c[1] - p[1]) * scale; if (d < 24 && d < best) { best = d; dragCorner = i; } });
    }
    if (dragCorner >= 0) { pushHistory(); return; }
    selected = hitTest(p); renderAnns(); draw(); return;
  }
  if (drawing?.stage === "ref") { drawing.refStart = p; drawing.data.ref_box = [...p, ...p]; draw(); return; }
  if (t.kind === "box") { drawing = { type: tool, kind: "box", start: p, data: { box: [...p, ...p] } }; draw(); return; }
  if (t.kind === "line" || t.kind === "poly") {
    const now = Date.now();
    if (drawing && now - lastTap < 350) { lastTap = 0; finish(); return; }  // 双击结束
    lastTap = now;
    if (!drawing) {
      drawing = { type: tool, kind: t.kind, data: t.kind === "line" ? { points: [p] } : { polygon: [p] } };
      if (tool === "ink") drawing.data.color = $("ink-color").value;
      if (tool === "folds") drawing.data.width = 10;
    } else (drawing.data.points || drawing.data.polygon).push(p);
    renderTools(); draw();
  }
});
canvas.addEventListener("pointermove", (e) => {
  if (!ann || !(e.buttons & 1)) return;
  const p = toImage(e);
  if (tool === "select" && dragCorner >= 0) { ann.corners[dragCorner] = p; draw(); return; }
  if (drawing?.stage === "ref" && drawing.refStart) { drawing.data.ref_box = norm([...drawing.refStart, ...p]); draw(); return; }
  if (drawing?.kind === "box") { drawing.data.box = norm([...drawing.start, ...p]); draw(); }
});
canvas.addEventListener("pointerup", () => {
  if (tool === "select" && dragCorner >= 0) { dragCorner = -1; cornersNote(); return; }
  if (drawing?.stage === "ref" && drawing.refStart) {
    const [x0, y0, x1, y1] = drawing.data.ref_box;
    if (x1 - x0 > 4 && y1 - y0 > 4) commit(drawing.type, drawing.data); else toast("参照框太小，请再拖一次");
    return;
  }
  if (drawing?.kind === "box") {
    const [x0, y0, x1, y1] = drawing.data.box;
    if (x1 - x0 > 4 && y1 - y0 > 4) commit(drawing.type, drawing.data); else { drawing = null; draw(); }
  }
});
const norm = ([x0, y0, x1, y1]) => [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];

function finish() {
  if (!drawing) return;
  const pts = drawing.data.points || drawing.data.polygon;
  if (drawing.kind === "line" && pts.length < 2) return toast("至少点两个点");
  if (drawing.kind === "poly" && pts.length < 3) return toast("至少点三个点围成区域");
  if (drawing.type === "haze") { drawing.stage = "ref"; $("drawing").hidden = true; $("hint").textContent = "现在拖一个框，框住同类但没被雾盖住的地方，作为参照。"; draw(); return; }
  commit(drawing.type, drawing.data);
}
$("finish").onclick = finish;
$("cancel").onclick = () => { drawing = null; renderTools(); draw(); };
document.addEventListener("keydown", (e) => {
  if (view !== "edit" || e.target.matches("input,select")) return;
  if (e.key === "Enter") finish();
  if (e.key === "Escape") { drawing = null; renderTools(); draw(); }
  if ((e.key === "Delete" || e.key === "Backspace") && selected) { e.preventDefault(); removeAnn(selected.type, selected.i); }
  if ((e.metaKey || e.ctrlKey) && e.key === "z") { e.preventDefault(); undo(); }
});

function commit(type, data) {
  pushHistory();
  (ann[type] = ann[type] || []).push(data);
  drawing = null; renderTools(); renderAnns(); draw();
}
function pushHistory() { undoStack.push(JSON.stringify(ann)); if (undoStack.length > 50) undoStack.shift(); }
function undo() {
  if (drawing) { drawing = null; renderTools(); draw(); return; }
  if (!undoStack.length) return toast("没有可撤销的操作");
  ann = JSON.parse(undoStack.pop()); selected = null; initEditor();
}
$("undo").onclick = undo;
$("clear").onclick = () => {
  pushHistory();
  for (const t of TOOLS.slice(1)) ann[t.id] = [];
  selected = null; renderAnns(); draw(); toast("已清除全部标注，可以撤销");
};

function hitTest(p) {
  const tol = 10 / scale;
  let hit = null;
  for (const t of TOOLS.slice(1)) {
    (ann[t.id] || []).forEach((a, i) => {
      const inBox = (b) => b && p[0] >= b[0] - tol && p[0] <= b[2] + tol && p[1] >= b[1] - tol && p[1] <= b[3] + tol;
      const nearLine = (pts) => pts && pts.some((q, j) => j > 0 && segDist(p, pts[j - 1], q) < tol);
      const inPoly = (pts) => pts && pointInPoly(p, pts);
      if (inBox(a.box) || inBox(a.ref_box) || nearLine(a.points) || inPoly(a.polygon)) hit = { type: t.id, i };
    });
  }
  return hit;
}
function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], l = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}
function pointInPoly(p, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function renderAnns() {
  const items = [];
  for (const t of TOOLS.slice(1)) (ann[t.id] || []).forEach((a, i) => items.push({ type: t.id, i, a }));
  const counts = {};
  $("anns").innerHTML = items.map(({ type, i, a }) => {
    counts[type] = (counts[type] || 0) + 1;
    const extra = type === "ink" ? (a.color === "blue" ? "（蓝）" : "（红）") : "";
    const sel = selected?.type === type && selected.i === i;
    return `<li class="${sel ? "sel" : ""}"><i style="background:${COLOR[type]}"></i><span data-t="${type}" data-i="${i}">${LABEL[type]} ${counts[type]}${extra}</span><button type="button" data-del-t="${type}" data-del-i="${i}">删除</button></li>`;
  }).join("");
}
$("anns").addEventListener("click", (e) => {
  const d = e.target.closest("[data-del-t]");
  if (d) return removeAnn(d.dataset.delT, +d.dataset.delI);
  const s = e.target.closest("[data-t]");
  if (s) { selected = { type: s.dataset.t, i: +s.dataset.i }; renderAnns(); draw(); }
});
function removeAnn(type, i) { pushHistory(); ann[type].splice(i, 1); selected = null; renderAnns(); draw(); }

// 四角
function cornersNote() {
  $("corners-note").textContent = ann.corners
    ? "修复时会按这四个角把照片拉正，框外的内容（相册膜、桌面、手机边框）会被裁掉。"
    : "不做透视校正，整张图原样修复。适合本来就是正面拍摄、铺满画面的照片。";
}
$("corners-auto").onclick = () => {
  if (!current.auto_corners) return toast("没有自动检测到照片边缘，请用「手动拖四角」");
  pushHistory(); ann.corners = structuredClone(current.auto_corners); cornersNote(); draw(); toast("已恢复自动检测的四角");
};
$("corners-full").onclick = () => {
  pushHistory();
  const w = img.naturalWidth, h = img.naturalHeight, m = Math.round(Math.min(w, h) * 0.06);
  if (!ann.corners) ann.corners = [[m, m], [w - m, m], [w - m, h - m], [m, h - m]];
  tool = "select"; renderTools(); cornersNote(); draw(); toast("用「选择」工具把四个圆点拖到照片的四个角");
};
$("corners-none").onclick = () => { pushHistory(); ann.corners = null; cornersNote(); draw(); };

// 选项与裁边
for (const k of ["top", "bottom", "left", "right"]) $("trim-" + k).addEventListener("change", (e) => { ann.trim[k] = Math.max(0, +e.target.value || 0); });
for (const k of ["digital", "screen_reflection", "stains", "veil", "auto_glare", "auto_scratch"]) $("opt-" + k).addEventListener("change", (e) => { ann.options[k] = e.target.checked; });

$("run").onclick = async () => {
  if (drawing) return toast("还有一条标注没画完，先点「完成这一条」或「放弃」");
  try {
    await api(`/api/photos/${current.id}/ann`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(ann) });
    await api(`/api/photos/${current.id}/restore`, { method: "POST" });
    current.status = "queued"; current.ann = structuredClone(ann);
    view = "compare"; showMain(); toast("已保存，开始重新修复"); refresh();
  } catch (e) { toast(e.message, 5000); }
};

// ------------------------------------------------------------------ 启动
(async () => {
  await refresh();
  const [id, v] = location.hash.slice(1).split("/");
  if (id && photos.some((p) => p.id === id)) open(id, v === "edit" ? "edit" : "compare");
  else if (photos.length) open(photos[0].id, "compare");  // 默认打开最新的一张
  setInterval(refresh, 2000);
})();
