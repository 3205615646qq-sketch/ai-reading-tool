# -*- coding: utf-8 -*-
"""
第三课 · 本地读数 read_one.py

模型只负责「看见」：检测给数字锚点，姿态给刻度线与叉丝的位置。
数值由本文件的几何协议算出：k 全局、b 每图、残差超限则拒读。

用法：
    python read_one.py --image "D:\\AII\\数据_clean\\IMG_0001.jpg"
    python read_one.py --image "D:\\AII\\数据_clean"          # 目录则批量
    python read_one.py --image xxx.jpg --det runs\\det_full\\weights\\best.pt

依赖：ultralytics、opencv-python（可视化用，缺失时跳过绘图）
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- 常量区
# 权重路径：优先按「脚本所在目录」找，这样整个文件夹拷到哪里都能跑
ROOT = Path(__file__).resolve().parent


def _resolve(p):
    p = Path(p)
    if not p.is_absolute() and not p.exists() and (ROOT / p).exists():
        return str(ROOT / p)
    return str(p)


DET_WEIGHTS = _resolve(r"runs\det_full\weights\best.pt")
POSE_WEIGHTS = _resolve(r"runs\pose_full\weights\best.pt")

# 全局斜率 mm/px。负号必须保留：望远镜成倒像，y 增大而刻度值减小。
# 由 check_protocol.py 在全量标注上标定：k = -(10 mm) / 平均刻度间距(px)
K_MM_PER_PX = -0.02388

REL_RESID_MAX = 0.015    # 相对残差阈值 = 最大残差 / 锚点跨度。判据是「斜率一致性」，
                         # 不是绝对毫米数：跨度越大，同一比例误差产生的绝对残差越大。
                         # 标定依据（全量 180 张标注，check_protocol.py --rel）：
                         #   跨度 10 mm 的 130 张 绝对残差 p95 = 0.096 mm，相对 p95 = 0.96%
                         #   跨度 20 mm 的  50 张 绝对残差 p95 = 0.324 mm，相对 p95 = 1.62%
                         # 绝对残差随跨度放大 3 倍，相对残差却同量级 → 只有相对量可比。
                         # 全量相对残差 p95 = 1.0%，取 1.5 倍余量。
RESIDUAL_MAX_MM = 0.10   # 仅用于报告里的「参考绝对量」（跨 10 mm 时约等于上述阈值），
                         # 不参与拒读判定。早期版本用它当阈值，会把跨度 20 mm 的图误杀。
REPORT_STEP_MM = 0.1     # 报告精度，与人工估读对齐
DIGIT_MM = 10.0          # 类别 num_i 对应 i*10 mm
TICK_MM = 10.0           # 相邻刻度线间距 10 mm（整厘米）
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 锚点取法：
#   "tick" —— 数字框只用来认出「这是几」，位置取该数字对应的刻度线 y（推荐）
#   "box"  —— 直接用数字框中心 y（讲义原写法，简单但带字形偏移）
# 依据：全量标注上，框中心相对对应刻度线中位偏 7.3 px（0.17 mm），
# 且同一刻度两个数字框彼此还能差 6 px 以上；换成刻度线后残差中位从 0.226 降到 0.037 mm。
ANCHOR_MODE = "tick"
ANCHOR_TOL_PX = 60.0     # 数字框与刻度线的最大配对距离，约 1.4 mm；超出则退回框中心
RESID_CONF_MIN = 0.60    # 参与「残差拒读」判定的锚点最低置信度；低于此只参与定标不参与判定



# ---------------------------------------------------------------- 协议本体
def _parts(a):
    """锚点兼容两种写法：(y, v) 或 (y, v, conf)"""
    return (a[0], a[1], a[2]) if len(a) >= 3 else (a[0], a[1], 1.0)


def fit_intercept(k, anchors):
    """
    b = Σw(vᵢ − k·yᵢ) / Σw ：k 已知，只解截距，按锚点置信度加权。
    低置信度的刻度线照样参与定标（信息总比丢弃多），但权重更低。
    """
    num = den = 0.0
    for a in anchors:
        y, v, w = _parts(a)
        w = max(w, 1e-3)
        num += w * (v - k * y)
        den += w
    return num / den


def check_residual(k, b, anchors, conf_min=RESID_CONF_MIN):
    """
    逐锚点检验，返回 (最大残差 mm, 残差明细 [(y, v, resid, conf, 是否参与判定)])。
    只把置信度 ≥ conf_min 的锚点计入「最大残差」——低置信度刻度线本身就不准，
    拿它当拒读依据会误杀好图（实测：糊图顶部刻度 conf 0.54 时残差虚高 0.12 mm）。
    若可信锚点不足 2 个，退回全部锚点做判定，不放松门槛。
    """
    detail = []
    for a in anchors:
        y, v, w = _parts(a)
        detail.append((y, v, abs(k * y + b - v), w, w >= conf_min))
    used = [d[2] for d in detail if d[4]]
    if len(used) < 2:
        used = [d[2] for d in detail]
        detail = [(y, v, r, w, True) for y, v, r, w, _ in detail]
    return max(used), detail


def combine(groups, k, rel_max=REL_RESID_MAX):
    """
    协议三步一次走完，返回 dict 或拒读原因。
    groups = {"anchors": [(y, v)], "ticks": [y], "cross": y or None}

    rel_max：相对残差上限（残差 / 锚点跨度）。比绝对毫米阈值更稳：
    同一张图的成像比例误差，在 10 mm 跨度上只产生 0.05 mm 残差，
    在 20 mm 跨度上就是 0.10 mm —— 用绝对值判定等于对量程大的图更严。
    """
    anchors, ticks, y_cross = groups["anchors"], groups["ticks"], groups["cross"]

    # ① 数量门槛：不足就拒读，绝不进入拟合（否则除零 / NaN）
    if y_cross is None:
        return {"ok": False, "reason": "无叉丝：没有被测量"}
    if len(anchors) < 2:
        return {"ok": False, "reason": f"锚点仅 {len(anchors)} 个：无法定标"}
    if len(ticks) < 2:
        return {"ok": False, "reason": f"刻度线仅 {len(ticks)} 条：无法校验斜率"}

    # ② 由锚点解 b，再逐锚点检验
    #    只用「可信锚点」定标：低置信度刻度线自己就不准，让它进拟合会把 b 拽偏，
    #    连带把好锚点的残差也顶上去（实测会误杀 0.12 mm 量级的正常图）。
    trusted = [a for a in anchors if _parts(a)[2] >= RESID_CONF_MIN]
    fit_set = trusted if len(trusted) >= 2 else anchors
    b = fit_intercept(k, fit_set)
    r_max, detail = check_residual(k, b, anchors)

    # 跨度取「参与判定的锚点」的值极差，与 r_max 同源，保证分母对得上
    used_v = [d[1] for d in detail if d[4]]
    span = (max(used_v) - min(used_v)) if len(used_v) >= 2 else 0.0
    rel = (r_max / span) if span > 0 else float("inf")

    out = {"ok": True, "k": k, "b": b, "resid_max": r_max,
           "span_mm": span, "rel_resid": rel,
           "detail": detail, "y_cross": y_cross,
           "anchors": anchors, "ticks": ticks}
    if rel > rel_max:
        out.update(ok=False,
                   reason=(f"相对残差 {rel * 100:.2f}% 超阈值 {rel_max * 100:.2f}%"
                           f"（最大 {r_max:.3f} mm / 跨度 {span:.0f} mm）"))
        return out

    # ③ 通过后才计算读数
    value = k * y_cross + b
    out["value_raw"] = value
    out["value"] = round(value / REPORT_STEP_MM) * REPORT_STEP_MM
    return out


# ---------------------------------------------------------------- 推理
def load_models(det_path, pose_path):
    from ultralytics import YOLO
    return YOLO(det_path), YOLO(pose_path)


def build_anchors(digits, ticks, mode=ANCHOR_MODE, tol=ANCHOR_TOL_PX):
    """
    数字框 → 锚点 (y, 刻度值, 置信度)。
    mode="tick"：位置换成该数字对应的刻度线 y，消掉字形相对刻度线的偏移；
    mode="box" ：直接用框中心 y。
    找不到近邻刻度线时退回框中心，并在返回的 warn 里计数。
    ticks 可以是 [y] 或 [(y, conf)]。
    """
    tk = [(t[0], t[1]) if isinstance(t, (tuple, list)) else (t, 1.0) for t in ticks]
    anchors, warn = [], 0
    for item in digits:
        y_box, value = item[0], item[1]
        d_conf = item[2] if len(item) >= 3 else 1.0
        y, conf = y_box, d_conf
        if mode == "tick" and tk:
            near, near_conf = min(tk, key=lambda t: abs(t[0] - y_box))
            if abs(near - y_box) <= tol:
                y, conf = near, near_conf
            else:
                warn += 1
        anchors.append((y, value, conf))

    # 同一刻度值可能有两个数字（双侧读数窗），按值合并：
    # 位置按置信度加权平均，置信度取最大 —— 避免同一刻度线被重复计数
    by_val = {}
    for y, v, c in anchors:
        by_val.setdefault(v, []).append((y, c))
    merged = []
    for v, items in sorted(by_val.items()):
        wsum = sum(c for _, c in items)
        y = sum(y * c for y, c in items) / wsum if wsum else sum(y for y, _ in items) / len(items)
        merged.append((y, v, max(c for _, c in items)))
    return merged, warn


def infer_groups(det_model, pose_model, img_path):
    """两个模型各推理一次，把结果整理成协议需要的 groups"""
    det = det_model(img_path, verbose=False)[0]
    digits = []
    for cls, box, conf in zip(det.boxes.cls.tolist(), det.boxes.xyxy.tolist(),
                              det.boxes.conf.tolist()):
        y_center = (box[1] + box[3]) / 2
        digits.append((y_center, (int(cls) + 1) * DIGIT_MM, float(conf)))

    pose = pose_model(img_path, verbose=False)[0]
    ticks, cross = [], None
    for cls, kps, conf in zip(pose.boxes.cls.tolist(),
                              pose.keypoints.xy.tolist(),
                              pose.boxes.conf.tolist()):
        ys = [p[1] for p in kps[:2]]
        if any(y <= 0 for y in ys):      # 端点未检出
            continue
        y = sum(ys) / 2                  # 两关键点均值，不用框中心
        if int(cls) == 0:
            ticks.append((y, conf))
        elif cross is None or conf > cross[1]:
            # 同一半面里同一叉丝可能被检出多次，取置信度最高的那条，
            # 不能把几条不同位置的检测平均成一个中间值
            cross = (y, conf)

    # 刻度线按邻近去重（相邻刻度线相距约 420 px，60 px 内视为同一条）
    kept = []
    for y, conf in sorted(ticks, key=lambda t: -t[1]):
        if all(abs(y - k) > 60 for k, _ in kept):
            kept.append((y, conf))
    ticks = sorted(k for k, _ in kept)
    y_cross = cross[0] if cross else None

    anchors, warn = build_anchors(digits, kept)
    return {"anchors": anchors, "digits": digits, "ticks": ticks,
            "cross": y_cross, "warn": warn}


# ---------------------------------------------------------------- 可视化
def _imread_unicode(path):
    """cv2.imread 读不了中文路径，绕道 imdecode"""
    import cv2
    import numpy as np
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def _imwrite_unicode(path, img):
    """把图像编码落盘，返回实际写入的路径。

    不用 ndarray.tofile()：目标文件被看图软件/预览窗占用时它会抛
    OSError(22, Invalid argument)，还会把整个读数流程带崩。
    这里改用 open(wb) 写入，占用则自动换名重试。
    """
    import cv2
    p = Path(path)
    ok, buf = cv2.imencode(p.suffix or ".png", img)
    if not ok:
        return None
    blob = buf.tobytes()
    cands = [p,
             p.with_name(f"{p.stem}_2{p.suffix}"),
             p.with_name(f"{p.stem}_3{p.suffix}")]
    for cand in cands:
        try:
            cand.parent.mkdir(parents=True, exist_ok=True)
            with open(cand, "wb") as f:
                f.write(blob)
            return cand
        except OSError:
            continue
    return None


def _open_path(path):
    """用系统默认程序打开图片或文件夹；失败时静默返回 False（不影响读数）"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))            # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        return True
    except Exception:
        return False


def draw(img_path, res, out_path):
    """画标注图并返回实际写入的路径；任何异常都不影响读数结果"""
    try:
        return _draw_impl(img_path, res, out_path)
    except Exception:                          # noqa: BLE001
        return None


def _draw_impl(img_path, res, out_path):
    try:
        import cv2
    except ImportError:
        return None
    img = _imread_unicode(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    k, b = res["k"], res["b"]

    for a in res["anchors"]:                          # 锚点
        y, v = a[0], a[1]
        low = len(a) >= 3 and a[2] < RESID_CONF_MIN
        cv2.circle(img, (int(w * 0.5), int(y)), 7,
                   (0, 165, 255) if low else (0, 0, 255), 2)
        label = f"{v:.0f}mm  y={y:.1f}" + ("  低置信" if low else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0, y0 = int(w * 0.52), int(y) - 30
        cv2.rectangle(img, (x0 - 6, y0 - 2), (x0 + tw + 6, y0 + th + 8), (30, 30, 30), -1)
        cv2.putText(img, label, (x0, y0 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for y in res["ticks"]:                            # 刻度线
        cv2.line(img, (0, int(y)), (w, int(y)), (0, 200, 0), 1)
    yc = int(res["y_cross"])                          # 叉丝
    cv2.line(img, (0, yc), (w, yc), (255, 128, 0), 3)
    for v in range(0, 61, 5):                         # 每 5 mm 一条标尺线
        y = int((v - b) / k)
        if 0 <= y < h:
            cv2.line(img, (w - 60, y), (w, y), (200, 200, 200), 1)
    txt = (f"read = {res['value']:.1f} mm   b={b:.2f}   "
           f"rel_resid={res['rel_resid'] * 100:.2f}% ({res['resid_max']:.3f} mm/{res['span_mm']:.0f} mm)")
    cv2.rectangle(img, (0, 0), (w, 46), (30, 30, 30), -1)
    cv2.putText(img, txt, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _imwrite_unicode(out_path, img)


# ---------------------------------------------------------------- 主流程
def report(name, res, out_path, warn=0, shown=False):
    print(f"\n===== {name} =====")
    if warn:
        print(f"提示 : {warn} 个数字框在 {ANCHOR_TOL_PX:.0f} px 内没找到对应刻度线，已退回框中心")
    if not res["ok"]:
        print(f"拒读 : {res['reason']}")
        if "resid_max" in res:
            print(f"截距 : b = {res['b']:.2f} mm   残差 : 最大 {res['resid_max']:.3f} mm"
                  f" / 跨度 {res.get('span_mm', 0):.0f} mm")
            print("      残差普遍偏大时先查标注与刻度线端点，不要先改阈值")
        return False
    parts = []
    for a in res["anchors"]:
        y, v = a[0], a[1]
        conf = a[2] if len(a) >= 3 else 1.0
        parts.append(f"y={y:.1f}→{v:.0f}mm" + (f"(conf {conf:.2f})" if conf < RESID_CONF_MIN else ""))
    print(f"锚点 : {'    '.join(parts)}")
    print(f"刻度 : {', '.join(f'{y:.1f}' for y in res['ticks'])}  （{len(res['ticks'])} 条）")
    print(f"斜率 : k = {res['k']:.5f} mm/px  （全局）")
    print(f"截距 : b = {res['b']:.2f} mm")
    print(f"残差 : 最大 {res['resid_max']:.3f} mm / 跨度 {res['span_mm']:.0f} mm"
          f" = {res['rel_resid'] * 100:.2f}%  → 通过（上限 {REL_RESID_MAX * 100:.1f}%）")
    print(f"叉丝 : y = {res['y_cross']:.1f}")
    print(f"读数 : {res['value']:.1f} mm  （计算值 {res['value_raw']:.2f}）")
    if out_path:
        print(f"可视化 : {out_path}" + ("   ← 已打开" if shown else ""))
    return True


def main():
    ap = argparse.ArgumentParser(description="第三课 · 读数协议")
    ap.add_argument("--image", required=True, help="图片路径或目录")
    ap.add_argument("--det", default=DET_WEIGHTS)
    ap.add_argument("--pose", default=POSE_WEIGHTS)
    ap.add_argument("--k", type=float, default=K_MM_PER_PX)
    ap.add_argument("--rel-resid", type=float, default=REL_RESID_MAX,
                    help="相对残差上限（残差/锚点跨度），默认 0.015")
    ap.add_argument("--anchor", choices=["tick", "box"], default=ANCHOR_MODE,
                    help="锚点取对应刻度线(tick)还是数字框中心(box)")
    ap.add_argument("--vis-dir", default=str(ROOT / "vis"))
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--show", dest="show", action="store_true", default=True,
                    help="读数后自动打开可视化图（默认开启）")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="只打印可视化路径，不自动打开")
    ap.add_argument("--save-csv", default="", help="把逐图结果写到 csv，便于与人工读数比对")
    args = ap.parse_args()

    target = Path(args.image)
    if not target.exists():
        sys.exit(f"路径不存在: {target}")
    if target.is_dir():
        images = sorted(p for p in target.iterdir() if p.suffix.lower() in IMG_EXTS)
    else:
        images = [target]
    if not images:
        sys.exit("没有找到图片")

    det_model, pose_model = load_models(args.det, args.pose)   # 只加载一次
    vis_dir = Path(args.vis_dir)
    ok_n = refuse_n = 0
    rows = []
    total = len(images)
    single = total == 1
    last_vis = None
    for i, img in enumerate(images, 1):
        if total > 1:
            print(f"\n进度 {i}/{total}")
        groups = infer_groups(det_model, pose_model, img)
        if args.anchor != ANCHOR_MODE:                          # 命令行覆盖锚点取法
            groups["anchors"], groups["warn"] = build_anchors(
                groups["digits"], groups["ticks"], args.anchor)
        res = combine(groups, args.k, args.rel_resid)
        out = None
        if res["ok"] and not args.no_vis:
            out = draw(img, res, vis_dir / (img.stem + ".png"))
        shown = bool(out and single and args.show and _open_path(out))
        if out:
            last_vis = out
        if report(img.name, res, out, groups.get("warn", 0), shown):
            ok_n += 1
            rows.append((img.name, f"{res['value']:.1f}", f"{res['b']:.2f}",
                         f"{res['rel_resid'] * 100:.2f}", "通过"))
            if res["ok"] and not args.no_vis and out is None:
                print("可视化 : 跳过（写入失败，多半是这张图正被看图软件占用）")
        else:
            refuse_n += 1
            rows.append((img.name, "", "", f"{res.get('rel_resid', 0) * 100:.2f}", res["reason"]))

    if args.save_csv:
        p = Path(args.save_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8-sig") as f:
            f.write("image,read_mm,b_mm,rel_resid_pct,status\n")
            for r in rows:
                f.write(",".join(r) + "\n")
        print(f"\n明细已写入 {p}")

    if len(images) > 1:
        print(f"\n合计 {len(images)} 张：读数 {ok_n} 张，拒读 {refuse_n} 张")
        if last_vis and args.show:                 # 批量没法一次弹一百张，直接开目录
            print(f"可视化目录 : {vis_dir}")
            _open_path(vis_dir)


if __name__ == "__main__":
    main()
