"""虹膜遮蔽度 -> PERCLOS / 眨眼率 / 长闭眼时长 的纯计算核心。

本模块只依赖 numpy，不 import cv2 / mediapipe，因此事件判定逻辑可以用合成信号在
本地离线验证（见 tests/test_eye_metrics_core.py），无需接触远程数据。

为什么用虹膜遮蔽而不是 EAR 幅度比
----------------------------------
EAR 路线的闭合度要除以"该被试全开时的开合度"，而关键点模型在闭眼时不会把下睑点
一路推到上睑点上，实测动态范围被压到 0~0.56，经典 PERCLOS 阈值根本够不到。
本实现改用教科书定义：眼睑遮住了虹膜垂直高度的多少。分母是虹膜直径——它在一个人
身上是解剖常数，且用**水平**直径估计，因为上下眼睑只遮蔽虹膜的垂直方向，不影响
水平方向。由此得到两个好处：
  1. 量程是真实的 0~1，P50/P80 阈值有意义；
  2. 不需要任何"全开参考值"，因此不需要用 alert 段定标——操纵检验的循环性问题消失。

三条贯穿全部指标的方法学约定
----------------------------
1. 所有比率指标的分母一律是"有效时长"而非"总时长"。被试在 sleepy 段若脸被遮挡或
   头部偏出更多，用总时长当分母会把 PERCLOS 系统性压低。
2. 条件水平的聚合是"分子分母各自求和"，不是"逐切片指标求均值"，避免比值的均值
   偏倚；同时长切片不会因数量差异被过度加权。
3. 索引错位一律"报告 + 可视化复核"，不加几何门控去拦截。本项目实测过：拿未核实的
   假设做过滤器，会把可疑数字变成零输出，更难查。
"""

import numpy as np

# 眼睑/眼角索引：在真实帧上用 --index_map 标定并人工复核过，不是凭记忆写的。
# 换数据集、换摄像头布局或换 mediapipe 版本后，先重跑 --index_map 复核再全量。
EYE_LANDMARKS = {
    "r": {"upper": (159, 160, 158), "lower": (145, 144, 153), "corner_a": 33, "corner_b": 133},
    "l": {"upper": (386, 387, 385), "lower": (374, 373, 380), "corner_a": 263, "corner_b": 398},
}

# 虹膜索引：需 FaceMesh(refine_landmarks=True)，共 478 点，468-477 为两眼虹膜各 5 点。
# 这组数字同样未经官方文档逐条核实，因此代码对它的用法刻意做到"不猜角色"：
#   - 不假设哪个点是圆心（取 5 点均值，且只用到相对量）
#   - 不假设哪个点是上下左右（水平半径取 |在眼轴上的投影| 的最大值）
# 唯一可能出错的是"这 5 个索引是否落在同一只眼的虹膜上"，由 iris_d/eye_w 比值
# 的合理性 + --dump_overlay 的虹膜描点共同把关。
EYE_IRIS_LANDMARKS = {"r": (468, 469, 470, 471, 472), "l": (473, 474, 475, 476, 477)}


def default_config():
    """全部可调阈值集中于此，并随结果一起写出，保证判据可复现。"""
    return {
        "fs": 30.0,
        # 逐帧有效性门控
        "width_ratio_lo": 0.50,     # 单帧眼裂宽度 / 被试中位数 的可接受区间
        "width_ratio_hi": 2.00,
        "iris_d_ratio_lo": 0.22,    # 虹膜直径 / 眼裂宽度 的生理合理区间
        "iris_d_ratio_hi": 0.70,    # （真人睑裂宽约 28mm、虹膜直径约 11.7mm -> ~0.42）
        "occ_min": -0.25,           # 遮蔽度下限：略负值容许（眼睑跨度大于虹膜属正常）
        "occ_max": 1.05,            # 上限：构造上 visible>=0 使 occ<=1，越界即关键点崩坏
        "sym_tol": 0.25,            # 双眼遮蔽度差上限（Hering 定律：双眼同步闭合）
        "width_parity_lo": 0.60,    # 左右眼尺度比的报告区间（只报告，不拦截）
        "width_parity_hi": 1.65,
        # 事件判定
        # P50 为主判据：遮蔽虹膜垂直高度一半以上即计为闭。PERCLOS 文献常用 P50/P70/P80
        # 不等，这里同时给出 0.40/0.60 作敏感性档，避免结论依赖单一阈值。
        "perclos_thresholds": (0.50, 0.40, 0.60),   # 首项为主判据
        "blink_min_dur_s": 0.10,    # 文献眨眼时程下限，仅作一档下限，不作隐藏过滤
        "blink_max_dur_s": 0.50,    # 超过则归入长闭眼，两类事件互斥不重复计数
        "blink_tier_ms": (100, 150, 200),   # 30 fps 下短眨眼被系统性漏计，须整组报告
        "closure_min_dur_s": 0.50,  # 长闭眼（方案中"微睡眠"的行为学代理）
        "closure_tier_s": (0.50, 1.00, 2.00),
        # 可行性判定（--dry_run 用）
        "min_eye_width_px": 18.0,   # 眼裂宽度中位数低于此 -> 建议放弃视频路线
        "min_valid_ratio": 0.80,    # 对齐方案 §12.1 的"有效追踪占比 >=80%"
        "min_occ_dyn_range": 0.35,  # 遮蔽度 p95-p05 的相对动态范围下限
        "min_iris_frame_share": 0.50,   # 虹膜几何可用的帧占比下限，低于此说明虹膜索引错
    }


# ── 逐帧几何 ────────────────────────────────────────────────────────
def eye_landmarks(points, side):
    """取某只眼的上/下眼睑点与内外眼角。索引越界时返回 None。"""
    spec = EYE_LANDMARKS[side]
    n = len(points)
    idx_u = [i for i in spec["upper"] if i < n]
    idx_l = [i for i in spec["lower"] if i < n]
    if not idx_u or not idx_l or spec["corner_a"] >= n or spec["corner_b"] >= n:
        return None, None, None, None
    return points[idx_u], points[idx_l], points[spec["corner_a"]], points[spec["corner_b"]]


def iris_points(points, side):
    """取该眼的 5 个虹膜点。索引越界（未开 refine_landmarks，只有 468 点）时返回 None。"""
    idx = [i for i in EYE_IRIS_LANDMARKS[side] if i < len(points)]
    if len(idx) < 5:
        return None
    return np.asarray(points[idx], dtype=float)


def eye_aperture(points, side):
    """经典 EAR：上下眼睑配对点的欧氏垂直距离均值 / 两眼角的欧氏距离。

    与 dlib 68 点版 EAR 同形——(‖p2−p6‖+‖p3−p5‖)/(2·‖p1−p4‖) 即"成对垂直距离的
    均值 ÷ 眼角间距"，按标定点数推广为 N 对；分子分母都用欧氏距离，配对按横坐标
    排序，故不依赖 EYE_LANDMARKS 的书写顺序。
    这里作为副信号保留（与上一轮 EAR 结果可比），主指标走虹膜遮蔽度。
    """
    up, lo, ca, cb = eye_landmarks(points, side)
    if up is None:
        return float("nan"), float("nan")
    d_w = float(np.linalg.norm(ca - cb))
    if not np.isfinite(d_w) or d_w <= 1e-6:
        return float("nan"), float("nan")
    u = up[np.argsort(up[:, 0])]
    l = lo[np.argsort(lo[:, 0])]
    n = min(len(u), len(l))
    d_v = np.linalg.norm(u[:n] - l[:n], axis=1)
    d_v = d_v[np.isfinite(d_v)]
    if d_v.size == 0:
        return float("nan"), d_w
    return float(np.mean(d_v)) / d_w, d_w


def iris_geometry(points, side):
    """虹膜遮蔽所需的逐帧几何量，全部在"眼坐标系"里表达。

    眼坐标系：u 沿两眼角连线（眼轴），v 与之垂直，原点取虹膜 5 点的均值。
    返回 (ap_lo, ap_hi, iris_d, eye_w)：
      ap_lo/ap_hi  上下眼睑在 v 轴上的跨度两端（不假设图像 y 轴朝向，故取 min/max）
      iris_d       虹膜水平直径 = 2·max|投影到 u|。用水平方向是因为上下眼睑只遮蔽
                   虹膜的垂直部分，水平直径不受闭合程度影响，是稳定尺度
      eye_w        两眼角欧氏距离（眼裂宽度）
    不可用时返回四个 nan，绝不返回一个看起来合理的假值。
    """
    up, lo, ca, cb = eye_landmarks(points, side)
    iris = iris_points(points, side)
    if up is None or iris is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    eye_w = float(np.linalg.norm(cb - ca))
    if not np.isfinite(eye_w) or eye_w <= 1e-6:
        return float("nan"), float("nan"), float("nan"), eye_w
    u = (cb - ca) / eye_w
    v = np.array([-u[1], u[0]])
    c = iris.mean(axis=0)
    pu = (iris - c) @ u
    pv = (iris - c) @ v
    iris_d = 2.0 * float(np.max(np.abs(pu)))
    lid_v = np.concatenate([(up - c) @ v, (lo - c) @ v])
    lid_v = lid_v[np.isfinite(lid_v)]
    if lid_v.size < 2 or not np.isfinite(iris_d) or iris_d <= 1e-6:
        return float("nan"), float("nan"), float("nan"), eye_w
    return float(np.min(lid_v)), float(np.max(lid_v)), iris_d, eye_w


def scale_parity_stats(a, b, cfg):
    """两侧尺度比的分布——只报告，不拦截。左右眼差异过大即提示某一侧索引错位。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / np.asarray(b, dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        return {"ratio_median": float("nan"), "ratio_bad_share": float("nan")}
    bad = float(np.mean((r < cfg["width_parity_lo"]) | (r > cfg["width_parity_hi"])))
    return {"ratio_median": float(np.median(r)), "ratio_bad_share": bad}


# ── 遮蔽度与有效性 ──────────────────────────────────────────────────
def iris_signals(ap_lo, ap_hi, iris_d, eye_w, ref, cfg):
    """单眼逐帧几何 -> (虹膜遮蔽度, 有效标记)。

    ref 为被试级常数 dict：iris_d（虹膜直径中位数）、eye_w（眼裂宽度中位数）。
    遮蔽度 o = 1 - 可见虹膜垂直高度 / 虹膜垂直直径，o=0 虹膜全可见、o=1 全被遮住。
    虹膜垂直直径用水平直径中位数代替（虹膜近似圆，且水平方向不受遮蔽影响）。
    """
    ap_lo = np.asarray(ap_lo, dtype=float)
    ap_hi = np.asarray(ap_hi, dtype=float)
    iris_d = np.asarray(iris_d, dtype=float)
    eye_w = np.asarray(eye_w, dtype=float)
    ok = np.isfinite(ap_lo) & np.isfinite(ap_hi) & np.isfinite(iris_d) & np.isfinite(eye_w)
    r_d = float(ref.get("iris_d", float("nan")))
    med_w = float(ref.get("eye_w", float("nan")))
    if not np.isfinite(r_d) or r_d <= 1e-6 or not np.isfinite(med_w) or med_w <= 1e-6:
        return np.full(ap_lo.shape, np.nan), ok
    with np.errstate(invalid="ignore"):
        half = 0.5 * r_d
        visible = np.clip(np.minimum(ap_hi, half) - np.maximum(ap_lo, -half), 0.0, 2.0 * half)
        occ = np.where(ok, 1.0 - visible / (2.0 * half), np.nan)
        ok_w = (eye_w >= cfg["width_ratio_lo"] * med_w) & (eye_w <= cfg["width_ratio_hi"] * med_w)
        ok_i = (iris_d / eye_w >= cfg["iris_d_ratio_lo"]) & (iris_d / eye_w <= cfg["iris_d_ratio_hi"])
        ok_r = (occ >= cfg["occ_min"]) & (occ <= cfg["occ_max"])
        ok = ok & ok_w & ok_i & ok_r
        occ = np.where(ok, np.clip(occ, 0.0, 1.0), np.nan)
    return occ, ok


def combine_eyes(occ_r, occ_l, ok_r, ok_l, cfg):
    """双眼共识：两眼都有效且遮蔽度接近才保留，取均值降噪。"""
    ok = ok_r & ok_l
    with np.errstate(invalid="ignore"):
        asym = np.abs(occ_r - occ_l)
    ok = ok & np.isfinite(asym) & (asym <= cfg["sym_tol"])
    occ = np.where(ok, 0.5 * (np.nan_to_num(occ_r) + np.nan_to_num(occ_l)), np.nan)
    return occ, ok


def median_scale(arrays):
    """被试级尺度常数：把若干一维数组的有限值 pooled 后取中位数。

    只用中位数这类与条件无关的量，不使用任何专注/低专注标签，因此 PERCLOS 的条件间
    差异不会被定标过程定义掉，也不引入"用 alert 段定尺子"的循环性。
    """
    vals = [np.asarray(a, dtype=float) for a in arrays if a is not None and len(a)]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if v.size]
    return float(np.median(np.concatenate(vals))) if vals else float("nan")


# ── 事件化 ──────────────────────────────────────────────────────────
def _runs(mask):
    """连续 True 段的 [起, 止) 索引。"""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def closure_events(occ, ok, fs, cfg):
    """达到主判据阈值的闭合事件，按时间顺序返回。

    事件时长取 n_frames/fs。30 fps 下至多低估 1 帧（期望 0.5 帧），对 >=500 ms 分档
    影响可忽略，但对 100 ms 级眨眼不可忽略——故眨眼率必须按多档下限整组报告。
    事件不跨无效帧合并：被无效帧隔断的两段算两个事件。
    """
    o = np.nan_to_num(np.asarray(occ, dtype=float))
    closed = (o >= cfg["perclos_thresholds"][0]) & np.asarray(ok, dtype=bool)
    ev = []
    for a, b in _runs(closed):
        seg = o[a:b]
        ev.append({"start_s": a / fs, "end_s": b / fs, "dur_s": (b - a) / fs,
                   "max_occ": float(np.max(seg)) if seg.size else float("nan")})
    return ev


def split_events(events, cfg):
    """眨眼与长闭眼按时长互斥分档。"""
    blinks = [e for e in events if e["dur_s"] < cfg["blink_max_dur_s"]]
    closures = [e for e in events if e["dur_s"] >= cfg["closure_min_dur_s"]]
    return blinks, closures


def perclos(occ, ok, fs, threshold):
    """遮蔽度 >= threshold 的时间占有效时间之比，返回 (占比, 闭合秒, 有效秒)。"""
    o = np.nan_to_num(np.asarray(occ, dtype=float))
    ok = np.asarray(ok, dtype=bool)
    valid_s = ok.sum() / fs
    closed_s = ((o >= threshold) & ok).sum() / fs
    return (closed_s / valid_s if valid_s > 0 else float("nan")), closed_s, valid_s


# ── 聚合 ────────────────────────────────────────────────────────────
def summarize_segment(occ, ok, fs, cfg):
    """一个连续记录单元（一段完整视频或一个切片）的指标。"""
    ev = closure_events(occ, ok, fs, cfg)
    blinks, _ = split_events(ev, cfg)
    o = np.nan_to_num(np.asarray(occ, dtype=float))
    out = {
        "n_frames": int(len(occ)),
        "valid_s": float(np.asarray(ok, dtype=bool).sum() / fs),
        "n_event": len(ev),
        "n_blink_lt500ms": len(blinks),
        "n_closure_ge500ms": len(ev) - len(blinks),
        "closure_dur_ge500ms_s": float(sum(e["dur_s"] for e in ev
                                           if e["dur_s"] >= cfg["closure_min_dur_s"])),
        "mean_dur_s": float(np.mean([e["dur_s"] for e in ev])) if ev else float("nan"),
        "max_dur_s": float(np.max([e["dur_s"] for e in ev])) if ev else 0.0,
        "mean_occ": float(np.mean(o[ok])) if np.any(ok) else float("nan"),
        "p95_occ": float(np.quantile(o[ok], 0.95)) if np.any(ok) else float("nan"),
    }
    for t in cfg["perclos_thresholds"]:
        p, cs, _ = perclos(occ, ok, fs, t)
        out[f"perclos_{int(round(t * 100))}"] = p
        out[f"closed_s_{int(round(t * 100))}"] = cs
    # 把 PERCLOS 拆成"眨眼贡献"与"长闭眼贡献"，判断条件间差异来自哪一侧。
    # 事件按主判据定义，故对更松的敏感性档需按事件峰值再过滤一次，保证与被减的
    # closed_s_k 同一把尺子。
    for t in cfg["perclos_thresholds"]:
        k = int(round(t * 100))
        long_s = sum(e["dur_s"] for e in ev
                     if e["dur_s"] >= cfg["closure_min_dur_s"] and e["max_occ"] >= t)
        long_s = min(long_s, out[f"closed_s_{k}"])
        out[f"long_closed_s_{k}"] = long_s
        out[f"perclos_{k}_excl_long_closure"] = \
            (out[f"closed_s_{k}"] - long_s) / out["valid_s"] if out["valid_s"] > 0 else float("nan")
    out["blink_rate_lt500ms_per_min"] = \
        len(blinks) / (out["valid_s"] / 60.0) if out["valid_s"] > 0 else float("nan")
    for ms in cfg["blink_tier_ms"]:
        n = sum(1 for e in blinks if e["dur_s"] + 1e-9 >= ms / 1000.0)
        out[f"n_blink_ge{ms}ms"] = n
        out[f"blink_rate_ge{ms}ms_per_min"] = \
            n / (out["valid_s"] / 60.0) if out["valid_s"] > 0 else float("nan")
    for s in cfg["closure_tier_s"]:
        tag = int(round(s * 1000))
        sel = [e for e in ev if e["dur_s"] + 1e-9 >= s]
        out[f"n_closure_ge{tag}ms"] = len(sel)
        out[f"closure_dur_ge{tag}ms_s"] = float(sum(e["dur_s"] for e in sel))
        out[f"closure_rate_ge{tag}ms_per_min"] = \
            len(sel) / (out["valid_s"] / 60.0) if out["valid_s"] > 0 else float("nan")
    return out


_ACCUM = ("n_frames", "valid_s", "n_event", "n_blink_lt500ms", "n_closure_ge500ms")


def aggregate_condition(segs, cfg):
    """把一个 (被试, 难度, 状态) 下所有连续单元的分子分母各自求和后再算指标。

    3 s / 6 s 切片上单次 PERCLOS 方差极大、期望眨眼数 <1，逐切片比较会让统计功效
    崩掉；条件水平聚合是这些指标能用的前提。
    """
    tot = {k: float(sum(s.get(k, 0.0) or 0.0 for s in segs)) for k in _ACCUM}
    valid_s = tot["valid_s"]
    per_min = valid_s / 60.0
    mins = [s["max_dur_s"] for s in segs if np.isfinite(s.get("max_dur_s", np.nan))]

    def rate(n):
        return n / per_min if per_min > 0 else float("nan")

    out = {"n_segments": len(segs), "total_frames": int(tot["n_frames"]),
           "valid_s": valid_s, "max_dur_s": float(np.max(mins)) if mins else 0.0}
    for t in cfg["perclos_thresholds"]:
        k = int(round(t * 100))
        num = sum(s.get(f"closed_s_{k}", 0.0) or 0.0 for s in segs)
        long_s = sum(s.get(f"long_closed_s_{k}", 0.0) or 0.0 for s in segs)
        out[f"closed_s_{k}"] = num
        out[f"long_closed_s_{k}"] = min(long_s, num)
        out[f"perclos_{k}"] = num / valid_s if valid_s > 0 else float("nan")
        out[f"perclos_{k}_excl_long_closure"] = (num - min(long_s, num)) / valid_s \
            if valid_s > 0 else float("nan")
    out["n_event"] = int(tot["n_event"])
    out["n_blink_lt500ms"] = int(tot["n_blink_lt500ms"])
    out["blink_rate_lt500ms_per_min"] = rate(tot["n_blink_lt500ms"])
    for ms in cfg["blink_tier_ms"]:
        n = float(sum(s.get(f"n_blink_ge{ms}ms", 0) or 0 for s in segs))
        out[f"n_blink_ge{ms}ms"] = int(n)
        out[f"blink_rate_ge{ms}ms_per_min"] = rate(n)
    for s_ in cfg["closure_tier_s"]:
        tag = int(round(s_ * 1000))
        n = float(sum(s.get(f"n_closure_ge{tag}ms", 0) or 0 for s in segs))
        d = float(sum(s.get(f"closure_dur_ge{tag}ms_s", 0.0) or 0.0 for s in segs))
        out[f"n_closure_ge{tag}ms"] = int(n)
        out[f"closure_dur_ge{tag}ms_s"] = d
        out[f"closure_rate_ge{tag}ms_per_min"] = rate(n)
        out[f"closure_dur_per_min_ge{tag}ms_s"] = rate(d)
    oc = [s["mean_occ"] for s in segs if np.isfinite(s.get("mean_occ", np.nan))]
    out["mean_occ"] = float(np.mean(oc)) if oc else float("nan")
    return out


def validity_ratio(per_segment, fs):
    """有效时长占比：对齐方案 §12.1 的 >=80% 纳入标准。

    分母逐段按该段实际 fps 折算，不能用标称 fs——否则分子按实际秒、分母按标称秒，
    容器 fps 与标称不符时会得到错误占比甚至大于 1。
    """
    valid_s = total_s = 0.0
    for s in per_segment:
        seg_fps = s.get("fps") or fs
        if not seg_fps or not np.isfinite(seg_fps) or seg_fps <= 0:
            seg_fps = fs
        valid_s += float(s.get("valid_s", 0.0) or 0.0)
        total_s += float(s.get("n_frames", 0) or 0) / float(seg_fps)
    return (valid_s / total_s if total_s > 0 else float("nan")), valid_s, total_s


def feasibility(samples, valid_ratio, cfg):
    """--dry_run 的可行性裁决：把"实现不了就不做"变成有量化判据的判断。

    samples 需含 eye_w / iris_d / occ 三个抽样数组（occ 为已定标的遮蔽度）。
    返回 (verdict, reasons, stats)，verdict 取 pass / marginal / reject。
    """
    def _fin(key):
        return np.asarray([v for v in samples.get(key, []) if np.isfinite(v)], dtype=float)

    w, d, o = _fin("eye_w"), _fin("iris_d"), _fin("occ")
    reasons, hard_fail = [], False
    idr = d / w if w.size == d.size and w.size else np.empty(0)
    idr = idr[np.isfinite(idr) & (idr > 0)]
    p95 = float(np.quantile(o, 0.95)) if o.size >= 20 else float("nan")
    p05 = float(np.quantile(o, 0.05)) if o.size >= 20 else float("nan")
    dyn = p95 - p05 if np.isfinite(p95) and np.isfinite(p05) else float("nan")
    stats = {
        "n_sample": int(o.size),
        "eye_width_median_px": float(np.median(w)) if w.size else float("nan"),
        "eye_width_p05_px": float(np.quantile(w, 0.05)) if w.size else float("nan"),
        "iris_d_over_eye_w": float(np.median(idr)) if idr.size else float("nan"),
        "occ_p05": p05, "occ_p95": p95, "occ_max": float(np.max(o)) if o.size else float("nan"),
        "occ_dyn_range": dyn,
        "valid_ratio": float(valid_ratio) if np.isfinite(valid_ratio) else float("nan"),
    }
    if not o.size:
        return "reject", ["抽样帧上没有可用的虹膜遮蔽度，几何量未产出"], stats
    if stats["eye_width_median_px"] < cfg["min_eye_width_px"]:
        hard_fail = True
        reasons.append(f"眼裂宽度中位数 {stats['eye_width_median_px']:.1f}px < "
                       f"{cfg['min_eye_width_px']}px，垂直方向只有几个像素，遮蔽度不可信")
    if not (cfg["iris_d_ratio_lo"] <= stats["iris_d_over_eye_w"] <= cfg["iris_d_ratio_hi"]):
        hard_fail = True
        reasons.append(f"虹膜直径/眼裂宽度 = {stats['iris_d_over_eye_w']:.2f}，"
                       f"不在生理区间 [{cfg['iris_d_ratio_lo']}, {cfg['iris_d_ratio_hi']}] "
                       f"内 -> 虹膜索引 EYE_IRIS_LANDMARKS 很可能错位，先跑 --index_map 复核")
    if not np.isfinite(dyn) or dyn < cfg["min_occ_dyn_range"]:
        hard_fail = True
        reasons.append(f"遮蔽度动态范围 p95-p05 = {dyn:.3f} < {cfg['min_occ_dyn_range']}，"
                       f"眼睑张合几乎没有变化")
    if np.isfinite(valid_ratio) and valid_ratio < cfg["min_valid_ratio"]:
        reasons.append(f"有效帧占比 {valid_ratio:.1%} < {cfg['min_valid_ratio']:.0%}"
                       f"（低于方案 §12.1 纳入标准）")
        if not hard_fail:
            return "marginal", reasons + ["建议只做长闭眼(>=500ms)，PERCLOS 与眨眼率"
                                          "需人工核对后再用"], stats
    if hard_fail:
        return "reject", reasons, stats
    reasons.append(f"眼裂尺度、虹膜比例与遮蔽度动态范围均达标"
                   f"（遮蔽度 p05={p05:.2f} p95={p95:.2f} max={stats['occ_max']:.2f}）")
    return "pass", reasons, stats
