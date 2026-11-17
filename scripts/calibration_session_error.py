#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充实验：alert / sleepy(fatigue) 两次 session 的四点校准误差对比
=================================================================

回应审稿意见（Session confounding）：证明 gaze 校准按 session 重复执行，
且 alert 与 fatigued 两次 session 的四点校准误差无显著差异（并做 TOST 等价检验）。

方法口径（与 homtransform.calibrate / _getCalibValuesOnScreen 完全一致）：
  1. 读取每个 session 的 results/Calibration.csv（原始校准记录，set_x/y/z 单位 mm）；
  2. 对每个校准点 idx，复现 _RemoveOutliers 的去离群掩码（gaze_x、gaze_y 各按
     mean±1*std 过滤后取 AND）；
  3. 取去离群后的 median gaze 向量，经该 session 的 STransG.npy 映射到屏幕坐标
     （_getScale 求交屏幕平面），得到 Sg；
  4. 单点误差 = ||Sg - SetVal|| 的屏幕内(x,y)欧氏距离（mm），并按 stats.txt 的
     屏幕尺寸换算 px；
  5. 每个 session 报告 4 个单点误差的均值与标准差（即"四点校准误差 mean±SD"）。

统计检验（被试内配对，alert vs sleepy）：
  - Shapiro-Wilk（配对差正态性）
  - 配对 t 检验 + Cohen's dz + 配对差 95%CI
  - Wilcoxon 符号秩检验（非参数稳健性）
  - TOST 等价检验（Schuirmann 1987；Lakens 2017）：绝对边界 --tost_margin_mm
    与相对边界 --tost_margin_rel（相对 alert 组均值）各做一次。

校准目录查找规则与 FatigueGuard_preprocess_batch.find_calibration_files 一致
（subject 级 → state/difficulty 级 → state 级），并记录命中的层级；
若同一被试的 alert 与 sleepy 解析到同一目录，则校准未按 session 重复，
脚本会如实标记 shared=True，该被试不进入配对统计。

仅依赖 numpy / pandas / scipy / matplotlib，无需 TensorFlow 或 GPU。

用法：
    python scripts/calibration_session_error.py \
        --data_root /root/autodl-tmp/shenxy/Data/DATA \
        --output    /root/autodl-tmp/shenxy/Data/calib_session_check

输出：
    calibration_session_error.csv           每行 = 一个 (subject, state, difficulty) session
    calibration_session_error_summary.md    协议 + 校准来源表 + 误差表 + 统计检验 + 结论
    calibration_session_paired.png          alert vs sleepy 配对点线图（有非共享数据时）
"""

from __future__ import annotations

import sys, os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJ_ROOT = _os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJ_ROOT)

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

STATES = ["alert", "sleepy"]          # sleepy = fatigued session（数据树命名）
DIFFICULTIES = ["easy", "hard"]
REQUIRED_NPY = ["STransG.npy", "StG.npy", "scaleWtG.npy", "STransW.npy", "StW.npy"]
N_FOUR_POINT = 4                      # 四点校准：屏幕四角 idx 0..3


# ── 校准目录查找（与 FatigueGuard_preprocess_batch.py 规则一致） ──────────

def find_calibration_dirs(subject_dir: Path, state: str, difficulty: str
                          ) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """返回 (results_dir, strans_dir, level)。level ∈ {subject, state_difficulty, state}"""
    candidates = [
        (subject_dir / "results", subject_dir / "STransG", "subject"),
        (subject_dir / state / difficulty / "results",
         subject_dir / state / difficulty / "STransG", "state_difficulty"),
        (subject_dir / state / "results",
         subject_dir / state / "STransG", "state"),
    ]
    for result_dir, strans_dir, level in candidates:
        if result_dir.exists() and strans_dir.exists():
            if all((strans_dir / f).exists() for f in REQUIRED_NPY):
                return result_dir, strans_dir, level
    return None, None, None


# ── 单 session 四点校准误差 ─────────────────────────────────────────────

def _mask_outliers(arr: np.ndarray, std_threshold: float = 1.0) -> np.ndarray:
    """复现 HomTransform._MaskOutliers：mean ± std_threshold*std 之内保留。"""
    mean = np.mean(arr)
    std = np.std(arr)
    return np.abs(arr - mean) < std_threshold * std


def _map_gaze_to_screen(gaze_vec: np.ndarray, stransg: np.ndarray) -> np.ndarray:
    """复现 _getScale + STransG 投影：单位 gaze 向量 → 屏幕平面坐标 (mm)。"""
    gtrans_s = np.linalg.inv(stransg)
    gt_s = gtrans_s[:3, 3].reshape(3, 1)
    gz = np.array([[0.0], [0.0], [1.0]])
    g = gaze_vec.reshape(3, 1)
    scale = (gz.T @ gt_s) / (gz.T @ g)
    sg = (stransg @ np.vstack((scale * g, [1.0])))[:3, 0]
    return sg


def _parse_screen_size(stats_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """从 stats.txt 解析屏幕物理尺寸 (width_mm, height_mm)，缺失返回 (None, None)。"""
    if not stats_path.exists():
        return None, None
    text = stats_path.read_text(encoding="utf-8", errors="ignore")
    mw = re.search(r"Width:\s*([\d.]+)px,\s*([\d.]+)mm", text)
    mh = re.search(r"Height:\s*([\d.]+)px,\s*([\d.]+)mm", text)
    width_mm = float(mw.group(2)) if mw else None
    height_mm = float(mh.group(2)) if mh else None
    return width_mm, height_mm


def _parse_screen_px(stats_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not stats_path.exists():
        return None, None
    text = stats_path.read_text(encoding="utf-8", errors="ignore")
    mw = re.search(r"Width:\s*([\d.]+)px", text)
    mh = re.search(r"Height:\s*([\d.]+)px", text)
    return (float(mw.group(1)) if mw else None,
            float(mh.group(1)) if mh else None)


def compute_session_errors(result_dir: Path, strans_dir: Path) -> Dict:
    """计算单个 session 的四点校准误差（mm），返回结构化结果。"""
    out: Dict = {
        "ok": False, "reason": "", "errors_mm": [], "errors_px": [],
        "mean_mm": float("nan"), "sd_mm": float("nan"),
        "mean_px": float("nan"), "sd_px": float("nan"),
        "mean_persample_mm": float("nan"),
        "n_calib_points_total": 0, "n_frames_used": 0,
        "screen_width_mm": float("nan"), "screen_height_mm": float("nan"),
    }

    calib_csv = result_dir / "Calibration.csv"
    if not calib_csv.exists():
        out["reason"] = f"缺少 Calibration.csv: {calib_csv}"
        return out

    stransg = np.asarray(np.load(strans_dir / "STransG.npy"), dtype=np.float64)
    if stransg.shape != (4, 4):
        out["reason"] = f"STransG.npy 形状异常 {stransg.shape}: {strans_dir}"
        return out

    stg = np.asarray(np.load(strans_dir / "StG.npy"), dtype=np.float64)
    n_points_total = stg.shape[0]
    out["n_calib_points_total"] = int(n_points_total)

    df = pd.read_csv(calib_csv, index_col=0)
    need_cols = ["idx", "gaze_x", "gaze_y", "gaze_z", "set_x", "set_y"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        out["reason"] = f"Calibration.csv 缺列 {missing}: {calib_csv}"
        return out

    df = df.dropna(subset=["gaze_x", "gaze_y", "gaze_z", "set_x", "set_y"])
    if df.empty:
        out["reason"] = f"Calibration.csv 无有效行: {calib_csv}"
        return out

    idx_unique = sorted(int(v) for v in pd.unique(df["idx"]))
    point_ids = idx_unique[:N_FOUR_POINT]
    if len(point_ids) < N_FOUR_POINT:
        out["reason"] = (f"校准点数不足 4（找到 idx={idx_unique}）: {calib_csv}")
        return out

    width_mm, height_mm = _parse_screen_size(result_dir / "stats.txt")
    width_px, height_px = _parse_screen_px(result_dir / "stats.txt")
    if width_mm is not None:
        out["screen_width_mm"] = width_mm
        out["screen_height_mm"] = height_mm

    errors_mm: List[float] = []
    errors_px: List[float] = []
    persample_mm: List[float] = []
    n_frames_used = 0

    for pid in point_ids:
        rows = df[df["idx"].astype(int) == pid]
        gaze = rows[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float64)
        setv = rows[["set_x", "set_y"]].to_numpy(dtype=np.float64)
        if gaze.shape[0] < 3:
            out["reason"] = f"校准点 idx={pid} 样本过少（{gaze.shape[0]}）"
            return out

        # 复现 _RemoveOutliers：对 gaze_x、gaze_y 分别按 mean±1*std 过滤后取 AND
        mask = _mask_outliers(gaze[:, 0]) & _mask_outliers(gaze[:, 1])
        if mask.sum() < 1:
            mask = np.ones(gaze.shape[0], dtype=bool)   # 全部被滤掉时退化为全保留
        gaze_m = gaze[mask]
        set_m = setv[mask]
        n_frames_used += int(mask.sum())

        target = np.median(set_m, axis=0)               # mm，理论上是常量
        med_gaze = np.median(gaze_m, axis=0)
        sg = _map_gaze_to_screen(med_gaze, stransg)
        err_mm = float(np.linalg.norm(sg[:2] - target))
        errors_mm.append(err_mm)

        # 辅助口径：逐样本误差均值（含校准期间的注视抖动）
        per_sample = []
        for gv, tv in zip(gaze_m, set_m):
            s = _map_gaze_to_screen(gv, stransg)
            per_sample.append(float(np.linalg.norm(s[:2] - tv)))
        persample_mm.append(float(np.mean(per_sample)))

        if width_mm is not None and width_px is not None:
            kx = width_px / width_mm
            ky = (height_px / height_mm) if height_mm else kx
            errors_px.append(float(math.hypot((sg[0] - target[0]) * kx,
                                              (sg[1] - target[1]) * ky)))

    out["ok"] = True
    out["errors_mm"] = errors_mm
    out["errors_px"] = errors_px
    out["mean_mm"] = float(np.mean(errors_mm))
    out["sd_mm"] = float(np.std(errors_mm, ddof=1)) if len(errors_mm) > 1 else float("nan")
    if errors_px:
        out["mean_px"] = float(np.mean(errors_px))
        out["sd_px"] = float(np.std(errors_px, ddof=1))
    out["mean_persample_mm"] = float(np.mean(persample_mm))
    out["n_frames_used"] = n_frames_used
    return out


# ── 统计检验 ────────────────────────────────────────────────────────────

def tost_paired(diff: np.ndarray, margin: float) -> float:
    """配对差的 TOST 等价检验（H0: |mean diff| >= margin），返回 max(p_lo, p_hi)。"""
    n = diff.size
    if n < 3 or not np.isfinite(margin) or margin <= 0:
        return float("nan")
    p_lo = stats.ttest_1samp(diff, -margin, alternative="greater").pvalue
    p_hi = stats.ttest_1samp(diff, margin, alternative="less").pvalue
    return float(max(p_lo, p_hi))


def paired_analysis(alert: np.ndarray, sleepy: np.ndarray,
                    margin_mm: float, margin_rel: float) -> Dict:
    """alert/sleepy 为按被试对齐的 session 平均四点误差（mm）。"""
    n = alert.size
    res: Dict = {"n": int(n)}
    res["alert_mean"] = float(np.mean(alert))
    res["alert_sd"] = float(np.std(alert, ddof=1)) if n > 1 else float("nan")
    res["sleepy_mean"] = float(np.mean(sleepy))
    res["sleepy_sd"] = float(np.std(sleepy, ddof=1)) if n > 1 else float("nan")

    diff = sleepy - alert
    res["diff_mean"] = float(np.mean(diff))
    if n >= 2:
        se = float(np.std(diff, ddof=1) / math.sqrt(n))
        tcrit = float(stats.t.ppf(0.975, n - 1))
        res["diff_se"] = se
        res["diff_ci95"] = (res["diff_mean"] - tcrit * se, res["diff_mean"] + tcrit * se)
        res["cohen_dz"] = res["diff_mean"] / float(np.std(diff, ddof=1)) \
            if np.std(diff, ddof=1) > 0 else float("nan")
        res["shapiro_p"] = float(stats.shapiro(diff).pvalue) if n >= 3 else float("nan")
        tt = stats.ttest_rel(sleepy, alert)
        res["ttest_t"] = float(tt.statistic)
        res["ttest_p"] = float(tt.pvalue)
        try:
            w = stats.wilcoxon(sleepy, alert)
            res["wilcoxon_stat"] = float(w.statistic)
            res["wilcoxon_p"] = float(w.pvalue)
        except ValueError:      # 例如差值全为 0
            res["wilcoxon_stat"] = float("nan")
            res["wilcoxon_p"] = float("nan")
    else:
        for k in ["diff_se", "cohen_dz", "shapiro_p", "ttest_t", "ttest_p",
                  "wilcoxon_stat", "wilcoxon_p"]:
            res[k] = float("nan")
        res["diff_ci95"] = (float("nan"), float("nan"))

    # TOST：绝对边界 + 相对边界（相对 alert 组均值）
    rel_margin = margin_rel * res["alert_mean"] if np.isfinite(res["alert_mean"]) else float("nan")
    res["tost_margin_mm"] = margin_mm
    res["tost_p_abs"] = tost_paired(diff, margin_mm)
    res["tost_margin_rel_mm"] = rel_margin
    res["tost_p_rel"] = tost_paired(diff, rel_margin)
    res["diff"] = diff
    return res


def verdict_text(res: Dict, label: str) -> str:
    lines = []
    alpha = 0.05
    p_t = res.get("ttest_p", float("nan"))
    p_w = res.get("wilcoxon_p", float("nan"))
    p_tost_a = res.get("tost_p_abs", float("nan"))
    if res["n"] < 3:
        return f"- **{label}**：有效配对样本不足（n={res['n']}），无法检验。"
    if np.isfinite(p_t) and p_t >= alpha:
        lines.append(f"- **{label}**：alert 与 fatigue session 的四点校准误差差异**不显著**"
                     f"（配对 t 检验 p={p_t:.3f}；Wilcoxon p={p_w:.3f}），"
                     f"配对差 Δ(fatigue−alert)={res['diff_mean']:+.2f} mm，"
                     f"95%CI [{res['diff_ci95'][0]:+.2f}, {res['diff_ci95'][1]:+.2f}] mm，"
                     f"Cohen's dz={res['cohen_dz']:.2f}。")
    else:
        lines.append(f"- **{label}**：⚠️ alert 与 fatigue session 的四点校准误差差异**显著**"
                     f"（配对 t 检验 p={p_t:.3g}；Wilcoxon p={p_w:.3g}），"
                     f"Δ(fatigue−alert)={res['diff_mean']:+.2f} mm。需在回复中如实讨论。")
    if np.isfinite(p_tost_a):
        if p_tost_a < alpha:
            lines.append(f"  - TOST 等价检验成立：在 ±{res['tost_margin_mm']:.1f} mm 边界内"
                         f"两 session 校准误差**统计等价**（p={p_tost_a:.3g}；"
                         f"相对边界 ±{res['tost_margin_rel_mm']:.1f} mm 时 p={res['tost_p_rel']:.3g}）。")
        else:
            lines.append(f"  - TOST 等价检验在 ±{res['tost_margin_mm']:.1f} mm 边界内未成立"
                         f"（p={p_tost_a:.3g}）。可放宽 --tost_margin_mm 或改报相对边界结果。")
    return "\n".join(lines)


# ── 图 ─────────────────────────────────────────────────────────────────

def make_figure(pairs: Dict[str, Dict], out_path: Path) -> bool:
    """pairs: label -> {alert, sleepy, subjects}；配对点线图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [k for k, v in pairs.items() if v["alert"].size >= 2]
    if not labels:
        return False
    fig, ax = plt.subplots(figsize=(3.2 * len(labels), 4.2))
    xs = np.arange(len(labels))
    for j, lab in enumerate(labels):
        a = pairs[lab]["alert"]
        s = pairs[lab]["sleepy"]
        xa = np.full(a.size, xs[j] - 0.12)
        xsl = np.full(s.size, xs[j] + 0.12)
        for xai, ai, xsi, si in zip(xa, a, xsl, s):
            ax.plot([xai, xsi], [ai, si], color="0.7", lw=0.8, zorder=1)
        ax.scatter(xa, a, color="tab:blue", zorder=2, label="alert" if j == 0 else None)
        ax.scatter(xsl, s, color="tab:red", zorder=2, label="fatigue (sleepy)" if j == 0 else None)
        ax.scatter([xs[j] - 0.12], [np.mean(a)], marker="_", s=400, color="tab:blue", zorder=3)
        ax.scatter([xs[j] + 0.12], [np.mean(s)], marker="_", s=400, color="tab:red", zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("4-point calibration error (mm)")
    ax.set_title("Per-session calibration error: alert vs fatigue")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


# ── 主流程 ──────────────────────────────────────────────────────────────

def fmt(v, p=2):
    return "-" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f"{v:.{p}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="alert vs fatigue session 四点校准误差对比")
    ap.add_argument("--data_root", required=True, help="数据集根目录（含 01/ 02/ ... 被试目录）")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--subjects", default=None, help="被试 ID 逗号分隔，如 01,02；默认全部")
    ap.add_argument("--tost_margin_mm", type=float, default=5.0,
                    help="TOST 等价检验的绝对边界（mm），默认 5")
    ap.add_argument("--tost_margin_rel", type=float, default=0.20,
                    help="TOST 相对边界（占 alert 组均值的比例），默认 0.20")
    ap.add_argument("--no_figure", action="store_true", help="不输出配图")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        subject_dirs = [data_root / s.strip() for s in args.subjects.split(",")]
        subject_dirs = [d for d in subject_dirs if d.exists()]
    else:
        subject_dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    if not subject_dirs:
        raise FileNotFoundError(f"在 {data_root} 下找不到被试目录")

    rows: List[Dict] = []
    notes: List[str] = []

    for subj_dir in subject_dirs:
        sid = subj_dir.name
        # 先解析每个 (state, difficulty) 的校准目录，用于检测 session 间共享
        resolved: Dict[Tuple[str, str], Tuple[Optional[Path], Optional[Path], Optional[str]]] = {}
        for state in STATES:
            for diff in DIFFICULTIES:
                resolved[(state, diff)] = find_calibration_dirs(subj_dir, state, diff)

        for diff in DIFFICULTIES:
            r_a = resolved[("alert", diff)]
            r_s = resolved[("sleepy", diff)]
            shared = (r_a[1] is not None and r_a[1] == r_s[1])
            for state in STATES:
                result_dir, strans_dir, level = resolved[(state, diff)]
                row: Dict = {
                    "subject": sid, "difficulty": diff, "state": state,
                    "calib_level": level or "", "shared_calibration": shared,
                    "calib_results_dir": str(result_dir) if result_dir else "",
                }
                if result_dir is None:
                    row.update({"ok": False, "reason": "未找到校准目录"})
                    rows.append(row)
                    continue
                res = compute_session_errors(result_dir, strans_dir)
                row.update({"ok": res["ok"], "reason": res["reason"]})
                for i in range(N_FOUR_POINT):
                    row[f"err_p{i+1}_mm"] = res["errors_mm"][i] if i < len(res["errors_mm"]) else np.nan
                    row[f"err_p{i+1}_px"] = res["errors_px"][i] if i < len(res["errors_px"]) else np.nan
                row.update({
                    "mean_err_mm": res["mean_mm"], "sd_err_mm": res["sd_mm"],
                    "mean_err_px": res["mean_px"], "sd_err_px": res["sd_px"],
                    "mean_persample_err_mm": res["mean_persample_mm"],
                    "n_calib_points_total": res["n_calib_points_total"],
                    "n_frames_used": res["n_frames_used"],
                    "screen_width_mm": res["screen_width_mm"],
                    "screen_height_mm": res["screen_height_mm"],
                })
                rows.append(row)
                if not res["ok"]:
                    notes.append(f"[{sid}/{diff}/{state}] {res['reason']}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "calibration_session_error.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已写出逐 session 误差表: {csv_path}")

    # ── 配对统计 ──────────────────────────────────────────────
    ok = df[df.get("ok") == True]  # noqa: E712
    pairs: Dict[str, Dict] = {}
    analyses: Dict[str, Dict] = {}

    def collect(diff_label: str, alert_vals, sleepy_vals, subjects):
        if len(subjects) >= 2:
            pairs[diff_label] = {"alert": alert_vals, "sleepy": sleepy_vals,
                                 "subjects": subjects}
            analyses[diff_label] = paired_analysis(
                alert_vals, sleepy_vals, args.tost_margin_mm, args.tost_margin_rel)

    for diff in DIFFICULTIES:
        sub_a, sub_s, subjects = [], [], []
        for sid in sorted(ok["subject"].unique()) if "subject" in ok else []:
            g = ok[(ok["subject"] == sid) & (ok["difficulty"] == diff) & (~ok["shared_calibration"])]
            a = g[g["state"] == "alert"]
            s = g[g["state"] == "sleepy"]
            if len(a) and len(s):
                sub_a.append(float(a.iloc[0]["mean_err_mm"]))
                sub_s.append(float(s.iloc[0]["mean_err_mm"]))
                subjects.append(sid)
        collect(diff, np.asarray(sub_a), np.asarray(sub_s), subjects)

    # combined：每被试取 easy/hard 两场景均值后再配对
    sub_a, sub_s, subjects = [], [], []
    for sid in sorted(ok["subject"].unique()) if "subject" in ok else []:
        g = ok[(ok["subject"] == sid) & (~ok["shared_calibration"])]
        a = g[g["state"] == "alert"]["mean_err_mm"].to_numpy(dtype=float)
        s = g[g["state"] == "sleepy"]["mean_err_mm"].to_numpy(dtype=float)
        if a.size and s.size:
            sub_a.append(float(np.mean(a)))
            sub_s.append(float(np.mean(s)))
            subjects.append(sid)
    collect("combined", np.asarray(sub_a), np.asarray(sub_s), subjects)

    # ── markdown 汇总 ─────────────────────────────────────────
    lines: List[str] = []
    lines.append("# alert vs fatigue session 四点校准误差对比（回应 Session confounding 意见）\n")
    lines.append(f"- 数据根目录: `{data_root}`")
    lines.append(f"- 误差口径: 每 session 的 `Calibration.csv` + `STransG.npy`，"
                 f"复现原始去离群（mean±1σ）后取每校准点 median gaze，经 STransG 映射到屏幕，"
                 f"与目标点求 (x,y) 欧氏距离（mm）；session 值 = 4 个单点误差的 mean±SD。")
    lines.append(f"- fatigue session 在数据树中命名为 `sleepy`。\n")

    # 校准来源与共享情况
    lines.append("## 1. 校准来源（回答“校准是否按 session 重复”）\n")
    if "calib_level" in df.columns:
        lvl_counts = df.groupby(["difficulty", "state", "calib_level"]).size()
        lines.append("| difficulty | state | 校准目录层级 | 条目数 |")
        lines.append("|---|---|---|---|")
        for (d_, s_, l_), c in lvl_counts.items():
            lines.append(f"| {d_} | {s_} | {l_ or '未找到'} | {c} |")
    shared_n = int((df.get("shared_calibration") == True).sum()) if "shared_calibration" in df else 0  # noqa: E712
    if shared_n > 0:
        shared_subj = sorted(df[df["shared_calibration"] == True]["subject"].unique())  # noqa: E712
        lines.append(f"\n⚠️ 以下被试的 alert 与 sleepy 解析到**同一**校准目录（校准未按 session 重复，"
                     f"两点误差恒等，已从配对统计中剔除）：{', '.join(shared_subj)}")
    else:
        lines.append("\n所有被试的 alert 与 sleepy session 均使用**各自独立**的校准目录，"
                     "即校准按 session 重复执行。")

    # 逐 session 误差表
    lines.append("\n## 2. 逐 session 四点校准误差\n")
    lines.append("| subject | difficulty | state | 校准层级 | p1 | p2 | p3 | p4 | mean±SD (mm) | mean±SD (px) | 备注 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        if r.get("ok"):
            mm = f"{fmt(r['mean_err_mm'])}±{fmt(r['sd_err_mm'])}"
            px = f"{fmt(r['mean_err_px'])}±{fmt(r['sd_err_px'])}" if np.isfinite(r.get("mean_err_px", np.nan)) else "-"
            note = "共享校准" if r.get("shared_calibration") else ""
            lines.append(f"| {r['subject']} | {r['difficulty']} | {r['state']} | {r.get('calib_level','')} "
                         f"| {fmt(r['err_p1_mm'])} | {fmt(r['err_p2_mm'])} | {fmt(r['err_p3_mm'])} | {fmt(r['err_p4_mm'])} "
                         f"| {mm} | {px} | {note} |")
        else:
            lines.append(f"| {r['subject']} | {r['difficulty']} | {r['state']} | {r.get('calib_level','')} "
                         f"| - | - | - | - | - | - | {r.get('reason','')} |")

    # 统计检验
    lines.append("\n## 3. 统计检验（被试内配对，alert vs fatigue）\n")
    lines.append("检验族：Shapiro-Wilk（正态性）→ 配对 t 检验 + Cohen's dz + 95%CI → "
                 "Wilcoxon 符号秩（稳健性）→ TOST 等价检验"
                 "（Schuirmann 1987; Lakens 2017, *PeerJ*）。")
    lines.append(f"TOST 边界：绝对 ±{args.tost_margin_mm:.1f} mm；相对 ±{args.tost_margin_rel:.0%}×alert 均值。\n")
    lines.append("| 条件 | n | alert mean±SD (mm) | fatigue mean±SD (mm) | Δ(fatigue−alert) [95%CI] | dz | Shapiro p | 配对 t (p) | Wilcoxon (p) | TOST p (绝对) | TOST p (相对) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in ["easy", "hard", "combined"]:
        if lab not in analyses:
            lines.append(f"| {lab} | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        r = analyses[lab]
        ci = r["diff_ci95"]
        lines.append(
            f"| {lab} | {r['n']} | {fmt(r['alert_mean'])}±{fmt(r['alert_sd'])} "
            f"| {fmt(r['sleepy_mean'])}±{fmt(r['sleepy_sd'])} "
            f"| {fmt(r['diff_mean'])} [{fmt(ci[0])}, {fmt(ci[1])}] "
            f"| {fmt(r['cohen_dz'])} | {fmt(r['shapiro_p'], 3)} "
            f"| {fmt(r['ttest_t'])} ({fmt(r['ttest_p'], 3)}) "
            f"| {fmt(r['wilcoxon_stat'], 1)} ({fmt(r['wilcoxon_p'], 3)}) "
            f"| {fmt(r['tost_p_abs'], 3)} | {fmt(r['tost_p_rel'], 3)} |")

    lines.append("\n### 结论\n")
    if analyses:
        for lab in ["easy", "hard", "combined"]:
            if lab in analyses:
                lines.append(verdict_text(analyses[lab], lab))
    else:
        lines.append("- ⚠️ 没有任何被试具备 alert/sleepy 各自独立且可复算的校准记录，"
                     "无法执行本补充实验。请先核对数据树中 Calibration.csv 是否存在、"
                     "以及校准目录是否按 state 分开存放。")

    if notes:
        lines.append("\n### 数据质量备注\n")
        for nt in notes:
            lines.append(f"- {nt}")

    md_path = out_dir / "calibration_session_error_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出汇总报告: {md_path}")

    if not args.no_figure and pairs:
        fig_path = out_dir / "calibration_session_paired.png"
        if make_figure(pairs, fig_path):
            print(f"已写出配图: {fig_path}")

    # 终端简报
    print("\n==== 简报 ====")
    for lab in ["easy", "hard", "combined"]:
        if lab in analyses:
            r = analyses[lab]
            print(f"[{lab}] n={r['n']} | alert {fmt(r['alert_mean'])}±{fmt(r['alert_sd'])} mm | "
                  f"fatigue {fmt(r['sleepy_mean'])}±{fmt(r['sleepy_sd'])} mm | "
                  f"Δ={fmt(r['diff_mean'])} mm | t-p={fmt(r['ttest_p'],3)} | "
                  f"Wilcoxon-p={fmt(r['wilcoxon_p'],3)} | TOST-p={fmt(r['tost_p_abs'],3)}")
    if shared_n:
        print(f"注意：{shared_n} 条记录来自 alert/sleepy 共享的校准目录，未进入配对统计。")


if __name__ == "__main__":
    main()
