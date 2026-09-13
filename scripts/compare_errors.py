#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比两套预处理结果的误差：

  旧（整段视频管线）:  Cali_result_20_shijie_2/{id}_{diff}_{task}/all_errors.txt
      - alert: all_errors.txt = 校准后逐样本误差（每行一个 float）
              error_before.txt / error_after.txt = 校准前/后平均误差（标量）
      - sleepy: all_errors.txt = 逐样本误差（未校准，即原始）
               average_error.txt = 平均误差（标量）

  新（切片管线 + tf_calibrate）: clip_20260826_after_calibrate/*.jsonl
      - 每帧记录 deviation_px_before_calibrate / deviation_px_after_calibrate
      - alert: after = TF 模型校准后误差
      - sleepy: after = 直通（= before，原始误差）

对比口径：
  - 主对比：新 deviation_px_after_calibrate  vs  旧 all_errors.txt
    （alert 两边都是“校准后”；sleepy 两边都是“原始”，口径一致）
  - 副对比：新 deviation_px_before_calibrate 均值  vs  旧 error_before.txt(alert)/average_error.txt(sleepy)

行数不一定对应（整段视频 vs 切片丢首帧 + 切片边界），故采用分布级（均值/中位数/分位数）
对比，不做逐行配对；并对每个条件报告行数差异。

用法：
    python scripts/compare_errors.py \
        --old_root /root/autodl-tmp/shenxy/XDU/Dataset/Cali_result_20_shijie_2 \
        --new_root /root/autodl-tmp/shenxy/XDU/Dataset/clip_20260826_after_calibrate \
        --output   /root/autodl-tmp/shenxy/XDU/Dataset/error_comparison
"""

import sys, os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJ_ROOT = _os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── 条件名解析：[id]_[difficulty]_[task]（容忍 task 上的数字后缀，如 sleepy1）──
def parse_cond(name: str) -> Optional[Tuple[str, str, str]]:
    tokens = name.split("_")
    if not tokens or not tokens[0].isdigit():
        return None
    id_tok = tokens[0]
    difficulty = task = None
    for tok in tokens[1:]:
        if difficulty is None and tok.startswith("easy"):
            difficulty = "easy"
        elif difficulty is None and tok.startswith("hard"):
            difficulty = "hard"
        elif tok.startswith("alert"):
            task = "alert"
        elif tok.startswith("sleep"):
            task = "sleepy"
    if difficulty is None or task is None:
        return None
    return id_tok, difficulty, task


# ── 旧结果读取 ──────────────────────────────────────────────
def load_float_lines(path: Path) -> np.ndarray:
    """读取每行一个 float 的文件（兼容逗号/空格分隔、空行、单标量）。"""
    if not path.exists():
        return np.array([], dtype=np.float64)
    vals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 取第一个可解析的数字
            for tok in re.split(r"[,\s]+", line):
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                    break  # 只取该行第一个
                except ValueError:
                    continue
    return np.asarray(vals, dtype=np.float64)


def load_scalar(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    arr = load_float_lines(path)
    if arr.size == 0:
        return None
    return float(arr[0])


def load_old_condition(folder: Path) -> Dict:
    """返回 {after_arr, before_mean, after_mean, state_note}。"""
    after_arr = load_float_lines(folder / "all_errors.txt")
    before_mean = load_scalar(folder / "error_before.txt")      # alert 有
    after_mean = load_scalar(folder / "error_after.txt")        # alert 有
    avg_mean = load_scalar(folder / "average_error.txt")        # sleepy 有
    # 统一：sleepy 没有 before/after 标量，用 average_error
    if before_mean is None:
        before_mean = avg_mean
    if after_mean is None:
        after_mean = avg_mean
    # 若标量缺失，用 all_errors 的均值兜底
    if after_arr.size and after_mean is None:
        after_mean = float(np.mean(after_arr))
    return {
        "after_arr": after_arr,
        "before_mean": before_mean,
        "after_mean": after_mean,
    }


# ── 新结果读取 ──────────────────────────────────────────────
def load_new_jsonl(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (before_arr, after_arr)，仅保留有限值。"""
    before, after = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            b = rec.get("deviation_px_before_calibrate")
            a = rec.get("deviation_px_after_calibrate")
            if isinstance(b, (int, float)) and np.isfinite(b):
                before.append(float(b))
            if isinstance(a, (int, float)) and np.isfinite(a):
                after.append(float(a))
    return np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)


def gather_new(new_root: Path) -> Dict[Tuple[str, str, str], Dict]:
    """扫描新结果目录所有 jsonl，按 (id,diff,task) 聚合（合并后缀变体如 sleepy1）。"""
    groups: Dict[Tuple[str, str, str], Dict] = {}
    files = sorted(new_root.rglob("*.jsonl"))
    for p in files:
        cond = parse_cond(p.stem)
        if cond is None:
            continue
        before, after = load_new_jsonl(p)
        g = groups.setdefault(cond, {"before": [], "after": [], "files": []})
        g["before"].append(before)
        g["after"].append(after)
        g["files"].append(p.name)
    out = {}
    for cond, g in groups.items():
        out[cond] = {
            "before": np.concatenate(g["before"]) if g["before"] else np.array([]),
            "after": np.concatenate(g["after"]) if g["after"] else np.array([]),
            "files": g["files"],
        }
    return out


def gather_old(old_root: Path) -> Dict[Tuple[str, str, str], Dict]:
    out = {}
    for d in sorted(old_root.iterdir()):
        if not d.is_dir():
            continue
        cond = parse_cond(d.name)
        if cond is None:
            continue
        out[cond] = load_old_condition(d)
    return out


# ── 统计 ─────────────────────────────────────────────────────
def stats(arr: np.ndarray) -> Dict:
    arr = arr[np.isfinite(arr)] if arr.size else arr
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "p10": float("nan"), "p90": float("nan"),
                "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def fmt(v, w=10, p=3):
    if v is None:
        return f"{'-':>{w}}"
    try:
        return f"{v:>{w}.{p}f}"
    except (TypeError, ValueError):
        return f"{'-':>{w}}"


def main():
    ap = argparse.ArgumentParser(description="对比两套预处理结果的误差")
    ap.add_argument("--old_root", default="/root/autodl-tmp/shenxy/XDU/Dataset/Cali_result_20_shijie_2", help="旧结果根目录（Cali_result_20_shijie_2）")
    ap.add_argument("--new_root", default="/root/autodl-tmp/shenxy/XDU/Dataset/clip_20260826_after_calibrate", help="新结果根目录（clip_20260826_after_calibrate）")
    ap.add_argument("--output", default="./compare_output", help="分析总结输出目录")
    args = ap.parse_args()

    old_root = Path(args.old_root)
    new_root = Path(args.new_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    old = gather_old(old_root)
    new = gather_new(new_root)

    keys = sorted(set(old) | set(new), key=lambda k: (k[0], k[1], k[2]))

    rows = []  # CSV 行
    lines = []  # markdown 文本

    lines.append("# 两套预处理结果误差对比\n")
    lines.append(f"- 旧结果: `{old_root}`")
    lines.append(f"- 新结果: `{new_root}`\n")
    lines.append("对比口径：新 `deviation_px_after_calibrate` vs 旧 `all_errors.txt`")
    lines.append("（alert 两边均为校准后；sleepy 两边均为原始未校准，口径一致）。\n")

    # 逐条件
    lines.append("## 1. 逐条件对比（主指标：after 校准/最终误差）\n")
    header = ("| 条件 | 旧n | 新n | n差 | 旧mean | 新mean | "
              "Δmean(新-旧) | 相对% | 旧median | 新median | 旧p95 | 新p95 |")
    lines.append(header)
    lines.append("|" + "|".join(["---"] * 12) + "|")

    agg = {}  # (diff,task) -> 聚合
    for k in keys:
        sid, diff, task = k
        o = old.get(k)
        n = new.get(k)
        o_after = o["after_arr"] if o else np.array([])
        n_after = n["after"] if n else np.array([])
        o_stats = stats(o_after)
        n_stats = stats(n_after)
        if o_stats["mean"] and np.isfinite(o_stats["mean"]) and o_stats["mean"] != 0:
            rel = (n_stats["mean"] - o_stats["mean"]) / o_stats["mean"] * 100
        else:
            rel = float("nan")
        delta = n_stats["mean"] - o_stats["mean"] if (np.isfinite(n_stats["mean"]) and np.isfinite(o_stats["mean"])) else float("nan")
        ndiff = n_stats["n"] - o_stats["n"]

        cond_str = f"{sid}_{diff}_{task}"
        lines.append(
            f"| {cond_str} | {o_stats['n']} | {n_stats['n']} | {ndiff:+d} | "
            f"{o_stats['mean']:.3f} | {n_stats['mean']:.3f} | {delta:+.3f} | "
            f"{rel:+.2f}% | {o_stats['median']:.3f} | {n_stats['median']:.3f} | "
            f"{o_stats['p95']:.3f} | {n_stats['p95']:.3f} |"
        )
        rows.append({
            "condition": cond_str, "id": sid, "difficulty": diff, "task": task,
            "old_n": o_stats["n"], "new_n": n_stats["n"], "n_diff": ndiff,
            "old_mean": o_stats["mean"], "new_mean": n_stats["mean"],
            "delta_mean": delta, "relative_pct": rel,
            "old_median": o_stats["median"], "new_median": n_stats["median"],
            "old_p95": o_stats["p95"], "new_p95": n_stats["p95"],
            "old_std": o_stats["std"], "new_std": n_stats["std"],
        })

        key2 = (diff, task)
        a = agg.setdefault(key2, {"old": [], "new": [], "old_before": [], "new_before": []})
        if o_stats["n"]:
            a["old"].append(o_stats["mean"])
        if n_stats["n"]:
            a["new"].append(n_stats["mean"])
        # 副指标 before
        ob = o["before_mean"] if o else None
        nb = float(np.mean(n["before"])) if (n and n["before"].size) else None
        if ob is not None:
            a["old_before"].append(ob)
        if nb is not None:
            a["new_before"].append(nb)

    # 按 (diff, task) 聚合
    lines.append("\n## 2. 按 (difficulty, task) 聚合（对受试者均值再取平均）\n")
    lines.append("| 条件 | 旧after均值 | 新after均值 | Δ | 相对% | 旧before均值 | 新before均值 | 旧校准增益% | 新校准增益% |")
    lines.append("|" + "|".join(["---"] * 9) + "|")
    for key2 in sorted(agg):
        diff, task = key2
        a = agg[key2]
        om = float(np.mean(a["old"])) if a["old"] else float("nan")
        nm = float(np.mean(a["new"])) if a["new"] else float("nan")
        delta = nm - om if (np.isfinite(nm) and np.isfinite(om)) else float("nan")
        rel = (nm - om) / om * 100 if (np.isfinite(om) and om) else float("nan")
        ob = float(np.mean(a["old_before"])) if a["old_before"] else float("nan")
        nb = float(np.mean(a["new_before"])) if a["new_before"] else float("nan")
        old_gain = (om - ob) / ob * 100 if (np.isfinite(ob) and ob) else float("nan")
        new_gain = (nm - nb) / nb * 100 if (np.isfinite(nb) and nb) else float("nan")
        lines.append(
            f"| {diff}_{task} | {om:.3f} | {nm:.3f} | {delta:+.3f} | {rel:+.2f}% | "
            f"{ob:.3f} | {nb:.3f} | {old_gain:+.2f}% | {new_gain:+.2f}% |"
        )

    # 总体
    all_old = [v for a in agg.values() for v in a["old"]]
    all_new = [v for a in agg.values() for v in a["new"]]
    om = float(np.mean(all_old)) if all_old else float("nan")
    nm = float(np.mean(all_new)) if all_new else float("nan")
    delta = nm - om if (np.isfinite(nm) and np.isfinite(om)) else float("nan")
    rel = (nm - om) / om * 100 if (np.isfinite(om) and om) else float("nan")
    lines.append("\n## 3. 总体（所有条件、所有受试者的 after 均值再平均）\n")
    lines.append(f"- 旧 after 平均误差: **{om:.3f} px**")
    lines.append(f"- 新 after 平均误差: **{nm:.3f} px**")
    lines.append(f"- 差值 Δ(新-旧): **{delta:+.3f} px**（相对 **{rel:+.2f}%**）")
    if rel < 0:
        verdict = f"新管线比旧管线平均误差**降低** {-rel:.2f}%（精度提升）。"
    elif rel > 0:
        verdict = f"新管线比旧管线平均误差**升高** {rel:.2f}%（精度下降）。"
    else:
        verdict = "两者平均误差基本一致。"
    lines.append(f"- 结论：{verdict}")

    # 行数差异说明
    lines.append("\n## 4. 行数差异与取舍说明\n")
    lines.append("- 旧管线处理整段连续视频（easy 每目标点扩 3s=90帧、hard 每 image_index 扩 180帧，"
                 "不足则用最后目标点补齐）；新管线按切片处理，且**每个切片丢弃第一帧**，"
                 "故两侧样本数通常不完全相等。")
    lines.append("- 本对比为**分布级**对比（均值/中位数/分位数），不做逐帧配对，行数差异不影响统计有效性。")
    lines.append("- 若某条件 n 差异显著，说明两套管线对该受试者的数据覆盖不同，需结合原始帧数核对；"
                 "差异较大的条件见下表。")
    big = [r for r in rows if abs(r["n_diff"]) > 0]
    if big:
        lines.append("\n| 条件 | 旧n | 新n | n差 |")
        lines.append("|" + "|".join(["---"] * 4) + "|")
        for r in big:
            lines.append(f"| {r['condition']} | {r['old_n']} | {r['new_n']} | {r['n_diff']:+d} |")
    else:
        lines.append("\n（无行数差异）")

    # 缺失条件
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    if only_old or only_new:
        lines.append("\n## 5. 仅一侧存在的条件\n")
        if only_old:
            lines.append("- 仅旧结果有: " + ", ".join(f"{a}_{b}_{c}" for a, b, c in only_old))
        if only_new:
            lines.append("- 仅新结果有: " + ", ".join(f"{a}_{b}_{c}" for a, b, c in only_new))

    # 写 markdown
    md_path = out_dir / "error_comparison_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出分析总结: {md_path}")

    # 写 CSV
    csv_path = out_dir / "error_comparison_table.csv"
    with csv_path.open("w", encoding="utf-8-sig") as f:
        cols = ["condition", "id", "difficulty", "task", "old_n", "new_n", "n_diff",
                "old_mean", "new_mean", "delta_mean", "relative_pct",
                "old_median", "new_median", "old_p95", "new_p95",
                "old_std", "new_std"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None or (isinstance(r[c], float) and not np.isfinite(r[c]))
                             else f"{r[c]}" for c in cols) + "\n")
    print(f"已写出 CSV 表: {csv_path}")

    # 终端简报
    print("\n==== 总体简报 ====")
    print(f"旧 after 平均误差: {om:.3f} px")
    print(f"新 after 平均误差: {nm:.3f} px")
    print(f"差值(新-旧): {delta:+.3f} px  ({rel:+.2f}%)")
    print(verdict)


if __name__ == "__main__":
    main()
