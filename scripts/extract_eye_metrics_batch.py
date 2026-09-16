"""从 30 fps 普通 RGB 摄像头视频提取 PERCLOS / 眨眼率 / 长闭眼时长。

对应方案 §12.2 的"独立客观通道"。事件判定与聚合算法在 eye_metrics_core.py（可离线
单测），本文件只负责视频 IO、关键点抽取、目录布局发现与结果落盘。

与现有预处理流程完全独立：不读写逐帧 gaze JSONL，不经过 L2CS / OpenVINO / SFM /
TF 校准，只读同一批源视频，结果按 (id, difficulty, state) 三元组与其它结果 join。
需要独立 conda 环境（mediapipe 与本仓库 openvino==2022.3.0 / numpy==1.23.4 冲突）。

输入布局自动识别，两种都支持、也可混在一棵树里：
  完整视频（主输入，20 被试 × 2 难度 × 2 状态 = 80 段 5 分钟）
      <root>/01/alert/easy/training_video.mp4
  切片（兼容，[id]_[difficulty]_[task]/clip_*.mp4）
      <root>/01_easy_alert/clip_1.mp4
同一条件两种布局都命中时优先采用连续视频（切片会引入边界效应并碎片化分母）。

数据树里混有校准视频，故 video 布局默认只收 --video_pattern=training_video.mp4；
被排除的项会报数量，加 --list_excluded 可逐条查看。不要静默过滤。

典型用法（服务器上）：

  # 1) 可行性诊断：几分钟决定"做还是不做"，务必先跑
  python scripts/extract_eye_metrics_batch.py --root <完整视频根目录> --dry_run 6

  # 2) 目视确认眼睑关键点索引落位（首次使用必做，否则后面全不可信）
  python scripts/extract_eye_metrics_batch.py --root <完整视频根目录> \\
      --dump_overlay ./results/eye_overlay --dump_units 2 --limit_units 1

  # 3) 正式提取。按被试分片开几个后台进程，比在脚本里塞多进程更可控
  for s in 01_02_03_04 05_06_07_08 09_10_11_12 13_14_15_16 17_18_19_20; do
    nohup python scripts/extract_eye_metrics_batch.py --root <完整视频根目录> \\
        --output_dir ./results/eye_metrics --subjects ${s//_/,.} \\
        > logs/eye_${s}.log 2>&1 &
  done

  nohup python scripts/extract_eye_metrics_batch.py --output_dir ./results/eye_metrics --save_signals > result_20260915_hongmo.out &
"""

import sys as _sys
import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJ_ROOT = _os.path.dirname(_SCRIPT_DIR)
_sys.path.insert(0, _PROJ_ROOT)
_sys.path.insert(0, _SCRIPT_DIR)

import argparse
import csv
import re
import time
from collections import defaultdict
from fnmatch import translate as fnmatch_translate
from pathlib import Path

import numpy as np

try:
    from eye_metrics_core import (
        EYE_IRIS_LANDMARKS, EYE_LANDMARKS, aggregate_condition, combine_eyes,
        default_config, eye_aperture, feasibility, iris_geometry, iris_signals,
        median_scale, scale_parity_stats, summarize_segment, validity_ratio,
    )
except ImportError:
    from scripts.eye_metrics_core import (
        EYE_IRIS_LANDMARKS, EYE_LANDMARKS, aggregate_condition, combine_eyes,
        default_config, eye_aperture, feasibility, iris_geometry, iris_signals,
        median_scale, scale_parity_stats, summarize_segment, validity_ratio,
    )


def _require_video_deps():
    """mediapipe 与本仓库 openvino==2022.3.0 / numpy==1.23.4 冲突，必须用独立环境。"""
    try:
        import cv2
    except ImportError:
        raise SystemExit("缺少 opencv-python：请在专用 conda 环境里 pip install opencv-python")
    try:
        import mediapipe as mp
    except ImportError:
        raise SystemExit(
            "缺少 mediapipe。需要独立环境（与本仓库 openvino 2022.3.0 / numpy 1.23.4 冲突）：\n"
            "  conda create -n eyemetrics python=3.10 -y && conda activate eyemetrics\n"
            "  pip install mediapipe numpy\n"
            '验证：python -c "import mediapipe; print(mediapipe.__version__)"')
    return cv2, mp


def open_face_mesh(mp, min_conf):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,       # 逐帧跟踪：固定机位连续视频，比 image 模式快且稳
        max_num_faces=1,
        refine_landmarks=True,         # 必须开：468-477 是两眼虹膜各 5 点，遮蔽度分母靠它
        min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )


def check_landmark_count(n_pts, where=""):
    """确认 mediapipe 版本真的给了虹膜点。开不到就直接停，不要静默产出全 nan。"""
    need = max(max(EYE_IRIS_LANDMARKS["r"]), max(EYE_IRIS_LANDMARKS["l"])) + 1
    if n_pts < need:
        raise SystemExit(
            f"{where}FaceMesh 只返回 {n_pts} 个关键点，取不到虹膜点（需 >= {need}）。\n"
            "  说明 refine_landmarks 没生效或该 mediapipe 版本行为不同。\n"
            "  排查：python -c \"import mediapipe as mp; "
            "f=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True); print('ok')\"")


# ── 目录布局发现 ────────────────────────────────────────────────────
def parse_folder(name):
    """[id]_[difficulty]_[task]，容忍 task 上的数字后缀。与 clip_preprocess_batch 保持一致。"""
    tokens = name.split("_")
    if not tokens or not tokens[0].isdigit():
        return None
    difficulty = task = task_raw = None
    for tok in tokens[1:]:
        if difficulty is None and tok.startswith("easy"):
            difficulty = "easy"
        elif difficulty is None and tok.startswith("hard"):
            difficulty = "hard"
        elif tok.startswith("alert"):
            task_raw, task = tok, "alert"
        elif tok.startswith("sleep"):
            task_raw, task = tok, "sleepy"
    if difficulty is None or task is None:
        return None
    return tokens[0], difficulty, task, task_raw


def clip_sort_key(p):
    m = re.search(r"clip_(\d+)", p.stem)
    return (int(m.group(1)) if m else 10 ** 9, p.name)


def discover_clip_layout(root, subjects, pattern="clip_*.mp4"):
    """root/[id]_[difficulty]_[task]/clip_*.mp4 -> (units, 未识别的含视频目录)"""
    units, skipped = [], []
    for d in sorted([p for p in Path(root).iterdir() if p.is_dir()], key=lambda p: p.name):
        vids = sorted(list(d.glob(pattern)), key=clip_sort_key)
        if not vids:
            continue
        parsed = parse_folder(d.name)
        if not parsed:
            skipped.append(str(d))
            continue
        sid, diff, task, _ = parsed
        if subjects and _sid_key(sid) not in subjects:
            continue
        units.append({"id": sid, "difficulty": diff, "state": task, "layout": "clip",
                      "name": d.name, "paths": vids})
    return units, skipped


def discover_video_layout(root, subjects, pattern="training_video.mp4"):
    """root/.../[id].../[difficulty].../[state].../<pattern>，整段连续视频。

    默认只收 training_video.mp4：数据里还混有校准视频，按 rglob("*.mp4") 会把它们
    当成实验片段拼进同一条件单元，既污染 PERCLOS 分母也污染被试级参考定标。
    """
    pat = re.compile(fnmatch_translate(pattern), re.IGNORECASE)
    found = defaultdict(list)
    skipped = []
    for path in sorted(Path(root).rglob("*.mp4")):
        if not pat.search(path.name):
            skipped.append(str(path))
            continue
        rel = path.relative_to(root)
        toks = list(rel.parts[:-1]) + [path.stem]
        sid = next((t for t in toks if re.fullmatch(r"\d{1,3}", t)), None)
        diff = next((t for t in toks if t.startswith(("easy", "hard"))), None)
        task = next((t for t in toks if t.startswith(("alert", "sleep"))), None)
        if not (sid and diff and task):
            skipped.append(f"{path}  (路径缺 id/难度/状态)")
            continue
        found[(sid, diff, "sleepy" if task.startswith("sleep") else "alert")].append(path)
    units = []
    for (sid, diff, task), paths in sorted(found.items()):
        paths = sorted(paths, key=clip_sort_key)
        if subjects and _sid_key(sid) not in subjects:
            continue
        if len(paths) > 1:
            print(f"  [WARN] {sid}/{diff}/{task} 命中 {len(paths)} 个视频，已全部并入该条件；"
                  f"5 分钟条件通常应为 1 段，请确认没有把校准视频命名成 {pattern}："
                  f"{[p.name for p in paths][:6]}")
        units.append({"id": sid, "difficulty": diff, "state": task, "layout": "video",
                      "name": f"{sid}_{diff}_{task}", "paths": paths})
    return units, skipped


def _sid_key(sid):
    return str(int(sid)) if str(sid).isdigit() else str(sid)


def discover_units(args):
    """按文件名白名单收集实验视频，返回 (units, 被排除项)。

    排除清单必须交给上层打印：静默过滤会让人以为自己看到的样本就是全部，
    这个项目已经在"凭假设静默筛掉数据"上吃过两次亏。
    """
    subjects = None
    if args.subjects:
        subjects = {_sid_key(s) for s in args.subjects.split(",") if s.strip()}
    roots = []
    if args.root:
        roots = [args.root]
    else:
        if args.layout in ("clip", "auto") and args.clip_root:
            roots.append(args.clip_root)
        if args.layout in ("video", "auto") and args.data_root:
            roots.append(args.data_root)
    roots = [r for r in roots if r and Path(r).is_dir()]
    if not roots:
        raise SystemExit("--root / --clip_root / --data_root 都不可用，检查目录是否存在")

    by_key, excluded = {}, []
    for r in roots:
        cands = []
        if args.layout in ("clip", "auto"):
            u, sk = discover_clip_layout(r, None, args.clip_pattern)
            cands += u
            excluded += [f"[clip目录未识别] {x}" for x in sk]
        if args.layout in ("video", "auto"):
            u, sk = discover_video_layout(r, None, args.video_pattern)
            cands += u
            excluded += [f"[video非实验片段] {x}" for x in sk]
        for u in cands:
            k = (_sid_key(u["id"]), u["difficulty"], u["state"])
            if subjects and k[0] not in subjects:
                continue
            prev = by_key.get(k)
            # 同一条件被两种布局都命中时优先连续视频：切片会引入边界效应且分母碎片化
            if prev is None or (prev["layout"] == "clip" and u["layout"] == "video"):
                by_key[k] = u
    return [by_key[k] for k in sorted(by_key)], excluded


# ── 逐帧提取 ────────────────────────────────────────────────────────
def landmarks_to_points(face_landmarks, w, h):
    pts = np.empty((len(face_landmarks), 2), dtype=float)
    for i, lm in enumerate(face_landmarks):
        pts[i, 0] = lm.x * w
        pts[i, 1] = lm.y * h
    return pts


def draw_overlay(cv2, frame, pts, out_dir, tag, frame_idx):
    """把实际用到的眼睑/眼角/虹膜点画成大标记并标索引，用于目视确认索引表没有错位。"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = frame.copy()
    for spec in EYE_LANDMARKS.values():
        for i in spec["upper"]:
            if i < len(pts):
                x, y = int(pts[i, 0]), int(pts[i, 1])
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(img, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        for i in spec["lower"]:
            if i < len(pts):
                x, y = int(pts[i, 0]), int(pts[i, 1])
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(img, str(i), (x + 4, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        for key, col in (("corner_a", (255, 0, 0)), ("corner_b", (255, 255, 0))):
            i = spec[key]
            if i < len(pts):
                x, y = int(pts[i, 0]), int(pts[i, 1])
                cv2.circle(img, (x, y), 4, col, -1)
                cv2.putText(img, str(i), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    for idxs in EYE_IRIS_LANDMARKS.values():
        for i in idxs:
            if i < len(pts):
                x, y = int(pts[i, 0]), int(pts[i, 1])
                cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 7, 1)
                cv2.putText(img, str(i), (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    cv2.imwrite(str(Path(out_dir) / f"{tag}_f{frame_idx:06d}.png"), img)


def extract_one_video(cv2, face_mesh, path, cfg, dump=None):
    """返回该连续单元的逐帧开合度与眼裂宽度。绝不跳帧：跳帧会毁掉眨眼检测。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (np.isfinite(fps) and fps > 1.0):
        print(f"  [WARN] {Path(path).name} 容器 fps 不可用（{fps}），按 {cfg['fs']} 计")
        fps = cfg["fs"]
    elif abs(fps - cfg["fs"]) > 1.0:
        print(f"  [WARN] {Path(path).name} 实际 fps={fps:.2f} 与标称 {cfg['fs']} 相差 >1，"
              f"时长按实际值算")
    n_read = n_face = n_eye = 0
    buf = {k: [] for k in ("lo_r", "hi_r", "lo_l", "hi_l", "id_r", "id_l",
                           "w_r", "w_l", "ear_r", "ear_l")}
    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            break
        if not ret or frame is None or frame.size == 0:
            break
        n_read += 1
        h, w = frame.shape[:2]
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        n_face += 1
        pts = landmarks_to_points(res.multi_face_landmarks[0].landmark, w, h)
        check_landmark_count(len(pts), where=f"{Path(path).name}: ")
        ar, wr = eye_aperture(pts, "r")
        al, wl = eye_aperture(pts, "l")
        gr = iris_geometry(pts, "r")
        gl = iris_geometry(pts, "l")
        # 两眼独立入列（缺失留 nan 占位，保持等长）。"任一眼失败就整帧丢弃"会让一只眼
        # 的索引错误连带废掉另一只正确的眼。
        if not (np.isfinite(ar) or np.isfinite(al)):
            continue
        n_eye += 1
        buf["ear_r"].append(ar); buf["ear_l"].append(al)
        buf["w_r"].append(wr); buf["w_l"].append(wl)
        buf["lo_r"].append(gr[0]); buf["hi_r"].append(gr[1]); buf["id_r"].append(gr[2])
        buf["lo_l"].append(gl[0]); buf["hi_l"].append(gl[1]); buf["id_l"].append(gl[2])
        if dump and n_read % dump["every"] == 0 and dump["left"] > 0:
            draw_overlay(cv2, frame, pts, dump["dir"], dump["tag"], n_read)
            dump["left"] -= 1
    cap.release()
    out = {"path": str(path), "name": Path(path).stem, "fps": float(fps),
           "n_frames": n_read, "n_face": n_face,
           "n_eye": n_eye}              # 至少一眼可用的帧数；每眼各自可用数由数组派生
    for k, v in buf.items():
        out[k] = np.asarray(v, dtype=float)
    return out


def extract_unit(cv2, mp, unit, cfg, dump=None):
    """同一单元内的多个切片共用一个 FaceMesh 实例（跟踪态连续），跨单元重置。"""
    face_mesh = open_face_mesh(mp, cfg["min_conf"])
    segs = []
    try:
        for p in unit["paths"]:
            if dump is not None:
                dump["tag"] = unit["name"]          # 配额在多个切片间共享，不会按切片翻倍
            s = extract_one_video(cv2, face_mesh, p, cfg, dump)
            if s and s["n_frames"] > 0:
                segs.append(s)
    finally:
        face_mesh.close()
    return segs


def subject_scales(segs, cfg):
    """被试级尺度常数：虹膜直径与眼裂宽度的中位数，左右眼分别估计。

    这两个量都是解剖常数、与专注/低专注标签无关，因此遮蔽度的条件间差异不会被定标
    过程定义掉——旧的"用 alert 段求全开参考值"的循环性问题随之消失。
    """
    def _pick(key):
        return [s[key] for s in segs if len(s.get(key, []))]

    out = {}
    for side in ("r", "l"):
        out[side] = {"iris_d": median_scale(_pick(f"id_{side}")),
                     "eye_w": median_scale(_pick(f"w_{side}"))}
    both = [v for v in (out["r"]["eye_w"], out["l"]["eye_w"]) if np.isfinite(v)]
    out["eye_w_mean"] = float(np.mean(both)) if both else float("nan")
    both_d = [v for v in (out["r"]["iris_d"], out["l"]["iris_d"]) if np.isfinite(v)]
    out["iris_d_mean"] = float(np.mean(both_d)) if both_d else float("nan")
    fin = [v for v in (out["r"]["iris_d"], out["l"]["iris_d"]) if np.isfinite(v) and v > 0]
    out["iris_ratio_lr"] = max(fin) / min(fin) if len(fin) == 2 else float("nan")
    return out


def summarize_unit(unit, segs, scales, cfg):
    """逐段指标 -> 条件水平聚合。定标常数一律用被试级，不逐单元各自定标。"""
    per_seg = []
    for s in segs:
        o_r, k_r = iris_signals(s["lo_r"], s["hi_r"], s["id_r"], s["w_r"], scales["r"], cfg)
        o_l, k_l = iris_signals(s["lo_l"], s["hi_l"], s["id_l"], s["w_l"], scales["l"], cfg)
        if cfg.get("eyes") == "r":
            occ, ok = o_r, k_r
        elif cfg.get("eyes") == "l":
            occ, ok = o_l, k_l
        else:
            occ, ok = combine_eyes(o_r, o_l, k_r, k_l, cfg)
        m = summarize_segment(occ, ok, s["fps"], cfg)
        # 逐级漏斗：只报告每级留下多少，不额外拦截，定位问题不再靠猜
        m["n_ear_ok"] = int(np.count_nonzero(np.isfinite(s["ear_r"]) | np.isfinite(s["ear_l"])))
        m["n_iris_r"] = int(np.count_nonzero(np.isfinite(s["id_r"])))
        m["n_iris_l"] = int(np.count_nonzero(np.isfinite(s["id_l"])))
        m["n_width_gate_ok"] = int(min(np.count_nonzero(k_r), np.count_nonzero(k_l)))
        m["n_consensus_ok"] = int(np.count_nonzero(ok))
        m.update({f"eye_{k}": v for k, v in
                  scale_parity_stats(s["w_r"], s["w_l"], cfg).items()})
        m.update({f"iris_{k}": v for k, v in
                  scale_parity_stats(s["id_r"], s["id_l"], cfg).items()})
        m.update({"unit": unit["name"], "segment": s["name"], "path": s["path"],
                  "fps": s["fps"], "n_face": s["n_face"], "n_eye": s["n_eye"]})
        per_seg.append(m)
    agg = aggregate_condition(per_seg, cfg)
    vr, _, _ = validity_ratio(per_seg, cfg["fs"])

    def _mean(key):
        vals = [m[key] for m in per_seg if np.isfinite(m.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    agg.update({"id": unit["id"], "difficulty": unit["difficulty"], "state": unit["state"],
                "unit": unit["name"], "layout": unit.get("layout", "unknown"),
                "eyes": cfg.get("eyes", "both"), "n_videos": len(segs), "valid_ratio": vr,
                "iris_d_r_px": scales["r"]["iris_d"], "iris_d_l_px": scales["l"]["iris_d"],
                "iris_d_mean_px": scales["iris_d_mean"], "iris_ratio_lr": scales["iris_ratio_lr"],
                "eye_w_r_px": scales["r"]["eye_w"], "eye_w_l_px": scales["l"]["eye_w"],
                "median_eye_width_px": scales["eye_w_mean"],
                "iris_d_over_eye_w": (scales["iris_d_mean"] / scales["eye_w_mean"]
                                      if scales["eye_w_mean"] else float("nan")),
                "fs": cfg["fs"], "n_face_total": sum(s["n_face"] for s in segs),
                "n_eye_total": sum(s["n_eye"] for s in segs),
                "n_frames_total": sum(s["n_frames"] for s in segs),
                "n_ear_ok": sum(m["n_ear_ok"] for m in per_seg),
                "n_iris_r": sum(m["n_iris_r"] for m in per_seg),
                "n_iris_l": sum(m["n_iris_l"] for m in per_seg),
                "n_width_gate_ok": sum(m["n_width_gate_ok"] for m in per_seg),
                "n_consensus_ok": sum(m["n_consensus_ok"] for m in per_seg),
                "eye_ratio_median": _mean("eye_ratio_median"),
                "eye_ratio_bad_share": _mean("eye_ratio_bad_share"),
                "iris_ratio_median": _mean("iris_ratio_median"),
                "iris_ratio_bad_share": _mean("iris_ratio_bad_share")})
    return agg, per_seg


def run_index_map(cv2, mp, unit, out_dir, frame_no):
    """把整张脸的全部关键点连号画出来 + 导出坐标 CSV，用于一次性标定索引。

    存在这个模式的原因很直接：EYE_LANDMARKS 是凭社区索引表写的、未经权威核实，
    连续几轮靠猜和加门控都没修对。索引应当从你自己的真实帧上读出来，一次定死。
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    face_mesh = open_face_mesh(mp, 0.5)
    cap = cv2.VideoCapture(str(unit["paths"][0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    k = min(max(frame_no, 0), max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release(); face_mesh.close()
        print(f"读不到第 {k} 帧，改用 --index_frame 指定一个更小的值")
        return
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        cap.release(); face_mesh.close()
        print(f"第 {k} 帧没检到人脸，换 --index_frame")
        return
    pts = landmarks_to_points(res.multi_face_landmarks[0].landmark, w, h)
    scale = 3
    big = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    for i, (x, y) in enumerate(pts):
        xi, yi = int(x * scale), int(y * scale)
        cv2.circle(big, (xi, yi), 2, (0, 255, 255), -1)
        cv2.putText(big, str(i), (xi + 4, yi - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1, cv2.LINE_AA)
    for spec in EYE_LANDMARKS.values():
        for kind, idxs in (("upper", spec["upper"]), ("lower", spec["lower"]),
                           ("corner", (spec["corner_a"], spec["corner_b"]))):
            for i in idxs:
                if i < len(pts):
                    cv2.circle(big, (int(pts[i, 0] * scale), int(pts[i, 1] * scale)),
                               7, (0, 0, 255) if kind != "corner" else (255, 0, 0), 1)
    for idxs in EYE_IRIS_LANDMARKS.values():
        for i in idxs:
            if i < len(pts):
                cv2.circle(big, (int(pts[i, 0] * scale), int(pts[i, 1] * scale)),
                           11, (255, 255, 255), 1)
    png = Path(out_dir) / f"index_map_{Path(unit['paths'][0]).stem}_f{k}.png"
    cv2.imwrite(str(png), big)
    csv_path = Path(out_dir) / f"index_coords_f{k}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "x_px", "y_px"])
        for i, (x, y) in enumerate(pts):
            wr.writerow([i, round(float(x), 2), round(float(y), 2)])
    cap.release(); face_mesh.close()
    print(f"标定图: {png}   （圈=眼睑/眼角，白色十字=虹膜）")
    print(f"坐标表: {csv_path}")
    print("请据此确认：绿点=上眼睑、红点=下眼睑、蓝/黄点=内外眼角、白色十字=虹膜 5 点。")
    print("虹膜点必须落在虹膜盘的轮廓上（4 个）加中心（1 个）；若落在巩膜或眼睑上，"
          "请修正 eye_metrics_core.EYE_IRIS_LANDMARKS 后重跑。")


def run_dry_run(units, cfg, n_units, per_unit_frames):
    """小样本量化裁决，把"实现不了就不做"变成有判据的判断而不是主观决定。

    两趟：先抽样收集虹膜几何量，再按被试级中位数定标算遮蔽度，最后裁决。
    这样报出的动态范围与正式跑同一把尺子，不会像旧的 EAR 版那样给出误导性的量程。
    """
    cv2, mp = _require_video_deps()
    face_mesh = open_face_mesh(mp, cfg["min_conf"])
    n_frame = n_face = n_eye = 0
    eye_w, iris_d, ap_lo, ap_hi = [], [], [], []
    for u in units[:max(n_units, 1)]:
        cap = cv2.VideoCapture(str(u["paths"][0]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        h = w = 0
        for k in np.linspace(0, max(total - 1, 1), per_unit_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            n_frame += 1
            h, w = frame.shape[:2]
            res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            n_face += 1
            pts = landmarks_to_points(res.multi_face_landmarks[0].landmark, w, h)
            check_landmark_count(len(pts), where=f"{u['name']}: ")
            got = False
            for side in ("r", "l"):
                lo, hi, d, ew = iris_geometry(pts, side)
                if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(d) and np.isfinite(ew):
                    ap_lo.append(lo); ap_hi.append(hi); iris_d.append(d); eye_w.append(ew)
                    got = True
            n_eye += int(got)
        print(f"  {u['name']}: fps={cap.get(cv2.CAP_PROP_FPS):.2f} 帧数={total} 分辨率={w}x{h}")
        cap.release()
    face_mesh.close()
    if not n_frame:
        print("没有读到任何帧，检查 --layout / --root / --clip_root")
        return "reject"
    scales = {"r": {"iris_d": median_scale([np.asarray(iris_d)]),
                    "eye_w": median_scale([np.asarray(eye_w)])}}
    scales["l"] = scales["r"]
    occ, ok = iris_signals(np.asarray(ap_lo), np.asarray(ap_hi),
                           np.asarray(iris_d), np.asarray(eye_w), scales["r"], cfg)
    det = n_face / n_frame
    ratio = n_eye / n_frame
    verdict, reasons, stats = feasibility({"eye_w": eye_w, "iris_d": iris_d, "occ": occ},
                                          ratio, cfg)
    print("\n" + "=" * 72)
    print(f"可行性裁决: {verdict.upper()}   抽样 {n_frame} 帧 | 检出人脸 {det:.1%} | "
          f"虹膜几何可用 {ratio:.1%}")
    for r in reasons:
        print(f"  - {r}")
    print(f"  眼裂宽度中位数 {stats['eye_width_median_px']:.1f}px（p05 "
          f"{stats['eye_width_p05_px']:.1f}px），虹膜/眼裂 = {stats['iris_d_over_eye_w']:.2f}")
    print(f"  遮蔽度 p05={stats['occ_p05']:.3f} p95={stats['occ_p95']:.3f} "
          f"max={stats['occ_max']:.3f} 动态范围={stats['occ_dyn_range']:.3f}")
    print("=" * 72)
    print({
        "pass": "下一步：先 --dump_overlay 目视确认索引落位，再全量跑。",
        "marginal": "人工核对若干部叠加图后再用；PERCLOS 与眨眼率须标注为下界估计。",
        "reject": "按'实现不了就不做'：放弃视频路线，并把该结论写入 §12.2 的局限说明。",
    }[verdict])
    return verdict


# ── 落盘 ────────────────────────────────────────────────────────────
# 只固定"标识列 + 质量列"的顺序；指标列（随 perclos_thresholds / blink_tier_ms /
# closure_tier_s 变化）由 ordered_columns 按实际结果键追加，避免列清单与配置漂移。
SUMMARY_COLS = [
    "id", "difficulty", "state", "unit", "layout", "eyes", "n_videos", "n_segments",
    "total_frames", "valid_s", "valid_ratio",
    "eye_w_r_px", "eye_w_l_px", "median_eye_width_px",
    "iris_d_r_px", "iris_d_l_px", "iris_d_mean_px", "iris_d_over_eye_w", "iris_ratio_lr",
    "iris_ratio_median", "iris_ratio_bad_share", "fs",
    "n_frames_total", "n_face_total", "n_eye_total", "n_ear_ok",
    "n_iris_r", "n_iris_l", "n_width_gate_ok", "n_consensus_ok",
    "eye_ratio_median", "eye_ratio_bad_share",
]


def ordered_columns(rows, prefix):
    """prefix 列在前，其余按首行键序追加（不丢列、也不会因配置改动而漏列）。"""
    rest = [k for k in rows[0] if k not in prefix]
    return [k for k in prefix if k in rows[0]] + sorted(rest)


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 8)
    if isinstance(v, np.integer):
        return int(v)
    return v


def write_csv(rows, path, columns=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = columns or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _clean(r.get(k)) for k in cols})


def save_signals(out_dir, unit, segs, scales, cfg):
    """存逐帧几何量与被试级定标常数。

    有了这些，之后改阈值、改分档、做敏感性分析都只需离线重算，不必再跑 mediapipe。
    """
    d = Path(out_dir) / "signals"
    d.mkdir(parents=True, exist_ok=True)
    for s in segs:
        np.savez_compressed(
            d / f"{unit['name']}__{s['name']}.npz",
            lo_r=s["lo_r"], hi_r=s["hi_r"], id_r=s["id_r"], w_r=s["w_r"], ear_r=s["ear_r"],
            lo_l=s["lo_l"], hi_l=s["hi_l"], id_l=s["id_l"], w_l=s["w_l"], ear_l=s["ear_l"],
            fps=np.asarray([s["fps"]], dtype=float),
            iris_d_ref=np.asarray([scales["r"]["iris_d"], scales["l"]["iris_d"]], dtype=float),
            eye_w_ref=np.asarray([scales["r"]["eye_w"], scales["l"]["eye_w"]], dtype=float))


def group_by_subject(units):
    g = defaultdict(list)
    for u in units:
        g[u["id"]].append(u)
    return sorted(g.items())


def main():
    ap = argparse.ArgumentParser(
        description="FatigueGuard §12.2 独立客观通道：PERCLOS / 眨眼率 / 长闭眼时长提取")
    ap.add_argument("--layout", default="auto", choices=["auto", "clip", "video"],
                    help="auto=同一目录里两种布局各认各的（默认）；clip=[id]_[difficulty]_[task]/clip_*.mp4；"
                         "video=整段连续视频，路径含 id/easy|hard/alert|sleepy")
    ap.add_argument("--root", default="/root/autodl-tmp/shenxy/XDU/Dataset/DATA2",
                    help="统一扫描目录；给了它就只扫这一个目录，两种布局自动识别")
    ap.add_argument("--clip_root", default="/root/autodl-tmp/shenxy/XDU/Dataset/20260821_clip_mini")
    ap.add_argument("--data_root", default="/root/autodl-tmp/shenxy/XDU/Dataset/DATA2")
    ap.add_argument("--output_dir", default="./results/eye_metrics")
    ap.add_argument("--subjects", default=None, help="只处理指定 ID，逗号分隔，如 01,02,03")
    ap.add_argument("--fs", default=30.0, type=float)
    ap.add_argument("--min_conf", default=0.5, type=float)
    ap.add_argument("--sym_tol", default=0.25, type=float, help="双眼遮蔽度差上限")
    ap.add_argument("--eyes", default="both", choices=["both", "r", "l"],
                    help="both=双眼共识（默认，最严谨）；r/l=只用被试右眼/左眼单通道，"
                         "供某一只眼索引尚未确认时先出可用结果")
    ap.add_argument("--min_eye_width_px", default=18.0, type=float,
                    help="眼裂宽度低于此判为不可行；首轮跑完按实际分布回填")
    ap.add_argument("--iris_ratio_lo", default=0.22, type=float,
                    help="虹膜直径/眼裂宽度的生理下限，低于此判该帧无效（虹膜索引错位的"
                         "主要征兆）")
    ap.add_argument("--iris_ratio_hi", default=0.70, type=float,
                    help="虹膜直径/眼裂宽度的生理上限")
    ap.add_argument("--video_pattern", default="training_video.mp4",
                    help="video 布局只收匹配此文件名的视频，用于排除混在数据里的校准视频；"
                         "需要放宽时改成如 *training*.mp4 并自行确认没混入校准段")
    ap.add_argument("--clip_pattern", default="clip_*.mp4", help="clip 布局的文件名模式")
    ap.add_argument("--list_excluded", action="store_true", help="逐条打印被排除的视频")
    ap.add_argument("--index_map", default=None,
                    help="标定模式：导出全 468 点带序号的放大图与坐标 CSV，用于一次性定死索引")
    ap.add_argument("--index_frame", default=300, type=int, help="标定模式取第几帧")
    ap.add_argument("--dry_run", default=0, type=int, help=">0 只抽样诊断可行性，值为抽样单元数")
    ap.add_argument("--dry_frames", default=60, type=int, help="每个抽样单元采多少帧")
    ap.add_argument("--dump_overlay", default=None, help="输出带索引标注的叠加图目录")
    ap.add_argument("--dump_units", default=2, type=int, help="叠加图覆盖前几个单元")
    ap.add_argument("--dump_every", default=45, type=int, help="每多少帧存一张")
    ap.add_argument("--dump_max", default=12, type=int, help="每单元最多存几张")
    ap.add_argument("--limit_units", default=0, type=int, help="最多处理几个单元，冒烟测试用")
    ap.add_argument("--save_signals", action="store_true", help="存逐帧开合度 npz，供阈值敏感性复用")
    args = ap.parse_args()

    cfg = default_config()
    cfg.update({"fs": args.fs, "sym_tol": args.sym_tol, "min_conf": args.min_conf,
                "min_eye_width_px": args.min_eye_width_px, "eyes": args.eyes,
                "iris_d_ratio_lo": args.iris_ratio_lo, "iris_d_ratio_hi": args.iris_ratio_hi})

    units, excluded = discover_units(args)
    if args.limit_units:
        units = units[:args.limit_units]
    if not units:
        print("没有匹配到任何视频单元，检查 --layout / --clip_root / --data_root / --subjects")
        return
    if excluded:
        print(f"按 --video_pattern='{args.video_pattern}' / --clip_pattern='{args.clip_pattern}' "
              f"排除了 {len(excluded)} 项非实验视频")
        if args.list_excluded:
            for x in excluded[:60]:
                print(f"   - {x}")
            if len(excluded) > 60:
                print(f"   ... 其余 {len(excluded) - 60} 项省略")
    else:
        print("未排除任何视频：确认数据里确实没有校准片段，或放宽 --video_pattern")
    n_vid = sum(len(u["paths"]) for u in units)
    by_layout = defaultdict(int)
    for u in units:
        by_layout[u["layout"]] += 1
    print(f"共 {len(units)} 个单元（被试×难度×状态）| 布局: "
          f"{dict(by_layout)} | 视频文件合计 {n_vid} 个 | "
          f"涉及被试 {len({u['id'] for u in units})} 人")
    if by_layout.get("clip"):
        print("  [提示] 含切片布局：PERCLOS/眨眼率一律在 (被试,难度,状态) 条件水平聚合后比较，"
              "不要逐切片比较")

    if args.index_map:
        cv2, mp = _require_video_deps()
        run_index_map(cv2, mp, units[0], args.index_map, args.index_frame)
        return

    if args.dry_run:
        run_dry_run(units, cfg, args.dry_run, args.dry_frames)
        return

    cv2, mp = _require_video_deps()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, seg_rows = [], []
    t0 = time.time()
    done = 0
    dump_state = None
    if args.dump_overlay:
        dump_state = {"dir": args.dump_overlay, "tag": "", "every": args.dump_every,
                      "left": max(args.dump_units, 1) * max(args.dump_max, 1)}
    groups = group_by_subject(units)
    for si, (sid, us) in enumerate(groups, 1):
        per_unit = {}
        for u in us:
            try:
                per_unit[u["name"]] = extract_unit(cv2, mp, u, cfg, dump_state)
            except Exception as e:
                print(f"  [FAIL] {u['name']}: {type(e).__name__}: {e}")
                per_unit[u["name"]] = []
        all_segs = [s for segs in per_unit.values() for s in segs]
        if not all_segs:
            print(f"  [SKIP] 被试 {sid}: 未读到任何有效帧")
            continue
        scales = subject_scales(all_segs, cfg)
        n_face = sum(s["n_face"] for s in all_segs)
        n_read = sum(s["n_frames"] for s in all_segs)
        n_iris_r = sum(int(np.count_nonzero(np.isfinite(s["id_r"]))) for s in all_segs)
        n_iris_l = sum(int(np.count_nonzero(np.isfinite(s["id_l"]))) for s in all_segs)
        need = ["r", "l"] if args.eyes == "both" else [args.eyes]
        missing = [k for k in need if not (np.isfinite(scales[k]["iris_d"])
                                           and np.isfinite(scales[k]["eye_w"]))]
        if missing:
            print(f"  [SKIP] 被试 {sid}: {'/'.join(missing)} 眼的虹膜几何不可用")
            print(f"         读帧 {n_read} -> 检到人脸 {n_face} -> "
                  f"虹膜可用 右 {n_iris_r} / 左 {n_iris_l}")
            if not n_face:
                print("         一帧人脸都没检到：检查视频能否解码、分辨率、--min_conf")
            elif not (n_iris_r or n_iris_l):
                print("         检到人脸但虹膜点全部取不到：refine_landmarks 未生效，"
                      "或 EYE_IRIS_LANDMARKS 索引错位 -> 跑 --index_map 复核")
            else:
                print("         一侧有虹膜另一侧没有：先用 --eyes r 或 --eyes l 出可用结果")
            continue
        for u in us:
            segs = per_unit.get(u["name"], [])
            if not segs:
                continue
            agg, per_seg = summarize_unit(u, segs, scales, cfg)
            rows.append(agg)
            seg_rows.extend(per_seg)
            if args.save_signals:
                save_signals(out_dir, u, segs, scales, cfg)
            done += 1
        el = time.time() - t0
        idr = (scales["iris_d_mean"] / scales["eye_w_mean"]
               if scales["eye_w_mean"] else float("nan"))
        print(f"  [{si}/{len(groups)}] 被试 {sid}: {len(us)} 单元，{len(all_segs)} 段，"
              f"眼裂中位 {scales['eye_w_mean']:.1f}px，虹膜中位 {scales['iris_d_mean']:.1f}px，"
              f"虹膜/眼裂={idr:.2f}，左右虹膜比={scales['iris_ratio_lr']:.2f}，累计 {el:.0f}s")
        if not (cfg["iris_d_ratio_lo"] <= idr <= cfg["iris_d_ratio_hi"]):
            print(f"    [索引可疑] 虹膜直径/眼裂宽度 = {idr:.2f} 不在生理区间 "
                  f"[{cfg['iris_d_ratio_lo']}, {cfg['iris_d_ratio_hi']}]，"
                  f"多半是 EYE_IRIS_LANDMARKS 错位；先跑 --index_map 复核再全量")
        if np.isfinite(scales["iris_ratio_lr"]) and scales["iris_ratio_lr"] > 2.0:
            print(f"    [索引可疑] 左右眼虹膜直径相差 {scales['iris_ratio_lr']:.1f} 倍，"
                  f"生理上不应如此，多半是某一只眼的索引错位")

    if not rows:
        print("\n全部单元都未产出结果：视频路线不可行，按'实现不了就不做'处理")
        return

    write_csv(rows, out_dir / "eye_metrics_by_condition.csv", ordered_columns(rows, SUMMARY_COLS))
    write_csv(seg_rows, out_dir / "eye_metrics_by_segment.csv")

    minv = cfg["min_valid_ratio"]
    bad = [r for r in rows if not np.isfinite(r.get("valid_ratio", float("nan")))
           or r["valid_ratio"] < minv]
    tiny = [r for r in rows if np.isfinite(r.get("median_eye_width_px", float("nan")))
            and r["median_eye_width_px"] < cfg["min_eye_width_px"]]
    idr = [r["iris_d_over_eye_w"] for r in rows if np.isfinite(r.get("iris_d_over_eye_w", np.nan))]
    idr_bad = [v for v in idr
               if not (cfg["iris_d_ratio_lo"] <= v <= cfg["iris_d_ratio_hi"])]
    print(f"\n完成 {done} 个单元，总耗时 {time.time() - t0:.0f}s")
    print(f"结果: {out_dir / 'eye_metrics_by_condition.csv'}（单元级）"
          f" / eye_metrics_by_segment.csv（片段级）")
    if idr:
        print(f"虹膜直径/眼裂宽度: 中位 {float(np.median(idr)):.2f}"
              f"（生理区间 [{cfg['iris_d_ratio_lo']}, {cfg['iris_d_ratio_hi']}]），"
              f"越界 {len(idr_bad)}/{len(idr)} 个单元"
              + ("" if not idr_bad else "  <-- 越界提示虹膜索引可能错位，先 --index_map 复核"))
    print(f"有效帧占比 < {minv:.0%} 的单元: {len(bad)} 个 "
          f"{[r['unit'] for r in bad][:8]}（低于方案 §12.1 纳入标准）")
    worst = min((r["valid_ratio"] for r in rows if np.isfinite(r.get("valid_ratio", float("nan")))),
                default=float("nan"))
    if np.isfinite(worst) and worst < 0.5:
        print("  [归因] n_eye_total 高但 valid_ratio 很低 -> 拒绝发生在双眼门控而非关键点抽取："
              "依次看 n_r_ok / n_l_ok（哪只眼是死的）、width_ratio_median（左右眼宽度是否失衡）、"
              "ref_ratio_lr（左右定标是否差数倍）。失衡多半是该侧 EYE_LANDMARKS 索引错位，"
              "用 --index_map 从真实帧标定，不要手改猜测")
    if tiny:
        print(f"[WARN] 眼裂宽度 < {cfg['min_eye_width_px']}px 的单元 {len(tiny)} 个："
              f"{[r['unit'] for r in tiny][:8]} —— 这些单元的 PERCLOS/眨眼率不可信，"
              f"考虑按被试剔除")
    if args.dump_overlay:
        print(f"叠加图在 {args.dump_overlay}：确认绿点=上睑、红点=下睑落在眼睑边缘；"
              f"若错位请修正 eye_metrics_core.EYE_LANDMARKS 后重跑")


if __name__ == "__main__":
    main()
