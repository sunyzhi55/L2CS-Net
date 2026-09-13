#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
切片版 FatigueGuard 预处理（第一步 + 第二步）

适配新的数据组织方式：
    clip_root/
    ├── 01_easy_alert/
    │   ├── clip_1.mp4
    │   ├── clip_2.mp4
    │   └── ...
    ├── 01_easy_sleepy/
    └── ...

每个子文件夹 `[id]_[difficulty]_[task]` 下有若干 `clip_N.mp4` 切片：
  - easy 任务每个切片 3 秒
  - hard 任务每个切片 6 秒

目标点文件与校准矩阵仍在“原来的数据路径”下（data_root/{id}/{task}/{difficulty}/）。
每个切片视频 ↔ 目标点文件中的一行（easy: centers_easy.txt 的一行；
hard: Gaze_hard_centers.npy 的一个 entry）。每个切片第一帧 gaze 提取为负，故丢弃。

本脚本把一个子文件夹下所有 clip 的预处理结果拼接为一个
`[id]_[difficulty]_[task].jsonl`（字段与原管线输出一致，额外附带 clip_idx /
clip_frame_idx 便于溯源）。第三步 TensorFlow 校准仍使用现有的
`scripts/tf_calibrate_jsonl_batch.py`，无需改动。

用法（在项目根目录运行）：
    python scripts/clip_preprocess_batch.py \
        --clip_root  /path/to/20260821_clip \
        --data_root  /path/to/FatigueGuardData/Data \
        --output_dir /path/to/output \
        --device cuda:0 --mode sfm
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

import cv2
import numpy as np

from FatigueGuard_preprocess_single import (
    GazeToPoint,
    pitch_yaw_to_gaze_vector,
    mm_to_pixel,
    _to_jsonable,
)
from gaze_tracking.model import EyeModel
from l2cs import Pipeline, select_device
import utilities.utils as util


# ── 目标点 / 校准矩阵辅助（与原批处理脚本保持一致）─────────────
def load_easy_targets(txt_path):
    targets = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                x, y = line.split(",")
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                x, y = parts[0], parts[1]
            targets.append((int(float(x)), int(float(y))))
    return targets


def load_hard_targets(npy_path):
    return list(np.load(npy_path, allow_pickle=True))


def find_calibration_files(subject_dir, state, difficulty):
    """与 FatigueGuard_preprocess_batch.py / ext_jsonl_to_screen.py 一致的查找规则。"""
    candidates = [
        (subject_dir / "results", subject_dir / "STransG"),
        (subject_dir / state / difficulty / "results", subject_dir / state / difficulty / "STransG"),
        (subject_dir / state / "results", subject_dir / state / "STransG"),
    ]
    for result_dir, strans_dir in candidates:
        if result_dir.exists() and strans_dir.exists():
            required = ["STransG.npy", "StG.npy", "scaleWtG.npy", "STransW.npy", "StW.npy"]
            if all((strans_dir / f).exists() for f in required):
                return result_dir, strans_dir
    return None, None


def find_task_dir(data_root, id_tok, task, task_raw, difficulty):
    """在原始数据路径下定位 {id}/{task}/{difficulty} 目录，兼容若干命名变体。"""
    candidates = [
        data_root / id_tok / task / difficulty,        # 规范化 task（如 sleepy）
        data_root / id_tok / task_raw / difficulty,    # 原始 task token（如 sleepy1）
        data_root / id_tok / difficulty / task,        # 备选顺序
        data_root / id_tok / difficulty / task_raw,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ── 文件夹名解析：[id]_[difficulty]_[task]（容忍 task 上的数字后缀）────
def parse_folder(name):
    tokens = name.split("_")
    if not tokens or not tokens[0].isdigit():
        return None
    id_tok = tokens[0]
    difficulty = None
    task = None
    task_raw = None
    for tok in tokens[1:]:
        if difficulty is None and tok.startswith("easy"):
            difficulty = "easy"
        elif difficulty is None and tok.startswith("hard"):
            difficulty = "hard"
        elif tok.startswith("alert"):
            task_raw = tok
            task = "alert"
        elif tok.startswith("sleep"):
            task_raw = tok
            task = "sleepy"
    if difficulty is None or task is None:
        return None
    return id_tok, difficulty, task, task_raw


def clip_sort_key(p: Path):
    m = re.search(r"clip_(\d+)", p.stem)
    return (int(m.group(1)) if m else 10 ** 9, p.name)


# ── 人脸特征提取（与 GazeToPoint._extract_face_features 等价，去除调试打印）──
def extract_face_features(model, frame, gaze_result):
    bbox = None
    landmarks = None
    try:
        face_boxes = model.face_detection.predict(frame)
    except Exception:
        face_boxes = []
    if face_boxes:
        face_box = face_boxes[0]
        bbox = [float(v) for v in face_box]
        try:
            face = model.get_crop_image(frame, face_box)
            if face is not None and face.size > 0:
                face_landmarks = model.facial_landmark_35.predict(face)
                xmin, ymin, _, _ = face_box
                landmarks = [[float(pt[0] + xmin), float(pt[1] + ymin)] for pt in face_landmarks]
        except Exception:
            landmarks = None

    return {
        "face_detection_bbox": bbox,
        "facial_landmark_35": landmarks,
        "RetinaFace_bbox": _to_jsonable(gaze_result.bboxes[0])
            if getattr(gaze_result, "bboxes", None) is not None and len(gaze_result.bboxes) else None,
        "RetinaFace_landmarks": _to_jsonable(gaze_result.landmarks[0])
            if getattr(gaze_result, "landmarks", None) is not None and len(gaze_result.landmarks) else None,
    }


def _finite_or_nan(x):
    return float(x) if np.isfinite(x) else np.nan


# ── 单切片处理：复用 GazeToPoint 的投影方法，逐帧提取并写出到 fp ──
def process_clip(gtp, gaze_pipeline, eye_model, cap, difficulty, target_for_clip,
                 clip_idx, sfm, out_fp, state):
    """
    state: 跨切片连续计数器 {"fi": int, "ts": float}
    target_for_clip: easy -> [x, y]；hard -> [[x, y], ...]
    """
    # 每个切片独立重置中值滤波队列，避免跨切片污染 (如果注释掉后，表示跨切片连续滤波)
    # =================================================
    gtp.QueueGaze = np.nan * np.zeros((3, 5))
    # =================================================

    width, height = gtp.width, gtp.height
    width_mm, height_mm = gtp.width_mm, gtp.height_mm

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (np.isfinite(fps) and fps and fps > 0):
        fps = 30.0
    dt = 1.0 / fps

    frame_prev = None
    WTransG1 = np.eye(4)
    clip_frame_idx = 0

    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            break
        if not ret or frame is None:
            break
        if frame.size == 0:
            clip_frame_idx += 1
            continue

        # L2CS 提取 (pitch, yaw)
        try:
            gaze_result = gaze_pipeline.step(frame)
        except Exception:
            gaze_result = None

        if gaze_result is not None:
            pitch = (float(gaze_result.pitch[0])
                     if getattr(gaze_result, "pitch", None) is not None and len(gaze_result.pitch)
                     else np.nan)
            yaw = (float(gaze_result.yaw[0])
                   if getattr(gaze_result, "yaw", None) is not None and len(gaze_result.yaw)
                   else np.nan)
            confidence = (float(gaze_result.scores[0])
                          if getattr(gaze_result, "scores", None) is not None and len(gaze_result.scores)
                          else np.nan)
            face_features = extract_face_features(eye_model, frame, gaze_result)
        else:
            pitch = yaw = confidence = np.nan
            face_features = {
                "face_detection_bbox": None,
                "facial_landmark_35": None,
                "RetinaFace_bbox": None,
                "RetinaFace_landmarks": None,
            }

        if np.isfinite(pitch) and np.isfinite(yaw):
            gaze = pitch_yaw_to_gaze_vector(pitch, yaw)
            
            gaze = util.MedianFilter(gtp.QueueGaze, gaze)
            if frame_prev is not None and sfm:
                try:
                    WTransG1, _, _ = gtp.sfm.get_GazeToWorld(eye_model, frame_prev, frame)
                except Exception:
                    pass  # SFM 失败则沿用上一帧的 WTransG1
            frame_prev = frame
            if sfm:
                # print(f"WTransG1: \n{WTransG1}")
                # print("gaze:", gaze)                       # 应一致（你已确认）
                # print("WTransG1:\n", WTransG1)            # 若不一致 → 原因 B
                # print("STransW:\n", gtp.STransW)          # 若不一致 → 原因 A
                # print("scaleWtG:", gtp.scaleWtG)
                # print("StW:", gtp.StW)
                FSgaze, _, _ = gtp._getGazeOnScreen_sfm(gaze, WTransG1)
            else:
                FSgaze, _, _ = gtp._getGazeOnScreen(gaze)
        else:
            gaze = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            FSgaze = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        
        FSgaze = np.asarray(FSgaze, dtype=np.float64).reshape(3)
        print(f"FSgaze: {FSgaze}")

        # mm -> px（仅在有限时计算，避免 int(NaN) 崩溃）
        if np.all(np.isfinite(FSgaze)):
            s_px = mm_to_pixel(FSgaze, width, height, width_mm, height_mm)
            x_px = int(np.clip(s_px[0], 0, width - 1))
            y_px = int(np.clip(s_px[1], 0, height - 1))
        else:
            x_px, y_px = -1, -1

        # 丢弃每个切片的第一帧（gaze 提取为负）
        if clip_frame_idx != 0:
            record = {
                "timestamp": state["ts"],
                "frame_idx": state["fi"],
                "clip_idx": clip_idx,
                "clip_frame_idx": clip_frame_idx,
                "pitch_yaw_rad": [pitch, yaw],
                "gaze_xyz": [
                    _finite_or_nan(gaze[0]),
                    _finite_or_nan(gaze[1]),
                    _finite_or_nan(gaze[2]),
                ],
                "gaze_screen_xy_mm": [
                    _finite_or_nan(FSgaze[0]),
                    _finite_or_nan(FSgaze[1]),
                ],
                "gaze_screen_xy_px": [x_px, y_px],
                "face_detection_bbox": face_features["face_detection_bbox"],
                "facial_landmark_35": face_features["facial_landmark_35"],
                "RetinaFace_bbox": face_features["RetinaFace_bbox"],
                "RetinaFace_landmarks": face_features["RetinaFace_landmarks"],
                "confidence": confidence,
            }
            if difficulty == "easy":
                record["target_xy_px"] = list(target_for_clip)
            else:
                record["target_centers_xy_px"] = [list(c) for c in target_for_clip]

            out_fp.write(json.dumps(_to_jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
            state["fi"] += 1
            state["ts"] += dt

        clip_frame_idx += 1

    return clip_frame_idx  # 该切片总帧数（含被丢弃的第一帧）


def get_clip_target(targets, ci, difficulty):
    """取第 ci 个切片对应的目标点；越界时用最后一个并返回告警标志。"""
    clipped = False
    if ci < len(targets):
        t = targets[ci]
    else:
        t = targets[-1]
        clipped = True
    if difficulty == "easy":
        return list(t), clipped
    centers = t["centers"] if isinstance(t, dict) else t
    return [list(c) for c in centers], clipped


def process_folder(folder, data_root, output_dir, args, gaze_pipeline, eye_model):
    name = folder.name
    parsed = parse_folder(name)
    if parsed is None:
        print(f"[SKIP] 无法解析文件夹名: {name}")
        return False
    id_tok, difficulty, task, task_raw = parsed
    print(f"\n[PROC] {name}  (id={id_tok}, difficulty={difficulty}, task={task})")

    # 定位原始数据目录（目标点 + 校准矩阵所在）
    task_dir = find_task_dir(data_root, id_tok, task, task_raw, difficulty)
    if task_dir is None:
        print(f"  [SKIP] 在 data_root 下找不到任务目录: {id_tok}/{task}/{difficulty}")
        return False

    # 目标点
    if difficulty == "easy":
        target_file = task_dir / "centers_easy.txt"
        if not target_file.exists():
            print(f"  [SKIP] 缺少目标点文件: {target_file}")
            return False
        targets = load_easy_targets(str(target_file))
    else:
        target_file = task_dir / "Gaze_hard_centers.npy"
        if not target_file.exists():
            print(f"  [SKIP] 缺少目标点文件: {target_file}")
            return False
        targets = load_hard_targets(str(target_file))
    if not targets:
        print(f"  [SKIP] 目标点文件为空: {target_file}")
        return False

    # 校准矩阵
    subject_dir = data_root / id_tok
    result_dir, strans_dir = find_calibration_files(subject_dir, task, difficulty)
    if result_dir is None:
        print(f"  [SKIP] 找不到校准矩阵: {id_tok}/{task}/{difficulty}")
        return False

    # 切片列表（按 clip_N 数值排序）
    clips = sorted(folder.glob("clip_*.mp4"), key=clip_sort_key)
    if not clips:
        print(f"  [SKIP] 未找到 clip_*.mp4: {folder}")
        return False
    if args.max_clips and args.max_clips > 0:
        clips = clips[: args.max_clips]

    if len(clips) != len(targets):
        print(f"  [WARN] 切片数({len(clips)}) != 目标点数({len(targets)})；"
              f"超出的切片将复用最后一个目标点")

    # 构建 GazeToPoint（加载该校准矩阵，整个文件夹复用）
    margs = argparse.Namespace(
        stg_npy=str(strans_dir / "STransG.npy"),
        stg_aux_npy=str(strans_dir / "StG.npy"),
        scale_wtg=str(strans_dir / "scaleWtG.npy"),
        stw_npy=str(strans_dir / "STransW.npy"),
        stw_aux_npy=str(strans_dir / "StW.npy"),
        directory=str(result_dir.parent),
        camera_data_dir=args.camera_data_dir,
        device=args.device,
        weights=args.weights,
        arch=args.arch,
        sfm_openvino_device=args.sfm_openvino_device,
    )
    gtp = GazeToPoint(Path(margs.directory), margs)
    sfm = (args.mode == "sfm")

    out_path = output_dir / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"fi": 0, "ts": 0.0}

    with out_path.open("w", encoding="utf-8") as fp:
        for ci, clip_path in enumerate(clips):
            target_for_clip, clipped = get_clip_target(targets, ci, difficulty)
            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                print(f"  [WARN] 无法打开 {clip_path.name}，跳过")
                continue
            n = process_clip(gtp, gaze_pipeline, eye_model, cap,
                             difficulty, target_for_clip, ci, sfm, fp, state)
            cap.release()
            tag = " (复用最后目标点)" if clipped else ""
            print(f"  [{ci + 1}/{len(clips)}] {clip_path.name} 帧数={n}{tag}")

    print(f"  [OK] -> {out_path}  (共 {state['fi']} 帧)")
    return True


def main():
    parser = argparse.ArgumentParser(description="切片版 FatigueGuard 预处理（第一步+第二步）")
    parser.add_argument("--clip_root", default="/root/autodl-tmp/shenxy/XDU/Dataset/20260821_clip_mini",
                        help="切片数据根目录，下含 [id]_[difficulty]_[task]/ 子文件夹")
    parser.add_argument("--data_root", default="/root/autodl-tmp/shenxy/XDU/Dataset/DATA2",
                        help="原始数据根目录（含目标点文件与校准矩阵）")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/shenxy/XDU/Dataset/20260821_clip_output",
                        help="输出 jsonl 目录")
    parser.add_argument("--camera_data_dir", default="./camera_data",
                        help="相机标定数据目录（默认 ./camera_data）")
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0")
    parser.add_argument("--weights", default="models/L2CSNet_gaze360.pkl",
                        help="L2CS 权重文件路径")
    parser.add_argument("--arch", default="ResNet50", help="ResNet18/34/50/101/152")
    parser.add_argument("--mode", default="sfm", choices=["global", "sfm"],
                        help="global=STransG, sfm=STransW@WTransG（逐帧头部运动补偿）")
    parser.add_argument("--sfm_openvino_device", default="CPU", help="SFM 的 OpenVINO 设备")
    parser.add_argument("--max_frames", default=0, type=int,
                        help="（保留参数，暂未启用）")
    parser.add_argument("--max_clips", default=0, type=int,
                        help="每个文件夹最多处理的切片数，0 表示全部（测试用）")
    parser.add_argument("--subjects", default=None,
                        help="只处理指定受试者 ID，逗号分隔，如 01,02,03")
    args = parser.parse_args()

    clip_root = Path(args.clip_root)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 子文件夹（只要包含 clip_*.mp4 的目录）
    folders = sorted([d for d in clip_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    if args.subjects:
        wanted = {s.strip() for s in args.subjects.split(",")}
        folders = [d for d in folders if parse_folder(d.name) and parse_folder(d.name)[0] in wanted]

    if not folders:
        print(f"在 {clip_root} 下未找到任何子文件夹")
        return

    print(f"共找到 {len(folders)} 个子文件夹")
    print(f"模式: {args.mode}")

    # 全局只加载一次 L2CS 推理管线与 OpenVINO EyeModel（与受试者无关）
    print("加载 L2CS 管线与 OpenVINO EyeModel ...")
    gaze_pipeline = Pipeline(
        weights=Path(args.weights),
        arch=args.arch,
        device=select_device(args.device, batch_size=1),
    )
    eye_model = EyeModel(".")

    success = skip = 0
    for folder in folders:
        try:
            ok = process_folder(folder, data_root, output_dir, args,
                                gaze_pipeline, eye_model)
        except Exception as e:
            print(f"[ERROR] {folder.name} 处理失败: {e}")
            ok = False
        if ok:
            success += 1
        else:
            skip += 1

    print(f"\n完成！成功: {success}, 跳过: {skip}")
    print("下一步：对输出 jsonl 运行 TensorFlow 校准：")
    print("  python scripts/tf_calibrate_jsonl_batch.py "
          f"--input_path {output_dir} --output_dir {output_dir}_tfCali")


if __name__ == "__main__":
    main()
