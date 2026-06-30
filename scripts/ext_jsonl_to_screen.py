#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将外部方法生成的 JSONL（含 pitch_yaw_rad）通过校准矩阵映射到屏幕坐标。

用法:
    python scripts/ext_jsonl_to_screen.py \
        --data_root /path/to/FatigueGuardData/Data \
        --jsonl_dir  /path/to/external_jsonl \
        --output_dir /path/to/screen_mapped \
        --mode global
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
    load_matrix,
    mm_to_pixel,
    pitch_yaw_to_gaze_vector,
)
from gaze_tracking.model import EyeModel
from utilities.utils import MedianFilter
from sfm.sfm_module import SFM


# ── 屏幕参数（与项目一致） ─────────────────────────────────────
SCREEN_WIDTH_PX  = 1920
SCREEN_HEIGHT_PX = 1080
SCREEN_WIDTH_MM  = 344.0
SCREEN_HEIGHT_MM = 193.0


def parse_jsonl_filename(filename: str):
    """解析文件名返回 (subject_id, difficulty, state) 或 None。"""
    stem = Path(filename).stem
    m = re.match(r"^(\d+)_(easy|hard)_(alert|sleepy)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_calibration_files(subject_dir, state, difficulty):
    """查找校准矩阵目录，与批处理脚本逻辑一致。"""
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


def augment_easy_targets(targets, total_frames, fps=30):
    frames_per_target = fps * 3
    augmented = []
    for t in targets:
        augmented.extend([t] * frames_per_target)
    if len(augmented) < total_frames:
        augmented.extend([targets[-1]] * (total_frames - len(augmented)))
    return augmented[:total_frames]


def augment_hard_targets(targets, total_frames):
    frames_per_index = 180
    augmented = []
    for item in targets:
        augmented.extend([item['centers']] * frames_per_index)
    if len(augmented) < total_frames:
        augmented.extend([targets[-1]['centers']] * (total_frames - len(augmented)))
    return augmented[:total_frames]


def _to_jsonable(value):
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def process_one_file(jsonl_path, data_root, output_dir, mode, camera_data_dir):
    """处理单个外部 JSONL 文件：投影到屏幕坐标并合并目标点。"""
    parsed = parse_jsonl_filename(jsonl_path.name)
    if parsed is None:
        print(f"  [SKIP] 无法解析文件名: {jsonl_path.name}")
        return False

    subject_id, difficulty, state = parsed
    subject_dir = data_root / subject_id

    # 查找校准矩阵
    result_dir, strans_dir = find_calibration_files(subject_dir, state, difficulty)
    if result_dir is None:
        print(f"  [SKIP] 找不到校准矩阵: {subject_id}/{state}/{difficulty}")
        return False

    # 查找目标点文件
    task_dir = subject_dir / state / difficulty
    target_data = None
    if difficulty == "easy":
        target_file = task_dir / "centers_easy.txt"
        if target_file.exists():
            target_data = load_easy_targets(str(target_file))
    else:
        target_file = task_dir / "Gaze_hard_centers.npy"
        if target_file.exists():
            target_data = load_hard_targets(str(target_file))

    # 加载 JSONL
    records = load_jsonl(jsonl_path)
    if not records:
        print(f"  [SKIP] JSONL 为空: {jsonl_path.name}")
        return False

    # 跳过第一行（与原管线 frame_idx==0 对应）
    records = records[1:]
    total_frames = len(records)

    # 构建 GazeToPoint（复用投影逻辑）
    import argparse as _ap
    args = _ap.Namespace()
    args.stg_npy = str(strans_dir / "STransG.npy")
    args.stw_npy = str(strans_dir / "STransW.npy")
    args.scale_wtg = str(strans_dir / "scaleWtG.npy")
    args.stg_aux_npy = str(strans_dir / "StG.npy")
    args.stw_aux_npy = str(strans_dir / "StW.npy")
    args.directory = str(result_dir.parent)
    args.camera_data_dir = camera_data_dir
    args.device = "cpu"
    args.weights = "models/L2CSNet_gaze360.pkl"
    args.arch = "ResNet50"
    args.sfm_openvino_device = "CPU"

    gaze_to_point = GazeToPoint(Path(args.directory), args)

    # SFM 模式：加载视频和模型
    sfm_model = None
    video_cap = None
    if mode == "sfm":
        video_path = task_dir / "training_video.mp4"
        if not video_path.exists():
            print(f"  [WARN] SFM 模式需要视频但不存在: {video_path}，回退到 global 模式")
            mode = "global"
        else:
            sfm_model = SFM(Path(args.directory), args)
            video_cap = cv2.VideoCapture(str(video_path))
            # 读取并丢弃第 0 帧（JSONL 从 frame 1 开始）
            ret0, _ = video_cap.read()
            if not ret0:
                print(f"  [WARN] 视频第 0 帧读取失败，回退到 global 模式")
                video_cap.release()
                video_cap = None
                mode = "global"

    # 初始化中值滤波器
    queue_gaze = np.nan * np.zeros((3, 5))
    eye_model = None
    frame_prev = None

    # 扩展目标点
    if target_data is not None:
        if difficulty == "easy":
            fps = 30
            if video_cap is not None:
                fps = int(video_cap.get(cv2.CAP_PROP_FPS)) or 30
            augmented_targets = augment_easy_targets(target_data, total_frames, fps)
        else:
            augmented_targets = augment_hard_targets(target_data, total_frames)
    else:
        augmented_targets = None

    # 写出 JSONL
    out_name = f"{subject_id}_{difficulty}_{state}.jsonl"
    out_path = output_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fp:
        for idx, record in enumerate(records):
            pitch_yaw = record.get("pitch_yaw_rad")
            if not pitch_yaw or len(pitch_yaw) < 2:
                gaze = np.array([np.nan, np.nan, np.nan])
            else:
                pitch_val = pitch_yaw[0]
                yaw_val = pitch_yaw[1]
                if pitch_val is None or yaw_val is None:
                    gaze = np.array([np.nan, np.nan, np.nan])
                else:
                    gaze = pitch_yaw_to_gaze_vector(float(pitch_val), float(yaw_val))
                    gaze = MedianFilter(queue_gaze, gaze)

            # SFM 模式：读取视频帧计算 WTransG
            w_trans_g = np.eye(4)
            if mode == "sfm" and video_cap is not None:
                ret, frame_curr = video_cap.read()
                if ret and frame_curr is not None:
                    if eye_model is None:
                        eye_model = EyeModel("./")
                    if frame_prev is not None:
                        try:
                            w_trans_g, _, _ = sfm_model.get_GazeToWorld(eye_model, frame_prev, frame_curr)
                        except Exception:
                            pass  # SFM 失败时用单位矩阵
                    frame_prev = frame_curr.copy()

            # 投影到屏幕
            if np.all(np.isfinite(gaze)):
                if mode == "sfm":
                    fs_gaze, _, _ = gaze_to_point._getGazeOnScreen_sfm(gaze, w_trans_g)
                else:
                    fs_gaze, _, _ = gaze_to_point._getGazeOnScreen(gaze)
                fs_gaze = np.asarray(fs_gaze, dtype=np.float64).reshape(3)
            else:
                fs_gaze = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

            # mm → 像素
            if np.all(np.isfinite(fs_gaze)):
                pt_px = mm_to_pixel(
                    fs_gaze,
                    SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX,
                    SCREEN_WIDTH_MM, SCREEN_HEIGHT_MM,
                )
                x_px = int(np.clip(pt_px[0], 0, SCREEN_WIDTH_PX - 1))
                y_px = int(np.clip(pt_px[1], 0, SCREEN_HEIGHT_PX - 1))
            else:
                x_px, y_px = -1, -1

            # 构建输出记录
            out_record = {
                "timestamp": record.get("timestamps", record.get("timestamp")),
                "frame_idx": record.get("frame_idx"),
                "pitch_yaw_rad": _to_jsonable(pitch_yaw) if pitch_yaw else None,
                "gaze_xyz": _to_jsonable(gaze.tolist()) if np.any(np.isfinite(gaze)) else None,
                "gaze_screen_xy_mm": _to_jsonable(fs_gaze[:2].tolist()),
                "gaze_screen_xy_px": [x_px, y_px],
            }

            # 合并目标点
            if augmented_targets is not None and idx < len(augmented_targets):
                if difficulty == "easy":
                    out_record["target_xy_px"] = list(augmented_targets[idx])
                else:
                    out_record["target_centers_xy_px"] = [list(c) for c in augmented_targets[idx]]

            fp.write(json.dumps(_to_jsonable(out_record), ensure_ascii=False, allow_nan=False) + "\n")

    if video_cap is not None:
        video_cap.release()

    print(f"  [OK] {out_name} ({total_frames} 帧)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="将外部 JSONL 通过校准矩阵映射到屏幕坐标。"
    )
    parser.add_argument("--data_root", default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Data_original", help="原始数据根目录")
    parser.add_argument("--jsonl_dir", default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_puregaze/GazeJsonLine", help="外部 JSONL 目录")
    parser.add_argument("--output_dir", default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_puregaze/screen_Output", help="输出目录")
    parser.add_argument("--mode", default="sfm", choices=["global", "sfm"],
                        help="投影模式：global（默认）或 sfm")
    parser.add_argument("--camera_data_dir", default="./camera_data",
                        help="相机标定数据目录")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    jsonl_dir = Path(args.jsonl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"在 {jsonl_dir} 下未找到 JSONL 文件")
        return

    print(f"共找到 {len(jsonl_files)} 个 JSONL 文件")
    print(f"模式: {args.mode}")
    print()

    success = 0
    skip = 0
    for jsonl_path in jsonl_files:
        print(f"[PROC] {jsonl_path.name}")
        ok = process_one_file(jsonl_path, data_root, output_dir, args.mode, args.camera_data_dir)
        if ok:
            success += 1
        else:
            skip += 1

    print(f"\n完成！成功: {success}, 跳过: {skip}")


if __name__ == "__main__":
    main()
