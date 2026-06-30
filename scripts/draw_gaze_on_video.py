#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在原始视频上逐帧绘制注视视线方向，输出可视化视频。

用法:
    python scripts/draw_gaze_on_video.py \
        --video_root /path/to/FatigueGuardData/Data \
        --jsonl_dir  /path/to/FatigueGuard_jsonl \
        --output_dir /path/to/gaze_videos
"""

import sys, os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJ_ROOT = _os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJ_ROOT)

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


# ── 颜色常量 (BGR) ──────────────────────────────────────────────
COLOR_BBOX = (0, 255, 0)        # 绿色 - 人脸框
COLOR_ARROW = (0, 0, 255)       # 红色 - 注视线
COLOR_KEYPOINT = (255, 180, 0)  # 蓝色 - 面部关键点
COLOR_GAZE_PT = (0, 255, 255)   # 黄色 - 屏幕注视点
COLOR_TEXT = (255, 255, 255)    # 白色 - 信息文字
COLOR_BG_TEXT = (0, 0, 0)       # 黑色 - 文字背景


def parse_jsonl_filename(filename: str):
    """
    解析 JSONL 文件名，返回 (subject_id, difficulty, state) 或 None。
    例如: '01_easy_alert.jsonl' -> ('01', 'easy', 'alert')
    """
    stem = Path(filename).stem
    m = re.match(r"^(\d+)_(easy|hard)_(alert|sleepy)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def load_jsonl(path: Path):
    """逐行读取 JSONL，返回 record 列表。"""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_valid_point(pt) -> bool:
    """判断坐标点是否有效（非 None、非 NaN）。"""
    if pt is None:
        return False
    if isinstance(pt, (list, tuple, np.ndarray)):
        return all(math.isfinite(float(v)) for v in pt[:2])
    return False


def draw_gaze_arrow(frame, bbox, pitch, yaw, arrow_length=150):
    """
    从人脸框中心画出注视视线箭头。
    采用 L2CS-Net 的球面投影公式。
    """
    if bbox is None:
        return

    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    if not (math.isfinite(pitch) and math.isfinite(yaw)):
        return

    dx = -arrow_length * math.sin(yaw) * math.cos(pitch)
    dy = -arrow_length * math.sin(pitch)

    pt_start = (cx, cy)
    pt_end = (int(cx + dx), int(cy + dy))

    cv2.arrowedLine(frame, pt_start, pt_end, COLOR_ARROW, 2, cv2.LINE_AA, tipLength=0.18)


def draw_face_bbox(frame, bbox):
    """绘制人脸检测框。"""
    if bbox is None:
        return
    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLOR_BBOX, 2)


def draw_keypoints(frame, landmarks, radius=2):
    """绘制 35 点面部关键点。"""
    if landmarks is None:
        return
    for pt in landmarks:
        if is_valid_point(pt):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), radius, COLOR_KEYPOINT, -1)


def draw_screen_gaze_point(frame, pt, radius=8, thickness=2):
    """在视频帧上标记屏幕注视点位置（黄色十字）。"""
    if not is_valid_point(pt):
        return
    x, y = int(pt[0]), int(pt[1])
    h, w = frame.shape[:2]
    # 限制在帧范围内
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    cv2.drawMarker(frame, (x, y), COLOR_GAZE_PT, cv2.MARKER_CROSS, radius, thickness)


def draw_info_text(frame, record, frame_idx):
    """在帧左上角绘制帧信息，包括 pitch/yaw/gaze_xyz。"""
    pitch_yaw = record.get("pitch_yaw_rad")
    gaze_xyz = record.get("gaze_xyz")

    pitch_val = float(pitch_yaw[0]) if pitch_yaw and len(pitch_yaw) >= 1 and math.isfinite(pitch_yaw[0]) else float("nan")
    yaw_val = float(pitch_yaw[1]) if pitch_yaw and len(pitch_yaw) >= 2 and math.isfinite(pitch_yaw[1]) else float("nan")
    gx = float(gaze_xyz[0]) if gaze_xyz and len(gaze_xyz) >= 1 and math.isfinite(gaze_xyz[0]) else float("nan")
    gy = float(gaze_xyz[1]) if gaze_xyz and len(gaze_xyz) >= 2 and math.isfinite(gaze_xyz[1]) else float("nan")
    gz = float(gaze_xyz[2]) if gaze_xyz and len(gaze_xyz) >= 3 and math.isfinite(gaze_xyz[2]) else float("nan")

    lines = [
        f"Frame: {frame_idx}  Time: {record.get('timestamp', 0):.2f}s  Conf: {record.get('confidence', 0):.3f}",
        f"Pitch: {pitch_val:+.4f} rad  Yaw: {yaw_val:+.4f} rad",
        f"GazeXYZ: ({gx:+.4f}, {gy:+.4f}, {gz:+.4f})",
    ]
    py = 30
    for text in lines:
        # 黑色背景
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (5, py - th - 4), (tw + 10, py + 4), COLOR_BG_TEXT, -1)
        cv2.putText(frame, text, (8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
        py += th + 10


def process_one_video(video_path: Path, records: list, output_path: Path,
                      arrow_length: int, draw_kp: bool, draw_gp: bool):
    """
    处理单个视频：逐帧叠加 JSONL 预测结果，写出新视频。

    注意：RunGazeOnScreen 中 frame_idx==0 不写入 JSONL，
    所以 JSONL 行数 = 视频帧数 - 1。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # JSONL 从 frame_idx=1 开始，所以 JSONL[i] 对应视频帧 i+1
    expected_jsonl_count = total_frames - 1
    if len(records) != expected_jsonl_count:
        print(f"  [WARN] JSONL 记录数 ({len(records)}) != 视频帧数-1 ({expected_jsonl_count})，按较短的处理")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    jsonl_idx = 0
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # frame_idx == 0 没有对应 JSONL 记录，直接写原帧
        if frame_idx == 0:
            writer.write(frame)
            continue

        # 取对应的 JSONL 记录
        if jsonl_idx < len(records):
            record = records[jsonl_idx]
            jsonl_idx += 1

            # 绘制各元素
            bbox = record.get("face_detection_bbox")
            draw_face_bbox(frame, bbox)

            if draw_kp:
                draw_keypoints(frame, record.get("facial_landmark_35"))

            pitch_yaw = record.get("pitch_yaw_rad")
            if pitch_yaw and len(pitch_yaw) >= 2:
                draw_gaze_arrow(frame, bbox, float(pitch_yaw[0]), float(pitch_yaw[1]), arrow_length)

            if draw_gp:
                draw_screen_gaze_point(frame, record.get("gaze_screen_xy_px"))

            draw_info_text(frame, record, frame_idx)

        writer.write(frame)

    cap.release()
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="在原始视频上逐帧绘制注视视线方向，输出可视化视频。"
    )
    parser.add_argument("--video_root", default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Data_original", type=str,
                        help="原始视频根目录（包含受试者子目录）")
    parser.add_argument("--jsonl_dir", default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Data0620_tf_calibrate", type=str,
                        help="JSONL 预测文件目录")
    parser.add_argument("--output_dir", default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Output_Videos", type=str,
                        help="输出视频目录")
    parser.add_argument("--arrow_length", default=150, type=int,
                        help="视线箭头像素长度（默认 150）")
    parser.add_argument("--no-keypoints", action="store_true",
                        help="不绘制面部关键点")
    parser.add_argument("--no-gaze-point", action="store_true",
                        help="不绘制屏幕注视点标记")
    args = parser.parse_args()

    video_root = Path(args.video_root)
    jsonl_dir = Path(args.jsonl_dir)
    output_dir = Path(args.output_dir)

    # 扫描 JSONL 文件
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"在 {jsonl_dir} 下未找到 JSONL 文件")
        return

    print(f"共找到 {len(jsonl_files)} 个 JSONL 文件")
    print(f"视频根目录: {video_root}")
    print(f"输出目录:   {output_dir}")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for jsonl_path in jsonl_files:
        parsed = parse_jsonl_filename(jsonl_path.name)
        if parsed is None:
            print(f"[SKIP] 无法解析文件名: {jsonl_path.name}")
            skip_count += 1
            continue

        subject_id, difficulty, state = parsed

        # 拼接视频路径
        video_path = video_root / subject_id / state / difficulty / "training_video.mp4"
        if not video_path.exists():
            print(f"[SKIP] 视频不存在: {video_path}")
            skip_count += 1
            continue

        # 输出路径
        out_name = f"{subject_id}_{difficulty}_{state}.mp4"
        out_path = output_dir / out_name

        print(f"[PROC] {subject_id}/{state}/{difficulty} -> {out_path}")

        records = load_jsonl(jsonl_path)
        ok = process_one_video(
            video_path, records, out_path,
            arrow_length=args.arrow_length,
            draw_kp=not args.no_keypoints,
            draw_gp=not args.no_gaze_point,
        )

        if ok:
            print(f"  [OK] 已保存 ({len(records)} 帧)")
            success_count += 1
        else:
            fail_count += 1

    print()
    print(f"完成！成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}")


if __name__ == "__main__":
    main()
