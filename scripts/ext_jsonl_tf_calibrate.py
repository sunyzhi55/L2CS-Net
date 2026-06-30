#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对屏幕映射后的 JSONL 施加 TensorFlow 校准网络，输出校准后的 JSONL。

用法:
    python scripts/ext_jsonl_tf_calibrate.py \
        --input_path /path/to/screen_mapped \
        --output_dir  /path/to/calibrated
"""

from __future__ import annotations

import sys, os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJ_ROOT = _os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MODEL_CKPT_DEFAULT = Path("tf_calibrate_model") / "gaze_calibration_model.ckpt"


# ── 工具函数 ────────────────────────────────────────────────────

def _is_finite_number(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _point_from_value(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    x, y = value[0], value[1]
    if not (_is_finite_number(x) and _is_finite_number(y)):
        return None
    return np.array([float(x), float(y)], dtype=np.float32)


def _point_to_json(point: Optional[Sequence[float]]) -> Optional[List[float]]:
    if point is None:
        return None
    return [float(point[0]), float(point[1])]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(
        np.asarray(a, dtype=np.float32)[:2] - np.asarray(b, dtype=np.float32)[:2]
    ))


def _nearest_distance(point: Sequence[float], targets: Sequence[Sequence[float]]) -> Optional[float]:
    if point is None or not targets:
        return None
    point_arr = np.asarray(point, dtype=np.float32)[:2]
    best = None
    for target in targets:
        target_arr = np.asarray(target, dtype=np.float32)[:2]
        if not np.all(np.isfinite(target_arr)):
            continue
        dist = float(np.linalg.norm(point_arr - target_arr))
        if best is None or dist < best:
            best = dist
    return best


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _dump_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


# ── TF 校准模型 ─────────────────────────────────────────────────

class TFCalibrationModel:
    """
    加载预训练校准网络并推理。
    网络结构：输入(2) → Dense(10,ReLU) → Dense(10,ReLU) → Dense(2)
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path)
        self._tf = None
        self._sess = None
        self._x = None
        self._y_true = None
        self._y_pred = None
        self._saver = None
        self._build()

    def _build(self) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required. Please install TensorFlow."
            ) from exc

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        graph = tf.Graph()
        with graph.as_default():
            self._x = tf.compat.v1.placeholder(tf.float32, [None, 2], name="x")
            self._y_true = tf.compat.v1.placeholder(tf.float32, [None, 2], name="y_true")

            l2_reg = tf.keras.regularizers.l2(0.01)
            h1 = tf.compat.v1.layers.dense(self._x, 10, activation=tf.nn.relu,
                                            kernel_regularizer=l2_reg, name="dense")
            h2 = tf.compat.v1.layers.dense(h1, 10, activation=tf.nn.relu,
                                            kernel_regularizer=l2_reg, name="dense_1")
            self._y_pred = tf.compat.v1.layers.dense(h2, 2, name="dense_2")

            loss = tf.reduce_mean(tf.square(self._y_pred - self._y_true))
            loss += tf.compat.v1.losses.get_regularization_loss()
            tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)
            self._saver = tf.compat.v1.train.Saver()

        self._tf = tf
        self._sess = tf.compat.v1.Session(graph=graph)
        self._saver.restore(self._sess, str(self.checkpoint_path))

    def predict(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return points.reshape(0, 2)
        pred = self._sess.run(
            self._y_pred,
            feed_dict={self._x: points, self._y_true: np.zeros_like(points)},
        )
        return np.asarray(pred, dtype=np.float32)

    def close(self) -> None:
        if self._sess is not None:
            self._sess.close()
            self._sess = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── 误差统计 ────────────────────────────────────────────────────

class ErrorStats:
    def __init__(self):
        self.count = 0
        self.sum_before = 0.0
        self.sum_after = 0.0

    def update(self, before: Optional[float], after: Optional[float]):
        if before is not None and after is not None:
            self.count += 1
            self.sum_before += float(before)
            self.sum_after += float(after)

    @property
    def mean_before(self) -> float:
        return self.sum_before / self.count if self.count else float("nan")

    @property
    def mean_after(self) -> float:
        return self.sum_after / self.count if self.count else float("nan")


# ── 任务类型推断 ────────────────────────────────────────────────

def _infer_task(record: Dict[str, Any], source_path: Path) -> str:
    if "target_xy_px" in record:
        return "easy"
    if "target_centers_xy_px" in record:
        return "hard"
    name = source_path.name.lower()
    if "easy" in name:
        return "easy"
    if "hard" in name:
        return "hard"
    raise ValueError(f"Cannot infer task type for {source_path}")


# ── 记录格式化 ──────────────────────────────────────────────────

def _format_record(record: Dict[str, Any], task: str,
                   calibrated_xy: Optional[Sequence[float]],
                   before: Optional[float], after: Optional[float]) -> Dict[str, Any]:
    """构建校准后的输出记录，只保留存在的字段。"""
    output = OrderedDict()

    # 基础字段（Script A 一定有）
    for key in ["timestamp", "frame_idx", "pitch_yaw_rad", "gaze_xyz",
                 "gaze_screen_xy_mm", "gaze_screen_xy_px"]:
        if key in record:
            output[key] = record[key]

    # 校准后的屏幕坐标
    output["gaze_screen_tf_calibrate_xy_px"] = _point_to_json(calibrated_xy)

    # 目标点
    if task == "easy" and "target_xy_px" in record:
        output["target_xy_px"] = record["target_xy_px"]
    elif task == "hard" and "target_centers_xy_px" in record:
        output["target_centers_xy_px"] = record["target_centers_xy_px"]

    # 误差
    output["deviation_px_before_calibrate"] = before
    output["deviation_px_after_calibrate"] = after

    # 如果原始记录有这些字段也保留
    for key in ["face_detection_bbox", "facial_landmark_35",
                 "RetinaFace_bbox", "RetinaFace_landmarks", "confidence"]:
        if key in record:
            output[key] = record[key]

    return output


# ── 核心处理 ────────────────────────────────────────────────────

def _calibrate_records(records: List[Dict[str, Any]], task: str,
                       model: TFCalibrationModel) -> Tuple[List[Dict[str, Any]], ErrorStats]:
    stats = ErrorStats()
    valid_points = []
    valid_indices = []

    for idx, record in enumerate(records):
        point = _point_from_value(record.get("gaze_screen_xy_px"))
        if point is not None:
            valid_points.append(point)
            valid_indices.append(idx)

    calibrated_map: Dict[int, np.ndarray] = {}
    if valid_points:
        calibrated_batch = model.predict(np.stack(valid_points, axis=0))
        for idx, cal_xy in zip(valid_indices, calibrated_batch):
            calibrated_map[idx] = np.asarray(cal_xy, dtype=np.float32)

    output_records = []
    for idx, record in enumerate(records):
        cal_xy = calibrated_map.get(idx)
        cal_xy_seq = cal_xy.tolist() if cal_xy is not None else None

        # 计算误差
        predicted = _point_from_value(record.get("gaze_screen_xy_px"))
        calibrated = _point_from_value(cal_xy) if cal_xy is not None else None

        before, after = None, None
        if task == "easy":
            target = _point_from_value(record.get("target_xy_px"))
            if predicted is not None and target is not None:
                before = _distance(predicted, target)
            if calibrated is not None and target is not None:
                after = _distance(calibrated, target)
        else:
            target_centers = record.get("target_centers_xy_px") or []
            if predicted is not None and target_centers:
                before = _nearest_distance(predicted, target_centers)
            if calibrated is not None and target_centers:
                after = _nearest_distance(calibrated, target_centers)

        output_records.append(_format_record(record, task, cal_xy_seq, before, after))
        stats.update(before, after)

    return output_records, stats


def _iter_jsonl_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.jsonl") if p.is_file())


def _output_path_for(source_file: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / source_file.name
    return output_root / source_file.relative_to(input_root)


# ── 主函数 ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="对屏幕映射后的 JSONL 施加 TF 校准网络。"
    )
    parser.add_argument("--input_path", default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_puregaze/screen_Output",
                        help="输入 JSONL 文件或目录（Script A 的输出）")
    parser.add_argument("--output_dir", default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_puregaze/tf_calibrate_Output",
                        help="校准后 JSONL 的输出目录")
    parser.add_argument("--model_ckpt", default=str(MODEL_CKPT_DEFAULT),
                        help="TensorFlow 校准模型 checkpoint 路径")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = _iter_jsonl_files(input_path)
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under: {input_path}")

    overall = ErrorStats()

    with TFCalibrationModel(Path(args.model_ckpt)) as model:
        for source_file in jsonl_files:
            records = _load_jsonl(source_file)
            if not records:
                print(f"[SKIP] {source_file} is empty.")
                continue

            task = _infer_task(records[0], source_file)
            calibrated, stats = _calibrate_records(records, task, model)

            out_path = _output_path_for(source_file, input_path, output_dir)
            _dump_jsonl(out_path, calibrated)

            overall.count += stats.count
            overall.sum_before += stats.sum_before
            overall.sum_after += stats.sum_after

            print(
                f"[{task.upper()}] {source_file.name} | "
                f"n={stats.count} | "
                f"before={stats.mean_before:.2f} px | "
                f"after={stats.mean_after:.2f} px"
            )
            print(f"  -> {out_path}")

    print(
        f"\n[OVERALL] n={overall.count} | "
        f"before={overall.mean_before:.2f} px | "
        f"after={overall.mean_after:.2f} px"
    )


if __name__ == "__main__":
    main()
