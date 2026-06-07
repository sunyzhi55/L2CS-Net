#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch calibrate FatigueGuard JSONL outputs with a trained TensorFlow model.

The script reads the preprocessed JSONL files, applies the pretrained
calibration network to the estimated screen points, writes calibrated JSONL
files to a new output directory, and prints mean pixel error before and after
calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MODEL_CKPT_DEFAULT = Path("tf_calibrate_model") / "gaze_calibration_model.ckpt"


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
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32)[:2] - np.asarray(b, dtype=np.float32)[:2]))


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
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _dump_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class TFCalibrationModel:
    """
    Restore the pretrained calibration network and run inference on (x, y).
    The graph matches the original training-time structure.
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
            import tensorflow as tf  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TensorFlow is required to run calibration inference. "
                "Please install TensorFlow in the execution environment."
            ) from exc

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        graph = tf.Graph()
        with graph.as_default():
            self._x = tf.compat.v1.placeholder(tf.float32, [None, 2], name="x")
            self._y_true = tf.compat.v1.placeholder(tf.float32, [None, 2], name="y_true")

            l2_regularizer = tf.keras.regularizers.l2(0.01)
            hidden1 = tf.compat.v1.layers.dense(
                self._x,
                units=10,
                activation=tf.nn.relu,
                kernel_regularizer=l2_regularizer,
                name="dense",
            )
            hidden2 = tf.compat.v1.layers.dense(
                hidden1,
                units=10,
                activation=tf.nn.relu,
                kernel_regularizer=l2_regularizer,
                name="dense_1",
            )
            self._y_pred = tf.compat.v1.layers.dense(hidden2, units=2, name="dense_2")

            loss = tf.reduce_mean(tf.square(self._y_pred - self._y_true)) + tf.compat.v1.losses.get_regularization_loss()
            tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

            self._saver = tf.compat.v1.train.Saver()

        self._tf = tf
        self._sess = tf.compat.v1.Session(graph=graph)
        self._saver.restore(self._sess, str(self.checkpoint_path))

    def predict(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return points.reshape(0, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected an array of shape (N, 2), got {points.shape}")

        pred = self._sess.run(
            self._y_pred,
            feed_dict={
                self._x: points,
                self._y_true: np.zeros_like(points, dtype=np.float32),
            },
        )
        return np.asarray(pred, dtype=np.float32)

    def close(self) -> None:
        if self._sess is not None:
            self._sess.close()
            self._sess = None

    def __enter__(self) -> "TFCalibrationModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class ErrorStats:
    count: int = 0
    sum_before: float = 0.0
    sum_after: float = 0.0

    def update(self, before: Optional[float], after: Optional[float]) -> None:
        if before is None or after is None:
            return
        self.count += 1
        self.sum_before += float(before)
        self.sum_after += float(after)

    @property
    def mean_before(self) -> float:
        return self.sum_before / self.count if self.count else float("nan")

    @property
    def mean_after(self) -> float:
        return self.sum_after / self.count if self.count else float("nan")


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


def _format_easy_record(record: Dict[str, Any], calibrated_xy: Optional[Sequence[float]]) -> Dict[str, Any]:
    output = OrderedDict()
    output["timestamp"] = record.get("timestamp")
    output["frame_idx"] = record.get("frame_idx")
    output["pitch_yaw_rad"] = record.get("pitch_yaw_rad")
    output["gaze_xyz"] = record.get("gaze_xyz")
    output["gaze_screen_xy_mm"] = record.get("gaze_screen_xy_mm")
    output["gaze_screen_xy_px"] = record.get("gaze_screen_xy_px")
    output["gaze_screen_tf_calibrate_xy_px"] = _point_to_json(calibrated_xy)
    output["target_xy_px"] = record.get("target_xy_px")
    output["bbox"] = record.get("bbox")
    output["landmarks"] = record.get("landmarks")
    output["confidence"] = record.get("confidence")
    return output


def _format_hard_record(record: Dict[str, Any], calibrated_xy: Optional[Sequence[float]]) -> Dict[str, Any]:
    output = OrderedDict()
    output["timestamp"] = record.get("timestamp")
    output["frame_idx"] = record.get("frame_idx")
    output["pitch_yaw_rad"] = record.get("pitch_yaw_rad")
    output["gaze_xyz"] = record.get("gaze_xyz")
    output["gaze_screen_xy_mm"] = record.get("gaze_screen_xy_mm")
    output["gaze_screen_xy_px"] = record.get("gaze_screen_xy_px")
    output["gaze_screen_tf_calibrate_xy_px"] = _point_to_json(calibrated_xy)
    output["target_centers_xy_px"] = record.get("target_centers_xy_px")
    output["bbox"] = record.get("bbox")
    output["landmarks"] = record.get("landmarks")
    output["confidence"] = record.get("confidence")
    return output


def _process_easy_record(record: Dict[str, Any], calibrated_xy: Optional[Sequence[float]]) -> Tuple[Optional[float], Optional[float]]:
    predicted = _point_from_value(record.get("gaze_screen_xy_px"))
    target = _point_from_value(record.get("target_xy_px"))
    calibrated = _point_from_value(calibrated_xy) if calibrated_xy is not None else None

    before = _distance(predicted, target) if predicted is not None and target is not None else None
    after = _distance(calibrated, target) if calibrated is not None and target is not None else None
    return before, after


def _process_hard_record(record: Dict[str, Any], calibrated_xy: Optional[Sequence[float]]) -> Tuple[Optional[float], Optional[float]]:
    predicted = _point_from_value(record.get("gaze_screen_xy_px"))
    target_centers = record.get("target_centers_xy_px") or []
    calibrated = _point_from_value(calibrated_xy) if calibrated_xy is not None else None

    if predicted is None or not target_centers:
        return None, None

    before = _nearest_distance(predicted, target_centers)
    after = _nearest_distance(calibrated, target_centers) if calibrated is not None else None
    return before, after


def _calibrate_records(
    records: List[Dict[str, Any]],
    task: str,
    model: TFCalibrationModel,
) -> Tuple[List[Dict[str, Any]], ErrorStats]:
    stats = ErrorStats()
    valid_points: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx, record in enumerate(records):
        point = _point_from_value(record.get("gaze_screen_xy_px"))
        if point is not None:
            valid_points.append(point)
            valid_indices.append(idx)

    calibrated_map: Dict[int, np.ndarray] = {}
    if valid_points:
        calibrated_batch = model.predict(np.stack(valid_points, axis=0))
        for idx, calibrated_xy in zip(valid_indices, calibrated_batch):
            calibrated_map[idx] = np.asarray(calibrated_xy, dtype=np.float32)

    output_records: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        calibrated_xy = calibrated_map.get(idx)
        if task == "easy":
            before, after = _process_easy_record(record, calibrated_xy)
            output_records.append(_format_easy_record(record, calibrated_xy))
        else:
            before, after = _process_hard_record(record, calibrated_xy)
            output_records.append(_format_hard_record(record, calibrated_xy))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply TF calibration to FatigueGuard JSONL files.")
    parser.add_argument("--input_path", default="/data3/wangchangmiao/shenxy/Code/gaze/DataOutput1", help="Input JSONL file or directory.")
    parser.add_argument("--output_dir", default="/data3/wangchangmiao/shenxy/Code/gaze/CalibratedData", help="Directory for calibrated JSONL files.")
    parser.add_argument("--model_ckpt", default=str(MODEL_CKPT_DEFAULT), help="TensorFlow checkpoint path.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = _iter_jsonl_files(input_path)
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under: {input_path}")

    overall_stats = ErrorStats()

    with TFCalibrationModel(Path(args.model_ckpt)) as model:
        for source_file in jsonl_files:
            records = _load_jsonl(source_file)
            if not records:
                print(f"[SKIP] {source_file} is empty.")
                continue

            task = _infer_task(records[0], source_file)
            calibrated_records, stats = _calibrate_records(records, task, model)

            out_path = _output_path_for(source_file, input_path, output_dir)
            _dump_jsonl(out_path, calibrated_records)

            overall_stats.count += stats.count
            overall_stats.sum_before += stats.sum_before
            overall_stats.sum_after += stats.sum_after

            print(
                f"[{task.upper()}] {source_file.name} | "
                f"n={stats.count} | "
                f"before={stats.mean_before:.6f} px | "
                f"after={stats.mean_after:.6f} px"
            )
            print(f"  -> {out_path}")

    print(
        f"[OVERALL] n={overall_stats.count} | "
        f"before={overall_stats.mean_before:.6f} px | "
        f"after={overall_stats.mean_after:.6f} px"
    )


if __name__ == "__main__":
    main()
