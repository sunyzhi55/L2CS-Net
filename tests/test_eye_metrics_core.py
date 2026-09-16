"""eye_metrics_core 的合成信号验证。

这些用例不需要视频、不需要 mediapipe/cv2，因此可以在本地（无数据环境）直接跑通，
用来锁定 PERCLOS / 眨眼 / 长闭眼的事件判定与聚合口径。上服务器前先跑这个。
"""

import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

TMP_ROOT = Path("C:/tmp")

from scripts.eye_metrics_core import (
    EYE_IRIS_LANDMARKS,
    EYE_LANDMARKS,
    aggregate_condition,
    combine_eyes,
    default_config,
    eye_aperture,
    feasibility,
    iris_geometry,
    iris_points,
    iris_signals,
    median_scale,
    perclos,
    scale_parity_stats,
    split_events,
    summarize_segment,
    validity_ratio,
    closure_events,
)

# 该模块顶层不 import cv2 / mediapipe（只在 _require_video_deps 里惰性导入），
# 因此定标与聚合逻辑可以在没有视频依赖的本地环境里直接测。
from scripts.extract_eye_metrics_batch import subject_scales, summarize_unit

FS = 30.0
W = 30.0        # 合成眼的眼裂宽度（像素）
R_IRIS = 6.0    # 合成虹膜半径 -> 直径 12，占眼裂 0.40，落在生理区间内


def series(closed_spans, n_frames, level=0.95, ok_spans=None):
    """构造遮蔽度序列：closed_spans 为 [(起帧, 帧数), ...]，其余帧遮蔽度为 0。

    ok_spans 为 None 时全部有效；否则只在 ok_spans 内有效（模拟丢脸/遮挡）。
    """
    occ = np.zeros(n_frames)
    for a, b in closed_spans:
        occ[a:a + b] = level
    ok = np.ones(n_frames, dtype=bool)
    if ok_spans is not None:
        ok[:] = False
        for a, b in ok_spans:
            ok[a:a + b] = True
    return occ, ok


class GeometryTests(unittest.TestCase):
    def test_aperture_is_height_over_width(self):
        """已知几何：眼裂高 6px、宽 30px -> 开合度 0.2。"""
        pts = np.full((478, 2), np.nan)
        for i, x in zip((159, 160, 158), (28.0, 35.0, 45.0)):
            pts[i] = (x, 100.0)                  # 上睑
        for i, x in zip((145, 144, 153), (28.0, 35.0, 45.0)):
            pts[i] = (x, 106.0)                  # 下睑，y 向下为正
        pts[33] = (50.0, 103.0)
        pts[133] = (20.0, 103.0)
        aper, width = eye_aperture(pts, "r")
        self.assertAlmostEqual(aper, 6.0 / 30.0, places=6)
        self.assertAlmostEqual(width, 30.0, places=6)

    def _eye_pts(self, lid_x, upper_y=100.0, lower_y=106.0, corner_a=50.0, corner_b=20.0):
        """按给定横坐标铺一个右眼，其余点置 0 不影响。"""
        pts = np.full((478, 2), 0.0)
        for i, x in zip((159, 160, 158), lid_x):
            pts[i] = (x, upper_y)
        for i, x in zip((145, 144, 153), lid_x):
            pts[i] = (x, lower_y)
        pts[33] = (corner_a, 103.0)
        pts[133] = (corner_b, 103.0)
        return pts

    def test_aperture_is_pure_geometry_and_never_rejects(self):
        """eye_aperture 只算几何，不做任何"看起来不像眼睛"的拦截。

        撤除门控是有意为之：门控以索引正确为前提，而索引未经核实，一旦错位就会把
        整批数据拦成零输出（实测发生），比产出可疑数字更糟。
        """
        pts = self._eye_pts((200.0, 210.0, 220.0))     # 眼睑点跑到眼角之外
        self.assertTrue(np.isfinite(eye_aperture(pts, "r")[0]))
        pts = self._eye_pts((28.0, 35.0, 45.0), upper_y=70.0)   # 等效于落在眉毛上
        self.assertTrue(np.isfinite(eye_aperture(pts, "r")[0]))

    def test_far_corner_yields_inflated_width_visible_in_parity_stats(self):
        """眼角错到对侧：本侧几何自洽、无人拦截，但宽度虚高会被对称统计暴露。"""
        cfg = default_config()
        pts = self._eye_pts((28.0, 35.0, 45.0), corner_a=50.0, corner_b=-120.0)
        aper, width = eye_aperture(pts, "r")
        self.assertAlmostEqual(width, 170.0, places=6)
        self.assertLess(aper, 0.05)                     # 被虚高宽度压成假小值，即 ref_l=0.032 的成因
        st = scale_parity_stats(np.array([30.0]), np.array([width]), cfg)
        self.assertGreater(st["ratio_bad_share"], 0.0)   # 报告层看得见

    def test_missing_landmarks_yield_nan_not_crash(self):
        pts = np.full((10, 2), 1.0)        # 长度不足，覆盖不到眼睑索引
        aper, width = eye_aperture(pts, "l")
        self.assertTrue(np.isnan(aper) and np.isnan(width))

    def test_zero_width_is_rejected_as_nan(self):
        pts = np.full((478, 2), 5.0)       # 内外眼角重合 -> 宽度 0
        aper, width = eye_aperture(pts, "r")
        self.assertTrue(np.isnan(aper))


class PerclosTests(unittest.TestCase):
    def test_perclos_is_time_fraction_over_valid_time(self):
        """300 帧全有效、其中 60 帧闭合 -> 0.20。"""
        cfg = default_config()
        occ, ok = series([(0, 60)], 300)
        p, cs, vs = perclos(occ, ok, FS, 0.80)
        self.assertAlmostEqual(p, 60 / 300, places=6)
        self.assertAlmostEqual(cs, 2.0, places=6)
        self.assertAlmostEqual(vs, 10.0, places=6)

    def test_denominator_is_valid_time_not_total_time(self):
        """关键反例：无效帧集中在低专注段时，用总时长当分母会低估 PERCLOS。"""
        occ, ok = series([(0, 20)], 100, ok_spans=[(0, 80)])
        p, _, vs = perclos(occ, ok, FS, 0.80)
        self.assertAlmostEqual(vs, 80 / FS, places=6)
        self.assertAlmostEqual(p, 20 / 80, places=6)          # 0.25，不是 0.20

    def test_sensitivity_tiers_bracket_the_primary_p50(self):
        """主判据 P50，0.40/0.60 只做敏感性；事件按主判据定义。"""
        cfg = default_config()
        self.assertEqual(cfg["perclos_thresholds"][0], 0.50)
        occ, ok = series([(0, 30)], 300, level=0.55)
        self.assertAlmostEqual(perclos(occ, ok, FS, 0.50)[0], 30 / 300, places=6)
        self.assertAlmostEqual(perclos(occ, ok, FS, 0.60)[0], 0.0, places=6)
        st = summarize_segment(occ, ok, FS, cfg)
        self.assertAlmostEqual(st["perclos_50"], 0.10, places=6)
        self.assertAlmostEqual(st["perclos_60"], 0.0, places=6)
        self.assertEqual(len(closure_events(occ, ok, FS, cfg)), 1)


class EventTests(unittest.TestCase):
    def test_blink_and_closure_are_mutually_exclusive(self):
        """100ms 眨眼 + 1000ms 长闭眼 -> 各计一次，不重复计数。"""
        cfg = default_config()
        occ, ok = series([(30, 3), (300, 30)], 600)          # 3 帧 = 0.1s，30 帧 = 1.0s
        ev = closure_events(occ, ok, FS, cfg)
        self.assertEqual(len(ev), 2)
        blinks, closures = split_events(ev, cfg)
        self.assertEqual(len(blinks), 1)
        self.assertEqual(len(closures), 1)
        self.assertAlmostEqual(blinks[0]["dur_s"], 0.1, places=6)
        self.assertAlmostEqual(closures[0]["dur_s"], 1.0, places=6)

    def test_event_does_not_bridge_invalid_frames(self):
        """被 4 帧无效隔断的两段闭合必须算两个事件，否则时长被虚增。"""
        cfg = default_config()
        occ = np.zeros(200)
        occ[20:40] = 0.95
        occ[44:64] = 0.95
        ok = np.ones(200, dtype=bool)
        ok[40:44] = False
        ev = closure_events(occ, ok, FS, cfg)
        self.assertEqual(len(ev), 2)
        self.assertAlmostEqual(ev[0]["dur_s"], 20 / FS, places=6)
        self.assertAlmostEqual(ev[1]["dur_s"], 20 / FS, places=6)

    def test_blink_tiers_separate_short_excursions_from_blinks(self):
        """30 fps 下一帧级抖动进不了任何时长下限，必须整档报告而不是挑一档。"""
        cfg = default_config()
        occ, ok = series([(10, 4), (100, 5)], 300)           # 133ms 与 167ms
        s = summarize_segment(occ, ok, FS, cfg)
        self.assertEqual(s["n_blink_lt500ms"], 2)
        self.assertEqual((s["n_blink_ge100ms"], s["n_blink_ge150ms"], s["n_blink_ge200ms"]), (2, 1, 0))
        self.assertAlmostEqual(s["blink_rate_lt500ms_per_min"], 2 / (300 / FS / 60), places=6)
        self.assertAlmostEqual(s["blink_rate_ge150ms_per_min"], 1 / (300 / FS / 60), places=6)
        self.assertEqual(s["n_closure_ge500ms"], 0)

    def test_closure_tiers_nested_counts(self):
        cfg = default_config()
        occ, ok = series([(0, 20), (100, 40), (200, 70)], 300)   # 0.67s / 1.33s / 2.33s
        s = summarize_segment(occ, ok, FS, cfg)
        self.assertEqual(s["n_closure_ge500ms"], 3)
        self.assertEqual(s["n_closure_ge1000ms"], 2)
        self.assertEqual(s["n_closure_ge2000ms"], 1)
        self.assertAlmostEqual(s["closure_dur_ge2000ms_s"], 70 / FS, places=6)

    def test_perclos_decomposes_into_blink_and_long_closure(self):
        """总 PERCLOS 与"剔除长闭眼"的 PERCLOS 之差应正好是长闭眼贡献。"""
        cfg = default_config()
        occ, ok = series([(10, 3), (200, 30)], 400)
        s = summarize_segment(occ, ok, FS, cfg)
        long_s = s["closure_dur_ge500ms_s"]
        self.assertAlmostEqual(
            s["perclos_50"] - s["perclos_50_excl_long_closure"],
            long_s / s["valid_s"], places=9)


class EarFormulaTests(unittest.TestCase):
    """EAR 的两个要点：分子分母都用欧氏距离、按横坐标配对。"""

    def _eye(self, up, lo, ca, cb, side="r"):
        pts = np.zeros((478, 2))
        for i, (x, y) in zip(EYE_LANDMARKS[side]["upper"], up):
            pts[i] = (x, y)
        for i, (x, y) in zip(EYE_LANDMARKS[side]["lower"], lo):
            pts[i] = (x, y)
        pts[EYE_LANDMARKS[side]["corner_a"]] = ca
        pts[EYE_LANDMARKS[side]["corner_b"]] = cb
        return pts

    def test_denominator_is_euclidean_canthus_distance_not_x_diff(self):
        """眼角有纵向偏移时（头部滚转），分母必须是欧氏距离。"""
        up = [(30.0, 100.0), (35.0, 100.0), (40.0, 100.0)]
        lo = [(30.0, 106.0), (35.0, 106.0), (40.0, 106.0)]
        flat = self._eye(up, lo, (50.0, 100.0), (20.0, 100.0))
        rolled = self._eye(up, lo, (50.0, 110.0), (20.0, 100.0))
        e_flat, w_flat = eye_aperture(flat, "r")
        e_roll, w_roll = eye_aperture(rolled, "r")
        self.assertAlmostEqual(w_flat, 30.0, places=6)
        self.assertAlmostEqual(w_roll, float(np.hypot(30.0, 10.0)), places=6)
        self.assertLess(e_roll, e_flat)

    def test_vertical_pair_uses_euclidean_not_y_only(self):
        """上下睑点横向错开时，垂直距离应是斜边而不是 y 差。"""
        up = [(30.0, 100.0), (35.0, 100.0), (40.0, 100.0)]
        lo = [(33.0, 106.0), (38.0, 106.0), (43.0, 106.0)]
        pts = self._eye(up, lo, (50.0, 100.0), (20.0, 100.0))
        ear, _ = eye_aperture(pts, "r")
        self.assertAlmostEqual(ear, float(np.hypot(3.0, 6.0)) / 30.0, places=6)

    def test_pairing_is_by_x_not_by_written_order(self):
        """标定点的书写顺序不该影响结果：按横坐标排序后配对。"""
        up = [(30.0, 100.0), (40.0, 100.0), (35.0, 100.0)]
        lo = [(40.0, 106.0), (30.0, 106.0), (35.0, 106.0)]
        pts = self._eye(up, lo, (50.0, 100.0), (20.0, 100.0))
        ear, _ = eye_aperture(pts, "r")
        self.assertAlmostEqual(ear, 6.0 / 30.0, places=6)


class IrisGeometryTests(unittest.TestCase):
    """虹膜遮蔽度的量程性质：0=全可见、0.5=遮一半、1=全遮。"""

    def _face(self, up_y=-R_IRIS, lo_y=R_IRIS, side="r", n_pts=478):
        """铺一只眼：眼角在 (0,0)/(W,0) 使眼轴为 x、v 轴为 y；虹膜 5 点。"""
        pts = np.zeros((n_pts, 2))
        spec = EYE_LANDMARKS[side]
        pts[spec["corner_a"]] = (0.0, 0.0)
        pts[spec["corner_b"]] = (W, 0.0)
        for i in spec["upper"]:
            pts[i] = (W / 2.0, up_y)
        for i in spec["lower"]:
            pts[i] = (W / 2.0, lo_y)
        for i, (dx, dy) in zip(EYE_IRIS_LANDMARKS[side],
                               ((0.0, 0.0), (-R_IRIS, 0.0), (R_IRIS, 0.0),
                                (0.0, -R_IRIS), (0.0, R_IRIS))):
            pts[i] = (W / 2.0 + dx, dy)
        return pts

    def test_geometry_matches_the_synthetic_eye(self):
        lo, hi, d, ew = iris_geometry(self._face(up_y=-R_IRIS, lo_y=R_IRIS), "r")
        self.assertAlmostEqual(d, 2 * R_IRIS, places=6)
        self.assertAlmostEqual(ew, W, places=6)
        self.assertAlmostEqual(lo, -R_IRIS, places=6)
        self.assertAlmostEqual(hi, R_IRIS, places=6)

    def test_horizontal_diameter_is_unaffected_by_eyelids(self):
        """分母的稳定性来源：上下眼睑只遮蔽垂直方向，不该改变虹膜水平直径。"""
        a = iris_geometry(self._face(up_y=-R_IRIS, lo_y=R_IRIS), "r")[2]
        b = iris_geometry(self._face(up_y=0.0, lo_y=0.0), "r")[2]
        self.assertAlmostEqual(a, b, places=9)

    def test_missing_iris_indices_return_nan(self):
        """未开 refine_landmarks 时只有 468 点，必须返回 nan 而不是编一个值。"""
        self.assertIsNone(iris_points(np.zeros((468, 2)), "r"))
        self.assertTrue(all(np.isnan(v) for v in iris_geometry(np.zeros((468, 2)), "r")))

    def test_axis_orientation_does_not_change_occlusion(self):
        """眼角连线方向反过来时 v 轴随之反向，遮蔽度必须不变。"""
        pts = self._face(up_y=-R_IRIS, lo_y=0.0)
        spec = EYE_LANDMARKS["r"]
        a = iris_geometry(pts, "r")
        flipped = pts.copy()
        flipped[spec["corner_a"]], flipped[spec["corner_b"]] = \
            pts[spec["corner_b"]].copy(), pts[spec["corner_a"]].copy()
        b = iris_geometry(flipped, "r")
        cfg = default_config()
        ref = {"iris_d": a[2], "eye_w": a[3]}
        occ_a, _ = iris_signals([a[0]], [a[1]], [a[2]], [a[3]], ref, cfg)
        occ_b, _ = iris_signals([b[0]], [b[1]], [b[2]], [b[3]], ref, cfg)
        self.assertAlmostEqual(float(occ_a[0]), float(occ_b[0]), places=9)


class IrisSignalsTests(unittest.TestCase):
    """遮蔽度换算与逐帧门控。"""

    def setUp(self):
        self.cfg = default_config()
        self.ref = {"iris_d": 2 * R_IRIS, "eye_w": W}

    def _sig(self, up_y, lo_y, n=4, d=2 * R_IRIS, w=W):
        lo = np.full(n, min(up_y, lo_y))
        hi = np.full(n, max(up_y, lo_y))
        return iris_signals(lo, hi, np.full(n, d), np.full(n, w), self.ref, self.cfg)

    def test_scale_is_anchored_on_the_iris_not_on_the_open_eye(self):
        occ_open, _ = self._sig(-R_IRIS, R_IRIS)
        occ_half, _ = self._sig(0.0, R_IRIS)          # 上睑压到中心，遮住上半
        occ_shut, _ = self._sig(0.0, 0.0)
        self.assertAlmostEqual(float(occ_open[0]), 0.0, places=9)
        self.assertAlmostEqual(float(occ_half[0]), 0.5, places=9)
        self.assertAlmostEqual(float(occ_shut[0]), 1.0, places=9)

    def test_closure_reaches_the_classic_perclos_thresholds(self):
        """EAR 路线上限只有 0.56，P80 永远够不到；虹膜路线必须能过 0.75。"""
        occ, _ = self._sig(-0.2 * R_IRIS, 0.2 * R_IRIS)     # 只剩 20% 虹膜高度可见
        self.assertGreater(float(occ[0]), 0.75)

    def test_nan_geometry_is_invalid_not_silently_open(self):
        lo = np.array([np.nan, -R_IRIS])
        hi = np.array([np.nan, R_IRIS])
        occ, ok = iris_signals(lo, hi, np.array([np.nan, 12.0]),
                               np.array([np.nan, W]), self.ref, self.cfg)
        self.assertFalse(bool(ok[0]))
        self.assertTrue(bool(ok[1]))

    def test_absurd_iris_ratio_is_rejected(self):
        """虹膜直径/眼裂宽度超出生理区间 = 虹膜索引错位的主要征兆。"""
        d = np.array([2 * R_IRIS, 0.2 * W])
        occ, ok = iris_signals(np.array([-6.0, -6.0]), np.array([6.0, 6.0]), d,
                               np.array([W, W]), self.ref, self.cfg)
        self.assertTrue(bool(ok[0]))
        self.assertFalse(bool(ok[1]))

    def test_extreme_yaw_frames_dropped_by_width_gate(self):
        w = np.array([W, W, 0.2 * W, 2.4 * W])
        occ, ok = iris_signals(np.full(4, -6.0), np.full(4, 6.0), np.full(4, 12.0),
                               w, self.ref, self.cfg)
        self.assertEqual([bool(v) for v in ok], [True, True, False, False])

    def test_unavailable_scales_yield_all_nan(self):
        occ, ok = iris_signals(np.array([-6.0]), np.array([6.0]), np.array([12.0]),
                               np.array([W]), {"iris_d": float("nan"), "eye_w": W}, self.cfg)
        self.assertTrue(np.isnan(occ[0]))


class ScaleParityTests(unittest.TestCase):
    """左右眼尺度对称只做报告，不拦帧：它是暴露索引错位的仪表。"""

    def setUp(self):
        self.cfg = default_config()

    def test_symmetric_scales_report_clean(self):
        st = scale_parity_stats(np.full(5, 40.0), np.full(5, 41.0), self.cfg)
        self.assertAlmostEqual(st["ratio_median"], 40.0 / 41.0, places=6)
        self.assertEqual(st["ratio_bad_share"], 0.0)

    def test_one_sided_inflation_is_flagged(self):
        st = scale_parity_stats(np.full(5, 40.0), np.full(5, 40.0 * 5.7), self.cfg)
        self.assertEqual(st["ratio_bad_share"], 1.0)

    def test_nan_or_zero_input_is_not_a_crash(self):
        st = scale_parity_stats(np.array([np.nan, 40.0, 40.0]),
                                np.array([40.0, 0.0, 40.0]), self.cfg)
        self.assertAlmostEqual(st["ratio_median"], 1.0, places=6)

    def test_all_unusable_returns_nan(self):
        st = scale_parity_stats(np.array([np.nan]), np.array([np.nan]), self.cfg)
        self.assertTrue(np.isnan(st["ratio_median"]))


class ConsensusTests(unittest.TestCase):
    def test_asymmetric_eyes_are_dropped(self):
        """一眼报闭合、一眼报睁开（头发遮挡或关键点崩坏）时整帧剔除。"""
        cfg = default_config()
        o_r = np.array([0.9, 0.9, 0.0, 0.9])
        o_l = np.array([0.9, 0.1, 0.9, 0.9])
        ok_r = np.array([True] * 4)
        ok_l = np.array([True] * 4)
        occ, ok = combine_eyes(o_r, o_l, ok_r, ok_l, cfg)
        self.assertTrue(ok[0] and ok[3])
        self.assertFalse(ok[1] or ok[2])
        self.assertAlmostEqual(occ[0], 0.9, places=6)

    def test_nan_in_one_eye_propagates_to_invalid(self):
        cfg = default_config()
        occ, ok = combine_eyes(np.array([np.nan]), np.array([0.1]),
                              np.array([True]), np.array([True]), cfg)
        self.assertFalse(bool(ok[0]))


class AggregationTests(unittest.TestCase):
    def test_condition_level_pools_numerators_not_averages(self):
        """长片段应当主导条件水平 PERCLOS，而不是被短片段按个数拉平。"""
        cfg = default_config()
        occ_a, ok_a = series([(0, 100)], 1000)               # 10%
        occ_b, ok_b = series([(0, 25)], 100)                 # 25%
        segs = [summarize_segment(occ_a, ok_a, FS, cfg), summarize_segment(occ_b, ok_b, FS, cfg)]
        agg = aggregate_condition(segs, cfg)
        pooled = 125 / 1100.0
        mean_of_ratios = 0.5 * (100 / 1000.0 + 25 / 100.0)
        self.assertAlmostEqual(agg["perclos_50"], pooled, places=6)
        self.assertNotAlmostEqual(agg["perclos_50"], mean_of_ratios, places=3)
        self.assertAlmostEqual(agg["valid_s"], 1100 / FS, places=6)
        self.assertEqual(agg["n_segments"], 2)

    def test_blink_rate_aggregation_matches_pooled_counts(self):
        cfg = default_config()
        occ_a, ok_a = series([(10, 3), (100, 4)], 300)
        occ_b, ok_b = series([(5, 3)], 150)
        segs = [summarize_segment(occ_a, ok_a, FS, cfg), summarize_segment(occ_b, ok_b, FS, cfg)]
        agg = aggregate_condition(segs, cfg)
        n = 3
        self.assertEqual(agg["n_blink_lt500ms"], n)
        self.assertAlmostEqual(agg["blink_rate_lt500ms_per_min"], n / ((450 / FS) / 60), places=4)

    def test_excl_long_closure_is_consistent_across_levels(self):
        """条件水平拆分必须等于逐段 closed_s / long_closed_s 分别求和再相除。"""
        cfg = default_config()
        occ_a, ok_a = series([(10, 3), (200, 30)], 400)          # 100ms 眨眼 + 1s 长闭眼
        occ_b, ok_b = series([(5, 4), (300, 45)], 500)           # 133ms 眨眼 + 1.5s 长闭眼
        segs = [summarize_segment(occ_a, ok_a, FS, cfg), summarize_segment(occ_b, ok_b, FS, cfg)]
        agg = aggregate_condition(segs, cfg)
        closed = sum(s["closed_s_50"] for s in segs)
        long_s = sum(s["long_closed_s_50"] for s in segs)
        valid = sum(s["valid_s"] for s in segs)
        self.assertGreater(long_s, 0.0)
        self.assertAlmostEqual(agg["perclos_50"], closed / valid, places=9)
        self.assertAlmostEqual(agg["perclos_50_excl_long_closure"],
                               (closed - long_s) / valid, places=9)
        self.assertAlmostEqual(agg["perclos_50"] - agg["perclos_50_excl_long_closure"],
                               long_s / valid, places=9)

    def test_validity_ratio_uses_per_segment_fps_not_nominal(self):
        """实际 25 fps 的片段若按标称 30 折算分母，占比会算成 1.2 这种非法值。"""
        cfg = default_config()
        occ, ok = series([(0, 100)], 1000)                    # 全部有效
        m = summarize_segment(occ, ok, 25.0, cfg)             # 按实际 25 fps 计 valid_s
        m["fps"] = 25.0
        vr, valid_s, total_s = validity_ratio([m], 30.0)      # 标称 fs=30
        self.assertAlmostEqual(total_s, 1000 / 25.0, places=6)
        self.assertAlmostEqual(valid_s, 1000 / 25.0, places=6)
        self.assertAlmostEqual(vr, 1.0, places=9)             # 旧写法这里会得到 1.2
        self.assertLessEqual(vr, 1.0 + 1e-9)

    def test_numerators_are_emitted_so_all_can_be_pooled(self):
        """方案里的 all = easy+hard 合并；比值不可相加，必须分子分母各自求和。"""
        cfg = default_config()
        occ_e, ok_e = series([(0, 100)], 1000)               # easy: 10%
        occ_h, ok_h = series([(0, 25)], 100)                 # hard: 25%
        easy = aggregate_condition([summarize_segment(occ_e, ok_e, FS, cfg)], cfg)
        hard = aggregate_condition([summarize_segment(occ_h, ok_h, FS, cfg)], cfg)
        for row in (easy, hard):
            for k in ("closed_s_50", "long_closed_s_50", "closed_s_40", "valid_s"):
                self.assertIn(k, row)
        closed = easy["closed_s_50"] + hard["closed_s_50"]
        valid = easy["valid_s"] + hard["valid_s"]
        all_perclos = closed / valid
        self.assertAlmostEqual(all_perclos, 125 / 1100.0, places=6)
        self.assertNotAlmostEqual(all_perclos,
                                  0.5 * (easy["perclos_50"] + hard["perclos_50"]), places=3)

    def test_all_invalid_gives_nan_not_exception(self):
        cfg = default_config()
        occ, ok = series([(0, 10)], 100, ok_spans=[])
        s = summarize_segment(occ, ok, FS, cfg)
        self.assertEqual(s["valid_s"], 0.0)
        self.assertTrue(np.isnan(s["perclos_50"]))
        agg = aggregate_condition([s], cfg)
        self.assertTrue(np.isnan(agg["perclos_50"]))


class FeasibilityTests(unittest.TestCase):
    """--dry_run 的裁决：把"实现不了就不做"变成有量化判据的判断。"""

    def _samples(self, eye_w=40.0, iris_d=16.0, occ=None, n=200):
        occ = np.linspace(0.0, 0.9, n) if occ is None else np.asarray(occ, dtype=float)
        return {"eye_w": np.full(n, eye_w), "iris_d": np.full(n, iris_d), "occ": occ}

    def test_healthy_signal_passes(self):
        v, reasons, st = feasibility(self._samples(), 0.95, default_config())
        self.assertEqual(v, "pass", reasons)
        self.assertAlmostEqual(st["iris_d_over_eye_w"], 0.40, places=6)

    def test_small_eye_rejected(self):
        v, reasons, _ = feasibility(self._samples(eye_w=9.0), 0.95, default_config())
        self.assertEqual(v, "reject")
        self.assertTrue(any("眼裂宽度" in r for r in reasons))

    def test_misindexed_iris_rejected_by_physiological_ratio(self):
        v, reasons, _ = feasibility(self._samples(iris_d=150.0), 0.95, default_config())
        self.assertEqual(v, "reject")
        self.assertTrue(any("EYE_IRIS_LANDMARKS" in r for r in reasons))

    def test_flat_signal_rejected_even_at_good_scale(self):
        v, _, _ = feasibility(self._samples(occ=np.full(200, 0.2)), 0.95, default_config())
        self.assertEqual(v, "reject")

    def test_low_valid_ratio_is_marginal(self):
        v, reasons, _ = feasibility(self._samples(), 0.60, default_config())
        self.assertEqual(v, "marginal")
        self.assertTrue(any("有效帧占比" in r for r in reasons))

    def test_no_samples_rejected(self):
        v, _, _ = feasibility({"eye_w": [], "iris_d": [], "occ": []},
                              float("nan"), default_config())
        self.assertEqual(v, "reject")


class SubjectScalesTests(unittest.TestCase):
    """被试级定标：只用虹膜直径/眼裂宽度这类解剖常数，完全不碰条件标签。"""

    def _segs(self, up_y, lo_y, n=600):
        lo, hi = min(up_y, lo_y), max(up_y, lo_y)
        one = lambda: {"path": "v.mp4", "name": "v", "fps": FS, "n_frames": n,
                       "n_face": n, "n_eye": n,
                       "lo_r": np.full(n, lo), "hi_r": np.full(n, hi),
                       "id_r": np.full(n, 2 * R_IRIS), "w_r": np.full(n, W),
                       "ear_r": np.full(n, 0.30),
                       "lo_l": np.full(n, lo), "hi_l": np.full(n, hi),
                       "id_l": np.full(n, 2 * R_IRIS), "w_l": np.full(n, W),
                       "ear_l": np.full(n, 0.31)}
        return [one()]

    def _unit(self, state):
        return {"id": "01", "difficulty": "easy", "state": state, "name": f"01_easy_{state}"}

    def test_scales_ignore_condition_labels(self):
        """定标只用中位数，因此不存在"用 alert 段定尺子再声称 sleepy 更闭"的循环。"""
        cfg = default_config()
        alert = self._segs(-R_IRIS, R_IRIS)
        sleepy = self._segs(0.0, 0.3 * R_IRIS)
        both = subject_scales(alert + sleepy, cfg)
        only_sleepy = subject_scales(sleepy, cfg)
        self.assertAlmostEqual(both["r"]["iris_d"], only_sleepy["r"]["iris_d"], places=9)
        self.assertAlmostEqual(both["r"]["eye_w"], only_sleepy["r"]["eye_w"], places=9)

    def test_droopy_low_focus_keeps_the_effect(self):
        """全程眼睑下垂的低专注段必须报出高遮蔽度（旧 EAR 版会报 0）。"""
        cfg = default_config()
        alert = self._segs(-R_IRIS, R_IRIS)
        sleepy = self._segs(-0.1 * R_IRIS, 0.1 * R_IRIS)      # 只露 10% 虹膜高度
        scales = subject_scales(alert + sleepy, cfg)
        a, _ = summarize_unit(self._unit("alert"), alert, scales, cfg)
        b, _ = summarize_unit(self._unit("sleepy"), sleepy, scales, cfg)
        self.assertAlmostEqual(a["perclos_50"], 0.0, places=6)
        self.assertGreater(b["perclos_50"], 0.9, "低专注段应几乎全程算闭合")
        self.assertGreater(b["mean_occ"], a["mean_occ"])

    def test_both_conditions_share_one_ruler(self):
        cfg = default_config()
        alert = self._segs(-R_IRIS, R_IRIS)
        sleepy = self._segs(0.0, 0.2 * R_IRIS)
        scales = subject_scales(alert + sleepy, cfg)
        a, _ = summarize_unit(self._unit("alert"), alert, scales, cfg)
        b, _ = summarize_unit(self._unit("sleepy"), sleepy, scales, cfg)
        self.assertEqual(a["iris_d_r_px"], b["iris_d_r_px"])
        self.assertEqual(a["eye_w_r_px"], b["eye_w_r_px"])

    def test_unusable_side_is_reported_not_silently_zeroed(self):
        """一只眼虹膜几何全 nan 时该侧尺度为 nan，另一侧不受牵连。"""
        cfg = default_config()
        segs = self._segs(-R_IRIS, R_IRIS)
        segs[0]["id_r"] = np.full(segs[0]["id_r"].shape, np.nan)
        scales = subject_scales(segs, cfg)
        self.assertTrue(np.isnan(scales["r"]["iris_d"]))
        self.assertTrue(np.isfinite(scales["l"]["iris_d"]))

    def test_median_scale_drops_nan_and_empty(self):
        self.assertTrue(np.isnan(median_scale([])))
        self.assertAlmostEqual(median_scale([np.array([np.nan, 4.0, 6.0])]), 5.0, places=9)


class CsvColumnTests(unittest.TestCase):
    """表头重复是静默缺陷：pandas 会把第二列改名成 xxx.1，下游按列名取值拿到 DataFrame。"""

    def test_summary_cols_have_no_duplicates(self):
        from scripts.extract_eye_metrics_batch import SUMMARY_COLS
        dup = [c for c in set(SUMMARY_COLS) if SUMMARY_COLS.count(c) > 1]
        self.assertEqual(dup, [], f"SUMMARY_COLS 存在重复列 {dup}，写出的 CSV 表头会重复")

    def test_ordered_columns_emits_no_duplicates(self):
        from scripts.extract_eye_metrics_batch import ordered_columns
        rows = [{"id": 1, "perclos_50": 0.1, "valid_s": 300.0},
                {"id": 2, "perclos_50": 0.2, "valid_s": 290.0}]
        cols = ordered_columns(rows, ["id", "valid_s"])
        self.assertEqual(len(cols), len(set(cols)))
        self.assertEqual(cols[:2], ["id", "valid_s"])


class DiscoveryTests(unittest.TestCase):
    """主输入是 80 个完整视频，但必须兼容切片树，两者还可能混在同一棵目录里。"""

    def setUp(self):
        import shutil, tempfile
        from pathlib import Path
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="eye_layout_", dir=TMP_ROOT))
        self.addCleanup(shutil.rmtree, str(self.root), True)
        # 连续视频布局：01/alert/easy/training_video.mp4
        for sid, st, df in (("01", "alert", "easy"), ("01", "sleepy", "easy"),
                            ("02", "alert", "hard")):
            d = self.root / sid / st / df
            d.mkdir(parents=True)
            (d / "training_video.mp4").write_text("")
        # 切片布局：01_easy_alert/clip_*.mp4（与上面的 01/easy/alert 条件重叠）
        for name, n in (("01_easy_alert", 3), ("02_hard_sleepy1", 2), ("bad_dir", 1)):
            d = self.root / name
            d.mkdir()
            for i in range(1, n + 1):
                (d / f"clip_{i}.mp4").write_text("")

    def _args(self, layout="auto", subjects=None, video_pattern="training_video.mp4",
              clip_pattern="clip_*.mp4"):
        return Namespace(layout=layout, subjects=subjects, root=str(self.root),
                         clip_root=None, data_root=None,
                         video_pattern=video_pattern, clip_pattern=clip_pattern)

    def test_auto_finds_both_layouts(self):
        from scripts.extract_eye_metrics_batch import discover_units
        units, _ = discover_units(self._args())
        got = {(u["id"], u["difficulty"], u["state"], u["layout"]) for u in units}
        assert_layouts = {x[3] for x in got}
        self.assertEqual(assert_layouts, {"video", "clip"})
        # 02_hard_sleepy1 只有切片版，02/alert/hard 只有连续版，都应保留
        self.assertIn(("02", "hard", "sleepy", "clip"), got)
        self.assertIn(("02", "hard", "alert", "video"), got)

    def test_calibration_video_is_excluded_not_concatenated(self):
        """校准视频与 training_video 同目录时，绝不能被拼进同一条件单元。"""
        from scripts.extract_eye_metrics_batch import discover_units
        calib = self.root / "01" / "alert" / "easy" / "calibration_video.mp4"
        calib.write_text("")
        units, excluded = discover_units(self._args())
        u = next(x for x in units if (x["id"], x["difficulty"], x["state"]) == ("01", "easy", "alert"))
        self.assertEqual([p.name for p in u["paths"]], ["training_video.mp4"])
        self.assertTrue(any("calibration_video.mp4" in e for e in excluded))

    def test_relaxed_pattern_picks_calibration_back(self):
        """放宽白名单应当真的放宽，并且用 >1 段的告警把它显式暴露出来。"""
        from scripts.extract_eye_metrics_batch import discover_units
        (self.root / "01" / "alert" / "easy" / "calibration_video.mp4").write_text("")
        units, excluded = discover_units(self._args(video_pattern="*video.mp4"))
        u = next(x for x in units if (x["id"], x["difficulty"], x["state"]) == ("01", "easy", "alert"))
        self.assertEqual(len(u["paths"]), 2)
        self.assertNotIn(str(self.root / "01" / "alert" / "easy" / "training_video.mp4"), excluded)

    def test_continuous_video_wins_when_both_match_same_condition(self):
        from scripts.extract_eye_metrics_batch import discover_units
        units, _ = discover_units(self._args())
        got = {(x["id"], x["difficulty"], x["state"]): x for x in units}
        u = got[("01", "easy", "alert")]
        self.assertEqual(u["layout"], "video")
        self.assertEqual(len(u["paths"]), 1)

    def test_layouts_do_not_cross_contaminate(self):
        """切片文件不能被 video 布局当成整段视频认走，反之亦然。"""
        from scripts.extract_eye_metrics_batch import discover_clip_layout, discover_video_layout
        vids, _ = discover_video_layout(self.root, None)
        self.assertTrue(all(len(u["paths"]) == 1 for u in vids))
        self.assertEqual({u["name"] for u in vids},
                         {"01_easy_alert", "01_easy_sleepy", "02_hard_alert"})
        clips, skipped = discover_clip_layout(self.root, None)
        self.assertEqual({u["name"] for u in clips}, {"01_easy_alert", "02_hard_sleepy1"})
        self.assertTrue(all(p.name.startswith("clip_") for u in clips for p in u["paths"]))
        self.assertTrue(skipped)                      # bad_dir 含 mp4 但目录名不合规 -> 被报告

    def test_subjects_are_matched_numerically_not_as_strings(self):
        from scripts.extract_eye_metrics_batch import discover_units
        units, _ = discover_units(self._args(subjects="1,2"))
        self.assertTrue(units)
        self.assertEqual({u["id"] for u in units}, {"01", "02"})
        units, _ = discover_units(self._args(subjects="07"))
        self.assertEqual(units, [])

    def test_explicit_layout_restricts_discovery(self):
        from scripts.extract_eye_metrics_batch import discover_units
        self.assertTrue(all(u["layout"] == "clip" for u in discover_units(self._args("clip"))[0]))
        self.assertTrue(all(u["layout"] == "video" for u in discover_units(self._args("video"))[0]))

    def test_clip_ordering_is_numeric_not_lexical(self):
        from scripts.extract_eye_metrics_batch import discover_clip_layout
        clips, _ = discover_clip_layout(self.root, None)
        u = next(x for x in clips if x["name"] == "01_easy_alert")
        self.assertEqual([p.name for p in u["paths"]],
                         ["clip_1.mp4", "clip_2.mp4", "clip_3.mp4"])
        (self.root / "01_easy_alert" / "clip_10.mp4").write_text("")
        clips, _ = discover_clip_layout(self.root, None)
        u = next(x for x in clips if x["name"] == "01_easy_alert")
        self.assertEqual([p.name for p in u["paths"]][-1], "clip_10.mp4")


if __name__ == "__main__":
    unittest.main()
