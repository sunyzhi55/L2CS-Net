import json
import math
import tempfile
import unittest
from pathlib import Path

TMP_ROOT = Path("C:/tmp")

from scripts.plot_gamma_fit_deviation import (
    collect_deviation_values,
    fit_gamma_distribution,
    plot_gamma_fit,
)


class PlotGammaFitDeviationTests(unittest.TestCase):
    def test_collect_deviation_values_reads_nested_jsonl_and_filters_invalid(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            root = Path(tmp)
            nested = root / "subject"
            nested.mkdir()
            jsonl_path = nested / "01_easy_alert.jsonl"
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps({"deviation_px_after_calibrate": 10.0}),
                        json.dumps({"deviation_px_after_calibrate": None}),
                        json.dumps({"deviation_px_after_calibrate": "20.5"}),
                        json.dumps({"deviation_px_after_calibrate": -1.0}),
                        json.dumps({"other": 99}),
                    ]
                ),
                encoding="utf-8",
            )

            values, files = collect_deviation_values(root)

        self.assertEqual(files, 1)
        self.assertEqual(values.tolist(), [10.0, 20.5])

    def test_collect_deviation_values_filters_by_state_from_filename(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            root = Path(tmp)
            samples = {
                "01_easy_alert.jsonl": 11.0,
                "01_hard_sleepy.jsonl": 22.0,
                "misc.jsonl": 99.0,
            }
            for filename, value in samples.items():
                (root / filename).write_text(
                    json.dumps({"deviation_px_after_calibrate": value}),
                    encoding="utf-8",
                )

            all_values, all_files = collect_deviation_values(root, state="all")
            alert_values, alert_files = collect_deviation_values(root, state="alert")
            sleepy_values, sleepy_files = collect_deviation_values(root, state="sleepy")

        self.assertEqual(all_files, 3)
        self.assertEqual(sorted(all_values.tolist()), [11.0, 22.0, 99.0])
        self.assertEqual(alert_files, 1)
        self.assertEqual(alert_values.tolist(), [11.0])
        self.assertEqual(sleepy_files, 1)
        self.assertEqual(sleepy_values.tolist(), [22.0])
    def test_fit_gamma_distribution_returns_ks_result(self):
        values = [12.0, 15.0, 18.0, 24.0, 31.0, 44.0, 60.0, 72.0]

        try:
            result = fit_gamma_distribution(values)
        except RuntimeError as exc:
            self.skipTest(str(exc))

        self.assertGreater(result.shape, 0)
        self.assertEqual(result.loc, 0.0)
        self.assertGreater(result.scale, 0)
        self.assertTrue(math.isfinite(result.ks_statistic))
        self.assertTrue(0.0 <= result.p_value <= 1.0)

    def test_plot_gamma_fit_writes_png(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            output_path = Path(tmp) / "gamma.png"
            values = [12.0, 15.0, 18.0, 24.0, 31.0, 44.0, 60.0, 72.0, 90.0]

            try:
                result = plot_gamma_fit(values, output_path, bins=5)
            except RuntimeError as exc:
                self.skipTest(str(exc))

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertGreater(result.shape, 0)


if __name__ == "__main__":
    unittest.main()

