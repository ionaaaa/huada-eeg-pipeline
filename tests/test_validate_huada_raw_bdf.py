from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_huada_raw_bdf.py"
SPEC = importlib.util.spec_from_file_location("validate_huada_raw_bdf", SCRIPT_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def valid_config() -> dict:
    return {
        "preprocessing": {
            "target_sfreq": 200.0,
            "window_seconds": 10.0,
            "hop_seconds": 10.0,
            "segment_samples": 2000,
            "bandpass_low": 0.3,
            "bandpass_high": 75.0,
            "notch_hz": 50.0,
            "unit_before_normalization": "uV",
            "normalization": "clip_divide_200",
            "normalization_detail": "x_uV / 200, then clip to [-1, 1]",
        },
        "windowing": {"drop_incomplete_tail": True, "padding": False},
        "source_expectations": {"format": "BDF", "sampling_rate_hz": 500.0, "n_channels": 64},
    }


class SubjectDirectoryParsingTests(unittest.TestCase):
    def test_full_subject_directory(self) -> None:
        row = validator.parse_subject_directory("SUB1_31_M")
        self.assertIsNotNone(row)
        self.assertEqual(row["subject_id"], "SUB1")
        self.assertEqual(row["subject_number"], 1)
        self.assertEqual(row["age"], 31)
        self.assertEqual(row["sex"], "male")

    def test_missing_age_subject_directory(self) -> None:
        row = validator.parse_subject_directory("SUB5_M")
        self.assertIsNotNone(row)
        self.assertEqual(row["subject_id"], "SUB5")
        self.assertIsNone(row["age"])
        self.assertEqual(row["sex"], "male")
        self.assertEqual(row["metadata_status"], "age_missing")

    def test_case_insensitive_sex_conversion(self) -> None:
        male = validator.parse_subject_directory("sub9_40_m")
        female = validator.parse_subject_directory("SuB10_24_f")
        self.assertEqual(male["sex"], "male")
        self.assertEqual(female["sex"], "female")


class UnitConversionTests(unittest.TestCase):
    def test_supported_units_convert_to_uv(self) -> None:
        expected = {"V": 1e6, "mV": 1e3, "µV": 1.0, "uV": 1.0, "μV": 1.0}
        for unit, multiplier in expected.items():
            with self.subTest(unit=unit):
                self.assertEqual(validator.unit_to_uv_multiplier(unit), multiplier)

    def test_unknown_unit_fails(self) -> None:
        self.assertIsNone(validator.unit_to_uv_multiplier("nV"))
        self.assertIsNone(validator.unit_to_uv_multiplier(""))
        self.assertIsNone(validator.unit_to_uv_multiplier(None))


class ChannelLayoutTests(unittest.TestCase):
    def test_channel_name_or_order_mismatch_is_detected(self) -> None:
        reference = [
            {"channel_name": "Fp1", "sampling_rate_hz": 500.0, "physical_dimension": "µV"},
            {"channel_name": "Fp2", "sampling_rate_hz": 500.0, "physical_dimension": "µV"},
        ]
        actual = [
            {"channel_name": "Fp2", "sampling_rate_hz": 500.0, "physical_dimension": "µV"},
            {"channel_name": "Fp1", "sampling_rate_hz": 500.0, "physical_dimension": "µV"},
        ]
        mismatches = validator.compare_channel_layout(reference, actual)
        self.assertEqual(len(mismatches), 2)
        self.assertTrue(all(row["field"] == "channel_name" for row in mismatches))


class ConfigTests(unittest.TestCase):
    def test_segment_samples_consistency_passes(self) -> None:
        validator.validate_config(valid_config())

    def test_segment_samples_inconsistency_fails(self) -> None:
        config = valid_config()
        config["preprocessing"]["segment_samples"] = 1999
        with self.assertRaises(validator.ConfigValidationError):
            validator.validate_config(config)


class BatchIsolationTests(unittest.TestCase):
    def test_one_failure_does_not_stop_batch(self) -> None:
        def fake_validator(value: int) -> int:
            if value == 2:
                raise RuntimeError("synthetic failure")
            return value * 10

        successes, failures = validator.run_batch([1, 2, 3], fake_validator)
        self.assertEqual(successes, [(1, 10), (3, 30)])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], 2)
        self.assertIsInstance(failures[0][1], RuntimeError)


if __name__ == "__main__":
    unittest.main()
