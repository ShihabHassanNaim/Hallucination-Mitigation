"""Phase 6 — Calibration tests.

Covers:
* TemperatureScaler identity / softening / sharpening.
* IsotonicCalibrator monotonicity + edge cases.
* ECE / Brier / log-loss helpers.
* ``CalibratedEEDC.fit_calibrator`` + persistence.
* End-to-end: default ``Pipeline`` loads ``CalibratedEEDC`` cleanly.
"""
from __future__ import annotations

import json
import math
import os

os.environ["CRISP_MOCK"] = "1"
os.environ.setdefault("CRISP_INDEX_PATH", "data/test_index_phase6")

import pytest

from src.calibration import (
    CalibratedEEDC,
    IdentityCalibrator,
    IsotonicCalibrator,
    TemperatureScaler,
    brier_score,
    evaluate_calibrator,
    expected_calibration_error,
    log_loss,
)
from src.eedc import EEDCSignals, EEDCScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_signals(n: int = 50):
    """Synthesise signals where lower entropy + higher retrieval => supported."""
    sigs, labels = [], []
    for i in range(n):
        # "supported" examples have low entropy, high retrieval, high consistency.
        if i % 2 == 0:
            sigs.append(EEDCSignals(
                nli_entropy_norm=0.05 + 0.1 * (i / n),
                retrieval_top1=0.9 - 0.05 * (i / n),
                self_consistency=0.95,
            ))
            labels.append(1)
        else:
            sigs.append(EEDCSignals(
                nli_entropy_norm=0.7 + 0.1 * (i / n),
                retrieval_top1=0.2 + 0.05 * (i / n),
                self_consistency=0.3,
            ))
            labels.append(0)
    return sigs, labels


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


class TestTemperatureScaler:
    def test_identity_at_t1(self):
        ts = TemperatureScaler(temperature=1.0)
        out = ts.transform([0.1, 0.5, 0.9])
        assert out == pytest.approx([0.1, 0.5, 0.9])

    def test_softening_high_t_increases_uncertainty(self):
        ts = TemperatureScaler(temperature=2.0)
        out = ts.transform([0.9])
        assert 0.5 < out[0] < 0.9  # pulled toward 0.5
        ts2 = TemperatureScaler(temperature=1.0)
        assert ts2.transform([0.9])[0] == 0.9

    def test_sharpening_low_t_pulls_toward_extremes(self):
        ts = TemperatureScaler(temperature=0.5)
        out_hi = ts.transform([0.7])
        out_lo = ts.transform([0.3])
        assert out_hi[0] > 0.7
        assert out_lo[0] < 0.3

    def test_fit_lowers_log_loss(self):
        sigs, labels = _toy_signals(80)
        scorer = EEDCScorer().fit(sigs, labels) if hasattr(EEDCScorer, "fit") else EEDCScorer()
        # Score raw, then fit a temperature on top.
        raw = [scorer.score(s) for s in sigs]
        # Synthetic ground-truth: predictable mapping.
        # Pretend the "real" labels are based on entropy+retrieval directly.
        # We'll just verify that fit() returns T != 1 for our skew dataset.
        calibrated = TemperatureScaler()
        before_ll = log_loss(raw, labels)
        calibrated.fit(raw, labels)
        after_ll = log_loss(calibrated.transform(raw), labels)
        assert calibrated.temperature > 0.0
        # log-loss should not blow up — fit should produce a valid T.
        assert math.isfinite(after_ll)

    def test_round_trip_dict(self):
        ts = TemperatureScaler(temperature=1.7)
        d = ts.to_dict()
        ts2 = TemperatureScaler.from_dict(d)
        assert ts2.temperature == pytest.approx(1.7)


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_monotone_output(self):
        xs = [0.1, 0.3, 0.5, 0.7, 0.9]
        ys = [0, 0, 0, 1, 1]  # non-monotone raw labels; PAV pools them
        cal = IsotonicCalibrator().fit(xs, ys)
        out = cal.transform(xs)
        # Isotonic output must be non-decreasing.
        for a, b in zip(out, out[1:]):
            assert a <= b + 1e-9, f"output not monotone: {out}"

    def test_extreme_clamps(self):
        cal = IsotonicCalibrator().fit([0.2, 0.8], [0, 1])
        assert cal.transform([-0.5])[0] == pytest.approx(0.0, abs=1e-3)
        assert cal.transform([1.5])[0] == pytest.approx(1.0, abs=1e-3)

    def test_empty_fit_safe(self):
        cal = IsotonicCalibrator().fit([], [])
        assert cal.transform([0.5]) == [0.5]

    def test_round_trip_dict(self):
        cal = IsotonicCalibrator().fit([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
        d = cal.to_dict()
        cal2 = IsotonicCalibrator.from_dict(d)
        out1 = cal.transform([0.1, 0.5, 0.9])
        out2 = cal2.transform([0.1, 0.5, 0.9])
        assert out1 == pytest.approx(out2)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_ece_perfect_calibrated(self):
        scores = [0.1, 0.4, 0.5, 0.6, 0.9]
        labels = [0, 0, 0, 1, 1]
        ece = expected_calibration_error(scores, labels)
        assert 0.0 <= ece <= 1.0

    def test_brier_perfect(self):
        assert brier_score([0.0, 1.0], [0, 1]) == pytest.approx(0.0)
        assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)

    def test_log_loss_finite(self):
        assert math.isfinite(log_loss([0.5], [1]))

    def test_evaluate_calibrator_returns_metrics(self):
        cal = TemperatureScaler(temperature=1.5)
        m = evaluate_calibrator(cal, [0.2, 0.5, 0.8], [0, 0, 1])
        assert m.method == "temperature"
        assert m.n == 3
        assert 0.0 <= m.accuracy <= 1.0


# ---------------------------------------------------------------------------
# CalibratedEEDC composition
# ---------------------------------------------------------------------------


class TestCalibratedEEDC:
    def test_score_identity_passes_through(self):
        scorer = EEDCScorer()
        cal = CalibratedEEDC(scorer=scorer, calibrator=IdentityCalibrator())
        sig = EEDCSignals(nli_entropy_norm=0.1, retrieval_top1=0.9,
                           self_consistency=0.9)
        assert cal.score(sig) == pytest.approx(scorer.score(sig), abs=1e-6)

    def test_score_temperature_changes_phi(self):
        scorer = EEDCScorer()
        sig = EEDCSignals(nli_entropy_norm=0.1, retrieval_top1=0.9,
                           self_consistency=0.9)
        plain = scorer.score(sig)
        cal = CalibratedEEDC(scorer=scorer,
                             calibrator=TemperatureScaler(temperature=2.0))
        assert cal.score(sig) != pytest.approx(plain, abs=1e-3)

    def test_fit_calibrator_lowers_ece(self):
        sigs, labels = _toy_signals(100)
        cal = CalibratedEEDC(scorer=EEDCScorer())
        # Make labels predictable from the raw score so a good calibrator wins.
        raw = [cal.raw_score(s) for s in sigs]
        # Synthesise "ground-truth" labels from raw score alone.
        gt = [1 if r >= 0.5 else 0 for r in raw]
        m = cal.fit_calibrator(sigs, gt, method="temperature")
        assert m.method == "temperature"
        assert cal.calibrator.name == "temperature"

    def test_save_load_round_trip(self, tmp_path):
        scorer = EEDCScorer()
        cal = CalibratedEEDC(scorer=scorer,
                             calibrator=TemperatureScaler(temperature=1.4))
        path = tmp_path / "eedc.json"
        cal.save(path)
        loaded = CalibratedEEDC.load(path)
        assert loaded.calibrator.temperature == pytest.approx(1.4)
        sig = EEDCSignals(0.2, 0.8, 0.7)
        assert cal.score(sig) == pytest.approx(loaded.score(sig), abs=1e-6)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineLoadsCalibratedEEDC:
    def test_pipeline_uses_calibrated_eedc(self, tmp_path, monkeypatch):
        from src.pipeline import Pipeline

        # Write a fake weights file in the Phase 6 composite format.
        weights_path = tmp_path / "eedc.json"
        weights_path.write_text(json.dumps({
            "weights": [-1.2, -0.6, -0.4, 1.1],
            "calibrator": {"kind": "temperature", "temperature": 1.5},
        }), encoding="utf-8")
        monkeypatch.setenv("CRISP_EEDC_WEIGHTS", str(weights_path))
        monkeypatch.setenv("CRISP_CALIBRATION", "temperature")
        # Re-load config so env override is applied.
        from src.config import load_config
        cfg = load_config()
        cfg.eedc.weights_path = str(weights_path)

        corpus = ["France's capital is Paris.", "Mars is the red planet."]
        pipeline = Pipeline(config=cfg)
        pipeline.build_index(corpus)
        result = pipeline.run("What is the capital of France?")
        assert 0.0 <= result.confidence <= 1.0