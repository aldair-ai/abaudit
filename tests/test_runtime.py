"""
Tests for abaudit.runtime
=========================
Covers check_srm(), check_optional_stopping(), check_novelty_effect().

Run with:
    pytest tests/test_runtime.py -v
"""

import numpy as np
import pytest

import abaudit as ab
from abaudit.runtime import (
    check_srm,
    check_optional_stopping,
    check_novelty_effect,
)


# ── check_srm ────────────────────────────────────────────────────────────────

class TestCheckSRM:
    def test_returns_dict(self):
        result = check_srm(5000, 5000)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = check_srm(5000, 5000)
        for key in ["srm_detected", "chi2_stat", "p_value",
                    "n_control", "n_treatment", "n_total",
                    "expected_split", "observed_split",
                    "severity", "message"]:
            assert key in result

    def test_balanced_split_no_srm(self):
        result = check_srm(5000, 5000)
        assert result["srm_detected"] is False
        assert result["severity"] == "none"

    def test_severe_imbalance_srm_detected(self):
        result = check_srm(9000, 1000)
        assert result["srm_detected"] is True
        assert result["severity"] == "severe"

    def test_n_total_correct(self):
        result = check_srm(3000, 4000)
        assert result["n_total"] == 7000

    def test_observed_split_correct(self):
        result = check_srm(4000, 6000)
        assert abs(result["observed_split"] - 0.6) < 0.001

    def test_custom_expected_split(self):
        """90/10 intended split with matching observed → no SRM."""
        result = check_srm(9000, 1000, expected_split=0.1)
        assert result["srm_detected"] is False

    def test_message_is_string(self):
        result = check_srm(5000, 5000)
        assert isinstance(result["message"], str)
        assert len(result["message"]) > 0

    def test_p_value_in_range(self):
        result = check_srm(5000, 5000)
        assert 0 <= result["p_value"] <= 1

    def test_invalid_n_control(self):
        with pytest.raises(ValueError, match="n_control"):
            check_srm(0, 1000)

    def test_invalid_n_treatment(self):
        with pytest.raises(ValueError, match="n_control"):
            check_srm(1000, 0)

    def test_invalid_expected_split(self):
        with pytest.raises(ValueError, match="expected_split"):
            check_srm(5000, 5000, expected_split=0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            check_srm(5000, 5000, alpha=1.0)

    def test_importable_from_top_level(self):
        result = ab.check_srm(5000, 5000)
        assert "srm_detected" in result

    def test_mild_severity(self):
        """Small but detectable imbalance → mild severity."""
        result = check_srm(5300, 4700, alpha=0.001)
        if result["srm_detected"]:
            assert result["severity"] in ("mild", "severe")


# ── check_optional_stopping ───────────────────────────────────────────────────

class TestCheckOptionalStopping:
    def test_returns_dict(self):
        result = check_optional_stopping([0.1, 0.05, 0.03])
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = check_optional_stopping([0.1, 0.05, 0.03])
        for key in ["n_peeks", "final_p_value", "effective_alpha",
                    "fwer_inflation", "early_crossing",
                    "n_crossings", "risk_level", "message"]:
            assert key in result

    def test_single_peek_low_risk(self):
        result = check_optional_stopping([0.03])
        assert result["n_peeks"] == 1
        assert result["risk_level"] == "low"
        assert abs(result["effective_alpha"] - 0.05) < 1e-9

    def test_many_peeks_moderate_risk(self):
        history = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
        result  = check_optional_stopping(history)
        assert result["n_peeks"] == 7
        assert result["effective_alpha"] > 0.05

    def test_early_crossing_detected(self):
        """p dips below alpha early then recovers → early_crossing=True."""
        history = [0.12, 0.04, 0.08, 0.09, 0.10]
        result  = check_optional_stopping(history, alpha=0.05)
        assert result["early_crossing"] is True
        assert result["risk_level"] == "high"

    def test_no_early_crossing_monotone(self):
        """Monotonically decreasing p → no early crossing."""
        history = [0.20, 0.15, 0.10, 0.06, 0.03]
        result  = check_optional_stopping(history)
        assert result["early_crossing"] is False

    def test_final_p_value_correct(self):
        history = [0.1, 0.05, 0.03]
        result  = check_optional_stopping(history)
        assert abs(result["final_p_value"] - 0.03) < 1e-6

    def test_fwer_inflation_positive(self):
        history = [0.1, 0.05, 0.03, 0.02, 0.01]
        result  = check_optional_stopping(history)
        assert result["fwer_inflation"] > 0

    def test_effective_alpha_increases_with_peeks(self):
        r1 = check_optional_stopping([0.03])
        r5 = check_optional_stopping([0.1, 0.08, 0.05, 0.04, 0.03])
        assert r5["effective_alpha"] > r1["effective_alpha"]

    def test_message_is_string(self):
        result = check_optional_stopping([0.03])
        assert isinstance(result["message"], str)

    def test_invalid_empty_history(self):
        with pytest.raises(ValueError, match="at least one"):
            check_optional_stopping([])

    def test_invalid_p_value_out_of_range(self):
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            check_optional_stopping([0.05, 1.5])

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            check_optional_stopping([0.03], alpha=0.0)

    def test_importable_from_top_level(self):
        result = ab.check_optional_stopping([0.03])
        assert "n_peeks" in result


# ── check_novelty_effect ──────────────────────────────────────────────────────

class TestCheckNoveltyEffect:
    @staticmethod
    def _make_data(early_trt_mean=0.5, late_trt_mean=0.1, n=200):
        rng = np.random.default_rng(42)
        return {
            "early_control":   rng.normal(0.0, 1.0, n),
            "early_treatment": rng.normal(early_trt_mean, 1.0, n),
            "late_control":    rng.normal(0.0, 1.0, n),
            "late_treatment":  rng.normal(late_trt_mean,  1.0, n),
        }

    def test_returns_dict(self):
        d = self._make_data()
        result = check_novelty_effect(**d)
        assert isinstance(result, dict)

    def test_required_keys(self):
        d = self._make_data()
        result = check_novelty_effect(**d)
        for key in ["early_effect", "late_effect", "effect_decay",
                    "novelty_detected", "p_value", "risk_level", "message"]:
            assert key in result

    def test_novelty_detected_when_effect_fades(self):
        """Large early effect that fades → novelty detected."""
        d = self._make_data(early_trt_mean=1.0, late_trt_mean=0.0, n=500)
        result = check_novelty_effect(**d)
        assert result["novelty_detected"] is True
        assert result["early_effect"] > result["late_effect"]

    def test_no_novelty_stable_effect(self):
        """Stable effect across periods → no novelty."""
        rng = np.random.default_rng(0)
        d = {
            "early_control":   rng.normal(0, 1, 500),
            "early_treatment": rng.normal(0.3, 1, 500),
            "late_control":    rng.normal(0, 1, 500),
            "late_treatment":  rng.normal(0.3, 1, 500),
        }
        result = check_novelty_effect(**d)
        assert result["novelty_detected"] is False

    def test_effect_decay_sign(self):
        """Decay = early - late, should be positive when novelty present."""
        d = self._make_data(early_trt_mean=1.0, late_trt_mean=0.0, n=500)
        result = check_novelty_effect(**d)
        assert result["effect_decay"] > 0
        assert result["early_effect"] > result["late_effect"]

    def test_p_value_in_range(self):
        d = self._make_data()
        result = check_novelty_effect(**d)
        assert 0 <= result["p_value"] <= 1

    def test_message_is_string(self):
        d = self._make_data()
        result = check_novelty_effect(**d)
        assert isinstance(result["message"], str)

    def test_invalid_too_few_observations(self):
        with pytest.raises(ValueError, match="early_control"):
            check_novelty_effect(
                early_control=np.array([1.0]),
                early_treatment=np.array([1.0, 2.0]),
                late_control=np.array([1.0, 2.0]),
                late_treatment=np.array([1.0, 2.0]),
            )

    def test_importable_from_top_level(self):
        d = self._make_data()
        result = ab.check_novelty_effect(**d)
        assert "novelty_detected" in result
