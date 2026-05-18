"""
Tests for abaudit.validity
==========================
Covers audit() and AuditResult for:
  - Happy path (clean experiment, real effect)
  - Each individual flag triggering correctly
  - Edge cases and invalid inputs
  - summary() and __repr__ output

Run with:
    pytest tests/test_validity.py -v
"""

import numpy as np
import pytest

import abaudit as ab
from abaudit.validity import audit, AuditResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_experiment():
    """Large, well-powered experiment with a real effect. Should be clean."""
    rng  = np.random.default_rng(42)
    ctrl = rng.normal(0.0, 1.0, 2000)
    trt  = rng.normal(0.3, 1.0, 2000)
    return ctrl, trt


@pytest.fixture
def null_experiment():
    """No true effect — control and treatment are identical."""
    rng  = np.random.default_rng(0)
    ctrl = rng.normal(0.0, 1.0, 500)
    trt  = rng.normal(0.0, 1.0, 500)
    return ctrl, trt


@pytest.fixture
def tiny_experiment():
    """Very small sample — underpowered."""
    rng  = np.random.default_rng(7)
    ctrl = rng.normal(0.0, 1.0, 20)
    trt  = rng.normal(0.5, 1.0, 20)
    return ctrl, trt


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_audit_result(self, clean_experiment):
        ctrl, trt = clean_experiment
        result = audit(ctrl, trt)
        assert isinstance(result, AuditResult)

    def test_importable_from_top_level(self, clean_experiment):
        """ab.audit() should work without importing submodules."""
        ctrl, trt = clean_experiment
        result = ab.audit(ctrl, trt)
        assert isinstance(result, AuditResult)


# ── Core fields ───────────────────────────────────────────────────────────────

class TestCoreFields:
    def test_p_value_in_range(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert 0.0 <= r.p_value <= 1.0

    def test_ppv_in_range(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, prior_f=0.2)
        assert 0.0 < r.ppv < 1.0

    def test_power_in_range(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert 0.0 < r.power <= 1.0

    def test_bias_score_in_range(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert 0.0 <= r.bias_score <= 1.0

    def test_n_correct(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert r.n_control   == len(ctrl)
        assert r.n_treatment == len(trt)

    def test_alpha_stored(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, alpha=0.01)
        assert r.alpha == 0.01

    def test_prior_f_stored(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, prior_f=0.3)
        assert r.prior_f == 0.3

    def test_flags_is_list(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert isinstance(r.flags, list)

    def test_recommendations_is_list(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert isinstance(r.recommendations, list)


# ── Clean experiment ──────────────────────────────────────────────────────────

class TestCleanExperiment:
    def test_no_srm_flag(self, clean_experiment):
        """Equal group sizes → no SRM."""
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert not r.srm_flag

    def test_no_optional_stopping_flag(self, clean_experiment):
        """n_peeks=1 → no optional stopping flag."""
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, n_peeks=1)
        assert not r.optional_stopping_flag

    def test_no_multiple_metrics_flag(self, clean_experiment):
        """Single metric → no multiple metrics flag."""
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert not r.multiple_metrics_flag

    def test_low_bias_score(self, clean_experiment):
        """Well-designed experiment with real effect → low bias score."""
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, prior_f=0.5)
        assert r.bias_score < 0.4


# ── Individual flags ──────────────────────────────────────────────────────────

class TestSRMFlag:
    def test_srm_triggered_on_imbalance(self):
        """Severe traffic imbalance should trigger SRM flag."""
        rng  = np.random.default_rng(1)
        ctrl = rng.normal(0, 1, 9000)
        trt  = rng.normal(0, 1, 1000)
        r    = audit(ctrl, trt, expected_split=0.5)
        assert r.srm_flag

    def test_srm_not_triggered_on_balanced(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert not r.srm_flag

    def test_srm_raises_bias_score(self):
        """SRM should increase bias score."""
        rng      = np.random.default_rng(1)
        ctrl_bal = np.random.default_rng(42).normal(0, 1, 1000)
        trt_bal  = np.random.default_rng(43).normal(0, 1, 1000)
        ctrl_imb = rng.normal(0, 1, 9000)
        trt_imb  = rng.normal(0, 1, 1000)
        r_clean  = audit(ctrl_bal, trt_bal)
        r_srm    = audit(ctrl_imb, trt_imb, expected_split=0.5)
        assert r_srm.bias_score > r_clean.bias_score


class TestMultipleMetrics:
    def test_corrected_p_value_inflated(self, clean_experiment):
        """Bonferroni correction should multiply p by n_metrics."""
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, metrics=['m1', 'm2', 'm3', 'm4', 'm5'])
        assert abs(r.p_value_corrected - min(r.p_value * 5, 1.0)) < 1e-10

    def test_n_metrics_stored(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, metrics=['a', 'b', 'c'])
        assert r.n_metrics == 3

    def test_single_metric_no_correction(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert r.p_value_corrected == r.p_value
        assert r.n_metrics == 1


class TestOptionalStopping:
    def test_flag_triggered_above_threshold(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, n_peeks=10)
        assert r.optional_stopping_flag

    def test_flag_not_triggered_below_threshold(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, n_peeks=2)
        assert not r.optional_stopping_flag

    def test_effective_alpha_inflated(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, n_peeks=10, alpha=0.05)
        assert r.effective_alpha > 0.05

    def test_effective_alpha_equals_alpha_when_no_peeking(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt, n_peeks=1, alpha=0.05)
        assert abs(r.effective_alpha - 0.05) < 1e-10


class TestUnderpowered:
    def test_underpowered_flag_small_sample(self, tiny_experiment):
        ctrl, trt = tiny_experiment
        r = audit(ctrl, trt)
        assert r.underpowered_flag

    def test_not_underpowered_large_sample(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        assert not r.underpowered_flag


class TestPPV:
    def test_ppv_increases_with_prior(self, clean_experiment):
        """Higher prior → higher PPV."""
        ctrl, trt = clean_experiment
        r_low  = audit(ctrl, trt, prior_f=0.05)
        r_high = audit(ctrl, trt, prior_f=0.50)
        assert r_high.ppv > r_low.ppv

    def test_ppv_low_prior_adds_flag(self, tiny_experiment):
        """Very low prior with low power → PPV flag."""
        ctrl, trt = tiny_experiment
        r = audit(ctrl, trt, prior_f=0.01)
        assert any("PPV" in f for f in r.flags)


# ── Invalid inputs ────────────────────────────────────────────────────────────

class TestInvalidInputs:
    def test_too_few_control(self):
        with pytest.raises(ValueError, match="at least 2"):
            audit(np.array([1.0]), np.array([1.0, 2.0, 3.0]))

    def test_too_few_treatment(self):
        with pytest.raises(ValueError, match="at least 2"):
            audit(np.array([1.0, 2.0]), np.array([1.0]))

    def test_invalid_prior_zero(self):
        with pytest.raises(ValueError, match="prior_f"):
            audit(np.ones(10), np.ones(10), prior_f=0.0)

    def test_invalid_prior_one(self):
        with pytest.raises(ValueError, match="prior_f"):
            audit(np.ones(10), np.ones(10), prior_f=1.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            audit(np.ones(10), np.ones(10), alpha=0.0)

    def test_invalid_n_peeks(self):
        with pytest.raises(ValueError, match="n_peeks"):
            audit(np.ones(10), np.ones(10), n_peeks=0)


# ── Output methods ────────────────────────────────────────────────────────────

class TestOutput:
    def test_summary_runs_without_error(self, clean_experiment, capsys):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        r.summary()
        captured = capsys.readouterr()
        assert "abaudit" in captured.out
        assert "PPV" in captured.out
        assert "Bias score" in captured.out

    def test_repr_contains_key_fields(self, clean_experiment):
        ctrl, trt = clean_experiment
        r = audit(ctrl, trt)
        s = repr(r)
        assert "AuditResult" in s
        assert "ppv="        in s
        assert "bias_score=" in s

    def test_summary_shows_flags(self, capsys):
        """Experiment with SRM should print a warning in summary."""
        rng  = np.random.default_rng(1)
        ctrl = rng.normal(0, 1, 9000)
        trt  = rng.normal(0, 1, 1000)
        r    = audit(ctrl, trt)
        r.summary()
        captured = capsys.readouterr()
        assert "Warnings" in captured.out
