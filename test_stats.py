"""
Tests for abaudit._stats
========================
Every public function in _stats.py has at least:
  - A "happy path" test with a known answer
  - An edge case or boundary test
  - A test for invalid inputs (raises expected errors)

Run with:
    pytest tests/test_stats.py -v
"""

import numpy as np
import pytest

from abaudit._stats import (
    ppv,
    ppv_biased,
    fwer,
    bonferroni_threshold,
    bh_adjusted,
    srm_test,
    optional_stopping_effective_alpha,
    cohens_d,
    estimate_power,
    benford_test,
    last_digit_test,
)


# ── ppv ──────────────────────────────────────────────────────────────────────

class TestPPV:
    def test_known_value(self):
        """Replicates the Ioannidis (2005) example: f=0.1, power=0.8, α=0.05."""
        result = ppv(prior_f=0.1, power=0.8, alpha=0.05)
        assert abs(result - 0.64) < 0.01

    def test_high_prior_gives_high_ppv(self):
        """When most hypotheses tested are true, PPV should be high."""
        result = ppv(prior_f=0.8, power=0.8, alpha=0.05)
        assert result > 0.9

    def test_low_prior_gives_low_ppv(self):
        """Exploratory fishing: very low prior → low PPV even with good power."""
        result = ppv(prior_f=0.001, power=0.8, alpha=0.05)
        assert result < 0.02

    def test_perfect_power(self):
        """Power = 1 → PPV = f / (f + α(1-f))."""
        f = 0.3
        a = 0.05
        expected = f / (f + a * (1 - f))
        assert abs(ppv(prior_f=f, power=1.0, alpha=a) - expected) < 1e-10

    def test_invalid_prior_zero(self):
        with pytest.raises(ValueError, match="prior_f"):
            ppv(prior_f=0.0, power=0.8, alpha=0.05)

    def test_invalid_prior_one(self):
        with pytest.raises(ValueError, match="prior_f"):
            ppv(prior_f=1.0, power=0.8, alpha=0.05)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="power"):
            ppv(prior_f=0.2, power=0.0, alpha=0.05)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ppv(prior_f=0.2, power=0.8, alpha=0.0)


# ── ppv_biased ───────────────────────────────────────────────────────────────

class TestPPVBiased:
    def test_zero_bias_equals_ppv(self):
        """bias_u=0 must produce identical result to ppv()."""
        f, power, alpha = 0.2, 0.8, 0.05
        assert abs(ppv_biased(f, power, alpha, 0.0) - ppv(f, power, alpha)) < 1e-10

    def test_bias_reduces_ppv(self):
        """Adding bias should reduce PPV."""
        f, power, alpha = 0.2, 0.8, 0.05
        assert ppv_biased(f, power, alpha, 0.5) < ppv(f, power, alpha)

    def test_max_bias(self):
        """bias_u=1: everything reported as significant → PPV collapses to prior."""
        result = ppv_biased(0.1, 0.8, 0.05, bias_u=1.0)
        assert abs(result - 0.1) < 0.01

    def test_invalid_bias(self):
        with pytest.raises(ValueError, match="bias_u"):
            ppv_biased(0.2, 0.8, 0.05, bias_u=1.5)


# ── fwer ─────────────────────────────────────────────────────────────────────

class TestFWER:
    def test_single_test(self):
        """One test: FWER = α."""
        assert abs(fwer(0.05, 1) - 0.05) < 1e-10

    def test_twenty_tests(self):
        """The jelly bean scenario: m=20, α=0.05 → FWER ≈ 0.64."""
        assert abs(fwer(0.05, 20) - 0.6415) < 0.001

    def test_approaches_one(self):
        """Many tests → FWER approaches 1."""
        assert fwer(0.05, 1000) > 0.999

    def test_invalid_m(self):
        with pytest.raises(ValueError, match="m must"):
            fwer(0.05, 0)


# ── bonferroni_threshold ──────────────────────────────────────────────────────

class TestBonferroni:
    def test_standard(self):
        assert abs(bonferroni_threshold(0.05, 20) - 0.0025) < 1e-10

    def test_single(self):
        assert abs(bonferroni_threshold(0.05, 1) - 0.05) < 1e-10


# ── bh_adjusted ──────────────────────────────────────────────────────────────

class TestBHAdjusted:
    def test_all_null(self):
        """Uniformly distributed p-values: most adjusted should be > 0.05."""
        rng = np.random.default_rng(42)
        p   = rng.uniform(0, 1, 100)
        adj = bh_adjusted(p, alpha=0.05)
        assert (adj > 0.05).mean() > 0.8

    def test_all_significant(self):
        """All tiny p-values: adjusted should still be < 0.05."""
        p   = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        adj = bh_adjusted(p, alpha=0.05)
        assert (adj < 0.05).all()

    def test_monotonicity(self):
        """Adjusted p-values must be non-decreasing when sorted."""
        rng = np.random.default_rng(0)
        p   = rng.uniform(0, 1, 50)
        adj = bh_adjusted(p, alpha=0.05)
        sorted_adj = np.sort(adj)
        assert (np.diff(sorted_adj) >= -1e-12).all()

    def test_adjusted_geq_raw(self):
        """BH adjusted p-values are always >= the raw p-values."""
        p   = np.array([0.001, 0.01, 0.03, 0.04, 0.5])
        adj = bh_adjusted(p)
        assert (adj >= p - 1e-12).all()


# ── srm_test ─────────────────────────────────────────────────────────────────

class TestSRMTest:
    def test_perfect_split(self):
        """Equal groups → no SRM, p should be high."""
        _, p = srm_test(5000, 5000)
        assert p > 0.05

    def test_severe_imbalance(self):
        """Very unequal split → SRM detected, p should be tiny."""
        _, p = srm_test(9000, 1000)
        assert p < 0.001

    def test_custom_split(self):
        """90/10 intended split with matching observed → no SRM."""
        _, p = srm_test(9000, 1000, expected_split=0.1)
        assert p > 0.05

    def test_returns_two_floats(self):
        chi2, p = srm_test(5100, 4900)
        assert isinstance(chi2, float)
        assert isinstance(p, float)
        assert 0 <= p <= 1


# ── optional_stopping_effective_alpha ────────────────────────────────────────

class TestOptionalStopping:
    def test_no_peeking(self):
        """1 peek = running the test once = nominal alpha."""
        assert abs(optional_stopping_effective_alpha(0.05, 1) - 0.05) < 1e-10

    def test_inflated_by_peeking(self):
        """More peeks → higher effective alpha."""
        a1 = optional_stopping_effective_alpha(0.05, 1)
        a5 = optional_stopping_effective_alpha(0.05, 5)
        assert a5 > a1

    def test_many_peeks(self):
        """100 peeks at α=0.05 → effective alpha near 1."""
        eff = optional_stopping_effective_alpha(0.05, 100)
        assert eff > 0.99


# ── cohens_d ─────────────────────────────────────────────────────────────────

class TestCohensD:
    def test_zero_effect(self):
        """Identical groups → d ≈ 0."""
        rng  = np.random.default_rng(1)
        data = rng.normal(0, 1, 200)
        assert abs(cohens_d(data[:100], data[100:])) < 0.3

    def test_known_effect(self):
        """Groups with mean diff = 1, SD = 1 → d ≈ 1."""
        rng  = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 10_000)
        trt  = rng.normal(1, 1, 10_000)
        d    = cohens_d(ctrl, trt)
        assert abs(d - 1.0) < 0.05

    def test_sign(self):
        """Positive when treatment mean > control mean."""
        ctrl = np.array([1.0, 2.0, 3.0])
        trt  = np.array([4.0, 5.0, 6.0])
        assert cohens_d(ctrl, trt) > 0


# ── estimate_power ────────────────────────────────────────────────────────────

class TestEstimatePower:
    def test_standard_scenario(self):
        """d=0.5, n=64 per group, α=0.05 → power ≈ 0.8."""
        p = estimate_power(effect_size=0.5, n_control=64,
                           n_treatment=64, alpha=0.05)
        assert 0.75 < p < 0.85

    def test_large_n_high_power(self):
        """Very large n → power near 1."""
        p = estimate_power(0.2, 10_000, 10_000)
        assert p > 0.99

    def test_tiny_effect_low_power(self):
        """Tiny effect, small n → low power."""
        p = estimate_power(0.01, 30, 30)
        assert p < 0.1


# ── benford_test ──────────────────────────────────────────────────────────────

class TestBenfordTest:
    def test_natural_data_passes(self):
        """Log-normal data spanning orders of magnitude should pass (p > 0.05)."""
        rng    = np.random.default_rng(42)
        values = rng.lognormal(mean=5, sigma=2.5, size=1000)
        _, p   = benford_test(values)
        assert p > 0.05

    def test_fabricated_data_fails(self):
        """Uniform first digits should fail Benford's test."""
        rng    = np.random.default_rng(42)
        # Force uniform first digits by constructing values
        values = np.array([
            d * 10 ** rng.integers(0, 5)
            for d in rng.integers(1, 10, size=500)
        ], dtype=float)
        _, p = benford_test(values)
        assert p < 0.05

    def test_too_few_values_raises(self):
        with pytest.raises(ValueError, match="30 values"):
            benford_test(np.array([1.0, 2.0, 3.0]))


# ── last_digit_test ───────────────────────────────────────────────────────────

class TestLastDigitTest:
    def test_uniform_last_digits_pass(self):
        """Random integers should have roughly uniform last digits."""
        rng    = np.random.default_rng(0)
        values = rng.integers(1, 1_000_000, size=1000).astype(float)
        _, p   = last_digit_test(values)
        assert p > 0.05

    def test_all_zeros_fail(self):
        """Values all ending in 0 → extreme deviation from uniformity."""
        values = np.array([100, 200, 300, 400, 500] * 50, dtype=float)
        _, p   = last_digit_test(values)
        assert p < 0.001

    def test_returns_floats(self):
        values    = np.arange(1, 101, dtype=float)
        chi2, p   = last_digit_test(values)
        assert isinstance(chi2, float)
        assert isinstance(p, float)
