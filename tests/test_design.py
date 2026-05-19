"""
Tests for abaudit.design
========================
Covers power_analysis(), ppv_given_design(), minimum_trustworthy_n(),
design_summary(), and DesignResult.

Run with:
    pytest tests/test_design.py -v
"""

import pytest
import abaudit as ab
from abaudit.design import (
    power_analysis,
    ppv_given_design,
    minimum_trustworthy_n,
    design_summary,
    DesignResult,
)


# ── power_analysis ────────────────────────────────────────────────────────────

class TestPowerAnalysis:
    def test_returns_dict(self):
        result = power_analysis(effect_size=0.5)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = power_analysis(effect_size=0.5)
        for key in ["n_per_group", "n_total", "effect_size",
                    "power_target", "power_achieved", "alpha"]:
            assert key in result

    def test_n_total_is_double_n_per_group(self):
        result = power_analysis(effect_size=0.5)
        assert result["n_total"] == result["n_per_group"] * 2

    def test_medium_effect_standard_scenario(self):
        """d=0.5, power=0.80, alpha=0.05 → n per group ≈ 64."""
        result = power_analysis(effect_size=0.5, power=0.80, alpha=0.05)
        assert 60 <= result["n_per_group"] <= 70

    def test_small_effect_needs_more_n(self):
        """Smaller effect → larger required n."""
        small  = power_analysis(effect_size=0.2)
        medium = power_analysis(effect_size=0.5)
        assert small["n_per_group"] > medium["n_per_group"]

    def test_higher_power_needs_more_n(self):
        result_80 = power_analysis(effect_size=0.5, power=0.80)
        result_95 = power_analysis(effect_size=0.5, power=0.95)
        assert result_95["n_per_group"] > result_80["n_per_group"]

    def test_power_achieved_close_to_target(self):
        """Achieved power should be at or just above the target."""
        result = power_analysis(effect_size=0.5, power=0.80)
        assert result["power_achieved"] >= 0.79

    def test_effect_size_stored(self):
        result = power_analysis(effect_size=0.3)
        assert result["effect_size"] == 0.3

    def test_alpha_stored(self):
        result = power_analysis(effect_size=0.5, alpha=0.01)
        assert result["alpha"] == 0.01

    def test_invalid_zero_effect(self):
        with pytest.raises(ValueError, match="non-zero"):
            power_analysis(effect_size=0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            power_analysis(effect_size=0.5, alpha=0.0)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="power"):
            power_analysis(effect_size=0.5, power=1.0)

    def test_importable_from_top_level(self):
        result = ab.power_analysis(effect_size=0.5)
        assert "n_per_group" in result


# ── ppv_given_design ──────────────────────────────────────────────────────────

class TestPPVGivenDesign:
    def test_returns_float_in_range(self):
        result = ppv_given_design(effect_size=0.5, n_per_group=64,
                                   prior_f=0.2)
        assert isinstance(result, float)
        assert 0 < result < 1

    def test_larger_n_gives_higher_ppv(self):
        """More data → higher power → higher PPV."""
        ppv_small = ppv_given_design(0.3, 50,   0.2)
        ppv_large = ppv_given_design(0.3, 1000, 0.2)
        assert ppv_large > ppv_small

    def test_higher_prior_gives_higher_ppv(self):
        ppv_low  = ppv_given_design(0.5, 100, prior_f=0.05)
        ppv_high = ppv_given_design(0.5, 100, prior_f=0.50)
        assert ppv_high > ppv_low

    def test_low_prior_low_ppv(self):
        """Very low prior → PPV < 0.5 even with decent n."""
        result = ppv_given_design(0.3, 100, prior_f=0.01)
        assert result < 0.5

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n_per_group"):
            ppv_given_design(0.5, 1, 0.2)

    def test_invalid_prior(self):
        with pytest.raises(ValueError, match="prior_f"):
            ppv_given_design(0.5, 100, prior_f=0.0)

    def test_importable_from_top_level(self):
        result = ab.ppv_given_design(0.5, 100, 0.2)
        assert 0 < result < 1


# ── minimum_trustworthy_n ─────────────────────────────────────────────────────

class TestMinimumTrustworthyN:
    def test_returns_dict(self):
        result = minimum_trustworthy_n(0.5, 0.2)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = minimum_trustworthy_n(0.5, 0.2)
        for key in ["n_per_group", "n_total", "ppv_achieved",
                    "power_at_n", "target_ppv", "feasible"]:
            assert key in result

    def test_feasible_result(self):
        result = minimum_trustworthy_n(0.5, 0.3, target_ppv=0.80)
        assert result["feasible"] is True
        assert result["n_per_group"] is not None
        assert result["n_total"] == result["n_per_group"] * 2

    def test_ppv_achieved_meets_target(self):
        target = 0.80
        result = minimum_trustworthy_n(0.5, 0.3, target_ppv=target)
        assert result["ppv_achieved"] >= target

    def test_infeasible_very_low_prior(self):
        """Extremely low prior → target PPV not achievable."""
        result = minimum_trustworthy_n(0.1, prior_f=0.001,
                                        target_ppv=0.99, max_n=10_000)
        assert result["feasible"] is False
        assert result["n_per_group"] is None

    def test_larger_effect_needs_less_n(self):
        small  = minimum_trustworthy_n(0.2, 0.3)
        large  = minimum_trustworthy_n(0.8, 0.3)
        assert large["n_per_group"] < small["n_per_group"]

    def test_higher_target_ppv_needs_more_n(self):
        low  = minimum_trustworthy_n(0.5, 0.3, target_ppv=0.70)
        high = minimum_trustworthy_n(0.5, 0.3, target_ppv=0.90)
        if low["feasible"] and high["feasible"]:
            assert high["n_per_group"] >= low["n_per_group"]

    def test_invalid_target_ppv(self):
        with pytest.raises(ValueError, match="target_ppv"):
            minimum_trustworthy_n(0.5, 0.2, target_ppv=0.0)

    def test_invalid_prior(self):
        with pytest.raises(ValueError, match="prior_f"):
            minimum_trustworthy_n(0.5, prior_f=1.0)

    def test_importable_from_top_level(self):
        result = ab.minimum_trustworthy_n(0.5, 0.2)
        assert "n_per_group" in result


# ── design_summary ────────────────────────────────────────────────────────────

class TestDesignSummary:
    def test_returns_design_result(self):
        result = design_summary(effect_size=0.5, prior_f=0.2)
        assert isinstance(result, DesignResult)

    def test_importable_from_top_level(self):
        result = ab.design_summary(effect_size=0.5, prior_f=0.2)
        assert isinstance(result, DesignResult)

    def test_n_recommended_at_least_n_power(self):
        """Recommended n must be >= what power analysis alone requires."""
        result = design_summary(0.5, 0.2)
        if result.ppv_feasible:
            assert result.n_recommended >= result.n_per_group_power

    def test_ppv_at_recommended_meets_target(self):
        result = design_summary(0.5, prior_f=0.3, target_ppv=0.80)
        if result.ppv_feasible:
            assert result.ppv_at_recommended >= 0.79

    def test_power_at_recommended_meets_target(self):
        result = design_summary(0.5, prior_f=0.3, target_power=0.80)
        if result.ppv_feasible:
            assert result.power_at_recommended >= 0.79

    def test_infeasible_low_prior(self):
        result = design_summary(0.1, prior_f=0.001,
                                 target_ppv=0.99)
        assert not result.ppv_feasible
        assert result.n_recommended is None
        assert any("not achievable" in w.lower() or
                   "not feasible"   in w.lower() or
                   "insufficient"   in w.lower()
                   for w in result.warnings)

    def test_small_effect_warning(self):
        result = design_summary(effect_size=0.05, prior_f=0.3)
        assert any("small effect" in w.lower() for w in result.warnings)

    def test_very_low_prior_warning(self):
        result = design_summary(effect_size=0.5, prior_f=0.01)
        assert any("prior" in w.lower() for w in result.warnings)

    def test_invalid_prior(self):
        with pytest.raises(ValueError, match="prior_f"):
            design_summary(0.5, prior_f=0.0)

    def test_invalid_effect(self):
        with pytest.raises(ValueError, match="effect_size"):
            design_summary(0.0, prior_f=0.2)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            design_summary(0.5, prior_f=0.2, alpha=1.0)

    def test_invalid_target_power(self):
        with pytest.raises(ValueError, match="target_power"):
            design_summary(0.5, prior_f=0.2, target_power=0.0)

    def test_invalid_target_ppv(self):
        with pytest.raises(ValueError, match="target_ppv"):
            design_summary(0.5, prior_f=0.2, target_ppv=1.0)


# ── DesignResult display ──────────────────────────────────────────────────────

class TestDesignResult:
    def test_summary_runs_without_error(self, capsys):
        result = design_summary(0.5, prior_f=0.2)
        result.summary()
        captured = capsys.readouterr()
        assert "abaudit" in captured.out
        assert "Effect size" in captured.out

    def test_summary_infeasible_runs_without_error(self, capsys):
        result = design_summary(0.1, prior_f=0.001, target_ppv=0.99)
        result.summary()
        captured = capsys.readouterr()
        assert "NOT FEASIBLE" in captured.out

    def test_repr_contains_key_fields(self):
        result = design_summary(0.5, prior_f=0.2)
        s = repr(result)
        assert "DesignResult" in s
        assert "prior_f"      in s
        assert "ppv"          in s
