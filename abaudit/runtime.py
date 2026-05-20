"""
abaudit.runtime
===============
During-experiment health checks.

These functions are designed to be called *while* an experiment is
running — not just at the end.  They help catch problems early:
traffic imbalances, peeking inflation, and novelty effects.

Each function returns a plain dict so results are easy to log,
serialize, or feed into a monitoring dashboard.

Main functions
--------------
- ``check_srm``               : Sample Ratio Mismatch test
- ``check_optional_stopping`` : Peeking / early stopping diagnosis
- ``check_novelty_effect``    : Early vs. late effect comparison

Example
-------
>>> from abaudit.runtime import check_srm, check_optional_stopping
>>> check_srm(n_control=4850, n_treatment=5150)
>>> check_optional_stopping(p_value_history=[0.12, 0.08, 0.04])
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy import stats as _scipy_stats

from abaudit._stats import (
    srm_test                          as _srm_test,
    optional_stopping_effective_alpha as _opt_stop,
    fwer                              as _fwer,
)

__all__ = [
    "check_srm",
    "check_optional_stopping",
    "check_novelty_effect",
]


# ── check_srm ────────────────────────────────────────────────────────────────

def check_srm(
    n_control:      int,
    n_treatment:    int,
    expected_split: float = 0.5,
    alpha:          float = 0.01,
) -> dict:
    """Sample Ratio Mismatch (SRM) test.

    Tests whether the observed traffic split matches the intended split.
    SRM is one of the most common and most dangerous bugs in A/B testing
    — it invalidates all downstream analysis because the groups are no
    longer comparable.

    Common causes: logging errors, bot traffic, cookie deletion,
    redirect bugs, or misconfigured assignment logic.

    Parameters
    ----------
    n_control : int
        Number of units assigned to control.
    n_treatment : int
        Number of units assigned to treatment.
    expected_split : float, optional
        Expected fraction of traffic in treatment.  Default 0.5.
    alpha : float, optional
        Significance threshold for SRM detection.  Default 0.01
        (conventional — more conservative than the usual 0.05).

    Returns
    -------
    dict with keys:
        srm_detected  : bool  — True if SRM is present
        chi2_stat     : float — chi-squared statistic
        p_value       : float — p-value of the SRM test
        n_control     : int
        n_treatment   : int
        expected_split: float
        observed_split: float — actual fraction in treatment
        severity      : str   — 'none', 'mild', 'severe'
        message       : str   — human-readable interpretation

    Examples
    --------
    >>> from abaudit.runtime import check_srm
    >>> check_srm(n_control=4850, n_treatment=5150)
    {'srm_detected': False, ...}
    >>> check_srm(n_control=9000, n_treatment=1000)
    {'srm_detected': True, 'severity': 'severe', ...}
    """
    if n_control < 1 or n_treatment < 1:
        raise ValueError("n_control and n_treatment must be >= 1.")
    if not (0 < expected_split < 1):
        raise ValueError("expected_split must be strictly between 0 and 1.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")

    chi2, p      = _srm_test(n_control, n_treatment, expected_split)
    n_total      = n_control + n_treatment
    observed_split = n_treatment / n_total
    srm_detected = p < alpha

    # Severity based on relative deviation from expected split
    deviation = abs(observed_split - expected_split) / expected_split
    if not srm_detected:
        severity = "none"
    elif deviation < 0.05:
        severity = "mild"
    else:
        severity = "severe"

    if not srm_detected:
        message = (
            f"No SRM detected (p = {p:.4f}). "
            f"Traffic split looks as expected "
            f"({n_control:,} control / {n_treatment:,} treatment)."
        )
    else:
        message = (
            f"SRM detected (p = {p:.4f}, severity: {severity}). "
            f"Expected {expected_split:.1%} treatment, "
            f"observed {observed_split:.1%}. "
            "Investigate before trusting any results."
        )

    return {
        "srm_detected":   srm_detected,
        "chi2_stat":      round(chi2, 4),
        "p_value":        round(p, 6),
        "n_control":      n_control,
        "n_treatment":    n_treatment,
        "n_total":        n_total,
        "expected_split": expected_split,
        "observed_split": round(observed_split, 4),
        "severity":       severity,
        "message":        message,
    }


# ── check_optional_stopping ───────────────────────────────────────────────────

def check_optional_stopping(
    p_value_history: list[float],
    alpha:           float = 0.05,
) -> dict:
    """Diagnose optional stopping / peeking from a p-value history.

    Takes a time series of p-values recorded during the experiment and
    assesses whether peeking has inflated the false positive rate.

    Two checks are run:
    1. **FWER inflation**: effective α after n_peeks interim looks.
    2. **Early crossing**: did the p-value cross alpha early then recover?
       This is a sign the team may have been tempted to stop.

    Parameters
    ----------
    p_value_history : list[float]
        Ordered list of p-values recorded during the experiment.
        The last entry is the final p-value.
    alpha : float, optional
        Nominal significance level.  Default 0.05.

    Returns
    -------
    dict with keys:
        n_peeks           : int   — number of times p-value was recorded
        final_p_value     : float — last p-value in the history
        effective_alpha   : float — inflated alpha from peeking
        fwer_inflation    : float — how much alpha was inflated
        early_crossing    : bool  — True if p dipped below alpha early
                                    then rose again
        n_crossings       : int   — total times p crossed alpha
        risk_level        : str   — 'low', 'moderate', 'high'
        message           : str   — human-readable interpretation

    Examples
    --------
    >>> from abaudit.runtime import check_optional_stopping
    >>> history = [0.12, 0.08, 0.04, 0.06, 0.03]
    >>> check_optional_stopping(history)
    {'n_peeks': 5, 'early_crossing': True, ...}
    """
    if len(p_value_history) < 1:
        raise ValueError("p_value_history must contain at least one value.")
    if not all(0 <= p <= 1 for p in p_value_history):
        raise ValueError("All p-values must be in [0, 1].")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")

    p_arr        = np.array(p_value_history)
    n_peeks      = len(p_arr)
    final_p      = float(p_arr[-1])
    eff_alpha    = _opt_stop(alpha, n_peeks)
    inflation    = round(eff_alpha - alpha, 4)

    # Check for early crossing: p went below alpha then came back up
    crossed      = p_arr < alpha
    n_crossings  = int(np.sum(np.diff(crossed.astype(int)) != 0))
    early_cross  = bool(np.any(crossed[:-1]) and not crossed[-1])

    # Risk level
    if n_peeks <= 3 and not early_cross:
        risk = "low"
    elif n_peeks <= 10 and not early_cross:
        risk = "moderate"
    else:
        risk = "high"

    if early_cross:
        risk = "high"

    if risk == "low":
        message = (
            f"{n_peeks} peek(s) detected. Effective α = {eff_alpha:.4f}. "
            "Low peeking risk."
        )
    elif early_cross:
        message = (
            f"Early crossing detected — p-value dipped below {alpha} "
            f"at an interim look but recovered. "
            f"Effective α = {eff_alpha:.4f}. "
            "High risk of optional stopping bias."
        )
    else:
        message = (
            f"{n_peeks} peeks detected. Effective α = {eff_alpha:.4f} "
            f"(inflated by {inflation:.4f}). "
            "Consider using sequential testing methods."
        )

    return {
        "n_peeks":         n_peeks,
        "final_p_value":   round(final_p, 6),
        "effective_alpha": round(eff_alpha, 6),
        "fwer_inflation":  inflation,
        "early_crossing":  early_cross,
        "n_crossings":     n_crossings,
        "risk_level":      risk,
        "message":         message,
    }


# ── check_novelty_effect ──────────────────────────────────────────────────────

def check_novelty_effect(
    early_control:   np.ndarray,
    early_treatment: np.ndarray,
    late_control:    np.ndarray,
    late_treatment:  np.ndarray,
    alpha:           float = 0.05,
) -> dict:
    """Test for novelty effect by comparing early vs. late effect sizes.

    A novelty effect occurs when users behave differently simply because
    something is new — not because the change is genuinely better.
    The effect inflates early measurements and then fades as users adapt.

    This function splits the experiment into an early period and a late
    period, estimates the effect size in each, and tests whether they
    differ significantly.

    Parameters
    ----------
    early_control : array-like
        Primary metric for control group in the early period.
    early_treatment : array-like
        Primary metric for treatment group in the early period.
    late_control : array-like
        Primary metric for control group in the late period.
    late_treatment : array-like
        Primary metric for treatment group in the late period.
    alpha : float, optional
        Significance level for the novelty effect test.  Default 0.05.

    Returns
    -------
    dict with keys:
        early_effect    : float — Cohen's d in the early period
        late_effect     : float — Cohen's d in the late period
        effect_decay    : float — early_effect - late_effect
        novelty_detected: bool  — True if early > late significantly
        p_value         : float — p-value of early vs. late difference
        risk_level      : str   — 'low', 'moderate', 'high'
        message         : str   — human-readable interpretation

    Examples
    --------
    >>> import numpy as np
    >>> from abaudit.runtime import check_novelty_effect
    >>> rng = np.random.default_rng(42)
    >>> check_novelty_effect(
    ...     early_control=rng.normal(0, 1, 200),
    ...     early_treatment=rng.normal(0.5, 1, 200),   # large early effect
    ...     late_control=rng.normal(0, 1, 200),
    ...     late_treatment=rng.normal(0.1, 1, 200),    # faded late effect
    ... )
    """
    early_control   = np.asarray(early_control,   dtype=float)
    early_treatment = np.asarray(early_treatment, dtype=float)
    late_control    = np.asarray(late_control,    dtype=float)
    late_treatment  = np.asarray(late_treatment,  dtype=float)

    for arr, name in [(early_control,   "early_control"),
                      (early_treatment, "early_treatment"),
                      (late_control,    "late_control"),
                      (late_treatment,  "late_treatment")]:
        if len(arr) < 2:
            raise ValueError(f"{name} must have at least 2 observations.")

    # Effect size in each period
    from abaudit._stats import cohens_d as _cohens_d
    early_d = _cohens_d(early_control, early_treatment)
    late_d  = _cohens_d(late_control,  late_treatment)
    decay   = early_d - late_d

    # Test: are the means in treatment significantly different
    # between early and late? (tests for time trend in treatment)
    _, p_value = _scipy_stats.ttest_ind(early_treatment, late_treatment)
    p_value    = float(p_value)

    novelty_detected = (early_d > late_d) and (p_value < alpha)

    if abs(decay) < 0.1:
        risk = "low"
    elif abs(decay) < 0.3:
        risk = "moderate"
    else:
        risk = "high"

    if not novelty_detected:
        message = (
            f"No significant novelty effect detected. "
            f"Early effect: d = {early_d:.3f}, "
            f"Late effect: d = {late_d:.3f}. "
            "Effect appears stable over time."
        )
    else:
        message = (
            f"Novelty effect detected. "
            f"Early effect: d = {early_d:.3f}, "
            f"Late effect: d = {late_d:.3f} "
            f"(decay: {decay:.3f}). "
            "The effect may fade after the initial novelty wears off. "
            "Wait for the late-period effect to stabilise before deciding."
        )

    return {
        "early_effect":     round(early_d,  4),
        "late_effect":      round(late_d,   4),
        "effect_decay":     round(decay,    4),
        "novelty_detected": novelty_detected,
        "p_value":          round(p_value,  6),
        "risk_level":       risk,
        "message":          message,
    }
