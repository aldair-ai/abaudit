"""
abaudit._stats
==============
Internal statistical helpers.  All functions here are **pure** — they
take numbers in and return numbers out, with no side effects, no I/O,
and no dependency on any other abaudit module.

Do NOT import this module directly in user-facing code.  Use the public
API in validity.py / design.py / runtime.py instead.

Every function has:
  - A NumPy-style docstring
  - Type hints
  - A clear formula reference where relevant
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _scipy_stats
from typing import Tuple


# ── PPV / Ioannidis framework ────────────────────────────────────────────────

def ppv(prior_f: float, power: float, alpha: float) -> float:
    """Positive Predictive Value of a statistically significant result.

    Implements the core formula from Ioannidis (2005):
    PPV = (power × f) / (power × f + α × (1 − f))

    This is the probability that the effect is real *given* that the
    test returned a significant result — the quantity that p-values
    alone cannot tell you.

    Parameters
    ----------
    prior_f : float
        Prior probability that a true effect exists (0 < f < 1).
    power : float
        Statistical power of the test, i.e. 1 − β (0 < power ≤ 1).
    alpha : float
        Significance threshold, e.g. 0.05.

    Returns
    -------
    float
        PPV ∈ (0, 1).

    Examples
    --------
    >>> ppv(prior_f=0.1, power=0.8, alpha=0.05)
    0.64
    """
    if not (0 < prior_f < 1):
        raise ValueError("prior_f must be strictly between 0 and 1.")
    if not (0 < power <= 1):
        raise ValueError("power must be in (0, 1].")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")

    numerator   = power * prior_f
    denominator = power * prior_f + alpha * (1.0 - prior_f)
    return numerator / denominator


def ppv_biased(
    prior_f: float,
    power: float,
    alpha: float,
    bias_u: float,
) -> float:
    """PPV under researcher bias u (Ioannidis 2005, extended model).

    bias_u is the fraction of non-significant results that get reported
    as significant anyway (p-hacking, selective reporting, HARKing).
    When bias_u = 0 this reduces to the standard ppv().

    Parameters
    ----------
    prior_f : float
        Prior probability that a true effect exists.
    power : float
        Statistical power (1 − β).
    alpha : float
        Significance threshold.
    bias_u : float
        Bias parameter ∈ [0, 1].  bias_u = 0 → no bias.

    Returns
    -------
    float
        Biased PPV ∈ (0, 1).
    """
    if not (0 <= bias_u <= 1):
        raise ValueError("bias_u must be in [0, 1].")

    p_finding_given_effect    = power   + bias_u * (1.0 - power)
    p_finding_given_no_effect = alpha   + bias_u * (1.0 - alpha)

    numerator   = p_finding_given_effect    * prior_f
    denominator = (p_finding_given_effect    * prior_f
                   + p_finding_given_no_effect * (1.0 - prior_f))
    return numerator / denominator


# ── Multiple testing ─────────────────────────────────────────────────────────

def fwer(alpha: float, m: int) -> float:
    """Family-Wise Error Rate for m independent tests.

    FWER = 1 − (1 − α)^m

    Parameters
    ----------
    alpha : float
        Per-test significance level.
    m : int
        Number of independent tests.

    Returns
    -------
    float
        FWER ∈ (0, 1).
    """
    if m < 1:
        raise ValueError("m must be >= 1.")
    return 1.0 - (1.0 - alpha) ** m


def bonferroni_threshold(alpha: float, m: int) -> float:
    """Bonferroni-corrected significance threshold.

    Parameters
    ----------
    alpha : float
        Desired family-wise error rate.
    m : int
        Number of tests.

    Returns
    -------
    float
        Per-test threshold α / m.
    """
    return alpha / m


def bh_adjusted(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR control).

    Parameters
    ----------
    p_values : np.ndarray
        Array of raw p-values.
    alpha : float
        Desired FDR level.

    Returns
    -------
    np.ndarray
        BH-adjusted p-values, same order as input.
    """
    m = len(p_values)
    order     = np.argsort(p_values)
    ranks     = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    adjusted = np.minimum(1.0, p_values * m / ranks)
    # Enforce monotonicity (cumulative minimum from the right)
    adjusted_sorted = adjusted[order]
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = adjusted_sorted
    return adjusted


# ── Sample Ratio Mismatch ────────────────────────────────────────────────────

def srm_test(
    n_control: int,
    n_treatment: int,
    expected_split: float = 0.5,
) -> Tuple[float, float]:
    """Sample Ratio Mismatch chi-squared test.

    Tests whether the observed traffic split matches the intended split.
    A significant result (p < 0.01 is conventional) flags a possible
    implementation bug — bot traffic, logging errors, cookie deletion, etc.

    Parameters
    ----------
    n_control : int
        Number of units in the control group.
    n_treatment : int
        Number of units in the treatment group.
    expected_split : float
        Expected fraction of traffic assigned to treatment (default 0.5).

    Returns
    -------
    (chi2_stat, p_value) : Tuple[float, float]
    """
    n_total   = n_control + n_treatment
    exp_treat = n_total * expected_split
    exp_ctrl  = n_total * (1.0 - expected_split)
    chi2, p   = _scipy_stats.chisquare(
        f_obs=[n_control, n_treatment],
        f_exp=[exp_ctrl,  exp_treat],
    )
    return float(chi2), float(p)


# ── Optional stopping ────────────────────────────────────────────────────────

def optional_stopping_effective_alpha(
    alpha: float,
    n_peeks: int,
) -> float:
    """Effective false-positive rate after n_peeks interim looks.

    Approximation: each peek is an independent Bernoulli trial at level α,
    so the effective FWER is 1 − (1 − α)^n_peeks.  This is a conservative
    upper bound; exact correction requires alpha-spending functions.

    Parameters
    ----------
    alpha : float
        Nominal per-test significance level.
    n_peeks : int
        Number of times the p-value was checked during data collection.

    Returns
    -------
    float
        Effective α (inflated).
    """
    return fwer(alpha, n_peeks)


# ── Effect size ──────────────────────────────────────────────────────────────

def cohens_d(
    control: np.ndarray,
    treatment: np.ndarray,
) -> float:
    """Cohen's d effect size (pooled SD).

    Parameters
    ----------
    control : np.ndarray
    treatment : np.ndarray

    Returns
    -------
    float
        Signed Cohen's d.  Positive means treatment > control.
    """
    n_c, n_t   = len(control), len(treatment)
    mean_diff  = treatment.mean() - control.mean()
    pooled_var = (
        (n_c - 1) * control.var(ddof=1)
        + (n_t - 1) * treatment.var(ddof=1)
    ) / (n_c + n_t - 2)
    return float(mean_diff / np.sqrt(pooled_var))


def estimate_power(
    effect_size: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> float:
    """Estimate achieved power for a two-sample t-test.

    Parameters
    ----------
    effect_size : float
        Cohen's d.
    n_control : int
    n_treatment : int
    alpha : float

    Returns
    -------
    float
        Estimated power ∈ (0, 1).
    """
    from scipy.stats import norm
    n_harm    = 2 / (1 / n_control + 1 / n_treatment)   # harmonic mean n
    se        = np.sqrt(2 / n_harm)
    z_alpha   = norm.ppf(1 - alpha / 2)
    z_power   = abs(effect_size) / se - z_alpha
    return float(norm.cdf(z_power))


# ── Data fabrication detection ───────────────────────────────────────────────

def benford_test(values: np.ndarray) -> Tuple[float, float]:
    """Chi-squared test of first significant digits against Benford's Law.

    Benford's Law: P(d) = log10(1 + 1/d) for d in {1,...,9}.
    Applies to data spanning several orders of magnitude.

    Parameters
    ----------
    values : np.ndarray
        Array of positive numbers.

    Returns
    -------
    (chi2_stat, p_value) : Tuple[float, float]
        High chi2 / low p-value → deviation from Benford → suspicious.
    """
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) < 30:
        raise ValueError("Need at least 30 values for a meaningful Benford test.")

    first_digits = np.array([int(str(abs(v)).lstrip('0').replace('.', '')[0])
                              for v in values])
    digits   = np.arange(1, 10)
    expected = np.log10(1 + 1 / digits)

    observed_counts = np.array([(first_digits == d).sum() for d in digits])
    # Normalise expected so it sums exactly to the observed total,
    # avoiding scipy's strict relative-tolerance check.
    expected_counts = expected / expected.sum() * observed_counts.sum()

    chi2, p = _scipy_stats.chisquare(observed_counts, f_exp=expected_counts)
    return float(chi2), float(p)


def last_digit_test(values: np.ndarray) -> Tuple[float, float]:
    """Chi-squared test of last digits for uniformity.

    In genuine precise measurements, last digits should be uniform over
    0–9.  Excess zeros indicate rounding; non-uniformity may indicate
    fabrication.

    Parameters
    ----------
    values : np.ndarray
        Array of numbers (integers or floats that will be cast to int).

    Returns
    -------
    (chi2_stat, p_value) : Tuple[float, float]
    """
    ints         = np.abs(values).astype(int)
    last_digits  = ints % 10
    counts       = np.array([(last_digits == d).sum() for d in range(10)])
    chi2, p      = _scipy_stats.chisquare(counts)
    return float(chi2), float(p)
