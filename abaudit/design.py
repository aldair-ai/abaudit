"""
abaudit.design
==============
Pre-experiment planning — power analysis with the Ioannidis PPV twist.

Standard power calculators tell you the sample size needed to achieve
80% power.  This module goes further: it also tells you whether the
experiment is *worth running* given your prior belief that the effect
exists.  A well-powered study with a low prior can still have a PPV
below 50% — meaning most significant results will be false positives.

Main functions
--------------
- ``power_analysis``          : sample size for a target power
- ``ppv_given_design``        : PPV you will achieve at a given n
- ``minimum_trustworthy_n``   : n needed so PPV >= a target threshold
- ``design_summary``          : full pre-experiment report

Example
-------
>>> import abaudit.design as abd
>>> plan = abd.design_summary(effect_size=0.3, prior_f=0.2)
>>> plan.summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Optional

import numpy as np
from scipy.stats import norm as _norm

from abaudit._stats import ppv as _ppv

__all__ = [
    "power_analysis",
    "ppv_given_design",
    "minimum_trustworthy_n",
    "design_summary",
    "DesignResult",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _power_from_n(
    effect_size: float,
    n_per_group: int,
    alpha: float,
) -> float:
    """Compute power for a two-sample t-test given n per group.

    Uses the normal approximation (valid for n >= 30).

    Parameters
    ----------
    effect_size : float
        Cohen's d (absolute value used).
    n_per_group : int
        Sample size per group (equal groups assumed).
    alpha : float
        Significance level (two-tailed).

    Returns
    -------
    float
        Power ∈ (0, 1).
    """
    z_alpha = _norm.ppf(1 - alpha / 2)
    se      = np.sqrt(2 / n_per_group)
    z_power = abs(effect_size) / se - z_alpha
    return float(_norm.cdf(z_power))


def _n_from_power(
    effect_size: float,
    power: float,
    alpha: float,
) -> int:
    """Compute n per group needed to achieve a target power.

    Parameters
    ----------
    effect_size : float
        Cohen's d (absolute value used).
    power : float
        Target power, e.g. 0.80.
    alpha : float
        Significance level (two-tailed).

    Returns
    -------
    int
        Minimum n per group (rounded up).
    """
    z_alpha = _norm.ppf(1 - alpha / 2)
    z_beta  = _norm.ppf(power)
    n       = 2 * ((z_alpha + z_beta) / abs(effect_size)) ** 2
    return ceil(n)


# ── Public functions ──────────────────────────────────────────────────────────

def power_analysis(
    effect_size: float,
    alpha:  float = 0.05,
    power:  float = 0.80,
) -> dict:
    """Standard power analysis: sample size needed to achieve target power.

    Parameters
    ----------
    effect_size : float
        Expected Cohen's d.  Use 0.2 (small), 0.5 (medium), 0.8 (large)
        as reference points (Cohen, 1988).
    alpha : float, optional
        Significance level.  Default 0.05.
    power : float, optional
        Target power (1 - β).  Default 0.80.

    Returns
    -------
    dict with keys:
        n_per_group   : int   — minimum sample size per group
        n_total       : int   — total sample size (both groups)
        effect_size   : float — Cohen's d used
        power_target  : float — target power
        power_achieved: float — actual power at n_per_group
        alpha         : float — significance level

    Examples
    --------
    >>> from abaudit.design import power_analysis
    >>> power_analysis(effect_size=0.3, power=0.80)
    {'n_per_group': 176, 'n_total': 352, ...}
    """
    if abs(effect_size) < 1e-6:
        raise ValueError("effect_size must be non-zero.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")
    if not (0 < power < 1):
        raise ValueError("power must be strictly between 0 and 1.")

    n_per_group    = _n_from_power(effect_size, power, alpha)
    power_achieved = _power_from_n(effect_size, n_per_group, alpha)

    return {
        "n_per_group":    n_per_group,
        "n_total":        n_per_group * 2,
        "effect_size":    effect_size,
        "power_target":   power,
        "power_achieved": round(power_achieved, 4),
        "alpha":          alpha,
    }


def ppv_given_design(
    effect_size: float,
    n_per_group: int,
    prior_f:     float,
    alpha:       float = 0.05,
) -> float:
    """PPV you will achieve if a significant result is found at this design.

    Combines the power estimate at the given n with the Ioannidis PPV
    formula.  This is the key quantity standard power calculators ignore:
    even at 80% power, a low prior can make PPV < 50%.

    Parameters
    ----------
    effect_size : float
        Expected Cohen's d.
    n_per_group : int
        Sample size per group.
    prior_f : float
        Prior probability the effect is real (0 < f < 1).
    alpha : float, optional
        Significance level.  Default 0.05.

    Returns
    -------
    float
        PPV ∈ (0, 1).

    Examples
    --------
    >>> from abaudit.design import ppv_given_design
    >>> ppv_given_design(effect_size=0.3, n_per_group=176, prior_f=0.1)
    0.64
    """
    if n_per_group < 2:
        raise ValueError("n_per_group must be >= 2.")
    if not (0 < prior_f < 1):
        raise ValueError("prior_f must be strictly between 0 and 1.")

    power = _power_from_n(abs(effect_size), n_per_group, alpha)
    return _ppv(prior_f=prior_f, power=power, alpha=alpha)


def minimum_trustworthy_n(
    effect_size: float,
    prior_f:     float,
    target_ppv:  float = 0.80,
    alpha:       float = 0.05,
    max_n:       int   = 1_000_000,
) -> dict:
    """Find the n per group needed so that PPV >= target_ppv.

    This answers the question: *how large does the experiment need to
    be before a significant result is actually trustworthy?*

    Unlike standard power analysis, this accounts for your prior belief
    via the Ioannidis PPV formula.  With a very low prior, no feasible
    sample size may achieve the target PPV — the function will tell you.

    Parameters
    ----------
    effect_size : float
        Expected Cohen's d.
    prior_f : float
        Prior probability the effect is real.
    target_ppv : float, optional
        Minimum acceptable PPV.  Default 0.80.
    alpha : float, optional
        Significance level.  Default 0.05.
    max_n : int, optional
        Search ceiling.  If no n below max_n achieves target_ppv,
        the result is marked infeasible.  Default 1,000,000.

    Returns
    -------
    dict with keys:
        n_per_group   : int   — minimum n per group to reach target PPV
                                (None if infeasible)
        n_total       : int   — total n (None if infeasible)
        ppv_achieved  : float — PPV at that n
        power_at_n    : float — power at that n
        target_ppv    : float — the requested target
        feasible      : bool  — False if target PPV is unreachable

    Examples
    --------
    >>> from abaudit.design import minimum_trustworthy_n
    >>> minimum_trustworthy_n(effect_size=0.3, prior_f=0.2, target_ppv=0.80)
    {'n_per_group': 176, 'n_total': 352, 'ppv_achieved': 0.82, ...}
    """
    if not (0 < target_ppv < 1):
        raise ValueError("target_ppv must be strictly between 0 and 1.")
    if not (0 < prior_f < 1):
        raise ValueError("prior_f must be strictly between 0 and 1.")

    # Binary search over n
    lo, hi = 2, max_n
    found  = None

    # First check if max_n is even sufficient
    if ppv_given_design(effect_size, max_n, prior_f, alpha) < target_ppv:
        return {
            "n_per_group":  None,
            "n_total":      None,
            "ppv_achieved": ppv_given_design(effect_size, max_n, prior_f, alpha),
            "power_at_n":   _power_from_n(abs(effect_size), max_n, alpha),
            "target_ppv":   target_ppv,
            "feasible":     False,
        }

    while lo <= hi:
        mid = (lo + hi) // 2
        if ppv_given_design(effect_size, mid, prior_f, alpha) >= target_ppv:
            found = mid
            hi    = mid - 1
        else:
            lo = mid + 1

    ppv_at_n   = ppv_given_design(effect_size, found, prior_f, alpha)
    power_at_n = _power_from_n(abs(effect_size), found, alpha)

    return {
        "n_per_group":  found,
        "n_total":      found * 2,
        "ppv_achieved": round(ppv_at_n, 4),
        "power_at_n":   round(power_at_n, 4),
        "target_ppv":   target_ppv,
        "feasible":     True,
    }


# ── DesignResult ──────────────────────────────────────────────────────────────

@dataclass
class DesignResult:
    """Full pre-experiment design report returned by ``design_summary()``.

    Attributes
    ----------
    effect_size : float
        Expected Cohen's d.
    prior_f : float
        Prior probability the effect is real.
    alpha : float
        Significance level.
    target_power : float
        Requested power level.
    n_per_group_power : int
        n per group needed to achieve target_power (standard calculation).
    n_per_group_ppv : int or None
        n per group needed so PPV >= target_ppv.  None if infeasible.
    n_recommended : int or None
        max(n_per_group_power, n_per_group_ppv) — the safe choice.
    power_at_recommended : float
        Power at n_recommended.
    ppv_at_recommended : float
        PPV at n_recommended.
    target_ppv : float
        Requested PPV threshold.
    ppv_feasible : bool
        False if no feasible n can achieve target_ppv given prior_f.
    warnings : list[str]
        Design-stage warnings.
    """

    effect_size:          float
    prior_f:              float
    alpha:                float
    target_power:         float
    target_ppv:           float

    n_per_group_power:    int
    n_per_group_ppv:      Optional[int]
    n_recommended:        Optional[int]

    power_at_recommended: float
    ppv_at_recommended:   float
    ppv_feasible:         bool

    warnings: list[str] = field(default_factory=list)

    def summary(self) -> None:
        """Print a formatted pre-experiment design report."""

        def _icon(ok: bool) -> str:
            return "✅" if ok else "⚠️ "

        print(f"\n{'abaudit — Pre-Experiment Design Report':^60}")
        print("=" * 60)
        print(f"  Effect size (Cohen's d)   : {self.effect_size:.2f}")
        print(f"  Prior f                   : {self.prior_f:.2f}  "
              f"({'{}% chance effect is real'.format(int(self.prior_f*100))})")
        print(f"  Significance level (α)    : {self.alpha}")
        print(f"  Target power              : {self.target_power:.0%}")
        print(f"  Target PPV                : {self.target_ppv:.0%}")
        print("─" * 60)
        print(f"  n per group (power only)  : {self.n_per_group_power:,}")

        if self.ppv_feasible:
            print(f"  n per group (PPV target)  : {self.n_per_group_ppv:,}")
            print(f"  {'─'*30}")
            print(f"  ➤  Recommended n / group  : {self.n_recommended:,}  "
                  f"(total: {self.n_recommended * 2:,})")
            print(f"  ➤  Power at recommended n : {self.power_at_recommended:.1%}  "
                  f"  {_icon(self.power_at_recommended >= self.target_power)}")
            print(f"  ➤  PPV  at recommended n  : {self.ppv_at_recommended:.1%}  "
                  f"  {_icon(self.ppv_at_recommended >= self.target_ppv)}")
        else:
            print(f"  n per group (PPV target)  : NOT FEASIBLE ❌")
            print(f"\n  Even with n = 1,000,000 the PPV cannot reach "
                  f"{self.target_ppv:.0%}")
            print(f"  given a prior of only f = {self.prior_f}.")
            print(f"  Consider increasing the prior (run a pilot study)")
            print(f"  or lowering your target PPV.")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for w in self.warnings:
                print(f"   • {w}")
        print()

    def __repr__(self) -> str:
        return (
            f"DesignResult("
            f"d={self.effect_size}, "
            f"prior_f={self.prior_f}, "
            f"n_recommended={self.n_recommended}, "
            f"ppv={self.ppv_at_recommended:.2f}, "
            f"power={self.power_at_recommended:.2f})"
        )


# ── design_summary ────────────────────────────────────────────────────────────

def design_summary(
    effect_size:  float,
    prior_f:      float,
    alpha:        float = 0.05,
    target_power: float = 0.80,
    target_ppv:   float = 0.80,
) -> DesignResult:
    """Full pre-experiment design report.

    Combines standard power analysis with PPV-aware sample size planning.
    Recommends the larger of the two n estimates so the experiment is both
    well-powered AND produces trustworthy results when significant.

    Parameters
    ----------
    effect_size : float
        Expected Cohen's d for the primary metric.
    prior_f : float
        Prior probability the effect is real (0 < f < 1).
    alpha : float, optional
        Significance level.  Default 0.05.
    target_power : float, optional
        Minimum acceptable power.  Default 0.80.
    target_ppv : float, optional
        Minimum acceptable PPV if the result is significant.  Default 0.80.

    Returns
    -------
    DesignResult
        Call ``.summary()`` for a formatted report.

    Examples
    --------
    >>> from abaudit.design import design_summary
    >>> plan = design_summary(effect_size=0.3, prior_f=0.2)
    >>> plan.summary()
    """
    if not (0 < prior_f < 1):
        raise ValueError("prior_f must be strictly between 0 and 1.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")
    if not (0 < target_power < 1):
        raise ValueError("target_power must be strictly between 0 and 1.")
    if not (0 < target_ppv < 1):
        raise ValueError("target_ppv must be strictly between 0 and 1.")
    if abs(effect_size) < 1e-6:
        raise ValueError("effect_size must be non-zero.")

    warnings: list[str] = []

    # Standard power analysis
    pa             = power_analysis(effect_size, alpha, target_power)
    n_power        = pa["n_per_group"]

    # PPV-aware sample size
    ppv_plan       = minimum_trustworthy_n(effect_size, prior_f,
                                           target_ppv, alpha)
    ppv_feasible   = ppv_plan["feasible"]
    n_ppv          = ppv_plan["n_per_group"]

    # Recommended n: take the larger of the two requirements
    if ppv_feasible:
        n_recommended = max(n_power, n_ppv)
    else:
        n_recommended = None
        warnings.append(
            f"Target PPV of {target_ppv:.0%} is not achievable with prior "
            f"f = {prior_f}. Even n = 1,000,000 is insufficient. "
            "Consider increasing your prior or lowering target_ppv."
        )

    # Stats at recommended n
    if n_recommended is not None:
        power_rec = _power_from_n(abs(effect_size), n_recommended, alpha)
        ppv_rec   = ppv_given_design(effect_size, n_recommended,
                                     prior_f, alpha)
    else:
        power_rec = _power_from_n(abs(effect_size), n_power, alpha)
        ppv_rec   = ppv_given_design(effect_size, n_power, prior_f, alpha)

    # Additional design warnings
    if abs(effect_size) < 0.1:
        warnings.append(
            f"Very small effect size (d = {effect_size:.2f}). "
            "Are you sure this effect is practically meaningful?"
        )
    if prior_f < 0.05:
        warnings.append(
            f"Very low prior (f = {prior_f}). "
            "Most significant results will be false positives "
            "regardless of sample size."
        )
    if n_recommended and n_recommended > 100_000:
        warnings.append(
            f"Required n per group ({n_recommended:,}) is very large. "
            "Consider whether the experiment is practically feasible."
        )

    return DesignResult(
        effect_size          = effect_size,
        prior_f              = prior_f,
        alpha                = alpha,
        target_power         = target_power,
        target_ppv           = target_ppv,
        n_per_group_power    = n_power,
        n_per_group_ppv      = n_ppv,
        n_recommended        = n_recommended,
        power_at_recommended = round(power_rec, 4),
        ppv_at_recommended   = round(ppv_rec, 4),
        ppv_feasible         = ppv_feasible,
        warnings             = warnings,
    )
