"""
abaudit.validity
================
Post-experiment audit — the core of abaudit.

The main entry point is ``audit()``.  It takes control and treatment
data, runs every validity check, and returns an ``AuditResult`` that
summarises what was found.

Example
-------
>>> import abaudit as ab
>>> result = ab.audit(
...     control=ctrl,
...     treatment=trt,
...     metrics=['conversion', 'revenue'],
...     primary='conversion',
...     prior_f=0.2,
... )
>>> result.summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats as _scipy_stats

from abaudit._stats import (
    ppv                               as _ppv,
    fwer                              as _fwer,
    srm_test                          as _srm_test,
    cohens_d                          as _cohens_d,
    estimate_power                    as _estimate_power,
    optional_stopping_effective_alpha as _opt_stop,
)

__all__ = ["AuditResult", "audit"]


# ── Thresholds ────────────────────────────────────────────────────────────────

_SRM_ALPHA       = 0.01   # conventional threshold for SRM detection
_PPV_WARN        = 0.80   # below this → PPV warning
_PPV_DANGER      = 0.50   # below this → PPV danger
_LARGE_EFFECT_D  = 1.50   # Cohen's d above this is suspiciously large
_MIN_POWER_WARN  = 0.80   # below this → underpowered warning
_OPT_STOP_WARN   = 3      # more peeks than this → optional stopping warning


# ── AuditResult ──────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    """Structured result returned by ``audit()``.

    You never instantiate this directly — ``audit()`` returns it.

    Attributes
    ----------
    p_value : float
        Raw p-value from the primary metric two-sample t-test.
    p_value_corrected : float
        Bonferroni-corrected p-value accounting for all metrics tested.
    effect_size : float
        Cohen's d for the primary metric.
    power : float
        Estimated achieved power given effect size and n.
    ppv : float
        Positive Predictive Value — P(true effect | significant result).
        The core quantity: how likely is this finding to actually be real?
    n_control : int
        Number of units in the control group.
    n_treatment : int
        Number of units in the treatment group.
    n_metrics : int
        Total number of metrics tested in the experiment.
    srm_p_value : float
        p-value from the Sample Ratio Mismatch chi-squared test.
    srm_flag : bool
        True if SRM detected (srm_p_value < 0.01).
    multiple_metrics_flag : bool
        True if significance disappears after Bonferroni correction.
    underpowered_flag : bool
        True if estimated power < 0.80.
    optional_stopping_flag : bool
        True if n_peeks exceeds the threshold.
    effective_alpha : float
        Inflated effective alpha from peeking; equals alpha if n_peeks=1.
    bias_score : float
        Composite score 0–1.  Higher means more red flags triggered.
    flags : list[str]
        Human-readable warnings.
    recommendations : list[str]
        Actionable suggestions based on triggered flags.
    alpha : float
        Significance level used.
    prior_f : float
        Prior probability of a true effect as supplied by the caller.
    """

    # Core statistics
    p_value:           float
    p_value_corrected: float
    effect_size:       float
    power:             float
    ppv:               float

    # Sample info
    n_control:   int
    n_treatment: int
    n_metrics:   int

    # Individual check results
    srm_p_value:           float
    srm_flag:              bool
    multiple_metrics_flag: bool
    underpowered_flag:     bool
    optional_stopping_flag:bool
    effective_alpha:       float

    # Summary
    bias_score:      float
    flags:           list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Parameters used
    alpha:   float = 0.05
    prior_f: float = 0.1

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a traffic-light audit table to stdout."""

        def _icon(ok: bool, warn: bool = False) -> str:
            if ok:
                return "✅"
            return "⚠️ " if warn else "❌"

        sig_raw       = self.p_value           < self.alpha
        sig_corrected = self.p_value_corrected < self.alpha
        ppv_ok        = self.ppv >= _PPV_WARN
        ppv_warn      = _PPV_DANGER <= self.ppv < _PPV_WARN
        power_ok      = self.power >= _MIN_POWER_WARN

        rows = [
            ("p-value (primary)",
             f"{self.p_value:.4f}",
             _icon(sig_raw)),

            ("p-value (Bonferroni corrected)",
             f"{self.p_value_corrected:.4f}",
             _icon(sig_corrected,
                   warn=sig_raw and not sig_corrected)),

            ("PPV — prob. effect is real",
             f"{self.ppv:.2f}",
             _icon(ppv_ok, warn=ppv_warn)),

            ("Statistical power",
             f"{self.power:.2f}",
             _icon(power_ok, warn=not power_ok)),

            ("Sample Ratio Mismatch",
             f"p = {self.srm_p_value:.3f}",
             _icon(not self.srm_flag)),

            ("Metrics tested",
             f"{self.n_metrics}",
             _icon(self.n_metrics == 1, warn=self.n_metrics > 1)),

            ("Optional stopping (peeks)",
             f"eff. α = {self.effective_alpha:.3f}",
             _icon(not self.optional_stopping_flag,
                   warn=self.optional_stopping_flag)),

            ("Effect size (Cohen's d)",
             f"{self.effect_size:.3f}",
             _icon(abs(self.effect_size) <= _LARGE_EFFECT_D,
                   warn=abs(self.effect_size) > _LARGE_EFFECT_D)),
        ]

        c1 = max(len(r[0]) for r in rows) + 2
        c2 = max(len(r[1]) for r in rows) + 2

        header = "abaudit — Experiment Validity Report"
        width  = c1 + c2 + 16
        print(f"\n{header:^{width}}")
        print(f"┌{'─'*(c1+2)}┬{'─'*(c2+2)}┬{'─'*8}┐")
        print(f"│ {'Check':<{c1}}│ {'Result':<{c2}}│ Status │")
        print(f"├{'─'*(c1+2)}┼{'─'*(c2+2)}┼{'─'*8}┤")
        for label, value, icon in rows:
            print(f"│ {label:<{c1}}│ {value:<{c2}}│ {icon:<6} │")
        print(f"└{'─'*(c1+2)}┴{'─'*(c2+2)}┴{'─'*8}┘")

        filled = int(self.bias_score * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        level  = ("🟢 Low concern"      if self.bias_score < 0.33 else
                  "🟡 Moderate concern" if self.bias_score < 0.66 else
                  "🔴 High concern")
        print(f"\nBias score: [{bar}] {self.bias_score:.2f} / 1.0  {level}")

        if self.flags:
            print("\n⚠️  Warnings:")
            for f in self.flags:
                print(f"   • {f}")

        if self.recommendations:
            print("\n💡 Recommendations:")
            for r in self.recommendations:
                print(f"   • {r}")
        print()

    def __repr__(self) -> str:
        return (
            f"AuditResult("
            f"p={self.p_value:.4f}, "
            f"ppv={self.ppv:.2f}, "
            f"power={self.power:.2f}, "
            f"bias_score={self.bias_score:.2f}, "
            f"flags={len(self.flags)})"
        )


# ── audit() ───────────────────────────────────────────────────────────────────

def audit(
    control:        np.ndarray,
    treatment:      np.ndarray,
    prior_f:        float = 0.1,
    alpha:          float = 0.05,
    metrics:        Optional[list[str]] = None,
    primary:        Optional[str] = None,
    n_peeks:        int = 1,
    expected_split: float = 0.5,
) -> AuditResult:
    """Run a full statistical validity audit on an A/B test result.

    Parameters
    ----------
    control : array-like
        Observed metric values for the control group.
    treatment : array-like
        Observed metric values for the treatment group.
    prior_f : float, optional
        Prior probability that a true effect exists (0 < f < 1).
        Default 0.1.  Use higher values for well-motivated hypotheses,
        lower for exploratory fishing.
    alpha : float, optional
        Significance threshold.  Default 0.05.
    metrics : list[str], optional
        Names of ALL metrics tested in this experiment.  Used for the
        Bonferroni correction.  If None, assumes one metric only.
    primary : str, optional
        Name of the primary (reported) metric — display only.
    n_peeks : int, optional
        Number of times the p-value was checked during collection.
        n_peeks=1 means no interim looks (the default safe assumption).
    expected_split : float, optional
        Expected fraction of traffic in treatment.  Default 0.5.

    Returns
    -------
    AuditResult
        Call ``.summary()`` to print the formatted report.

    Raises
    ------
    ValueError
        If inputs are invalid (too few observations, bad prior, etc.).

    Examples
    --------
    >>> import numpy as np, abaudit as ab
    >>> rng  = np.random.default_rng(42)
    >>> ctrl = rng.normal(0, 1, 500)
    >>> trt  = rng.normal(0.3, 1, 500)
    >>> result = ab.audit(ctrl, trt, prior_f=0.2)
    >>> result.summary()
    """
    # ── 0. Validate and coerce inputs ────────────────────────────────────────
    control   = np.asarray(control,   dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    if len(control) < 2 or len(treatment) < 2:
        raise ValueError(
            "Both control and treatment must have at least 2 observations."
        )
    if not (0 < prior_f < 1):
        raise ValueError("prior_f must be strictly between 0 and 1.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")
    if n_peeks < 1:
        raise ValueError("n_peeks must be >= 1.")

    n_control   = len(control)
    n_treatment = len(treatment)
    n_metrics   = len(metrics) if metrics else 1

    flags:           list[str] = []
    recommendations: list[str] = []

    # ── 1. Primary metric test ───────────────────────────────────────────────
    _, p_value = _scipy_stats.ttest_ind(control, treatment)
    p_value    = float(p_value)

    # ── 2. Multiple metrics correction (Bonferroni) ──────────────────────────
    p_value_corrected     = min(p_value * n_metrics, 1.0)
    multiple_metrics_flag = (
        n_metrics > 1
        and p_value < alpha
        and p_value_corrected >= alpha
    )

    if n_metrics > 1:
        flags.append(
            f"{n_metrics} metrics tested — Bonferroni-corrected p = "
            f"{p_value_corrected:.4f} (raw p = {p_value:.4f})."
        )
    if multiple_metrics_flag:
        flags.append(
            "Significance does NOT survive Bonferroni correction. "
            "Result may be a false positive from metric fishing."
        )
        recommendations.append(
            "Pre-specify a single primary metric before running the experiment."
        )

    # ── 3. Effect size ────────────────────────────────────────────────────────
    effect_size = _cohens_d(control, treatment)

    if abs(effect_size) > _LARGE_EFFECT_D:
        flags.append(
            f"Very large effect size (d = {effect_size:.2f}). "
            "Large effects from moderate samples are often inflated "
            "(winner's curse)."
        )
        recommendations.append(
            "Treat large effect sizes sceptically — wait for replication "
            "before acting on them."
        )

    # ── 4. Statistical power ──────────────────────────────────────────────────
    power             = _estimate_power(abs(effect_size), n_control,
                                        n_treatment, alpha)
    underpowered_flag = power < _MIN_POWER_WARN

    if underpowered_flag:
        flags.append(
            f"Low power: estimated power = {power:.2f} "
            f"(threshold: {_MIN_POWER_WARN}). "
            "Underpowered studies produce unreliable, inflated estimates."
        )
        recommendations.append(
            f"Increase sample size. Total n = {n_control + n_treatment:,}. "
            "Run a power analysis before the next experiment."
        )

    # ── 5. PPV (Ioannidis framework) ─────────────────────────────────────────
    ppv_val = _ppv(prior_f=prior_f, power=power, alpha=alpha)

    if ppv_val < _PPV_DANGER:
        flags.append(
            f"Low PPV = {ppv_val:.2f} — less than {ppv_val:.0%} chance "
            f"the effect is real given prior f = {prior_f}."
        )
        recommendations.append(
            "Run a pilot study to increase prior confidence, or replicate "
            "in an independent experiment before rolling out."
        )
    elif ppv_val < _PPV_WARN:
        flags.append(
            f"Moderate PPV = {ppv_val:.2f}. "
            "Consider replicating before a full rollout."
        )

    # ── 6. Sample Ratio Mismatch ──────────────────────────────────────────────
    _, srm_p  = _srm_test(n_control, n_treatment, expected_split)
    srm_flag  = srm_p < _SRM_ALPHA

    if srm_flag:
        flags.append(
            f"Sample Ratio Mismatch detected (SRM p = {srm_p:.4f}). "
            f"Expected {expected_split:.0%}/{1-expected_split:.0%} split, "
            f"observed {n_control:,}/{n_treatment:,}. "
            "Likely cause: logging bug, bot traffic, or cookie issues."
        )
        recommendations.append(
            "Investigate the traffic split before trusting any results. "
            "Fix the root cause and re-run the experiment."
        )

    # ── 7. Optional stopping ──────────────────────────────────────────────────
    effective_alpha         = _opt_stop(alpha, n_peeks)
    optional_stopping_flag  = n_peeks > _OPT_STOP_WARN

    if optional_stopping_flag:
        flags.append(
            f"Optional stopping risk: p-value checked {n_peeks} times. "
            f"Effective α ≈ {effective_alpha:.3f} (nominal: {alpha})."
        )
        recommendations.append(
            "Use sequential testing (SPRT) or an alpha-spending function "
            "when interim looks are necessary."
        )

    # ── 8. Bias score ─────────────────────────────────────────────────────────
    penalties = [
        (0.30, srm_flag),
        (0.25, ppv_val < _PPV_DANGER),
        (0.20, multiple_metrics_flag),
        (0.15, underpowered_flag),
        (0.15, optional_stopping_flag),
        (0.10, _PPV_DANGER <= ppv_val < _PPV_WARN),
        (0.10, abs(effect_size) > _LARGE_EFFECT_D),
    ]
    bias_score = min(1.0, sum(w for w, triggered in penalties if triggered))

    # ── 9. Return ─────────────────────────────────────────────────────────────
    return AuditResult(
        p_value                = p_value,
        p_value_corrected      = p_value_corrected,
        effect_size            = effect_size,
        power                  = power,
        ppv                    = ppv_val,
        n_control              = n_control,
        n_treatment            = n_treatment,
        n_metrics              = n_metrics,
        srm_p_value            = srm_p,
        srm_flag               = srm_flag,
        multiple_metrics_flag  = multiple_metrics_flag,
        underpowered_flag      = underpowered_flag,
        optional_stopping_flag = optional_stopping_flag,
        effective_alpha        = effective_alpha,
        bias_score             = bias_score,
        flags                  = flags,
        recommendations        = recommendations,
        alpha                  = alpha,
        prior_f                = prior_f,
    )
