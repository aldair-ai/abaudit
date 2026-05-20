"""
abaudit — Statistical Validity Auditor for A/B Tests
=====================================================

Because a significant p-value answers the wrong question.
abaudit asks: given that the result is significant, how likely
is it to be real?

Quick start
-----------
>>> import abaudit as ab

>>> # Post-experiment audit
>>> result = ab.audit(control=ctrl, treatment=trt, prior_f=0.2)
>>> result.summary()
>>> ab.generate_report(result, "report.html")

>>> # Pre-experiment design
>>> plan = ab.design_summary(effect_size=0.3, prior_f=0.2)
>>> plan.summary()

>>> # During-experiment checks
>>> ab.check_srm(n_control=4850, n_treatment=5150)
>>> ab.check_optional_stopping([0.12, 0.08, 0.04, 0.06, 0.03])
"""

from abaudit._version import __version__

# Post-experiment
from abaudit.validity import audit, AuditResult

# Pre-experiment
from abaudit.design import (
    power_analysis,
    ppv_given_design,
    minimum_trustworthy_n,
    design_summary,
    DesignResult,
)

# During-experiment
from abaudit.runtime import (
    check_srm,
    check_optional_stopping,
    check_novelty_effect,
)

# Reporting
from abaudit.report import generate_report

__all__ = [
    "__version__",
    # Post-experiment
    "audit",
    "AuditResult",
    # Pre-experiment
    "power_analysis",
    "ppv_given_design",
    "minimum_trustworthy_n",
    "design_summary",
    "DesignResult",
    # During-experiment
    "check_srm",
    "check_optional_stopping",
    "check_novelty_effect",
    # Reporting
    "generate_report",
]
