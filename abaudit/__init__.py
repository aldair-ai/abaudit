"""
abaudit — Statistical Validity Auditor for A/B Tests
=====================================================

Because a significant p-value answers the wrong question.
abaudit asks: *given that the result is significant, how likely
is it to be real?*

Quick start
-----------
>>> import abaudit as ab
>>> result = ab.audit(control=ctrl, treatment=trt, prior_f=0.2)
>>> result.summary()

Modules
-------
- abaudit.validity   : post-experiment audit (the core)
- abaudit.design     : pre-experiment power & PPV planning
- abaudit.runtime    : during-experiment health checks
- abaudit.report     : HTML report generation
"""

from abaudit._version  import __version__
from abaudit.validity  import audit, AuditResult

# ── Public API ──────────────────────────────────────────────────────────────
# These are the names users get with `import abaudit as ab`.
# Everything else is internal (prefix with _) or accessed via submodule.

# Imported lazily once the modules exist — scaffolded as stubs for now.
# Uncomment each line as you build the corresponding module in Phase 1+.

# from abaudit.validity import audit
# from abaudit.design   import power_analysis, minimum_trustworthy_n
# from abaudit.runtime  import check_srm, check_optional_stopping
# from abaudit.report   import generate_report

__all__ = [
    "__version__",
    "audit",
    "AuditResult",
    # "power_analysis",
    # "minimum_trustworthy_n",
    # "check_srm",
    # "check_optional_stopping",
    # "generate_report",
]
