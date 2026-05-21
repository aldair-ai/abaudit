# abaudit

**Statistical Validity Auditor for A/B Tests**

[![Tests](https://github.com/aldair-ai/abaudit/actions/workflows/tests.yml/badge.svg)](https://github.com/aldair-ai/abaudit/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/abaudit)](https://pypi.org/project/abaudit/)
[![Python](https://img.shields.io/pypi/pyversions/abaudit)](https://pypi.org/project/abaudit/)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/aldair-ai/abaudit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A significant p-value answers the wrong question.  
> **abaudit** asks: *given that the result is significant, how likely is it to actually be real?*

---

## Why abaudit?

Every A/B testing tool tells you **whether** your result is significant.  
None of them tell you **whether to trust it**.

A p-value is P(data | no effect) — the probability of seeing your data if there's no effect.  
What you actually want is P(true effect | significant result) — the **Positive Predictive Value (PPV)**.

These are not the same thing. With a low prior, multiple metrics tested, and a few interim peeks, a p = 0.03 result might only have a **20% chance of being real**. Standard tools report it as significant and move on. `abaudit` doesn't.

The math comes from Ioannidis (2005):

$$\text{PPV} = \frac{(1-\beta) \cdot f}{(1-\beta) \cdot f + \alpha \cdot (1-f)}$$

Where $f$ is your prior probability the effect exists, $1-\beta$ is power, and $\alpha$ is your significance threshold. This is Bayes' rule applied to hypothesis testing — and it's what p-values completely ignore.

---

## Quickstart

```bash
pip install abaudit
```

```python
import numpy as np
import abaudit as ab

rng  = np.random.default_rng(42)
ctrl = rng.normal(0.0, 1.0, 500)
trt  = rng.normal(0.3, 1.0, 500)

result = ab.audit(
    ctrl, trt,
    prior_f = 0.2,
    metrics = ['conversion', 'revenue', 'session_time'],
    n_peeks = 5,
)

result.summary()
```

```
         abaudit — Experiment Validity Report
┌──────────────────────────────────┬─────────────────┬────────┐
│ Check                            │ Result          │ Status │
├──────────────────────────────────┼─────────────────┼────────┤
│ p-value (primary)                │ 0.0000          │ ✅     │
│ p-value (Bonferroni corrected)   │ 0.0001          │ ✅     │
│ PPV — prob. effect is real       │ 0.83            │ ✅     │
│ Statistical power                │ 0.99            │ ✅     │
│ Sample Ratio Mismatch            │ p = 1.000       │ ✅     │
│ Metrics tested                   │ 3               │ ⚠️     │
│ Optional stopping (peeks)        │ eff. α = 0.226  │ ⚠️     │
│ Effect size (Cohen's d)          │ 0.271           │ ✅     │
└──────────────────────────────────┴─────────────────┴────────┘

Bias score: [███░░░░░░░░░░░░░░░░░] 0.15 / 1.0  🟢 Low concern

⚠️  Warnings:
   • 3 metrics tested — Bonferroni-corrected p = 0.0001 (raw p = 0.0000).
   • Optional stopping risk: p-value checked 5 times. Effective α ≈ 0.226 (nominal: 0.05).

💡 Recommendations:
   • Use sequential testing (SPRT) or an alpha-spending function
     when interim looks are necessary.
```

```python
# Save a shareable HTML report
ab.generate_report(result, path="audit_report.html")

# Pre-experiment: is this worth running?
plan = ab.design_summary(effect_size=0.3, prior_f=0.2)
plan.summary()

# During-experiment: health checks
ab.check_srm(n_control=4850, n_treatment=5150)
ab.check_optional_stopping([0.12, 0.08, 0.04, 0.06, 0.03])
```

---

## What abaudit checks

| Module | Check | Answers |
|--------|-------|---------|
| `validity` | **PPV** (Ioannidis 2005) | Given the significant result, what's the probability it's real? |
| `validity` | **Multiple metric correction** | You tested 3 things — what's the Bonferroni-corrected p? |
| `validity` | **Effect size plausibility** | Is the reported effect suspiciously large (winner's curse)? |
| `validity` | **Statistical power** | Was the study large enough to detect the effect reliably? |
| `runtime`  | **Sample Ratio Mismatch** | Was traffic split as intended? |
| `runtime`  | **Optional stopping** | Was the p-value checked multiple times during collection? |
| `runtime`  | **Novelty effect** | Did the effect fade after the initial novelty wore off? |
| `design`   | **PPV-aware power analysis** | How large does n need to be so results are actually trustworthy? |
| `report`   | **HTML report** | Self-contained report for sharing with stakeholders |

---

## What abaudit gives you that standard tools don't

| Standard tool | abaudit |
|--------------|---------|
| Reports p-value | Reports p-value **and PPV** |
| Ignores your prior | Uses Ioannidis PPV framework |
| Ignores multiple metrics | Applies Bonferroni correction automatically |
| Ignores peeking | Diagnoses optional stopping and inflated α |
| Ignores traffic split | Runs Sample Ratio Mismatch test |
| No composite score | Bias score 0–1 with breakdown |
| No HTML output | Self-contained shareable report |

---

## Full API

```python
import abaudit as ab

# ── Post-experiment audit ─────────────────────────────────────
result = ab.audit(
    control        = ctrl,          # array-like, control group
    treatment      = trt,           # array-like, treatment group
    prior_f        = 0.2,           # prior probability effect is real
    alpha          = 0.05,          # significance threshold
    metrics        = ['conversion', 'revenue'],  # all metrics tested
    primary        = 'conversion',  # the one being reported
    n_peeks        = 3,             # number of interim looks
    expected_split = 0.5,           # intended traffic split
)
result.summary()                    # traffic-light table
result.ppv                          # float: prob. effect is real
result.bias_score                   # float 0–1: composite red flags
result.flags                        # list[str]: warnings
ab.generate_report(result, "report.html")

# ── Pre-experiment planning ───────────────────────────────────
plan = ab.design_summary(
    effect_size  = 0.3,             # expected Cohen's d
    prior_f      = 0.2,             # prior probability
    target_power = 0.80,
    target_ppv   = 0.80,
)
plan.summary()
plan.n_recommended                  # n per group to achieve both targets

ab.power_analysis(effect_size=0.3)
ab.ppv_given_design(effect_size=0.3, n_per_group=176, prior_f=0.2)
ab.minimum_trustworthy_n(effect_size=0.3, prior_f=0.2, target_ppv=0.80)

# ── During-experiment checks ──────────────────────────────────
ab.check_srm(n_control=4850, n_treatment=5150)
ab.check_optional_stopping(p_value_history=[0.12, 0.08, 0.04])
ab.check_novelty_effect(
    early_control, early_treatment,
    late_control,  late_treatment,
)
```

---

## Demo notebook

See [`examples/demo.ipynb`](./examples/demo.ipynb) for a complete end-to-end walkthrough:
a realistic e-commerce A/B test from experiment design to HTML audit report,
with visualizations of PPV vs. prior, peeking inflation, and the bias score breakdown.

---

## Statistical foundation

- Ioannidis, J.P.A. (2005). *Why Most Published Research Findings Are False.* PLOS Medicine 2(8): e124.
- Simmons, J.P., Nelson, L.D., & Simonsohn, U. (2011). *False-positive psychology.* Psychological Science 22(11).
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments.* Cambridge University Press.
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the false discovery rate.* JRSS-B 57(1).

---

## Development

```bash
git clone https://github.com/aldair-ai/abaudit.git
cd abaudit
pip install -e ".[dev]"
pytest tests/ -v
```

| Phase | Module | Tests | Status |
|-------|--------|-------|--------|
| 0 | Scaffold + `_stats.py` | 27 | ✅ Complete |
| 1 | `validity.py` — core audit | 42 | ✅ Complete |
| 2 | `design.py` — pre-experiment | 35 | ✅ Complete |
| 3 | `runtime.py` — health checks | 35 | ✅ Complete |
| 4 | `report.py` — HTML reports | 11 | ✅ Complete |

**Total: 184 tests · 99% coverage · Python 3.9 – 3.12**

---

## License

MIT © [Edwin Aldair Espinoza Zegarra](https://github.com/aldair-ai)
