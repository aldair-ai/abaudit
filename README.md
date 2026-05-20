# abaudit

**Statistical Validity Auditor for A/B Tests**

[![Tests](https://github.com/aldair-ai/abaudit/actions/workflows/tests.yml/badge.svg)](https://github.com/aldair-ai/abaudit/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/abaudit)](https://pypi.org/project/abaudit/)
[![Python](https://img.shields.io/pypi/pyversions/abaudit)](https://pypi.org/project/abaudit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A significant p-value answers the wrong question.  
> **abaudit** asks: *given that the result is significant, how likely is it to actually be real?*

Most A/B testing tools tell you **whether** your result is significant.  
`abaudit` tells you **whether you should trust it**.

---

## The Problem

Your A/B test returned p = 0.031. The team is ready to ship. But:

- You tested 8 metrics and reported the best one
- Someone peeked at the results on Day 3 and almost stopped the test
- The traffic split is 52/48 instead of 50/50
- Your prior belief that this variant would work was maybe 15%

Given all of that, what is the actual probability this effect is real?  
`abaudit` computes that number.

---

## Quickstart

```bash
pip install abaudit
```

```python
import abaudit as ab

result = ab.audit(
    control=control_data,
    treatment=treatment_data,
    metrics=['conversion', 'revenue', 'time_on_site'],
    primary='conversion',
    prior_f=0.2,              # your belief the effect exists
    alpha=0.05,
    peeking_log=p_value_history,
)

result.summary()
# ┌─────────────────────────────┬────────┬────────┐
# │ Check                       │ Result │ Status │
# ├─────────────────────────────┼────────┼────────┤
# │ p-value                     │ 0.031  │  ✅    │
# │ PPV (prob. effect is real)  │ 0.41   │  ⚠️    │
# │ Sample Ratio Mismatch       │ 0.892  │  ✅    │
# │ Multiple metrics correction │ 0.093  │  ❌    │
# │ Optional stopping           │ 3 peeks│  ⚠️    │
# │ Effect size plausibility    │ d=0.8  │  ⚠️    │
# └─────────────────────────────┴────────┴────────┘
# Bias score: 0.42 / 1.0  ⚠️  Moderate concern

result.report("audit_report.html")   # full HTML report
result.ppv                            # 0.41
result.bias_score                     # 0.42
result.flags                          # list of warnings
```

---

## What abaudit Checks

| Module | Check | Answers |
|--------|-------|---------|
| `validity` | **PPV** (Ioannidis 2005) | Given the significant result, what's the probability it's real? |
| `validity` | **Multiple metric correction** | You tested 8 things — what's the corrected p-value for the best one? |
| `validity` | **Effect size plausibility** | Is the reported effect size realistic or suspiciously large? |
| `validity` | **Benford's Law** | Do the summary statistics look fabricated? |
| `runtime`  | **Sample Ratio Mismatch** | Was traffic split as intended? |
| `runtime`  | **Optional stopping** | Was the test stopped early after peeking? |
| `design`   | **PPV-aware power analysis** | Given your prior, how large does n need to be for results to be trustworthy? |

---

## Statistical Foundation

The core of `abaudit` is the **Positive Predictive Value** framework from:

> Ioannidis, J.P.A. (2005). *Why Most Published Research Findings Are False.*  
> PLOS Medicine 2(8): e124.

$$\text{PPV} = \frac{(1-\beta) \cdot f}{(1-\beta) \cdot f + \alpha \cdot (1-f)}$$

Where $f$ is your prior probability that the effect exists, $1-\beta$ is your test's power, and $\alpha$ is the significance threshold. This is exactly Bayes' rule applied to hypothesis testing.

---

## Development Status

| Phase | Module | Status |
|-------|--------|--------|
| 0 | Scaffold + `_stats.py` | ✅ Complete |
| 1 | `validity.py` — core audit | ✅ Complete |
| 2 | `design.py` — pre-experiment | ✅ Complete |
| 3 | `runtime.py` — health checks | ✅ Complete |
| 4 | `report.py` — HTML reports | ✅ Complete |

---

## Contributing

```bash
git clone https://github.com/aldair-ai/abaudit.git
cd abaudit
pip install -e ".[dev]"
pytest
```

---

## License

MIT © [Edwin Aldair Espinoza Zegarra](https://github.com/aldair-ai)
