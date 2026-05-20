"""
abaudit.report
==============
HTML audit report generation.

Generates a self-contained HTML file from an AuditResult.  The report
is styled, readable without any external dependencies, and suitable for
sharing with non-technical stakeholders.

Main function
-------------
- ``generate_report`` : write an HTML report from an AuditResult

Example
-------
>>> import abaudit as ab
>>> result = ab.audit(control=ctrl, treatment=trt, prior_f=0.2)
>>> ab.generate_report(result, path="audit_report.html")
>>> # Opens in any browser — no server needed
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abaudit.validity import AuditResult

__all__ = ["generate_report"]


# ── HTML template ─────────────────────────────────────────────────────────────
# Self-contained — no external CSS/JS dependencies.
# Uses inline styles so the file can be emailed or shared directly.

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>abaudit — Experiment Validity Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f8fafc;
    color: #1e293b;
    padding: 2rem;
    line-height: 1.6;
  }}
  .container {{ max-width: 860px; margin: 0 auto; }}
  header {{
    background: #1e293b;
    color: #f8fafc;
    padding: 1.5rem 2rem;
    border-radius: 8px 8px 0 0;
  }}
  header h1 {{ font-size: 1.4rem; font-weight: 700; }}
  header p  {{ font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem; }}
  .card {{
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0 0 8px 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
  }}
  h2 {{
    font-size: 1rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th {{
    text-align: left;
    padding: 0.6rem 0.8rem;
    background: #f1f5f9;
    color: #64748b;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  td {{
    padding: 0.65rem 0.8rem;
    border-bottom: 1px solid #f1f5f9;
  }}
  tr:last-child td {{ border-bottom: none; }}
  .icon-ok   {{ color: #16a34a; font-size: 1.1rem; }}
  .icon-warn {{ color: #d97706; font-size: 1.1rem; }}
  .icon-fail {{ color: #dc2626; font-size: 1.1rem; }}
  .badge {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
  }}
  .badge-ok   {{ background: #dcfce7; color: #15803d; }}
  .badge-warn {{ background: #fef9c3; color: #854d0e; }}
  .badge-fail {{ background: #fee2e2; color: #991b1b; }}
  .bias-bar-wrap {{
    background: #e2e8f0;
    border-radius: 9999px;
    height: 12px;
    margin: 0.5rem 0;
    overflow: hidden;
  }}
  .bias-bar {{
    height: 100%;
    border-radius: 9999px;
    transition: width 0.3s;
  }}
  .flag-item {{
    background: #fff7ed;
    border-left: 3px solid #f97316;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    border-radius: 0 4px 4px 0;
    font-size: 0.88rem;
  }}
  .rec-item {{
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    border-radius: 0 4px 4px 0;
    font-size: 0.88rem;
  }}
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
  }}
  .stat-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    text-align: center;
  }}
  .stat-box .val {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
  }}
  .stat-box .lbl {{
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
  }}
  footer {{
    text-align: center;
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 1.5rem;
  }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>abaudit — Experiment Validity Report</h1>
  <p>Generated {timestamp} &nbsp;·&nbsp; abaudit {version}</p>
</header>

<div class="card">
  <h2>Key Numbers</h2>
  <div class="stat-grid">
    <div class="stat-box">
      <div class="val">{p_value}</div>
      <div class="lbl">p-value (primary)</div>
    </div>
    <div class="stat-box">
      <div class="val">{ppv}</div>
      <div class="lbl">PPV — prob. effect is real</div>
    </div>
    <div class="stat-box">
      <div class="val">{power}</div>
      <div class="lbl">Statistical power</div>
    </div>
    <div class="stat-box">
      <div class="val">{effect_size}</div>
      <div class="lbl">Cohen's d</div>
    </div>
    <div class="stat-box">
      <div class="val">{n_total:,}</div>
      <div class="lbl">Total sample size</div>
    </div>
    <div class="stat-box">
      <div class="val">{n_metrics}</div>
      <div class="lbl">Metrics tested</div>
    </div>
  </div>

  <p style="margin-top:0.5rem;">
    <strong>Bias score: {bias_score_pct}</strong> &nbsp;
    <span class="badge {bias_badge_class}">{bias_level}</span>
  </p>
  <div class="bias-bar-wrap">
    <div class="bias-bar" style="width:{bias_pct}%; background:{bias_color};"></div>
  </div>
</div>

<div class="card">
  <h2>Validity Checks</h2>
  <table>
    <thead>
      <tr>
        <th>Check</th>
        <th>Result</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
{check_rows}
    </tbody>
  </table>
</div>

{flags_section}

{recs_section}

<div class="card">
  <h2>Parameters Used</h2>
  <table>
    <tbody>
      <tr><td><strong>Prior f</strong></td><td>{prior_f}</td></tr>
      <tr><td><strong>Alpha (α)</strong></td><td>{alpha}</td></tr>
      <tr><td><strong>n control</strong></td><td>{n_control:,}</td></tr>
      <tr><td><strong>n treatment</strong></td><td>{n_treatment:,}</td></tr>
      <tr><td><strong>Corrected p-value</strong></td><td>{p_value_corrected}</td></tr>
      <tr><td><strong>Effective alpha</strong></td><td>{effective_alpha}</td></tr>
    </tbody>
  </table>
</div>

<footer>
  Generated by <strong>abaudit</strong> {version} &nbsp;·&nbsp;
  <a href="https://github.com/aldair-ai/abaudit">github.com/aldair-ai/abaudit</a>
</footer>

</div>
</body>
</html>
"""

_CHECK_ROW = """      <tr>
        <td>{label}</td>
        <td>{value}</td>
        <td><span class="{icon_class}">{icon}</span></td>
      </tr>"""


# ── generate_report ───────────────────────────────────────────────────────────

def generate_report(
    result: "AuditResult",
    path:   str = "audit_report.html",
) -> str:
    """Generate a self-contained HTML validity report from an AuditResult.

    Parameters
    ----------
    result : AuditResult
        The result returned by ``abaudit.audit()``.
    path : str, optional
        File path to write the HTML report.  Default: 'audit_report.html'.

    Returns
    -------
    str
        Absolute path of the written file.

    Examples
    --------
    >>> import abaudit as ab
    >>> result = ab.audit(ctrl, trt, prior_f=0.2)
    >>> ab.generate_report(result, path="report.html")
    '/home/user/project/report.html'
    """
    from abaudit._version import __version__

    # ── Build check rows ──────────────────────────────────────────────────────
    def _row(label: str, value: str, ok: bool, warn: bool = False) -> str:
        if ok:
            icon, cls = "✔", "icon-ok"
        elif warn:
            icon, cls = "⚠", "icon-warn"
        else:
            icon, cls = "✘", "icon-fail"
        return _CHECK_ROW.format(label=label, value=value,
                                  icon=icon, icon_class=cls)

    sig_raw       = result.p_value           < result.alpha
    sig_corrected = result.p_value_corrected < result.alpha
    ppv_ok        = result.ppv >= 0.80
    ppv_warn      = 0.50 <= result.ppv < 0.80
    power_ok      = result.power >= 0.80

    rows = "\n".join([
        _row("p-value (primary)",
             f"{result.p_value:.4f}", sig_raw),
        _row("p-value (Bonferroni corrected)",
             f"{result.p_value_corrected:.4f}",
             sig_corrected,
             warn=sig_raw and not sig_corrected),
        _row("PPV — prob. effect is real",
             f"{result.ppv:.2f}", ppv_ok, warn=ppv_warn),
        _row("Statistical power",
             f"{result.power:.2f}", power_ok, warn=not power_ok),
        _row("Sample Ratio Mismatch",
             f"p = {result.srm_p_value:.3f}", not result.srm_flag),
        _row("Metrics tested",
             str(result.n_metrics),
             result.n_metrics == 1,
             warn=result.n_metrics > 1),
        _row("Optional stopping",
             f"eff. α = {result.effective_alpha:.3f}",
             not result.optional_stopping_flag,
             warn=result.optional_stopping_flag),
        _row("Effect size (Cohen's d)",
             f"{result.effect_size:.3f}",
             abs(result.effect_size) <= 1.5,
             warn=abs(result.effect_size) > 1.5),
    ])

    # ── Bias score visuals ────────────────────────────────────────────────────
    bias_pct = int(result.bias_score * 100)
    if result.bias_score < 0.33:
        bias_color, bias_level, bias_badge = "#16a34a", "Low concern", "badge-ok"
    elif result.bias_score < 0.66:
        bias_color, bias_level, bias_badge = "#d97706", "Moderate concern", "badge-warn"
    else:
        bias_color, bias_level, bias_badge = "#dc2626", "High concern", "badge-fail"

    # ── Flags section ─────────────────────────────────────────────────────────
    if result.flags:
        flag_items = "\n".join(
            f'  <div class="flag-item">⚠ {f}</div>' for f in result.flags
        )
        flags_section = (
            '<div class="card">\n'
            '  <h2>⚠ Warnings</h2>\n'
            f'{flag_items}\n'
            '</div>'
        )
    else:
        flags_section = (
            '<div class="card">\n'
            '  <h2>Warnings</h2>\n'
            '  <p style="color:#16a34a;">✔ No warnings — experiment looks clean.</p>\n'
            '</div>'
        )

    # ── Recommendations section ───────────────────────────────────────────────
    if result.recommendations:
        rec_items = "\n".join(
            f'  <div class="rec-item">💡 {r}</div>' for r in result.recommendations
        )
        recs_section = (
            '<div class="card">\n'
            '  <h2>💡 Recommendations</h2>\n'
            f'{rec_items}\n'
            '</div>'
        )
    else:
        recs_section = ""

    # ── Render ────────────────────────────────────────────────────────────────
    html = _TEMPLATE.format(
        timestamp          = datetime.now().strftime("%Y-%m-%d %H:%M"),
        version            = __version__,
        p_value            = f"{result.p_value:.4f}",
        ppv                = f"{result.ppv:.2f}",
        power              = f"{result.power:.2f}",
        effect_size        = f"{result.effect_size:.3f}",
        n_total            = result.n_control + result.n_treatment,
        n_metrics          = result.n_metrics,
        bias_score_pct     = f"{result.bias_score:.0%}",
        bias_pct           = bias_pct,
        bias_color         = bias_color,
        bias_level         = bias_level,
        bias_badge_class   = bias_badge,
        check_rows         = rows,
        flags_section      = flags_section,
        recs_section       = recs_section,
        prior_f            = result.prior_f,
        alpha              = result.alpha,
        n_control          = result.n_control,
        n_treatment        = result.n_treatment,
        p_value_corrected  = f"{result.p_value_corrected:.4f}",
        effective_alpha    = f"{result.effective_alpha:.4f}",
    )

    abs_path = os.path.abspath(path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✓ Report saved: {abs_path}")
    return abs_path
