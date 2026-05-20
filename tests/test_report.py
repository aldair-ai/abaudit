"""
Tests for abaudit.report
========================
Covers generate_report() — HTML output, file creation, content checks.

Run with:
    pytest tests/test_report.py -v
"""

import os
import tempfile
import numpy as np
import pytest

import abaudit as ab
from abaudit.report import generate_report


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_result():
    """A realistic AuditResult with some flags triggered."""
    rng  = np.random.default_rng(42)
    ctrl = rng.normal(0.0, 1.0, 500)
    trt  = rng.normal(0.3, 1.0, 500)
    return ab.audit(
        ctrl, trt,
        prior_f=0.2,
        metrics=["conversion", "revenue", "session_time"],
        n_peeks=5,
    )


@pytest.fixture
def clean_result():
    """A clean experiment — no flags."""
    rng  = np.random.default_rng(0)
    ctrl = rng.normal(0.0, 1.0, 2000)
    trt  = rng.normal(0.3, 1.0, 2000)
    return ab.audit(ctrl, trt, prior_f=0.5)


# ── File creation ─────────────────────────────────────────────────────────────

class TestFileCreation:
    def test_creates_file(self, sample_result, tmp_path):
        path = str(tmp_path / "report.html")
        generate_report(sample_result, path=path)
        assert os.path.exists(path)

    def test_returns_absolute_path(self, sample_result, tmp_path):
        path = str(tmp_path / "report.html")
        returned = generate_report(sample_result, path=path)
        assert os.path.isabs(returned)

    def test_default_filename(self, sample_result, tmp_path, monkeypatch):
        """Default path is 'audit_report.html' in cwd."""
        monkeypatch.chdir(tmp_path)
        generate_report(sample_result)
        assert os.path.exists(tmp_path / "audit_report.html")

    def test_importable_from_top_level(self, sample_result, tmp_path):
        path = str(tmp_path / "report.html")
        ab.generate_report(sample_result, path=path)
        assert os.path.exists(path)


# ── HTML content ──────────────────────────────────────────────────────────────

class TestHTMLContent:
    def _read(self, result, tmp_path):
        path = str(tmp_path / "r.html")
        generate_report(result, path=path)
        return open(path, encoding="utf-8").read()

    def test_is_valid_html(self, sample_result, tmp_path):
        html = self._read(sample_result, tmp_path)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_abaudit_title(self, sample_result, tmp_path):
        html = self._read(sample_result, tmp_path)
        assert "abaudit" in html

    def test_contains_ppv(self, sample_result, tmp_path):
        html = self._read(sample_result, tmp_path)
        assert "PPV" in html

    def test_contains_version(self, sample_result, tmp_path):
        html = self._read(sample_result, tmp_path)
        from abaudit._version import __version__
        assert __version__ in html

    def test_contains_p_value(self, sample_result, tmp_path):
        html = self._read(sample_result, tmp_path)
        assert f"{sample_result.p_value:.4f}" in html

    def test_warnings_section_present_when_flags(self, sample_result, tmp_path):
        html = self._read(sample_result, tmp_path)
        assert "Warnings" in html

    def test_no_warnings_message_when_clean(self, clean_result, tmp_path):
        html = self._read(clean_result, tmp_path)
        assert "No warnings" in html

    def test_recommendations_present_when_applicable(self, sample_result, tmp_path):
        html = self._read(sample_result, tmp_path)
        if sample_result.recommendations:
            assert "Recommendations" in html

    def test_bias_score_present(self, sample_result, tmp_path):
        html = self._read(sample_result, tmp_path)
        assert "Bias score" in html

    def test_file_not_empty(self, sample_result, tmp_path):
        path = str(tmp_path / "r.html")
        generate_report(sample_result, path=path)
        assert os.path.getsize(path) > 1000   # at least 1KB
