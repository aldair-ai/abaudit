"""Smoke tests — verify the package imports and version is set."""

import abaudit


def test_version_exists():
    assert hasattr(abaudit, "__version__")


def test_version_is_string():
    assert isinstance(abaudit.__version__, str)


def test_version_format():
    """Version must follow semver X.Y.Z."""
    parts = abaudit.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
