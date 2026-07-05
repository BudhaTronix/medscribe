"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest

from app.config import reset_settings_cache


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Ensure environment changes are visible to settings tests."""
    reset_settings_cache()
    yield
    reset_settings_cache()
