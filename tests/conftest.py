import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # make tests/helpers.py importable

from drowsiness.config.settings import Settings  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings()
