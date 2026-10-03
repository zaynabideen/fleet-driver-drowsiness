from pathlib import Path

import pytest
from pydantic import ValidationError

from drowsiness.config.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[2]


def test_defaults_are_valid():
    s = Settings()
    assert s.eyes.reopen_ratio > s.eyes.closed_ratio
    assert s.risk.ongoing_closure_s == (1.0, 2.0, 3.0)


def test_yaml_file_matches_code_defaults():
    """configs/default.yaml documents the defaults; it must not drift from the code."""
    assert load_settings(ROOT / "configs" / "default.yaml") == Settings()


def test_overrides_merge_deeply():
    s = load_settings(eyes={"closed_ratio": 0.6})
    assert s.eyes.closed_ratio == 0.6
    assert s.eyes.reopen_ratio == Settings().eyes.reopen_ratio


def test_hysteresis_must_be_consistent():
    with pytest.raises(ValidationError):
        load_settings(eyes={"closed_ratio": 0.8, "reopen_ratio": 0.7})


def test_severity_thresholds_must_be_ordered():
    with pytest.raises(ValidationError):
        load_settings(risk={"perclos": [0.3, 0.2, 0.4]})


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        load_settings(eyes={"closed_ratoi": 0.6})
