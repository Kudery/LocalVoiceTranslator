import sys
import types
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Seg:
    def __init__(self, text):
        self.text = text


class Info:
    def __init__(self, language):
        self.language = language


@pytest.fixture
def whisper_model(monkeypatch):
    """Replace faster_whisper.WhisperModel with a MagicMock (no real import, no GPU)."""
    cls = mock.MagicMock(name="WhisperModel")
    cls.return_value.transcribe.return_value = (iter([Seg(" Buna "), Seg("ziua.")]), Info("ro"))
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = cls
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    return cls
