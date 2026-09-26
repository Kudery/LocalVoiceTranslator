import json
from unittest import mock

import pytest

from conftest import Info, Seg
from localvoicetranslator.config import AUTO_DETECT, NO_TRANSLATION
from localvoicetranslator.pipeline import process
from localvoicetranslator.transcription import Transcriber, TranscriptionSettings

SETTINGS = TranscriptionSettings(device="cpu", compute_type="int8")


def _run(target, source=AUTO_DETECT):
    messages = []
    process("a.ogg", source, target, Transcriber(), SETTINGS,
            "http://localhost:11434", "qwen2.5:7b",
            lambda k, v: messages.append((k, v)))
    return messages


def _ollama(text):
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = json.dumps({"response": text}).encode()
    return mock.patch("urllib.request.urlopen", return_value=resp)


@pytest.mark.parametrize("target", ["English", "Dutch", "Spanish", "Japanese"])
def test_translates_via_ollama(whisper_model, target):
    with _ollama("TRANSLATED") as up:
        m = _run(target, "Romanian")
    assert ("label", f"Translation ({target}):") in m
    assert ("orig", "Buna ziua.") in m
    assert ("trans", "TRANSLATED") in m
    assert ("status", "Detected: ro - translating via Ollama...") in m
    assert m[-1] == ("status", "Done.")
    body = json.loads(up.call_args.args[0].data.decode("utf-8"))
    assert f"into {target}." in body["prompt"]
    kwargs = whisper_model.return_value.transcribe.call_args.kwargs
    assert kwargs["language"] == "ro" and kwargs["task"] == "transcribe"
    assert whisper_model.return_value.transcribe.call_count == 1


def test_auto_detect_passes_no_language(whisper_model):
    with _ollama("x"):
        _run("English")
    assert whisper_model.return_value.transcribe.call_args.kwargs["language"] is None


def test_source_already_in_target_language_skips_ollama(whisper_model):
    whisper_model.return_value.transcribe.return_value = (iter([Seg("Hello there.")]), Info("en"))
    with _ollama("x") as up:
        m = _run("English")
    up.assert_not_called()
    assert ("trans", "Hello there.") in m


def test_empty_transcript_skips_ollama(whisper_model):
    whisper_model.return_value.transcribe.return_value = (iter([]), Info("ro"))
    with _ollama("x") as up:
        m = _run("English")
    up.assert_not_called()
    assert not any(k == "trans" for k, _ in m)


def test_no_translation(whisper_model):
    with _ollama("x") as up:
        m = _run(NO_TRANSLATION, "French")
    up.assert_not_called()
    assert ("label", "Translation:") in m
    assert ("orig", "Buna ziua.") in m
    assert not any(k == "trans" for k, _ in m)


def test_transcription_error_propagates(whisper_model):
    whisper_model.return_value.transcribe.side_effect = RuntimeError("broken")
    with pytest.raises(RuntimeError, match="broken"):
        _run("English")
