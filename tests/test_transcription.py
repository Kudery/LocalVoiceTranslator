import os
import sys
import types
from unittest import mock

import pytest

from conftest import Info, Seg
from localvoicetranslator import transcription
from localvoicetranslator.transcription import Transcriber, TranscriptionSettings


def gpu_settings(**kw):
    return TranscriptionSettings(**{"model_size": "large-v3", "device": "cuda",
                                    "compute_type": "float16", **kw})


def test_transcribe_returns_text_and_info(whisper_model):
    text, info = Transcriber().transcribe("a.ogg", gpu_settings(), "ro")
    assert text == "Buna ziua."
    assert info.language == "ro"
    whisper_model.assert_called_once_with("large-v3", device="cuda", compute_type="float16")
    whisper_model.return_value.transcribe.assert_called_once_with(
        "a.ogg", language="ro", task="transcribe", vad_filter=True,
        beam_size=5, condition_on_previous_text=False)


def test_empty_model_name_uses_default(whisper_model):
    Transcriber().transcribe("a.ogg", gpu_settings(model_size="  "), None)
    assert whisper_model.call_args.args[0] == "large-v3-turbo"


def test_model_is_cached(whisper_model):
    t = Transcriber()
    t.transcribe("a.ogg", gpu_settings(), None)
    whisper_model.return_value.transcribe.return_value = (iter([Seg("x")]), Info("ro"))
    t.transcribe("b.ogg", gpu_settings(), None)
    assert whisper_model.call_count == 1


def test_other_model_reloads(whisper_model):
    t = Transcriber()
    t.transcribe("a.ogg", gpu_settings(), None)
    whisper_model.return_value.transcribe.return_value = (iter([]), Info("ro"))
    t.transcribe("a.ogg", gpu_settings(model_size="small"), None)
    assert whisper_model.call_count == 2


def test_model_dir_becomes_download_root(whisper_model, tmp_path):
    target = tmp_path / "models"
    Transcriber().transcribe("a.ogg", gpu_settings(model_dir=str(target)), None)
    assert target.is_dir()
    assert whisper_model.call_args.kwargs["download_root"] == str(target)


def _gpu_fails_cpu_works(whisper_model):
    cpu_model = mock.MagicMock(name="cpu_model")
    cpu_model.transcribe.return_value = (iter([Seg("cpu text")]), Info("ro"))

    def make(size, device, compute_type, **kw):
        if device == "cuda":
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")
        return cpu_model

    whisper_model.side_effect = make
    return cpu_model


def test_gpu_error_on_load_falls_back_to_cpu(whisper_model):
    cpu_model = _gpu_fails_cpu_works(whisper_model)
    messages = []
    t = Transcriber(notify=lambda k, v: messages.append((k, v)))

    text, _ = t.transcribe("a.ogg", gpu_settings(), "ro")

    assert text == "cpu text"
    assert t.forced_cpu
    assert whisper_model.call_args_list[-1] == mock.call("large-v3", device="cpu", compute_type="int8")
    cpu_model.transcribe.assert_called_once()
    assert any(k == "status" and "GPU error (CUBLAS_STATUS_NOT_SUPPORTED)" in v for k, v in messages)


def test_gpu_error_while_decoding_falls_back_to_cpu(whisper_model):
    """Error only raised while iterating the segments generator (typical for CUDA)."""
    def broken_generator():
        raise RuntimeError("cuDNN failed")
        yield  # pragma: no cover

    gpu_model = mock.MagicMock(name="gpu_model")
    gpu_model.transcribe.return_value = (broken_generator(), Info("ro"))
    cpu_model = mock.MagicMock(name="cpu_model")
    cpu_model.transcribe.return_value = (iter([Seg("ok")]), Info("ro"))
    whisper_model.side_effect = lambda size, device, compute_type, **kw: (
        gpu_model if device == "cuda" else cpu_model)

    text, _ = Transcriber().transcribe("a.ogg", gpu_settings(), None)

    assert text == "ok"
    gpu_model.transcribe.assert_called_once()
    cpu_model.transcribe.assert_called_once_with(
        "a.ogg", language=None, task="transcribe", vad_filter=True,
        beam_size=5, condition_on_previous_text=False)


def test_session_stays_on_cpu_after_fallback(whisper_model):
    cpu_model = _gpu_fails_cpu_works(whisper_model)
    t = Transcriber()
    t.transcribe("a.ogg", gpu_settings(), None)
    cpu_model.transcribe.return_value = (iter([Seg("second")]), Info("ro"))

    text, _ = t.transcribe("b.ogg", gpu_settings(), None)

    assert text == "second"
    assert [c.kwargs["device"] for c in whisper_model.call_args_list] == ["cuda", "cpu"]


def test_error_on_cpu_is_raised(whisper_model):
    whisper_model.return_value.transcribe.side_effect = RuntimeError("bad audio")
    with pytest.raises(RuntimeError, match="bad audio"):
        Transcriber().transcribe("a.ogg", gpu_settings(device="cpu"), None)
    assert whisper_model.return_value.transcribe.call_count == 1


def test_error_after_cpu_fallback_is_raised(whisper_model):
    whisper_model.return_value.transcribe.side_effect = RuntimeError("bad audio")
    with pytest.raises(RuntimeError, match="bad audio"):
        Transcriber().transcribe("a.ogg", gpu_settings(), None)
    # exactly one GPU attempt + one CPU attempt
    assert whisper_model.return_value.transcribe.call_count == 2


def test_cuda_dlls_registered_on_windows(monkeypatch, tmp_path):
    for lib in ("cublas", "cudnn"):
        (tmp_path / lib / "bin").mkdir(parents=True)
    nvidia = types.ModuleType("nvidia")
    nvidia.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "nvidia", nvidia)
    monkeypatch.setattr(sys, "platform", "win32")
    added = []
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setenv("PATH", "C:\\old")

    transcription.register_cuda_dlls()

    expected = [str(tmp_path / "cublas" / "bin"), str(tmp_path / "cudnn" / "bin")]
    assert added == expected
    assert all(p in os.environ["PATH"].split(os.pathsep) for p in expected)


def test_cuda_dlls_noop_outside_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "add_dll_directory", mock.Mock(), raising=False)
    transcription.register_cuda_dlls()
    os.add_dll_directory.assert_not_called()
