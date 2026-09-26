"""Transcription with faster-whisper, with model caching and automatic CPU fallback."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "large-v3-turbo"


@dataclass
class TranscriptionSettings:
    model_size: str = DEFAULT_MODEL
    device: str = "auto"
    compute_type: str = "auto"
    model_dir: str = ""
    vad: bool = True


def register_cuda_dlls():
    """Make the DLLs from nvidia-cublas-cu12 / nvidia-cudnn-cu12 findable (Windows only).

    pip installs them in site-packages/nvidia/<lib>/bin, which Windows does not search.
    Must run before ctranslate2 (via faster_whisper) is imported.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return  # packages not installed -> CUDA fails later and the CPU fallback kicks in
    for base in getattr(nvidia, "__path__", []):
        for lib in ("cublas", "cudnn"):
            bin_dir = Path(base) / lib / "bin"
            if not bin_dir.is_dir():
                continue
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
            # cuDNN loads its sub-libraries through the regular search order -> PATH too
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


class Transcriber:
    """Keeps one WhisperModel cached and falls back to CPU/int8 when the GPU fails."""

    def __init__(self, notify=None):
        self._notify = notify or (lambda kind, value: None)
        self._model = None
        self._model_sig = None      # (size, device, compute_type) of the loaded model
        self._forced_cpu = False    # set once the GPU fails -> CPU for the rest of the session

    @property
    def forced_cpu(self):
        return self._forced_cpu

    def _model_settings(self, settings):
        size = settings.model_size.strip() or DEFAULT_MODEL
        device = "cpu" if self._forced_cpu else settings.device
        compute = "int8" if self._forced_cpu else settings.compute_type
        return size, device, compute

    def _load_model(self, size, device, compute, model_dir):
        register_cuda_dlls()
        from faster_whisper import WhisperModel
        sig = (size, device, compute)
        if self._model is not None and self._model_sig == sig:
            return self._model
        self._notify("status", f"Loading model ({size}, {device})...")
        kwargs = {"device": device, "compute_type": compute}
        model_dir = model_dir.strip()
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            kwargs["download_root"] = model_dir
        model = WhisperModel(size, **kwargs)
        self._model = model
        self._model_sig = sig
        return model

    @staticmethod
    def _run(model, audio_path, language, vad):
        segments, info = model.transcribe(
            audio_path,
            language=language,          # None = auto-detect
            task="transcribe",
            vad_filter=vad,
            beam_size=5,
            condition_on_previous_text=False,
        )
        # segments is a generator: the actual (GPU) inference happens here,
        # so this must stay inside the try block in transcribe().
        return "".join(seg.text for seg in segments).strip(), info

    def transcribe(self, audio_path, settings, language):
        size, device, compute = self._model_settings(settings)
        vad = bool(settings.vad)
        try:
            model = self._load_model(size, device, compute, settings.model_dir)
            return self._run(model, audio_path, language, vad)
        except Exception as e:
            if device == "cpu":
                raise
            # GPU not usable (missing CUDA libs, unsupported card, out of memory, ...):
            # fall back to CPU/int8 for the rest of the session.
            self._notify("status", f"GPU error ({str(e)[:60]}) - falling back to CPU (int8)...")
            self._forced_cpu = True
            self._model = None
            model = self._load_model(size, "cpu", "int8", settings.model_dir)
            return self._run(model, audio_path, language, vad)
