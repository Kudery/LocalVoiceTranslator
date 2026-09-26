# Changelog

## 1.0.0

First public release.

- Transcribe audio locally with faster-whisper (`large-v3-turbo` by default)
- NVIDIA GPU support via pip-installed CUDA 12 libraries (no CUDA Toolkit needed), with automatic fallback to CPU
- Translate into 23 languages with any Ollama model, or transcribe only
- Auto-detect or force the source language; Ollama is skipped when the audio is already in the target language
- Reads `.ogg`/`.opus` (WhatsApp, Telegram), `.mp3`, `.m4a`, `.wav`, `.flac` and more
- Copy or save the original and the translation
- Windows scripts: `setup.bat`, `start.bat`, `create-shortcut.ps1`

**Install:** download `Source code (zip)` below, extract it, double-click `setup.bat`, then `start.bat`. See the [README](https://github.com/Kudery/LocalVoiceTranslator#readme) for requirements and troubleshooting.
